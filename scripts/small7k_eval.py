"""Native uncertainty, including CRPS and its numerical approximation details."""
import argparse,csv,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.small7k_train import ROOT,DATA,Pairs,KINDS,model_for,sha,gaussian_crps,sample_crps
from baselines.mc_dropout import dropout_inference

PR=ROOT/'predictions'
Q=(torch.arange(31,dtype=torch.float32)+.5)/31
PIXELS=1024

def pixel_indices(start,count):
    return np.stack([np.random.default_rng(17001+i).choice(3*256*256,PIXELS,replace=False) for i in range(start,start+count)])

@torch.inference_mode()
def infer(worker):
    torch.set_num_threads(2);torch.manual_seed(12345)
    kind=KINDS[worker];out=PR/f'worker_{worker}';out.mkdir(parents=True,exist_ok=True)
    model=model_for(worker);ck=torch.load(ROOT/f'worker_{worker}/best.pt',map_location='cpu',weights_only=False)
    assert ck['manifest_sha256']==sha(DATA/'manifest.json')
    epoch=ck['epoch'];model.load_state_dict(ck['model']);del ck;model.cuda().eval()
    dataset=Pairs('test');n=len(dataset)
    mean=np.lib.format.open_memmap(out/'mean.npy',mode='w+',dtype=np.float32,shape=(n,3,256,256))
    channels=30 if kind.startswith('mc') else len(Q) if kind=='conditioned' else 2
    dist=np.lib.format.open_memmap(out/'distribution.npy',mode='w+',dtype=np.float32,shape=(n,channels,PIXELS))
    endpoints=np.lib.format.open_memmap(out/'endpoints.npy',mode='w+',dtype=np.float32,shape=(n,2,PIXELS)) if kind=='conditioned' else None
    examples={};seconds=0.;position=0
    from torch.utils.data import DataLoader
    for batch in DataLoader(dataset,batch_size=2,num_workers=2):
        y=batch[:,0].cuda();count=len(y);idx=torch.from_numpy(pixel_indices(position,count)).cuda()
        def sample(v):return torch.gather(v.reshape(count,-1),1,idx)
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if kind.startswith('ensemble'):
                r,var=model(y);r,var=r.float(),var.float();values=torch.stack([sample(r),sample(var)],1)
                if position==0:examples={'mean':r.cpu().numpy(),'variance':var.cpu().numpy()}
            elif kind.startswith('mc'):
                with dropout_inference(model):draws=torch.stack([model(y).float() for _ in range(30)])
                r=draws.mean(0);values=torch.stack([sample(d) for d in draws],1)
                if position==0:examples={'mean':r.cpu().numpy(),'lo':torch.quantile(draws,.025,dim=0).cpu().numpy(),'hi':torch.quantile(draws,.975,dim=0).cpu().numpy()}
            else:
                f,r=model.features(y);r=r.float()
                if kind=='fixed':
                    bounds=model.predict(y,f).float();values=torch.stack([sample(bounds[:,j]) for j in range(2)],1)
                else:
                    values=torch.stack([sample(model.predict(y,f,y.new_full((count,),float(q))).float()) for q in Q],1)
                    bounds=torch.stack([model.predict(y,f,y.new_full((count,),q)).float() for q in [.025,.975]],1)
                    endpoints[position:position+count]=torch.stack([sample(bounds[:,j]) for j in range(2)],1).cpu().numpy()
                if position==0:examples={'mean':r.cpu().numpy(),'lo':bounds[:,0].cpu().numpy(),'hi':bounds[:,1].cpu().numpy()}
        torch.cuda.synchronize();seconds+=time.perf_counter()-tick
        assert torch.isfinite(r).all() and torch.isfinite(values).all()
        mean[position:position+count]=r.cpu().numpy();dist[position:position+count]=values.cpu().numpy();position+=count
        if position%200==0:print(kind,position,flush=True)
    mean.flush();dist.flush()
    if endpoints is not None:endpoints.flush()
    np.savez_compressed(out/'examples.npz',**examples)
    (out/'complete.json').write_text(json.dumps({'kind':kind,'selected_epoch':epoch,'count':n,'sampled_pixel_channels_per_image':PIXELS,'inference_compute_seconds':seconds,'manifest_sha256':sha(DATA/'manifest.json'),'q_midpoints':Q.tolist() if kind=='conditioned' else None},indent=2))

def gaussian_mixture_crps(means,variances,x):
    # E|X-x| - .5 E|X-X'|; preserve each member's Gaussian variance.
    def absolute_normal(delta,sigma):
        z=delta/sigma
        return sigma*(2*torch.exp(-.5*z.square())/(2*torch.pi)**.5+z*torch.erf(z/2**.5))
    first=absolute_normal(means-x,variances.clamp_min(1e-8).sqrt()).mean(0)
    second=torch.zeros_like(first)
    for j in range(len(means)):
        for k in range(len(means)):
            second+=absolute_normal(means[j]-means[k],(variances[j]+variances[k]).clamp_min(1e-8).sqrt())/(2*len(means)**2)
    return first-second

def mixture_quantiles(means,variances,levels):
    sigma=variances.clamp_min(1e-8).sqrt()
    lower=(means-10*sigma).min(0).values;upper=(means+10*sigma).max(0).values
    result=[]
    for q in levels:
        lo,hi=lower.clone(),upper.clone()
        for _ in range(30):
            mid=(lo+hi)/2;cdf=(.5*(1+torch.erf((mid-means)/(sigma*2**.5)))).mean(0)
            lo=torch.where(cdf<q,mid,lo);hi=torch.where(cdf>=q,mid,hi)
        result.append((lo+hi)/2)
    return torch.stack(result,1)

def evaluate():
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    torch.set_num_threads(4);dataset=Pairs('test');manifest=json.loads((DATA/'manifest.json').read_text())
    methods={'ensemble_nll':list(range(5)),'ensemble_crps':list(range(9,14)),'mc_mse':[5],'mc_crps':[14],'q_conditioned':[7],'fixed_head':[8]}
    metrics={};records=[];examples={};levels=[.025,.1,.25,.5,.75,.9,.975]
    for name,workers in methods.items():
        metadata=[json.loads((PR/f'worker_{w}/complete.json').read_text()) for w in workers]
        assert all(v['count']==1000 and v['manifest_sha256']==sha(DATA/'manifest.json') for v in metadata)
        means=[np.load(PR/f'worker_{w}/mean.npy',mmap_mode='r') for w in workers]
        distributions=[np.load(PR/f'worker_{w}/distribution.npy',mmap_mode='r') for w in workers]
        native=np.load(PR/f'worker_{workers[0]}/endpoints.npy',mmap_mode='r') if name=='q_conditioned' else None
        rows=[];hits=np.zeros(len(levels));nvalues=0;crossed=0;cross_total=0
        for start in range(0,1000,8):
            end=min(start+8,1000);n=end-start
            target=torch.stack([dataset[i][1] for i in range(start,end)])
            pred=torch.from_numpy(np.mean(np.stack([v[start:end] for v in means]),axis=0))
            idx=torch.from_numpy(pixel_indices(start,n));x=torch.gather(target.reshape(n,-1),1,idx)
            d=torch.from_numpy(np.stack([v[start:end] for v in distributions]))
            score=None;raw_score=None
            if name.startswith('ensemble'):
                mu,var=d[:,:,0],d[:,:,1];score=gaussian_mixture_crps(mu,var,x)
                qs=mixture_quantiles(mu,var,levels);lo,hi=qs[:,0],qs[:,-1]
            elif name.startswith('mc'):
                draws=d[0].transpose(0,1)
                score=sample_crps(draws,x)
                qs=torch.quantile(draws,torch.tensor(levels),dim=0).transpose(0,1);lo,hi=qs[:,0],qs[:,-1]
            elif name=='q_conditioned':
                qvalues=d[0];crossed+=(qvalues[:,1:]<qvalues[:,:-1]).sum().item();cross_total+=n*(len(Q)-1)*PIXELS
                residual=x[:,None]-qvalues
                raw_score=2*torch.maximum(Q[None,:,None]*residual,(Q[None,:,None]-1)*residual).mean(1)
                # Rearranged 31 midpoint quantiles define a discrete approximation.
                ordered=qvalues.sort(1).values;score=sample_crps(ordered.transpose(0,1),x)
                qs=torch.quantile(ordered,torch.tensor(levels),dim=1).transpose(0,1)
                raw_end=torch.from_numpy(np.array(native[start:end]));lo=raw_end.min(1).values;hi=raw_end.max(1).values
            else:
                bounds=d[0];crossed+=(bounds[:,0]>bounds[:,1]).sum().item();cross_total+=n*PIXELS
                lo,hi=bounds.min(1).values,bounds.max(1).values;qs=None
            if qs is not None:hits+=(x[:,None]<=qs).sum((0,2)).numpy();nvalues+=n*PIXELS
            psnr=-10*torch.log10((pred.double()-target.double()).square().mean((1,2,3)))
            sim=ssim(pred,target,data_range=1.,reduction='none')
            covered=((x>=lo)&(x<=hi)).float().mean(1);width=(hi-lo).mean(1)
            interval_score=(hi-lo)+40*(lo-x).clamp_min(0)+40*(x-hi).clamp_min(0)
            for j in range(n):
                row={'method':name,'id':manifest['test'][start+j]['id'],'psnr':psnr[j].item(),'ssim':sim[j].item(),'coverage95':covered[j].item(),'width95':width[j].item(),'interval_score95':interval_score[j].mean().item(),'crps':score[j].mean().item() if score is not None else None,'raw_quantile_integral':raw_score[j].mean().item() if raw_score is not None else None}
                rows.append(row);records.append(row)
        m={k:float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None for k in ['psnr','ssim','coverage95','width95','interval_score95','crps','raw_quantile_integral']}
        m.update(selected_epochs=[v['selected_epoch'] for v in metadata],quantile_crossing_rate=crossed/cross_total if cross_total else None,
                 quantile_calibration_mae=float(np.abs(hits/nvalues-np.array(levels)).mean()) if nvalues else None,
                 observed_cdf=(hits/nvalues).tolist() if nvalues else None,inference_compute_seconds=sum(v['inference_compute_seconds'] for v in metadata))
        metrics[name]=m
        ex=[np.load(PR/f'worker_{w}/examples.npz') for w in workers]
        pred=np.mean(np.stack([v['mean'] for v in ex]),axis=0)
        if name.startswith('ensemble'):
            mu=torch.from_numpy(np.stack([v['mean'] for v in ex]));var=torch.from_numpy(np.stack([v['variance'] for v in ex]))
            # Flatten image/channel dimensions only for this exact mixture interval.
            bounds=mixture_quantiles(mu.reshape(len(workers),2,-1),var.reshape(len(workers),2,-1),[.025,.975])
            lo=bounds[:,0].reshape(2,3,256,256).numpy();hi=bounds[:,1].reshape(2,3,256,256).numpy()
        else:lo=np.minimum(ex[0]['lo'],ex[0]['hi']);hi=np.maximum(ex[0]['lo'],ex[0]['hi'])
        examples.update({name+'_mean':pred,name+'_width':(hi-lo).mean(1)})
        print(name,json.dumps(m),flush=True)
    paired={}
    for prefix,baseline in [('ensemble','ensemble_nll'),('mc','mc_mse')]:
        a=np.array([r['crps'] for r in records if r['method']==prefix+'_crps'])
        b=np.array([r['crps'] for r in records if r['method']==baseline]);delta=a-b
        ix=np.random.default_rng(7103).integers(0,1000,(1000,1000))
        paired[prefix]={'crps_trained_minus_original':float(delta.mean()),'image_bootstrap95':np.quantile(delta[ix].mean(1),[.025,.975]).tolist()}
    protocol=json.loads((ROOT/'protocol.json').read_text())
    output={'protocol':protocol,'metrics':metrics,'paired_crps':paired,'evaluation_notes':['Raw native predictions; no calibration. Reconstruction metrics use all pixels; uncertainty metrics use 1024 identical sampled RGB values per image.',
              'Ensemble CRPS and quantiles use the full equal-weight Gaussian mixture, not a moment-matched Gaussian.',
              'MC evaluation is CRPS of its 30-draw empirical distribution; training uses a four-draw unbiased U-statistic estimator.',
              'Q CRPS uses a discrete approximation from sorted 31 midpoint quantiles. Raw integrated pinball and crossings are separately reported; q coverage/width use exact endpoint predictions with endpoint ordering repair.',
              'Fixed head has interval score only; no full distribution is assumed. All intervals nominal95; crossing endpoints reordered.',
              'Best native validation-loss checkpoint per model/stage; this is one model seed per paired run, five independent ensemble members. Bootstrap is across test images, not model seeds.']}
    (ROOT/'comparison.json').write_text(json.dumps(output,indent=2))
    with (ROOT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 5k training: original losses versus CRPS','','| Method | PSNR | SSIM | CRPS | Coverage95 | Width95 |','|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        score=f"{m['crps']:.6f}" if m['crps'] is not None else 'N/A (two endpoints)'
        lines.append(f"| {name} | {m['psnr']:.3f} | {m['ssim']:.4f} | {score} | {m['coverage95']:.2%} | {m['width95']:.5f} |")
    lines+=['',*output['evaluation_notes']];(ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    examples.update(measurement=torch.stack([dataset[i][0] for i in range(2)]).numpy(),target=torch.stack([dataset[i][1] for i in range(2)]).numpy())
    np.savez_compressed(ROOT/'examples.npz',**examples)
    render(examples,methods)

def render(examples,methods,title='5k / 1k / 1k: native versus CRPS training; no calibration',width_label='Raw 95% interval width'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    vmax=float(np.quantile(np.stack([examples[k+'_width'] for k in methods]),.995))
    with PdfPages(ROOT/'comparison_examples.pdf') as pdf:
        for i in range(2):
            fig,axes=plt.subplots(len(methods),4,figsize=(12,16))
            for row,name in enumerate(methods):
                for col,v in enumerate([examples['measurement'][i],examples['target'][i],examples[name+'_mean'][i],examples[name+'_width'][i]]):
                    ax=axes[row,col];ax.axis('off')
                    if col<3:ax.imshow(v.transpose(1,2,0).clip(0,1))
                    else:im=ax.imshow(v,cmap='magma',vmin=0,vmax=max(vmax,1e-6))
                    if row==0:ax.set_title(['Measurement','Ground truth','Prediction',width_label][col])
                axes[row,0].text(-.06,.5,name.replace('_','\n'),transform=axes[row,0].transAxes,ha='right',va='center')
            fig.suptitle(title)
            fig.subplots_adjust(left=.15,right=.93,bottom=.05,top=.95)
            fig.colorbar(im,ax=axes[:,-1],fraction=.025,pad=.02)
            pdf.savefig(fig);fig.savefig(ROOT/f'comparison_example_{i+1}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['infer','report']);p.add_argument('--worker',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)));a=p.parse_args()
    if a.action=='infer':infer(a.worker)
    else:evaluate()
