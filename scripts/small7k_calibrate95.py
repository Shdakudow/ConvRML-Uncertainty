"""Independent pooled-pixel 95% interval calibration of the 5k experiment."""
import argparse,json,os,sys,time,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.small7k_train import ROOT,DATA,KINDS,Pairs,model_for,sha
from scripts.small7k_eval import PR,PIXELS,pixel_indices,mixture_quantiles
from baselines.mc_dropout import dropout_inference

OUT=ROOT/'calibrated95';CD=Path('data/formal_rml_calibration95')
METHODS={'ensemble_nll':list(range(5)),'ensemble_crps':list(range(9,14)),
         'mc_mse':[5],'mc_crps':[14],'q_conditioned':[7],'fixed_head':[8]}

def audit():
    cal=json.loads((CD/'manifest.json').read_text())['calibration']
    split=json.loads((DATA/'manifest.json').read_text())
    used=split['train']+split['val']+split['test']
    assert len(cal)==2048
    assert not ({v['id'] for v in cal}&{v['id'] for v in used})
    assert not ({v['target_sha256'] for v in cal}&{v['target_sha256'] for v in used})
    return {'calibration_manifest_sha256':sha(CD/'manifest.json'),'training_manifest_sha256':sha(DATA/'manifest.json'),
            'calibration_images':2048,'test_images':1000,'pixel_channel_samples_per_image':PIXELS,
            'alpha':.05,'method':'One pooled empirical 95th-percentile factor per method, multiplying ordered interval half-width about its midpoint; minimum half-width 1e-6.',
            'scope':'Independent calibration images; no test fitting, no weight changes. Marginal sampled-pixel coverage target, not a per-image guarantee. Correlated pixels do not supply independent conformal samples. Only intervals calibrated; no assertion of improved full-distribution CRPS.'}

@torch.inference_mode()
def infer(worker):
    metadata=audit();torch.set_num_threads(2);torch.manual_seed(54321)
    kind=KINDS[worker];out=OUT/f'worker_{worker}';out.mkdir(parents=True,exist_ok=True)
    model=model_for(worker);ck=torch.load(ROOT/f'worker_{worker}/best.pt',map_location='cpu',weights_only=False)
    assert ck['manifest_sha256']==metadata['training_manifest_sha256']
    assert ck['epoch']==json.loads((PR/f'worker_{worker}/complete.json').read_text())['selected_epoch']
    epoch=ck['epoch'];model.load_state_dict(ck['model']);del ck;model.cuda().eval()
    pairs=np.load(CD/'pairs.npy',mmap_mode='r')
    count=30 if kind.startswith('mc') else 2
    values=np.lib.format.open_memmap(out/'distribution.npy',mode='w+',dtype=np.float32,shape=(2048,count,PIXELS))
    start=time.monotonic()
    for s in range(0,2048,2):
        y=torch.from_numpy(np.array(pairs[s:s+2,0],copy=True)).cuda().float()/255
        idx=torch.from_numpy(pixel_indices(s,2)).cuda()
        def sample(v):return torch.gather(v.reshape(2,-1),1,idx)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if kind.startswith('ensemble'):
                mean,var=model(y);v=torch.stack([sample(mean.float()),sample(var.float())],1)
            elif kind.startswith('mc'):
                with dropout_inference(model):v=torch.stack([sample(model(y).float()) for _ in range(30)],1)
            else:
                f,_=model.features(y)
                if kind=='fixed':b=model.predict(y,f).float();v=torch.stack([sample(b[:,i]) for i in range(2)],1)
                else:v=torch.stack([sample(model.predict(y,f,y.new_full((2,),q)).float()) for q in [.025,.975]],1)
        assert torch.isfinite(v).all();values[s:s+2]=v.cpu().numpy()
        if s%400==0:print(worker,s,flush=True)
    values.flush();(out/'complete.json').write_text(json.dumps({**metadata,'worker':worker,'epoch':epoch,'seconds':time.monotonic()-start},indent=2))

def intervals(name,arrays,start,end):
    d=torch.from_numpy(np.stack([v[start:end] for v in arrays]))
    if name.startswith('ensemble'):q=mixture_quantiles(d[:,:,0],d[:,:,1],[.025,.975]);lo,hi=q[:,0],q[:,1]
    elif name.startswith('mc'):
        q=torch.quantile(d[0],torch.tensor([.025,.975]),dim=1);lo,hi=q[0],q[1]
    else:lo=d[0].min(1).values;hi=d[0].max(1).values
    return (lo+hi)/2,((hi-lo)/2).clamp_min(1e-6)

def report():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);protocol=audit()
    cp=np.load(CD/'pairs.npy',mmap_mode='r');dataset=Pairs('test')
    original=json.loads((ROOT/'comparison.json').read_text())
    factors={};metrics={};records=[]
    for name,workers in METHODS.items():
        arrays=[]
        for worker in workers:
            meta=json.loads((OUT/f'worker_{worker}/complete.json').read_text())
            assert meta['calibration_manifest_sha256']==protocol['calibration_manifest_sha256']
            arrays.append(np.load(OUT/f'worker_{worker}/distribution.npy',mmap_mode='r'))
        scores=[]
        for start in range(0,2048,8):
            end=min(start+8,2048);mid,rad=intervals(name,arrays,start,end)
            x=cp[start:end,1].astype(np.float32).reshape(end-start,-1)/255
            target=torch.from_numpy(np.take_along_axis(x,pixel_indices(start,end-start),axis=1))
            scores.append(((target-mid).abs()/rad).numpy())
        factor=float(np.quantile(np.concatenate(scores),.95,method='higher'));assert np.isfinite(factor)
        factors[name]=factor
    # Freeze every factor before touching test targets.
    (OUT/'calibration.json').write_text(json.dumps({'protocol':protocol,'factors':factors},indent=2))
    for name,workers in METHODS.items():
        arrays=[np.load(PR/f'worker_{w}'/('endpoints.npy' if name=='q_conditioned' else 'distribution.npy'),mmap_mode='r') for w in workers]
        rows=[]
        for start in range(0,1000,8):
            end=min(start+8,1000);mid,rad=intervals(name,arrays,start,end)
            x=torch.stack([dataset[i][1] for i in range(start,end)]).reshape(end-start,-1)
            target=torch.gather(x,1,torch.from_numpy(pixel_indices(start,end-start)))
            lo=mid-factors[name]*rad;hi=mid+factors[name]*rad
            coverage=((lo<=target)&(target<=hi)).float().mean(1);width=(hi-lo).mean(1)
            score=(hi-lo)+40*(lo-target).clamp_min(0)+40*(target-hi).clamp_min(0)
            for i in range(end-start):
                row={'method':name,'test_index':start+i,'coverage95':coverage[i].item(),'width95':width[i].item(),'interval_score95':score[i].mean().item()};rows.append(row);records.append(row)
        metrics[name]={k:float(np.mean([r[k] for r in rows])) for k in ['coverage95','width95','interval_score95']}
        metrics[name].update(factor=factors[name],psnr=original['metrics'][name]['psnr'],ssim=original['metrics'][name]['ssim'],raw_coverage95=original['metrics'][name]['coverage95'],raw_width95=original['metrics'][name]['width95'])
        print(name,json.dumps(metrics[name]),flush=True)
    (OUT/'comparison.json').write_text(json.dumps({'protocol':protocol,'metrics':metrics},indent=2))
    with (OUT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 5k experiment: independent 95% interval calibration','',protocol['scope'],'',
           '| Method | Factor | Raw coverage | Calibrated coverage | Raw width | Calibrated width |','|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append(f"| {name} | {m['factor']:.4f} | {m['raw_coverage95']:.2%} | {m['coverage95']:.2%} | {m['raw_width95']:.5f} | {m['width95']:.5f} |")
    lines+=['','PSNR and SSIM are unchanged. Coverage/width estimated at 1024 shared sampled RGB pixel-channel values per test image. Factors fitted on 2048 separate images, no test tuning.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    examples=dict(np.load(ROOT/'examples.npz'))
    for name,workers in METHODS.items():
        ex=[np.load(PR/f'worker_{w}/examples.npz') for w in workers]
        if name.startswith('ensemble'):
            mu=torch.from_numpy(np.stack([v['mean'] for v in ex])).reshape(len(workers),2,-1)
            var=torch.from_numpy(np.stack([v['variance'] for v in ex])).reshape(len(workers),2,-1)
            q=mixture_quantiles(mu,var,[.025,.975]);rad=((q[:,1]-q[:,0])/2).clamp_min(1e-6).reshape(2,3,256,256).numpy()
        else:rad=np.maximum(np.abs(ex[0]['hi']-ex[0]['lo'])/2,1e-6)
        examples[name+'_width']=(2*factors[name]*rad).mean(1)
    np.savez_compressed(OUT/'examples.npz',**examples)
    import scripts.small7k_eval as evaluation
    evaluation.ROOT=OUT
    evaluation.render(examples,METHODS,title='5k models: independent calibration toward 95% coverage',width_label='Calibrated 95% interval width')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','infer','report']);p.add_argument('--worker',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)));a=p.parse_args()
    if a.action=='audit':print(json.dumps(audit(),indent=2))
    elif a.action=='infer':infer(a.worker)
    else:report()
