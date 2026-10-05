"""Frozen-checkpoint empirical calibration comparison; no optimizer or training."""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from toy_dataset import FourierCropDataset
from models.convnext import QuantileConvRecon
from interval_calibration import fit, apply, metrics

METHODS=['raw','repaired','additive','shared','asymmetric']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/motion_uniform_gn/best_model.pth')
    p.add_argument('--data',default='/share/monakhova/shamus_data/parallel_100k')
    p.add_argument('--out',default='results/toy_calibration')
    p.add_argument('--count',type=int,default=1024)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=OmegaConf.create(ck['config'])
    m=QuantileConvRecon(3,64,64,model_size=c.model.size,drop_path_rate=c.model.drop_path_rate,layer_scale_init_value=c.model.layer_scale_init_value,decoder_channels=c.model.decoder_channels,quantile_normalization=c.quantile.normalization,quantile_embedding_dim=c.quantile.embedding_dim,quantile_condition_channels=c.quantile.condition_channels).cuda()
    m.load_state_dict(ck['model_state_dict'],strict=True);m.eval();m.requires_grad_(False)
    old_start=int(c.toy.source_offset)
    old_count=sum(int(c.dataset[k]) for k in ('train_count','val_count','test_count'))
    def ds(offset,count,seed):
        return FourierCropDataset(a.data,offset,count,64,48,seed,psf=c.toy.psf)
    old=ds(old_start,old_count,0)
    calibration=ds(24000,a.count,30260920)
    testing=ds(24000+a.count,a.count,40260920)
    sets=[set(d.files) for d in (old,calibration,testing)]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Source split overlap')
    manifest={'previous_train_val_test':old.files,'calibration':calibration.files,'test':testing.files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def predict(d,role):
        cache={k:[] for k in ('x','r','lo','hi')};examples=[]
        with torch.inference_mode():
            for batch,(y,x,names) in enumerate(DataLoader(d,batch_size=32,num_workers=2)):
                y=y.cuda();q=torch.full((len(y),),.05,device='cuda')
                r,(lo,hi)=m.forward_quantiles(y,[q,torch.full_like(q,.95)])
                for k,v in dict(x=x,r=r,lo=lo,hi=hi).items():cache[k].append(v.cpu().numpy())
                if batch==0:examples=y[:10].cpu().numpy()
                if batch%8==0:print(role,batch*32,'/',len(d),flush=True)
        cache={k:np.concatenate(v) for k,v in cache.items()}
        for k,v in cache.items():np.save(out/f'{role}_{k}.npy',v)
        return cache,examples
    cal,_=predict(calibration,'calibration')
    t=time.monotonic();params=fit(**cal);fit_seconds=time.monotonic()-t
    (out/'calibration.json').write_text(json.dumps(params,indent=2))
    cal_metrics={method:metrics(cal['x'],*apply(cal['r'],cal['lo'],cal['hi'],params,method)) for method in METHODS}
    del cal
    test,examples=predict(testing,'test')
    test_metrics={method:metrics(test['x'],*apply(test['r'],test['lo'],test['hi'],params,method)) for method in METHODS}
    mse=float(np.mean((test['r']-test['x'])**2))
    report={'checkpoint':a.checkpoint,'epoch':int(ck['epoch'])+1,'checkpoint_sha256':hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),
            'new_training_seconds':0,'fit_seconds':fit_seconds,'seconds':time.monotonic()-start,
            'parameters':params,'calibration':cal_metrics,'test':test_metrics,
            'reconstruction_psnr_all_methods':float(-10*np.log10(mse)),
            'notes':['Empirical calibration, not certified population coverage.','Equal-sized images: each image has equal weight.','No clipping of interval bounds; metrics evaluate all RGB values.','Repaired control anchors bounds at reconstruction with width floor 0.001.','Fresh contiguous source blocks; source duplicates or acquisition dependence may remain.','First ten test images selected before examining results.']}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'examples.npz',y=examples,**{k:v[:10] for k,v in test.items()})
    make_pdf(out,report,test,examples)
    print(json.dumps(report,indent=2),flush=True)

def make_pdf(out,report,test,y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    p=report['parameters']
    with PdfPages(out/'comparison.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        rows=[[k]+[f'{report["test"][k][v]:.4f}' for v in ('coverage','lower_tail','upper_tail','mean_width','image_coverage_p10')] for k in METHODS]
        ax.table(cellText=rows,colLabels=['Method','Coverage','Lower miss','Upper miss','Width','Image cov p10'],loc='center',cellLoc='center')
        ax.set_title('Toy motion-blur: frozen uniform-quantile model\nIndependent calibration / test comparison')
        fig.text(.08,.15,f'Calibration images: {p["calibration_images"]}; test images: {len(test["x"])}\nTarget: 90% mean pixel/channel coverage. Empirical, not a statistical certificate.\nAll methods have identical reconstruction. No additional training.\nFactors: shared={p["shared"]:.3f}, lower={p["lower"]:.3f}, upper={p["upper"]:.3f}; additive={p["additive"]:.4f}')
        pdf.savefig(fig);plt.close(fig)
        for i in range(min(10,len(y))):
            r,lo,hi=[test[k][i:i+1] for k in ('r','lo','hi')];x=test['x'][i:i+1]
            fig,axes=plt.subplots(3,5,figsize=(15,9))
            for ax in axes.flat:ax.axis('off')
            for j,(title,v) in enumerate([('Measurement',y[i]),('Target',x[0]),('Reconstruction',r[0])]):
                axes[0,j].imshow(v.transpose(1,2,0).clip(0,1));axes[0,j].set_title(title)
            axes[0,3].imshow(np.abs(r-x)[0].mean(0),vmin=0,vmax=.2,cmap='magma');axes[0,3].set_title('MAE (scale 0–0.2)')
            bounds=[apply(r,lo,hi,p,k) for k in METHODS]
            vmax=max(float(np.maximum(h-l,0).mean(1).max()) for l,h in bounds)
            for j,(method,(l,h)) in enumerate(zip(METHODS,bounds)):
                im=axes[1,j].imshow(np.maximum(h-l,0)[0].mean(0),vmin=0,vmax=vmax,cmap='magma');axes[1,j].set_title(method+' width')
                covered=((l<=x)&(x<=h))[0].mean(0)
                axes[2,j].imshow(1-covered,vmin=0,vmax=1,cmap='gray');axes[2,j].set_title(f'Miss fraction; coverage {covered.mean():.1%}')
            fig.colorbar(im,ax=list(axes[1]),shrink=.6,label='Mean RGB interval width')
            fig.suptitle(f'Test example {i+1}: same width color scale across methods; white = missed channels')
            pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':main()
