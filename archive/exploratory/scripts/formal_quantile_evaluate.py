"""Calibrate fixed and q-conditioned bounds identically; evaluate paired test."""
import sys,json,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
ROOT=Path('results/formal_quantile_comparison')
TEST=Path('data/formal_rml_25k');CAL=Path('data/formal_rml_calibration95')

def infer(kind):
    import torch
    from models.shamus_rml import ShamusRML
    from models.formal_quantile import QuantileRML
    torch.set_num_threads(2)
    model=QuantileRML(ShamusRML(),kind).cuda()
    ck=torch.load(ROOT/kind/'best.pt',map_location='cpu',weights_only=False)
    assert ck['manifest_sha256']==hashlib.sha256((TEST/'manifest.json').read_bytes()).hexdigest()
    model.load_state_dict(ck['model'],strict=True);model.eval()
    for role,path,offset,count in [('calibration',CAL,0,2048),('test',TEST,22500,2500)]:
        pairs=np.load(path/'pairs.npy',mmap_mode='r')
        outputs={k:np.lib.format.open_memmap(ROOT/kind/f'{role}_{k}.npy',mode='w+',dtype=np.float32,shape=(count,3,256,256)) for k in ('r','lo','hi')}
        with torch.inference_mode():
            for s in range(0,count,2):
                y=torch.from_numpy(np.array(pairs[offset+s:offset+s+2,0],copy=True)).float().cuda()/255
                with torch.autocast('cuda',dtype=torch.bfloat16):r,lo,hi=model.bounds(y)
                for k,v in zip(outputs,(r,lo,hi)):outputs[k][s:s+2]=v.float().cpu().numpy()
        for a in outputs.values():a.flush()
    (ROOT/kind/'inference.json').write_text(json.dumps({'selected_epoch':ck['epoch'],'kind':kind,'train_manifest_sha256':ck['manifest_sha256'],'calibration_manifest_sha256':hashlib.sha256((CAL/'manifest.json').read_bytes()).hexdigest()},indent=2))

def arrays(kind,role):return {k:np.load(ROOT/kind/f'{role}_{k}.npy',mmap_mode='r') for k in ('r','lo','hi')}
def widths(r,lo,hi):return np.maximum(r-lo,.001),np.maximum(hi-r,.001)

def report():
    factors={};meta={};examples={};calpairs=np.load(CAL/'pairs.npy',mmap_mode='r')
    for kind in ('fixed','conditioned'):
        meta[kind]=json.loads((ROOT/kind/'inference.json').read_text())
        assert meta[kind]['train_manifest_sha256']==hashlib.sha256((TEST/'manifest.json').read_bytes()).hexdigest()
        assert meta[kind]['calibration_manifest_sha256']==hashlib.sha256((CAL/'manifest.json').read_bytes()).hexdigest()
        a=arrays(kind,'calibration')
        scores=np.lib.format.open_memmap(ROOT/kind/'calibration_scores.npy',mode='w+',dtype=np.float32,shape=a['r'].shape)
        for s in range(0,2048,8):
            r,lo,hi=[a[k][s:s+8] for k in ('r','lo','hi')];x=calpairs[s:s+8,1].astype(np.float32)/255
            wl,wu=widths(r,lo,hi);v=np.maximum((r-x)/wl,(x-r)/wu)
            if not np.isfinite(v).all():raise ValueError('Invalid calibration scores')
            scores[s:s+8]=v
        scores.flush();factors[kind]=max(0.,float(np.quantile(scores,.95,method='higher',overwrite_input=True)));del scores
    (ROOT/'calibration_factors.json').write_text(json.dumps({'target':.95,'factors':factors,'width_floor':.001,'method':'empirical shared-factor scaling; no RCPS certificate'},indent=2))
    metrics={}
    for role,path,offset,count in [('calibration',CAL,0,2048),('test',TEST,22500,2500)]:
        pairs=np.load(path/'pairs.npy',mmap_mode='r');metrics[role]={}
        for kind in factors:
            a=arrays(kind,role);sums={k:np.zeros(6) for k in ('raw','repaired','calibrated95')};cover={k:[] for k in sums};mse=0.;pb=0.
            for s in range(0,count,8):
                e=min(s+8,count);r,lo,hi=[a[k][s:e] for k in ('r','lo','hi')];x=pairs[offset+s:offset+e,1].astype(np.float32)/255
                wl,wu=widths(r,lo,hi);cl,ch=r-factors[kind]*wl,r+factors[kind]*wu
                mse+=np.mean((r-x)**2)*(e-s)
                for p,q in ((lo,.025),(hi,.975)):
                    err=x-p;pb+=.5*np.maximum(q*err,(q-1)*err).mean()*(e-s)
                for name,(l,h) in {'raw':(lo,hi),'repaired':(r-wl,r+wu),'calibrated95':(cl,ch)}.items():
                    cv=((l<=x)&(x<=h)).mean((1,2,3));cover[name].extend(cv.tolist())
                    sums[name]+=np.array([cv.mean(),np.maximum(h-l,0).mean(),(x<l).mean(),(x>h).mean(),(l>h).mean(),np.mean((r-x)**2)])*(e-s)
                if role=='test' and s==0:examples[kind]={'r':r,'x':x,'raw_width':np.maximum(hi-lo,0),'width':ch-cl,'miss':~((cl<=x)&(x<=ch))}
            metrics[role][kind]={'psnr':float(-10*np.log10(mse/count)),'endpoint_pinball':float(pb/count),'intervals':{k:dict(zip(('coverage','width','lower_miss','upper_miss','crossing','mse'),(v/count).tolist()))|{'image_coverage_p10':float(np.quantile(cover[k],.1))} for k,v in sums.items()}}
    payload={'target':.95,'factors':factors,'metrics':metrics,'selected_checkpoints':meta,'notes':['ConvRML adaptation of fixed-endpoint im2im quantile regression, not an exact paper reproduction.','Fixed baseline uses 20 normal reconstruction epochs plus 20 head-only epochs; conditioned model uses 20 joint epochs. Training budgets differ.','Same 20k training, 2500 validation, 2048 calibration and 2500 test images.','Both use identical empirical calibration; no guaranteed population coverage.','Final-layer fixed head leaves the normal MSE model frozen; conditioned training updates reconstruction as well.']}
    (ROOT/'comparison.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(ROOT/'comparison.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,7));ax.axis('off');rows=[]
        for kind in factors:
            m=metrics['test'][kind];v=m['intervals']['calibrated95'];rows.append([kind,f'{m["psnr"]:.2f}',f'{m["endpoint_pinball"]:.5f}',f'{v["coverage"]:.2%}',f'{v["width"]:.5f}'])
        ax.table(cellText=rows,colLabels=['Model','PSNR','Raw pinball','Calibrated coverage','Width'],cellLoc='center',loc='center')
        ax.set_title('Fixed-endpoint im2im adaptation vs general q-conditioned ConvRML')
        fig.text(.1,.13,'Independent empirical 95% calibration. No test-driven factor adjustment.\nDifferent training protocols; compare uncertainty and reconstruction separately.')
        pdf.savefig(fig);plt.close(fig)
        for i in range(8):
            fig,axes=plt.subplots(2,5,figsize=(16,7))
            vmax=max(float(e['width'][i].mean(0).max()) for e in examples.values())
            for row,kind in enumerate(factors):
                e=examples[kind]
                for col,(title,a) in enumerate([('Target',e['x'][i]),('Reconstruction',e['r'][i]),('Absolute error',np.abs(e['r'][i]-e['x'][i]).mean(0)),('95% interval width',e['width'][i].mean(0)),('Miss fraction',e['miss'][i].mean(0))]):
                    ax=axes[row,col];ax.axis('off');ax.set_title(kind+': '+title)
                    if a.ndim==3:ax.imshow(a.transpose(1,2,0).clip(0,1))
                    else:ax.imshow(a,cmap='magma',vmin=0,vmax=vmax if col==3 else (1 if col==4 else .2))
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','report']);p.add_argument('--kind',choices=['fixed','conditioned']);a=p.parse_args()
    if a.stage=='infer':
        if not a.kind:p.error('--kind required for inference')
        infer(a.kind)
    else:report()
