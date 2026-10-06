"""Export 20 real test examples, native quantiles and calibrated interval maps."""
import sys,json,base64,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML
from baselines.formal_rml import FormalRML
from baselines.mc_dropout import dropout_inference
from scripts.small7k_eval import mixture_quantiles
ROOT=Path('results/fixed_backbone50_20260927');OLD=Path('results/four_models_epoch50_20260925')
OUT=Path('results/uncertainty_viewer_20260928/data');OUT.mkdir(parents=True,exist_ok=True)
Q=[.05,.25,.5,.75,.95];N=20

def enc(a):
    a=np.asarray(a);assert np.isfinite(a).all()
    return base64.b64encode(a.astype('<f2').tobytes()).decode()
def load_model(net,path):
    ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==50
    net.load_state_dict(ck['model']);return net.cuda().eval()
def saved(worker,key):return np.load(ROOT/'predictions'/f'worker_{worker}'/f'test_{key}.npy',mmap_mode='r')[:N].copy()

@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.manual_seed(20260928)
    pairs=np.load('data/formal_rml_25k/pairs.npy',mmap_mode='r')[22500:22500+N].copy()
    ids=[v['id'] for v in json.loads(Path('data/formal_rml_25k/manifest.json').read_text())['test'][:N]]
    factors=json.loads((ROOT/'calibration.json').read_text())['factors']
    values={}
    r,lo,hi=[saved(6,k) for k in ['r','lo','hi']]
    values['fixed']={'mean':r,'endpoints':np.stack([lo,hi],1),'lo':r-factors['fixed']*np.maximum(r-lo,.001),'hi':r+factors['fixed']*np.maximum(hi-r,.001)}
    model=load_model(QuantileRML(ShamusRML(),'conditioned'),OLD/'quantiles/conditioned/latest.pt')
    qr=[];qm=[];qe=[]
    for i in range(N):
        y=torch.from_numpy(pairs[i:i+1,0].copy()).float().cuda()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f,r=model.features(y)
            qs=torch.stack([model.predict(y,f,y.new_full((1,),q)).float() for q in Q],1)
            ep=torch.stack([model.predict(y,f,y.new_full((1,),q)).float() for q in [.025,.975]],1)
        qm.append(r.float().cpu().numpy()[0]);qr.append(qs.cpu().numpy()[0]);qe.append(ep.cpu().numpy()[0])
    del model;torch.cuda.empty_cache()
    r=np.array(qm);ep=np.array(qe)
    values['conditioned']={'mean':r,'q':np.array(qr),'lo':r-factors['conditioned']*np.maximum(r-ep[:,0],.001),'hi':r+factors['conditioned']*np.maximum(ep[:,1]-r,.001)}
    print('Q-conditioned complete',flush=True)
    model=load_model(FormalRML('mc'),OLD/'baselines/worker_5/latest.pt');means=[];quant=[];stds=[]
    for i in range(N):
        y=torch.from_numpy(pairs[i:i+1,0].copy()).float().cuda()/255
        with torch.autocast('cuda',dtype=torch.bfloat16),dropout_inference(model):
            draws=torch.stack([model(y).float() for _ in range(30)])[:,0]
        means.append(draws.mean(0).cpu().numpy());stds.append(draws.std(0,unbiased=False).cpu().numpy())
        quant.append(torch.quantile(draws,torch.tensor(Q,device='cuda'),dim=0).cpu().numpy())
    del model;torch.cuda.empty_cache()
    r=np.array(means);std=np.maximum(np.array(stds),1e-4)
    values['mc']={'mean':r,'q':np.array(quant),'lo':r-factors['mc']*std,'hi':r+factors['mc']*std}
    print('MC complete',flush=True)
    means=np.stack([saved(w,'mean') for w in range(5)]);vars=np.stack([saved(w,'variance') for w in range(5)])
    r=means.mean(0);std=np.maximum(np.sqrt(vars.mean(0)+means.var(0)),1e-4)
    quant=[]
    for i in range(N):
        q=mixture_quantiles(torch.from_numpy(means[:,i:i+1]).cuda(),torch.from_numpy(vars[:,i:i+1]).cuda(),Q)
        quant.append(q[0].cpu().numpy())
    values['ensemble']={'mean':r,'q':np.array(quant),'lo':r-factors['ensemble']*std,'hi':r+factors['ensemble']*std}
    widths=np.concatenate([(v['hi']-v['lo']).mean(1).ravel() for v in values.values()])
    errors=np.concatenate([np.abs(v['mean']-pairs[:,1]/255).mean(1).ravel() for v in values.values()])
    meta={'count':N,'ids':ids,'quantiles':Q,'resolution':256,'widthMax':float(np.quantile(widths,.995)), 'errorMax':float(np.quantile(errors,.995)),
          'calibrationTarget':.9,'calibrationCount':2048,'aggregate':json.loads((ROOT/'comparison.json').read_text())['metrics'],
          'notes':'First 20 shared test images, no cherry-picking. Full precision metrics; float16 display arrays. RGB values are normalized to [0,1]; predictions are not clipped for metrics. MC: new reproducible 30-draw inference, seed 20260928. Quantile curves are raw, uncalibrated; interval maps use previously fitted empirical 90% factors. Fixed head has only native .025/.975 endpoints; optional linear interpolation is an assumption, not learned quantiles.'}
    for i in range(N):
        x=pairs[i,1].astype(np.float32)/255
        obj={'id':ids[i],'measurement':base64.b64encode(pairs[i,0].tobytes()).decode(),'target':base64.b64encode(pairs[i,1].tobytes()).decode(),'models':{}}
        for name,v in values.items():
            r=v['mean'][i];lo=v['lo'][i];hi=v['hi'][i]
            assert np.all(hi>=lo)
            obj['models'][name]={'mean':enc(r),'width':enc((hi-lo).mean(0)),
                'metrics':{'psnr':float(-10*np.log10(np.mean((r.astype(float)-x)**2))), 'ssim':float(ssim(torch.tensor(r[None]),torch.tensor(x[None]),data_range=1.)), 'coverage':float(((x>=lo)&(x<=hi)).mean()),'width':float((hi-lo).mean())}}
            key='endpoints' if name=='fixed' else 'q';obj['models'][name][key]=enc(v[key][i])
        (OUT/f'example_{i:02d}.js').write_text('window.receiveExample('+str(i)+','+json.dumps(obj,separators=(',',':'))+');')
    (OUT/'manifest.js').write_text('window.DATASET='+json.dumps(meta,separators=(',',':'))+';')
    (OUT/'protocol.json').write_text(json.dumps(meta,indent=2))
    print('Export complete',N,flush=True)
if __name__=='__main__':main()
