"""Independent empirical 95% interval calibration; never fit on test targets."""
import sys,json,hashlib,random,argparse,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

DATA=Path('data/formal_rml_calibration95')
OUT=Path('results/formal_rml_calibration95')
SOURCE=Path('results/formal_rml_25k_epoch20')
N=2048
Z95=1.959963984540054
STD_FLOOR=1e-4

def prepare():
    import torch
    from scripts.prepare_formal_rml import image,ROOT
    torch.set_num_threads(2)
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA/'manifest.json')
    DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('data/formal_rml_25k/manifest.json').read_text())
    previous=old['train']+old['val']+old['test']
    excluded={r['id'] for r in previous}|{5,6}
    excluded.update(r['id'] for r in old['rejected'])
    seen={r['target_sha256'] for r in previous}
    ids=[int(p.name.split('_')[2]) for p in (ROOT/'rml').glob('4x_img_*_cam_1.tiff')]
    ids=sorted(set(ids)-excluded);random.Random(20260924).shuffle(ids)
    pairs=np.lib.format.open_memmap(DATA/'pairs.npy',mode='w+',dtype=np.uint8,shape=(N,2,3,256,256))
    records=[];rejected=[]
    for i in ids:
        xp=ROOT/'gt2rml'/f'warped_4x_undistorted_img_{i}_cam_2.tiff';yp=ROOT/'rml'/f'4x_img_{i}_cam_1.tiff'
        try:
            x=image(xp);h=hashlib.sha256(x.tobytes()).hexdigest()
            if h in seen:continue
            y=image(yp)
        except Exception as e:rejected.append({'id':i,'reason':str(e)});continue
        pairs[len(records),0]=y;pairs[len(records),1]=x;seen.add(h)
        records.append({'id':i,'target_sha256':h,'measurement':str(yp),'target':str(xp)})
        if len(records)%256==0:print('CALIBRATION PAIRS',len(records),flush=True)
        if len(records)==N:break
    if len(records)!=N:raise RuntimeError('Insufficient disjoint calibration pairs')
    pairs.flush()
    (DATA/'manifest.json').write_text(json.dumps({'calibration':records,'rejected':rejected,'seed':20260924,'disjoint_from':'all 25000 original train/validation/test pairs, by ID and processed target SHA256'},indent=2))

def infer(worker):
    import torch
    from baselines.formal_rml import FormalRML
    from baselines.deep_ensemble import DeepEnsembleMember
    from baselines.mc_dropout import dropout_inference
    torch.set_num_threads(2);torch.manual_seed(20260924+worker)
    method='mc' if worker==5 else 'ensemble'
    net=FormalRML(method).cuda();model=DeepEnsembleMember(net).cuda() if method=='ensemble' else net
    ck=torch.load(SOURCE/f'worker_{worker}/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    pairs=np.load(DATA/'pairs.npy',mmap_mode='r')
    out=OUT/f'worker_{worker}';out.mkdir(parents=True,exist_ok=True)
    means=np.lib.format.open_memmap(out/'mean.npy',mode='w+',dtype=np.float32,shape=(N,3,256,256))
    variances=np.lib.format.open_memmap(out/'variance.npy',mode='w+',dtype=np.float32,shape=means.shape)
    with torch.inference_mode():
        for s in range(0,N,2):
            y=torch.from_numpy(np.array(pairs[s:s+2,0],copy=True)).float().cuda()/255
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if method=='ensemble':mu,var=model(y);mu,var=mu.float(),var.float()
                else:
                    with dropout_inference(net):
                        mu=None;m2=None
                        for j in range(30):
                            p=net(y).float()
                            if mu is None:mu=p.clone();m2=torch.zeros_like(p)
                            else:d=p-mu;mu=mu+d/(j+1);m2=m2+d*(p-mu)
                    var=m2/30
            means[s:s+2]=mu.cpu().numpy();variances[s:s+2]=var.cpu().numpy()
            if s%256==0:print(worker,'INFERENCE',s,'/',N,flush=True)
    means.flush();variances.flush()
    (out/'complete.json').write_text(json.dumps({'worker':worker,'epoch':ck['epoch'],'count':N,'manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()},indent=2))

def predictions(root,method):
    ids=range(5) if method=='ensemble' else [5]
    return ([np.load(root/f'worker_{i}/mean.npy',mmap_mode='r') for i in ids],
            [np.load(root/f'worker_{i}/variance.npy',mmap_mode='r') for i in ids])

def moments(pred,s,e):
    means,variances=pred
    m=np.stack([p[s:e] for p in means]);v=np.stack([p[s:e] for p in variances])
    mu=m.mean(0);var=v.mean(0)+m.var(0)
    if not np.isfinite(mu).all() or not np.isfinite(var).all() or (var<0).any():raise ValueError('Invalid moments')
    return mu,np.maximum(np.sqrt(var),STD_FLOOR)

def fit_factor(scores):
    if not np.isfinite(scores).all():raise ValueError('Nonfinite normalized errors')
    return float(np.quantile(scores,.95,method='higher',overwrite_input=True))

def report():
    cal_hash=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    test_report=json.loads((SOURCE/'comparison.json').read_text())
    for i in range(6):
        r=json.loads((OUT/f'worker_{i}/complete.json').read_text())
        assert r['manifest_sha256']==cal_hash and r['count']==N and r['worker']==i
        assert r['epoch']==test_report['members'][i]['selected_epoch']
    cpairs=np.load(DATA/'pairs.npy',mmap_mode='r');factors={}
    for method in ('ensemble','mc'):
        pred=predictions(OUT,method)
        scores=np.lib.format.open_memmap(OUT/f'{method}_scores.npy',mode='w+',dtype=np.float32,shape=(N,3,256,256))
        for s in range(0,N,8):
            mu,std=moments(pred,s,s+8);x=cpairs[s:s+8,1].astype(np.float32)/255
            scores[s:s+8]=np.abs(x-mu)/std
        scores.flush();factors[method]=fit_factor(scores);del scores
    # Freeze factors before opening test targets.
    (OUT/'factors.json').write_text(json.dumps({'coverage_target':.95,'std_floor':STD_FLOOR,'calibration_count':N,'radius_factors':factors,'relative_to_gaussian95':{k:v/Z95 for k,v in factors.items()},'manifest_sha256':cal_hash,'scope':'Empirical mean image pixel/channel coverage; not certified coverage'},indent=2))
    result={};examples={}
    for role,root,pairs,count,offset in [('calibration',OUT,cpairs,N,0),('test',SOURCE,np.load('data/formal_rml_25k/pairs.npy',mmap_mode='r'),2500,22500)]:
        result[role]={}
        for method in ('ensemble','mc'):
            pred=predictions(root,method);acc={k:[] for k in ('original90','gaussian95','calibrated95')}
            widths={k:[] for k in acc};tails={k:np.zeros(2) for k in acc}
            for s in range(0,count,8):
                e=min(s+8,count);mu,std=moments(pred,s,e);x=pairs[offset+s:offset+e,1].astype(np.float32)/255
                for name,factor in [('original90',1.6448536269514722),('gaussian95',Z95),('calibrated95',factors[method])]:
                    radius=factor*std;lo,hi=mu-radius,mu+radius
                    acc[name].extend(((lo<=x)&(x<=hi)).mean((1,2,3)).tolist());widths[name].extend((2*radius).mean((1,2,3)).tolist())
                    tails[name]+=np.array([(x<lo).mean(),(x>hi).mean()])*(e-s)
                if role=='test' and s==0:examples[method]={'mean':mu,'target':x,'std':std}
            result[role][method]={k:{'coverage':float(np.mean(v)),'mean_width':float(np.mean(widths[k])),'image_coverage_p10':float(np.quantile(v,.1)),'image_coverage_se':float(np.std(v,ddof=1)/np.sqrt(count)),'lower_miss':float(tails[k][0]/count),'upper_miss':float(tails[k][1]/count)} for k,v in acc.items()}
    payload={'target':.95,'radius_factors':factors,'metrics':result,'notes':['Reconstruction is unchanged.','Calibration targets empirical average coverage across equal-sized images, not each pixel or image individually.','No test-driven factor adjustment; test coverage may differ from 95%.','Exact duplicates excluded; near-duplicate/acquisition dependence remains possible.','Original90 and gaussian95 also use the documented standard-deviation floor.']}
    (OUT/'comparison.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2),flush=True)
    make_pdf(payload,examples)

def make_pdf(payload,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUT/'comparison.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,7));ax.axis('off')
        rows=[]
        for method,metrics in payload['metrics']['test'].items():
            for name,r in metrics.items():rows.append([method,name,f'{r["coverage"]:.2%}',f'{r["mean_width"]:.5f}'])
        ax.table(cellText=rows,colLabels=['Method','Interval','Test coverage','Width'],cellLoc='center',loc='center')
        ax.set_title('Epoch-20 real RML: empirical calibration targeting 95% coverage')
        fig.text(.1,.12,'2,048 new calibration images; 2,500 existing test images.\nFactors fixed before test evaluation. No reconstruction change or certified guarantee.')
        pdf.savefig(fig);plt.close(fig)
        for i in range(8):
            fig,axes=plt.subplots(2,4,figsize=(14,7))
            for row,method in enumerate(('ensemble','mc')):
                e=examples[method]
                arrays=[e['target'][i],e['mean'][i],(2*Z95*e['std'][i]).mean(0),(2*payload['radius_factors'][method]*e['std'][i]).mean(0)]
                for col,(title,a) in enumerate(zip(['Target','Reconstruction','Gaussian 95% width','Calibrated 95% width'],arrays)):
                    ax=axes[row,col];ax.axis('off');ax.set_title(method+': '+title)
                    if a.ndim==3:ax.imshow(a.transpose(1,2,0).clip(0,1))
                    else:ax.imshow(a,cmap='magma',vmin=0,vmax=.5)
            fig.suptitle(f'Test example {i+1}; shared width scale 0–0.5');fig.tight_layout();pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report']);p.add_argument('--worker',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)));a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare()
    elif a.stage=='infer':infer(a.worker)
    else:report()
