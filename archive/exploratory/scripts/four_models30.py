"""Continue all original four methods from exact epoch 20 to exact epoch 30."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

ROOT=Path('results/four_models_epoch30_20260925')
BR=ROOT/'baselines';QR=ROOT/'quantiles';PR=ROOT/'predictions'
TD=Path('data/formal_rml_25k');CD=Path('data/formal_rml_calibration95')
BS=Path('results/formal_rml_25k_epoch20');QS=Path('results/formal_quantile_comparison')
METHODS={'fixed':'ConvRML + fixed quantile head','conditioned':'Q-conditioned ConvRML','ensemble':'Deep ensemble','mc':'MC dropout'}

def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    if (ROOT/'prepared.json').exists():print('Already prepared');return
    th=fingerprint(TD/'manifest.json');ROOT.mkdir(parents=True,exist_ok=True)
    records=[]
    for worker in range(8):
        kind='fixed' if worker==6 else 'conditioned'
        src=BS/f'worker_{worker}' if worker<6 else QS/kind
        dest=BR/f'worker_{worker}' if worker<6 else QR/kind
        ck=torch.load(src/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==20
        if worker<6:assert json.loads((src/'complete.json').read_text())['manifest_sha256']==th
        else:assert ck['manifest_sha256']==th
        records.append({'worker':worker,'source':str(src),'start_epoch':ck['epoch'],'lr':[g['lr'] for g in ck['optimizer']['param_groups']]})
        del ck
        dest.mkdir(parents=True,exist_ok=True)
        for name in ['latest.pt','best.pt','history.json']:shutil.copy2(src/name,dest/name)
    (QR/'normal').mkdir(parents=True,exist_ok=True)
    shutil.copy2(QS/'normal/best.pt',QR/'normal/best.pt')
    shutil.copy2(QS/'smoke_conditioned.json',QR/'smoke_conditioned.json')
    for method in ['ensemble','mc']:shutil.copy2(Path('results/formal_rml_25k')/f'smoke_{method}.json',BR/f'smoke_{method}.json')
    record={'manifest_sha256':th,'calibration_manifest_sha256':fingerprint(CD/'manifest.json'),
            'workers':records,'end_epoch':30,'inference_checkpoint':'exact latest epoch30',
            'fixed_semantics':'20 original normal reconstruction epochs, frozen normal checkpoint, quantile head extended from 20 to 30 epochs',
            'conditioned_semantics':'original jointly trained Q-conditioned model, MSE + pinball weight 1, random uniform q',
            'calibration_alpha':.1,'test_count':1000,'calibration_count':2048}
    (ROOT/'prepared.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def train(worker,epoch):
    assert epoch in [22,24,26,28,30]
    assert (ROOT/'prepared.json').exists()
    if worker<6:
        command=[sys.executable,'-u','scripts/formal_rml_train.py','--worker',str(worker),'--epochs',str(epoch),
                 '--max-new-epochs','2','--skip-inference','--patience','0','--out',str(BR)]
    else:
        command=[sys.executable,'-u','scripts/formal_quantile_train.py','fixed' if worker==6 else 'conditioned',
                 '--epochs',str(epoch),'--root',str(QR)]
    subprocess.run(command,check=True)

@torch.inference_mode()
def infer(worker):
    from baselines.formal_rml import FormalRML
    from baselines.deep_ensemble import DeepEnsembleMember
    from baselines.mc_dropout import dropout_inference
    from models.shamus_rml import ShamusRML
    from models.formal_quantile import QuantileRML
    torch.set_num_threads(2);torch.manual_seed(20260924+worker)
    metadata=json.loads((ROOT/'prepared.json').read_text())
    assert metadata['manifest_sha256']==fingerprint(TD/'manifest.json')
    assert metadata['calibration_manifest_sha256']==fingerprint(CD/'manifest.json')
    kind='fixed' if worker==6 else 'conditioned'
    if worker<6:
        net=FormalRML('mc' if worker==5 else 'ensemble')
        model=net if worker==5 else DeepEnsembleMember(net)
        path=BR/f'worker_{worker}'/'latest.pt';keys=['mean','variance']
    else:model=QuantileRML(ShamusRML(),kind);path=QR/kind/'latest.pt';keys=['r','lo','hi']
    ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==30
    model.load_state_dict(ck['model']);del ck
    model.cuda().eval();out=PR/f'worker_{worker}';out.mkdir(parents=True,exist_ok=True)
    for role,data,offset,count in [('calibration',CD,0,2048),('test',TD,22500,1000)]:
        pairs=np.load(data/'pairs.npy',mmap_mode='r')
        arrays={key:np.lib.format.open_memmap(out/f'{role}_{key}.npy',mode='w+',dtype=np.float32,shape=(count,3,256,256)) for key in keys}
        for s in range(0,count,2):
            y=torch.from_numpy(np.array(pairs[offset+s:offset+s+2,0],copy=True)).float().cuda()/255
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if worker<5:mu,var=model(y);values=[mu.float(),var.float()]
                elif worker==5:
                    with dropout_inference(model):
                        mu=None;m2=None
                        for j in range(30):
                            pred=model(y).float()
                            if mu is None:mu=pred.clone();m2=torch.zeros_like(pred)
                            else:delta=pred-mu;mu=mu+delta/(j+1);m2=m2+delta*(pred-mu)
                    values=[mu,m2/30]
                else:values=[v.float() for v in model.bounds(y)]
            for key,value in zip(keys,values):
                if not torch.isfinite(value).all():raise ValueError('Nonfinite inference')
                arrays[key][s:s+2]=value.cpu().numpy()
        for array in arrays.values():array.flush()
        print(worker,role,'finished',count,flush=True)
    (out/'complete.json').write_text(json.dumps({'worker':worker,'epoch':30,'test_count':1000,'calibration_count':2048,'manifest_sha256':metadata['manifest_sha256']},indent=2))

def prediction(method,role):
    if method in ['fixed','conditioned']:
        worker=6 if method=='fixed' else 7
        a={k:np.load(PR/f'worker_{worker}'/f'{role}_{k}.npy',mmap_mode='r') for k in ['r','lo','hi']}
        def batch(s,e):
            r=a['r'][s:e];return r,np.maximum(r-a['lo'][s:e],.001),np.maximum(a['hi'][s:e]-r,.001)
    else:
        workers=range(5) if method=='ensemble' else [5]
        means=[np.load(PR/f'worker_{w}'/f'{role}_mean.npy',mmap_mode='r') for w in workers]
        variances=[np.load(PR/f'worker_{w}'/f'{role}_variance.npy',mmap_mode='r') for w in workers]
        def batch(s,e):
            m=np.stack([v[s:e] for v in means]);v=np.stack([a[s:e] for a in variances])
            r=m.mean(0);std=np.maximum(np.sqrt(v.mean(0)+m.var(0)),1e-4)
            return r,std,std
    return batch

def report():
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    torch.set_num_threads(4)
    protocol=json.loads((ROOT/'prepared.json').read_text())
    for worker in range(8):
        complete=json.loads((PR/f'worker_{worker}'/'complete.json').read_text())
        assert complete['epoch']==30 and complete['manifest_sha256']==protocol['manifest_sha256']
    cp=np.load(CD/'pairs.npy',mmap_mode='r');tp=np.load(TD/'pairs.npy',mmap_mode='r')
    manifest=json.loads((TD/'manifest.json').read_text());factors={};metrics={};rows=[];samples={}
    for method in METHODS:
        batch=prediction(method,'calibration')
        scores=np.lib.format.open_memmap(ROOT/f'{method}_scores.npy',mode='w+',dtype=np.float32,shape=(2048,3,256,256))
        for s in range(0,2048,8):
            r,wl,wu=batch(s,s+8);x=cp[s:s+8,1].astype(np.float32)/255
            score=np.maximum((r-x)/wl,(x-r)/wu);assert np.isfinite(score).all();scores[s:s+8]=score
        scores.flush();factors[method]=max(0.,float(np.quantile(scores,.9,method='higher',overwrite_input=True)));del scores
    (ROOT/'calibration.json').write_text(json.dumps({'alpha':.1,'count':2048,'factors':factors,'manifest_sha256':protocol['calibration_manifest_sha256']},indent=2))
    for method,label in METHODS.items():
        batch=prediction(method,'test');per=[]
        for s in range(0,1000,8):
            e=min(s+8,1000);r,wl,wu=batch(s,e);x=tp[22500+s:22500+e,1].astype(np.float32)/255
            lo=r-factors[method]*wl;hi=r+factors[method]*wu
            mse=((r.astype(np.float64)-x)**2).mean((1,2,3));psnr=-10*np.log10(mse)
            sim=ssim(torch.from_numpy(np.array(r)),torch.from_numpy(x),data_range=1.,reduction='none').numpy()
            coverage=((lo<=x)&(x<=hi)).mean((1,2,3));width=(hi-lo).mean((1,2,3))
            for j in range(e-s):
                row={'method':method,'id':manifest['test'][s+j]['id'],'psnr_db':float(psnr[j]),'ssim':float(sim[j]),'coverage':float(coverage[j]),'width':float(width[j])}
                per.append(row);rows.append(row)
            if s==0:samples.update({method+'_mean':r.copy(),method+'_lo':lo.copy(),method+'_hi':hi.copy()})
        metrics[method]={key:float(np.mean([v[key] for v in per])) for key in ['psnr_db','ssim','coverage','width']}
    with (ROOT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    old=json.loads(Path('results/four_models_1000_alpha10/summary.json').read_text())
    assert old['test_manifest_sha256']==protocol['manifest_sha256']
    output={'protocol':protocol,'metrics':metrics,'calibration_factors':factors,'previous20_best_selected':old['metrics']}
    (ROOT/'comparison.json').write_text(json.dumps(output,indent=2))
    lines=['# All four original methods continued to epoch 30','',
           'Exact epoch-30 checkpoints; 1,000 shared test images; empirical 90% calibration fitted separately on 2,048 calibration images.', '',
           '| Model | PSNR | SSIM | Coverage | Width |','|---|---:|---:|---:|---:|']
    for method,r in metrics.items():lines.append(f"| {METHODS[method]} | {r['psnr_db']:.4f} | {r['ssim']:.5f} | {r['coverage']:.2%} | {r['width']:.6f} |")
    lines+=['','Fixed-head epoch 30 means 30 head-training epochs on its original frozen normal backbone (which had 20 reconstruction-training epochs). Q-conditioned uses the original joint loss coefficient 1, not the later 0.02 control. Each of five ensemble members and the MC model also trained through epoch 30. Original results used best-validation checkpoints within 20 epochs; these results use exact epoch 30, so checkpoint selection differs. Best-validation checkpoints through 30 are preserved separately. No test-based parameter tuning; empirical calibration is not a per-image coverage guarantee.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    examples=ROOT/'examples';examples.mkdir(exist_ok=True)
    samples.update(measurement=tp[22500:22508,0].astype(np.float32)/255,target=tp[22500:22508,1].astype(np.float32)/255,ids=np.array([v['id'] for v in manifest['test'][:8]]))
    np.savez_compressed(examples/'examples.npz',**samples)
    (examples/'metadata.json').write_text(json.dumps({'coverage_target':.9,'selection':'first eight shared test images','methods':{k:{'factor':factors[k],'label':METHODS[k],'selected_epoch':30} for k in METHODS}},indent=2))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','infer','report']);p.add_argument('--worker',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)));p.add_argument('--epochs',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.worker,a.epochs)
    elif a.action=='infer':infer(a.worker)
    else:report()
