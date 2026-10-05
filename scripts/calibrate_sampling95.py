"""Matched 95% empirical calibration for the two frozen Q-sampling pilots."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
ROOT=Path('results/sampling_pilot_calibration95_20260925')
SOURCES={'uniform':Path('results/gradient_conflict_20260924/pilot/frozen'),
         'endpoints':Path('results/quantile_followups_20260924/endpoints/pilot/frozen')}
TD=Path('data/formal_rml_25k');CD=Path('data/formal_rml_calibration95')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def infer():
    from models.shamus_rml import ShamusRML
    from models.formal_quantile import QuantileRML
    torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
    configs=[]
    for method,source in SOURCES.items():
        out=ROOT/method;out.mkdir(exist_ok=True)
        ck=torch.load(source/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==2 and ck['config']['arm']=='frozen'
        assert ck['config']['manifest_sha256']==sha(TD/'manifest.json')
        configs.append(ck['config'])
        model=QuantileRML(ShamusRML(),'conditioned').cuda().eval();model.load_state_dict(ck['model']);del ck
        for role,path,offset,count in [('calibration',CD,0,2048),('test',TD,22500,1000)]:
            pairs=np.load(path/'pairs.npy',mmap_mode='r')
            outputs={k:np.lib.format.open_memmap(out/f'{role}_{k}.npy',mode='w+',dtype=np.float32,shape=(count,3,256,256)) for k in ['r','lo','hi']}
            for s in range(0,count,2):
                y=torch.from_numpy(np.array(pairs[offset+s:offset+s+2,0])).float().cuda()/255
                values=model.bounds(y) # FP32, matching the two-epoch pilot evaluation.
                for key,value in zip(outputs,values):
                    assert torch.isfinite(value).all();outputs[key][s:s+2]=value.cpu().numpy()
            for array in outputs.values():array.flush()
            print(method,role,count,'complete',flush=True)
        del model
    for key in ['initial_model_sha256','base_sha256','manifest_sha256','train_count','seed']:
        assert configs[0][key]==configs[1][key],key
    (ROOT/'inference.json').write_text(json.dumps({'sources':{k:str(v) for k,v in SOURCES.items()},'configs':configs,'checkpoint_epoch':2,'dtype':'float32','test_manifest_sha256':sha(TD/'manifest.json'),'calibration_manifest_sha256':sha(CD/'manifest.json')},indent=2))

def arrays(method,role):return {k:np.load(ROOT/method/f'{role}_{k}.npy',mmap_mode='r') for k in ['r','lo','hi']}
def widths(r,lo,hi):return np.maximum(r-lo,.001),np.maximum(hi-r,.001)

def report():
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    torch.set_num_threads(4)
    info=json.loads((ROOT/'inference.json').read_text())
    assert info['test_manifest_sha256']==sha(TD/'manifest.json') and info['calibration_manifest_sha256']==sha(CD/'manifest.json')
    cp=np.load(CD/'pairs.npy',mmap_mode='r');factors={}
    for method in SOURCES:
        a=arrays(method,'calibration')
        scores=np.lib.format.open_memmap(ROOT/method/'scores.npy',mode='w+',dtype=np.float32,shape=(2048,3,256,256))
        for s in range(0,2048,8):
            r,lo,hi=[a[k][s:s+8] for k in ['r','lo','hi']];x=cp[s:s+8,1].astype(np.float32)/255
            wl,wu=widths(r,lo,hi);score=np.maximum((r-x)/wl,(x-r)/wu)
            assert np.isfinite(score).all();scores[s:s+8]=score
        scores.flush();factors[method]=max(0.,float(np.quantile(scores,.95,method='higher',overwrite_input=True)));del scores
    # Persist calibration factors before any test-based comparisons.
    (ROOT/'calibration.json').write_text(json.dumps({'alpha':.05,'count':2048,'factors':factors,'width_floor':.001,'manifest_sha256':info['calibration_manifest_sha256']},indent=2))
    tp=np.load(TD/'pairs.npy',mmap_mode='r');manifest=json.loads((TD/'manifest.json').read_text())
    rows=[];metrics={};samples={};cal_coverage={}
    for method in SOURCES:
        a=arrays(method,'calibration');hits=0;count=0
        for s in range(0,2048,8):
            r,lo,hi=[a[k][s:s+8] for k in ['r','lo','hi']];x=cp[s:s+8,1].astype(np.float32)/255;wl,wu=widths(r,lo,hi)
            hits+=int(((r-factors[method]*wl<=x)&(x<=r+factors[method]*wu)).sum());count+=x.size
        cal_coverage[method]=hits/count
        a=arrays(method,'test');per=[]
        for s in range(0,1000,8):
            e=min(s+8,1000);r,lo,hi=[a[k][s:e] for k in ['r','lo','hi']];x=tp[22500+s:22500+e,1].astype(np.float32)/255
            wl,wu=widths(r,lo,hi);l=r-factors[method]*wl;h=r+factors[method]*wu
            mse=((r.astype(np.float64)-x)**2).mean((1,2,3));psnr=-10*np.log10(mse)
            sim=ssim(torch.from_numpy(np.array(r)),torch.from_numpy(x),data_range=1.,reduction='none').numpy()
            cov=((l<=x)&(x<=h)).mean((1,2,3));width=(h-l).mean((1,2,3));raw=((lo<=x)&(x<=hi)).mean((1,2,3))
            for j in range(e-s):
                row={'method':method,'id':manifest['test'][s+j]['id'],'psnr_db':float(psnr[j]),'ssim':float(sim[j]),'coverage95':float(cov[j]),'width95':float(width[j]),'raw_coverage':float(raw[j])};rows.append(row);per.append(row)
            if s==0:samples.update({method+'_mean':r.copy(),method+'_lo':l.copy(),method+'_hi':h.copy()})
        metrics[method]={key:float(np.mean([row[key] for row in per])) for key in ['psnr_db','ssim','coverage95','width95','raw_coverage']}
    # The frozen normal reconstructions must be identical across the sampling arms.
    u=arrays('uniform','test')['r'];v=arrays('endpoints','test')['r']
    for s in range(0,1000,8):np.testing.assert_array_equal(u[s:s+8],v[s:s+8])
    with (ROOT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload={'alpha':.05,'test_count':1000,'calibration_count':2048,'factors':factors,'calibration_coverage':cal_coverage,'metrics':metrics,'inference':info}
    (ROOT/'comparison.json').write_text(json.dumps(payload,indent=2))
    lines=['# Frozen Q-head sampling: alpha = 0.05','',
           'Existing two-epoch pilot checkpoints; no retraining. Independent 2048-image calibration; same first 1000 test images. FP32 inference matches the pilot precision.', '',
           '| Sampling | Calibration factor | Test coverage | Mean width | PSNR | SSIM |','|---|---:|---:|---:|---:|---:|']
    for k,r in metrics.items():lines.append(f"| {k} | {factors[k]:.6f} | {r['coverage95']:.2%} | {r['width95']:.6f} | {r['psnr_db']:.4f} | {r['ssim']:.5f} |")
    lines+=['','Both use the same frozen backbone and Q-conditioned architecture. Endpoints are q=0.025 and 0.975 in both inference passes. Calibrated bounds are r minus/plus one fitted factor times asymmetric positive widths, floored at 0.001. The factor is the pooled empirical 95th percentile (higher rule) of normalized pixel errors. This is empirical average pixel-channel calibration, not a per-image or distribution-free guarantee. Narrower intervals are favorable only at comparable held-out coverage. Earlier raw pilot numbers used 256 validation images and are not the same evaluation set.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    samples.update(measurement=tp[22500:22508,0].astype(np.float32)/255,target=tp[22500:22508,1].astype(np.float32)/255,ids=np.array([r['id'] for r in manifest['test'][:8]]))
    np.savez_compressed(ROOT/'examples.npz',**samples)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['infer','report']);a=p.parse_args()
    infer() if a.action=='infer' else report()
