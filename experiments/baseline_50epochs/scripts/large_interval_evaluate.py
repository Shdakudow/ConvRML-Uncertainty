"""Exact milestone evaluation, common empirical calibration and learning curves."""
import csv
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.large_interval_study import ROOT, DATA
from scripts.training_records import manifest_digest, split_bounds

METHODS = ('ensemble', 'mc', 'fixed', 'conditioned')
CALIBRATION = Path('data/formal_rml_calibration95')


def checkpoint(path, epoch):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['epoch'] != epoch or ck['manifest_sha256'] != manifest_digest(DATA):
        raise ValueError('Checkpoint epoch or dataset mismatch')
    return ck


def load_predictor(method, epoch):
    from baselines.formal_rml import FormalRML
    from baselines.deep_ensemble import DeepEnsembleMember
    from baselines.mc_dropout import dropout_inference
    from models.formal_quantile import QuantileRML
    from models.shamus_rml import ShamusRML
    histories=[]
    if method in ('ensemble', 'mc'):
        models=[]
        for worker in (range(5) if method=='ensemble' else [5]):
            net=FormalRML(method)
            net=DeepEnsembleMember(net) if method=='ensemble' else net
            ck=checkpoint(ROOT/'baselines'/f'worker_{worker}'/'checkpoints'/f'epoch_{epoch:03d}.pt',epoch)
            net.load_state_dict(ck['model']);histories.append(ck['history']);del ck
            models.append(net.cuda().eval())
        def predict(y):
            if method=='ensemble':
                moments=[net(y) for net in models]
                means=torch.stack([m.float() for m,v in moments])
                variance=torch.stack([v.float() for m,v in moments]).mean(0)+means.var(0,unbiased=False)
                r=means.mean(0)
            else:
                r=None;m2=None
                with dropout_inference(models[0]):
                    for i in range(30):
                        sample=models[0](y).float()
                        if r is None:r=sample.clone();m2=torch.zeros_like(r)
                        else:
                            delta=sample-r;r=r+delta/(i+1);m2=m2+delta*(sample-r)
                variance=m2/30
            radius=1.6448536269514722*variance.clamp_min(0).sqrt()
            return r,r-radius,r+radius
    else:
        path=(ROOT/'heads'/f'backbone_{epoch:03d}'/'fixed/checkpoints/epoch_010.pt'
              if method=='fixed' else ROOT/'quantiles/conditioned/checkpoints'/f'epoch_{epoch:03d}.pt')
        ck=checkpoint(path,10 if method=='fixed' else epoch)
        if tuple(ck['levels'])!=(.05,.95):raise ValueError('Incorrect interval levels')
        net=QuantileRML(ShamusRML(),method,levels=ck['levels'])
        net.load_state_dict(ck['model']);histories.append(ck['history']);del ck
        net.cuda().eval()
        def predict(y):return tuple(t.float() for t in net.bounds(y))
        if method=='fixed':
            base=checkpoint(ROOT/'quantiles/normal/checkpoints'/f'epoch_{epoch:03d}.pt',epoch)
            histories.append(base['history']);del base
    gpu_hours=sum(row['seconds'] for history in histories for row in history)/3600
    return predict,gpu_hours


@torch.inference_mode()
def evaluate(worker, epoch):
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    if worker not in range(4) or epoch not in range(5,51,5):raise ValueError('Invalid milestone evaluation')
    method=METHODS[worker]
    torch.set_num_threads(2)
    torch.manual_seed(20260928+worker)  # Same stochastic evaluation seed at each milestone.
    predict,gpu_hours=load_predictor(method,epoch)
    out=ROOT/'evaluation'/f'epoch_{epoch:03d}'
    out.mkdir(parents=True,exist_ok=True)
    cal=np.load(CALIBRATION/'pairs.npy',mmap_mode='r')
    test=np.load(DATA/'pairs.npy',mmap_mode='r')
    offset,count,_=split_bounds(DATA,'test')
    def batches(pairs,offset,count):
        for start in range(0,count,2):
            b=torch.from_numpy(np.array(pairs[offset+start:offset+min(start+2,count)],copy=True)).cuda().float()/255
            with torch.autocast('cuda',dtype=torch.bfloat16):r,lo,hi=predict(b[:,0])
            if not all(torch.isfinite(v).all() for v in (r,lo,hi)):raise ValueError('Nonfinite predictions')
            # Identical rescaling rule for all methods. Retain native endpoints separately.
            wl=(r-lo).clamp_min(1e-6);wu=(hi-r).clamp_min(1e-6)
            yield start,b[:,1],r,lo,hi,wl,wu
    score_path=out/f'{method}_calibration_scores.tmp.npy'
    scores=np.lib.format.open_memmap(score_path,mode='w+',dtype=np.float32,shape=(len(cal),3,256,256))
    for start,x,r,lo,hi,wl,wu in batches(cal,0,len(cal)):
        scores[start:start+len(x)]=torch.maximum((r-x)/wl,(x-r)/wu).cpu().numpy()
    scores.flush()
    factor=max(0.,float(np.quantile(scores,.90,method='higher',overwrite_input=True)))
    del scores
    score_path.unlink()
    rows=[]
    for start,x,r,lo,hi,wl,wu in batches(test,offset,count):
        lower=r-factor*wl;upper=r+factor*wu
        for i in range(len(x)):
            error=(r[i]-x[i]).abs()
            rows.append({'test_index':start+i,
                         'psnr':-10*math.log10(max(float(error.square().mean()),1e-12)),
                         'ssim':float(ssim(r[i:i+1],x[i:i+1],data_range=1.)),
                         'mae':float(error.mean()),
                         'coverage':float(((x[i]>=lower[i])&(x[i]<=upper[i])).float().mean()),
                         'width':float((upper[i]-lower[i]).mean()),
                         'native_coverage':float(((x[i]>=lo[i])&(x[i]<=hi[i])).float().mean()),
                         'native_signed_width':float((hi[i]-lo[i]).mean()),
                         'native_crossing_fraction':float((lo[i]>hi[i]).float().mean())})
    with (out/f'{method}_per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'method':method,'backbone_epoch':epoch,'head_epochs':10 if method=='fixed' else None,
            'alpha':.10,'calibration_count':len(cal),'test_count':count,'factor':factor,
            'manifest_sha256':manifest_digest(DATA),'calibration_sha256':manifest_digest(CALIBRATION),
            'training_gpu_hours_including_validation':gpu_hours,
            'gpu':torch.cuda.get_device_name(),
            'calibration':'Empirical pooled-pixel multiplicative scaling around reconstruction; minimum sided radius 1e-6; no image-level coverage guarantee',
            'metrics':{key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key!='test_index'}}
    temporary=out/f'{method}.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(out/f'{method}.json')
    print(json.dumps(result),flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'learning_curves';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for epoch in range(5,51,5):
        for method in METHODS:
            result=json.loads((ROOT/'evaluation'/f'epoch_{epoch:03d}'/f'{method}.json').read_text())
            rows.append({'method':method,'backbone_epoch':epoch,'head_epochs':result['head_epochs'],
                         'training_gpu_hours':result['training_gpu_hours_including_validation'],**result['metrics']})
    with (out/'metrics_by_epoch.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(metric,label) in zip(axes.flat,[('psnr','PSNR (dB)'),('ssim','SSIM'),('width','Mean calibrated interval width (UC)'),('coverage','Test pixel coverage')]):
        for method in METHODS:
            selected=[row for row in rows if row['method']==method]
            ax.plot([r['backbone_epoch'] for r in selected],[r[metric] for r in selected],marker='o',label=method)
        if metric=='coverage':ax.axhline(.9,color='gray',linestyle='--')
        ax.set(xlabel='Backbone training epoch',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('60k training images · fixed head: fresh 10 head epochs per frozen backbone')
    fig.tight_layout()
    fig.savefig(out/'metrics_by_epoch.png',dpi=180);fig.savefig(out/'metrics_by_epoch.pdf')
    plt.close(fig)


if __name__=='__main__':
    plot()
