"""Resumeable image-adaptive QUTCC continuation on the existing 60k split.

Each batch updates the reconstruction model with pinball and the q policy with
interval score. Calibration and test data are never used for these updates.
"""
import argparse, json, math, signal, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.image_adaptive_qutcc import ImageAdaptiveQUTCC, adaptive_step, PROTOCOL
from models.learned_quantile_selector import interval_score
from scripts.corrected_comparison_study import load_model, paths
from scripts.formal_rml_train import Pairs

ROOT=Path('results/image_adaptive_q_60k_20261014')
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

@torch.inference_mode()
def validate(model, loader):
    model.eval();total=0; qs=[]
    sums=dict(psnr=0.,ssim=0.,interval_score=0.,coverage=0.,width=0.)
    for b in loader:
        y,x=b[:,0].cuda(),b[:,1].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):r=model.interval_prediction(y)
        pred,lo,hi=r['prediction'],r['lower'],r['upper']
        if not all(torch.isfinite(t).all() for t in r.values()):raise ValueError('Nonfinite validation')
        n=len(y);mse=(pred-x).square().mean((1,2,3))
        sums['psnr']+=float((-10*mse.clamp_min(1e-12).log10()).sum())
        sums['ssim']+=float(ssim(pred,x,data_range=1.,reduction='none').sum())
        sums['interval_score']+=float(interval_score(lo,hi,x,model.alpha))*n
        sums['coverage']+=float(((x>=lo)&(x<=hi)).float().mean())*n
        sums['width']+=float((hi-lo).mean())*n
        qs.append(torch.stack([r['q_lower'],r['q_upper']],dim=1).float().cpu());total+=n
    q=torch.cat(qs)
    return dict(**{k:v/total for k,v in sums.items()},count=total,
                q_mean=q.mean(0).tolist(),q_std=q.std(0,unbiased=False).tolist(),
                q_min=q.min(0).values.tolist(),q_max=q.max(0).values.tolist(),
                calibrated=False,clipped=True)


def load_adaptive_checkpoint(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['protocol']!=PROTOCOL:raise ValueError('Wrong adaptive protocol')
    from models.corrected_uncertainty import PinballQUTCC
    model=ImageAdaptiveQUTCC(PinballQUTCC(),ck['alpha'])
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['smoke','train'])
    p.add_argument('--camera',choices=['rml','diffuser'],required=True)
    p.add_argument('--source-epoch',type=int,default=40)
    p.add_argument('--target-epoch',type=int,default=1,help='Adaptive continuation epoch, not total backbone epoch')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--save-steps',type=int,default=2000)
    a=p.parse_args()
    if a.target_epoch not in [1,2] or a.batch_size<1 or a.save_steps<1:raise ValueError('Invalid configuration')
    torch.set_num_threads(2);torch.manual_seed(20261014);torch.cuda.manual_seed_all(20261014)
    out=ROOT/a.camera;out.mkdir(parents=True,exist_ok=True)
    train=Pairs('train',paths(a.camera)[0]);val=Pairs('val',paths(a.camera)[0])
    if len(train)!=60000 or len(val)!=5000:raise ValueError('Expected 60k/5k split')
    base,source=load_model(a.camera,'conditioned',a.source_epoch)
    digest=source['manifest_sha256'];del source
    model=ImageAdaptiveQUTCC(base).cuda()
    backbone_opt=torch.optim.AdamW(model.quantile_model.parameters(),lr=1e-5,weight_decay=1e-4)
    selector_opt=torch.optim.AdamW(model.selector.parameters(),lr=1e-4,weight_decay=1e-4)
    qrng=torch.Generator(device='cuda').manual_seed(20261015)
    meta=dict(protocol=PROTOCOL,camera=a.camera,source_epoch=a.source_epoch,
              manifest_sha256=digest,alpha=.1,batch_size=a.batch_size,train_count=len(train),validation_count=len(val),
              backbone_lr=1e-5,selector_lr=1e-4,weight_decay=1e-4,seed=20261014,
              training_q_mixture=dict(uniform=.5,learned_lower=.25,learned_upper=.25),
              backbone_objective='pinball, detached quantile labels',selector_objective='fixed-alpha interval score',
              update_policy='alternating updates on every batch; backbone frozen only for selector gradient',
              calibration_used=False,test_used=False,precision='BF16 backbone; float32 q policy',gpu=torch.cuda.get_device_name())
    if a.action=='smoke':
        small=DataLoader(Subset(val,list(range(32))),batch_size=a.batch_size)
        before=validate(model,small)
        original_base=next(model.quantile_model.parameters()).detach().clone()
        original_head=model.selector.head.bias.detach().clone()
        times=[];gradients=[]
        for b in DataLoader(Subset(train,list(range(a.batch_size*8))),batch_size=a.batch_size):
            torch.cuda.synchronize();t=time.monotonic()
            row=adaptive_step(model,b[:,0].cuda(),b[:,1].cuda(),backbone_opt,selector_opt,qrng)
            torch.cuda.synchronize();times.append(time.monotonic()-t);gradients.append(row['selector_grad_norm'])
        t=time.monotonic();after=validate(model,small);torch.cuda.synchronize();vtime=time.monotonic()-t
        changed_base=bool((original_base!=next(model.quantile_model.parameters())).any())
        changed_head=bool((original_head!=model.selector.head.bias).any())
        estimate=(sum(times[2:])/len(times[2:])*math.ceil(len(train)/a.batch_size)+vtime*len(val)/32)*1.3
        passed=changed_base and changed_head and max(gradients)>0 and max(after['q_std'])>0 and estimate<10.5*3600
        report=dict(**meta,passed=passed,backbone_changed=changed_base,selector_changed=changed_head,
                    selector_grad_norms=gradients,estimated_epoch_hours=estimate/3600,
                    peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,before=before,after=after)
        write(out/'smoke.json',report);print(json.dumps(report),flush=True)
        if not passed:raise RuntimeError('Smoke gate failed; training must not start')
        return
    smoke=json.loads((out/'smoke.json').read_text())
    if not smoke['passed'] or any(smoke[k]!=meta[k] for k in ['protocol','source_epoch','manifest_sha256','batch_size']):
        raise ValueError('Missing/mismatched successful smoke gate')
    epoch,offset,history=1,0,[];best=float('inf');running=dict(pinball=0.,interval_score=0.,batches=0)
    latest=out/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        for key in ['protocol','camera','source_epoch','manifest_sha256','batch_size','backbone_lr','selector_lr']:
            if ck[key]!=meta[key]:raise ValueError('Resume mismatch: '+key)
        model.load_state_dict(ck['model'],strict=True)
        backbone_opt.load_state_dict(ck['backbone_optimizer']);selector_opt.load_state_dict(ck['selector_optimizer'])
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);qrng.set_state(ck['q_rng'])
        epoch,offset,history,best,running=ck['next_epoch'],ck['next_offset'],ck['history'],ck['best_score'],ck['running']
        del ck
    write(out/'config.json',meta)
    def save(next_epoch,next_offset):
        state=dict(**meta,model=model.state_dict(),backbone_optimizer=backbone_opt.state_dict(),
                   selector_optimizer=selector_opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                   q_rng=qrng.get_state(),next_epoch=next_epoch,next_offset=next_offset,history=history,best_score=best,running=running)
        temp=latest.with_suffix('.tmp');torch.save(state,temp);temp.replace(latest)
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if not history:
        baseline=out/'baseline_validation.json'
        if not baseline.exists():write(baseline,validate(model,DataLoader(val,batch_size=a.batch_size)))
    while epoch<=a.target_epoch:
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(20261014+epoch)).tolist()
        loader=DataLoader(Subset(train,order[offset:]),batch_size=a.batch_size,num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(20262014+epoch))
        begin=time.monotonic()
        for b in loader:
            row=adaptive_step(model,b[:,0].cuda(non_blocking=True),b[:,1].cuda(non_blocking=True),backbone_opt,selector_opt,qrng)
            offset+=len(b);running['batches']+=1
            for k in ['pinball','interval_score']:running[k]+=row[k]
            if running['batches']%100==0:
                progress=dict(epoch=epoch,processed=offset,total=len(train),**row)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            if running['batches']%a.save_steps==0 or STOP:save(epoch,offset)
            if STOP:raise SystemExit(75)
        metrics=validate(model,DataLoader(val,batch_size=a.batch_size,num_workers=2))
        row=dict(epoch=epoch,source_epoch=a.source_epoch,total_backbone_epochs=a.source_epoch+epoch,
                 validation=metrics,seconds_this_segment=time.monotonic()-begin,
                 training={k:running[k]/running['batches'] for k in ['pinball','interval_score']})
        history.append(row)
        compact=dict(**meta,model=model.state_dict(),adaptive_epoch=epoch,history=history)
        destination=out/('epoch_%03d.pt'%epoch)
        torch.save(compact,destination.with_suffix('.tmp'));destination.with_suffix('.tmp').replace(destination)
        if metrics['interval_score']<best:
            best=metrics['interval_score'];write(out/'best.json',dict(checkpoint=destination.name,score=best))
        write(out/'history.json',history);print(json.dumps(row),flush=True)
        epoch+=1;offset=0;running=dict(pinball=0.,interval_score=0.,batches=0);save(epoch,offset)

if __name__=='__main__':main()
