"""Resume corrupted-input adaptation epoch-10 states through epoch 30 in five-epoch jobs."""
import argparse,hashlib,json,math,signal,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from model import Model,training_loss
from continuation import learning_rate,load_initial,parent_identity
from data import Pairs,identity,audit
ROOT=Path(__file__).resolve().parent
TASKS=[('raw','qutcc'),('raw','im2im')]
PROTOCOL='raw_corrupted_resume10_to30_20261019_v1';STOP=0

def stop(sig,*_):
    global STOP
    STOP=75 if sig==signal.SIGUSR1 else 1

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)
def source_hash():
    h=hashlib.sha256()
    for p in sorted(ROOT.rglob('*.py')):
        if any(part in ('results','reference') for part in p.relative_to(ROOT).parts):continue
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()

@torch.inference_mode()
def validate(model,loader):
    model.eval();sums={'psnr':0.,'ssim':0.,'mae':0.};n=0
    for b in loader:
        y,x=b[:,0].cuda(),b[:,1].cuda();pred=model.center(y).clamp(0,1)
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite validation output')
        count=len(b);sums['psnr']+=float((-10*(pred-x).square().mean((1,2,3)).clamp_min(1e-12).log10()).sum())
        sums['ssim']+=float(ssim(pred,x,data_range=1.,reduction='none').sum())
        sums['mae']+=float((pred-x).abs().mean())*count;n+=count
    return dict(**{k:v/n for k,v in sums.items()},count=n,clipped=True,reconstruction='median q=0.5' if model.method=='qutcc' else 'MSE center')

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,choices=range(2),required=True);p.add_argument('--target-epoch',type=int,default=15);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert 11<=a.target_epoch<=30
    camera,method=TASKS[a.task];out=ROOT/'results'/camera/method;out.mkdir(parents=True,exist_ok=True)
    counts=audit(camera);torch.set_num_threads(2)
    seed=202610180+{'rml':0,'diffuser':1,'raw':2}[camera];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Model(method).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=0)
    rng=torch.Generator(device='cuda').manual_seed(seed+1000)
    effective=4;micro=1 if camera=='raw' else 2;steps=math.ceil(counts['train']/effective)
    meta=dict(protocol=PROTOCOL,camera=camera,method=method,epochs=30,seed=seed,initialization='resume_full_corrupted_epoch10_weights_optimizer_rng',parent=parent_identity(method),dataset_identity=identity(camera),source_sha256=source_hash(),counts=counts,
        effective_batch=effective,microbatch=micro,precision='float32',optimizer='Adam',betas=[.9,.99],weight_decay=0.,peak_lr=1e-5,initial_scheduled_lr=1e-5,
        schedule='adaptation epochs 11-30: continue constant 1e-5, existing Adam moments retained',quantiles='independent U(0,1) per image' if method=='qutcc' else [.05,.95],
        loss='pinball only' if method=='qutcc' else 'MSE center + pinball .05 + pinball .95',film_sites=48 if method=='qutcc' else 0,
        calibration='separate Algorithm 1 q search; alpha=.10' if method=='qutcc' else 'separate asymmetric endpoint scales',normalization='existing RGB [0,1] dataset; ImageNet input normalization inside existing ConvNeXt',
        deviations=['ConvNeXtV2 Base + FPN replaces attention U-Net as explicitly requested','lensless datasets and existing splits','RGB channels','internal pad-to-32 and crop for raw image shape','microbatch accumulation implements effective batch four'],test_used=False,calibration_used=False,corruptions='exact same reviewed fixed PNG inputs and unchanged targets as adaptation epochs 1-10',clean_pretraining_epochs=30)
    train,val=Pairs(camera,'train'),Pairs(camera,'val')
    if a.smoke:
        load_initial(model,opt,rng,method)
        for group in opt.param_groups:group["lr"]=learning_rate(10*steps,steps)
        before=next(model.parameters()).detach().clone();losses=[];times=[]
        for b in DataLoader(Subset(train,range(16)),batch_size=effective):
            model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic();total=0.
            for sub in b.split(micro):
                loss=training_loss(model,sub[:,0].cuda(),sub[:,1].cuda(),rng)*len(sub)/len(b)
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t);losses.append(total)
        assert not torch.equal(before,next(model.parameters()))
        t=time.monotonic();v=validate(model,DataLoader(Subset(val,range(8)),batch_size=micro));val_time=time.monotonic()-t
        estimate=sum(times[1:])/len(times[1:])*steps+val_time/8*len(val)
        report=dict(**meta,passed=estimate*5<10*3600,losses=losses,seconds_per_step=sum(times[1:])/len(times[1:]),estimated_epoch_hours=estimate/3600,gpu=torch.cuda.get_device_name(),peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,validation=v)
        write(out/'smoke.json',report);print(json.dumps(report),flush=True)
        if not report['passed']:raise RuntimeError('Runtime gate: five epochs may exceed allocation')
        return
    smoke=json.loads((out/'smoke.json').read_text());assert smoke['passed'] and smoke['source_sha256']==meta['source_sha256']
    latest=out/'latest.pt';epoch=1;offset=0;global_step=0;history=[];loss_sum=0.;loss_count=0
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['config']==meta,'Resume protocol/config mismatch'
    else:
        ck=load_initial(model,opt,rng,method,return_checkpoint=True)
    if ck is not None:
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);rng.set_state(ck['q_rng']);torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
        epoch,offset,global_step,history,loss_sum,loss_count=[ck[k] for k in ('next_epoch','offset','global_step','history','loss_sum','loss_count')];del ck
    def checkpoint():
        save(latest,dict(config=meta,model=model.state_dict(),optimizer=opt.state_dict(),q_rng=rng.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),next_epoch=epoch,offset=offset,global_step=global_step,history=history,loss_sum=loss_sum,loss_count=loss_count))
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop);write(out/'config.json',meta)
    # Recover an epoch-boundary milestone if preemption happened after the
    # resumable checkpoint commit but before its compact milestone was written.
    if offset==0 and epoch>1 and (epoch-1)%5==0:
        dest=out/'checkpoints'/f'epoch_{epoch-1:03d}.pt'
        if not dest.exists():save(dest,dict(config=meta,epoch=epoch-1,model=model.state_dict(),history=history))
    started=time.monotonic()
    while epoch<=a.target_epoch:
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        loader=DataLoader(Subset(train,order[offset:]),batch_size=effective,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(seed+epoch+10000))
        t=time.monotonic();model.train()
        for b in loader:
            opt.zero_grad(set_to_none=True);lr=learning_rate(global_step,steps)
            for group in opt.param_groups:group['lr']=lr
            value=0.
            for sub in b.split(micro):
                loss=training_loss(model,sub[:,0].cuda(non_blocking=True),sub[:,1].cuda(non_blocking=True),rng)*len(sub)/len(b)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();value+=float(loss.detach())
            # Match reference: no added clipping, auxiliary losses, or policy updates.
            opt.step();offset+=len(b);global_step+=1;loss_sum+=value*len(b);loss_count+=len(b)
            if global_step%100==0:
                progress=dict(epoch=epoch,processed=offset,total=len(train),loss=value,lr=lr,global_step=global_step);write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            should_requeue=time.monotonic()-started>10.5*3600
            if global_step%500==0 or STOP or should_requeue:checkpoint()
            if STOP or should_requeue:raise SystemExit(STOP or 75)
        v=validate(model,DataLoader(val,batch_size=micro,num_workers=2));row=dict(epoch=epoch,training_loss=loss_sum/loss_count,validation=v,segment_seconds=time.monotonic()-t,lr=opt.param_groups[0]['lr'],gpu=torch.cuda.get_device_name())
        history.append(row);write(out/'history.json',history);print(json.dumps(row),flush=True)
        finished_epoch=epoch
        write(out/'progress.json',dict(epoch=epoch,processed=len(train),total=len(train),stage='validated'))
        epoch+=1;offset=0;loss_sum=0.;loss_count=0;checkpoint()
        if finished_epoch%5==0:
            dest=out/'checkpoints'/f'epoch_{finished_epoch:03d}.pt'
            if not dest.exists():save(dest,dict(config=meta,epoch=finished_epoch,model=model.state_dict(),history=history))
if __name__=='__main__':main()
