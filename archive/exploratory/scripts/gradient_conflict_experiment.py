"""Matched two-epoch pilot: frozen Q, joint Q, and MSE-only continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML, pinball, LEVELS
from scripts.formal_rml_train import Pairs, DATA

ARMS = ['frozen', 'joint', 'mse']
ROOT = Path('results/gradient_conflict_20260924')

def digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def shared_parameters(model):
    return [p for name, p in model.base.named_parameters() if not name.startswith('head.')]

def gradient_stats(first, second):
    dot = sum((a.double()*b.double()).sum().item() for a,b in zip(first,second))
    aa = sum(a.double().square().sum().item() for a in first)
    bb = sum(b.double().square().sum().item() for b in second)
    return {'mse_norm': aa**.5, 'pinball_norm': bb**.5,
            'pinball_to_mse_norm': (bb/max(aa,1e-30))**.5,
            'cosine': dot/max((aa*bb)**.5,1e-30),
            'dot': dot, 'mse_dot_joint_gradient': aa+dot}

def probe(model, batches):
    # FP32, fixed training images, same q across every arm and checkpoint.
    model.eval(); previous=[p.requires_grad for p in model.base.parameters()]
    model.base.requires_grad_(True)
    params=shared_parameters(model); rows=[]
    rng=torch.Generator().manual_seed(4401)
    try:
        for batch in batches:
            y,x=batch[:,0].cuda(),batch[:,1].cuda()
            q=torch.rand(len(y),generator=rng).cuda()
            f,r=model.features(y); pred=model.predict(y,f,q)
            mse=(r-x).square().mean(); quantile=pinball(pred,x,q)
            a=torch.autograd.grad(mse,params,retain_graph=True)
            b=torch.autograd.grad(quantile,params)
            row=gradient_stats(a,b);row.update(mse=mse.item(),pinball=quantile.item())
            rows.append(row)
            del a,b,f,r,pred,mse,quantile
    finally:
        for p,flag in zip(model.base.parameters(),previous):p.requires_grad_(flag)
    return rows

@torch.inference_mode()
def evaluate(model, loader):
    model.eval(); sums=dict(mse=0.,psnr_db=0.,ssim=0.,endpoint_pinball=0.,raw_coverage=0.,raw_width=0.,crossing=0.)
    count=0
    for batch in loader:
        y,x=batch[:,0].cuda(),batch[:,1].cuda()
        # FP32 evaluation for a consistent before/after frozen invariance check.
        r,lo,hi=model.bounds(y)
        mse=(r-x).square().mean((1,2,3)); n=len(y);count+=n
        sums['mse']+=mse.sum().item()
        sums['psnr_db']+=(-10*torch.log10(mse.clamp_min(1e-30))).sum().item()
        sums['ssim']+=ssim(r,x,data_range=1.,reduction='none').sum().item()
        sums['endpoint_pinball']+=n*sum(pinball(p,x,x.new_full((n,),q)).item() for p,q in zip([lo,hi],LEVELS))/2
        sums['raw_coverage']+=n*((x>=lo)&(x<=hi)).float().mean().item()
        sums['raw_width']+=n*(hi-lo).mean().item()
        sums['crossing']+=n*(lo>hi).float().mean().item()
    return {**{k:v/count for k,v in sums.items()},'count':count}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default=ARMS[int(os.environ.get('SLURM_ARRAY_TASK_ID',0))])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--sampling',choices=['uniform','endpoints'],default='uniform')
    p.add_argument('--pinball-weight',type=float,default=1.)
    p.add_argument('--out',type=Path,default=ROOT)
    a=p.parse_args()
    if not 0<a.pinball_weight:raise ValueError('Pinball weight must be positive')
    torch.set_num_threads(2);torch.manual_seed(20260924);torch.cuda.manual_seed_all(20260924)
    torch.backends.cudnn.benchmark=False
    out=a.out/('smoke' if a.smoke else 'pilot')/a.arm;out.mkdir(parents=True,exist_ok=True)
    source=Path('results/formal_quantile_comparison/normal/best.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    manifest=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    assert ck['trained_stage']=='normal' and ck['manifest_sha256']==manifest
    base=ShamusRML();base.load_state_dict(ck['model']);base_hash=digest(base)
    model=QuantileRML(base,'conditioned').cuda();initial_hash=digest(model)
    if a.arm=='frozen':model.base.requires_grad_(False)
    if a.arm=='mse':model.conditioner.requires_grad_(False);model.head.requires_grad_(False)
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=1e-4)
    train_count=4 if a.smoke else 20000;val_count=4 if a.smoke else 256
    val=DataLoader(Subset(Pairs('val'),range(val_count)),batch_size=2,num_workers=2)
    probes=list(DataLoader(Subset(Pairs('train'),range(2 if a.smoke else 8)),batch_size=2))
    config={'arm':a.arm,'source':str(source),'source_epoch':ck['epoch'],'base_sha256':base_hash,
            'initial_model_sha256':initial_hash,'manifest_sha256':manifest,'train_count':train_count,
            'validation_count':val_count,'epochs':2,'seed':20260924,'gpu':torch.cuda.get_device_name(),
            'test_used':False,'calibration':False,'selection':'final epoch; no best-checkpoint selection',
            'gradient_probe':'first 8 training images (2 in smoke), fixed q, FP32, shared base excluding RGB head'}
    if a.out!=ROOT or a.sampling!='uniform' or a.pinball_weight!=1.:
        config.update(sampling=a.sampling,pinball_weight=a.pinball_weight)
    del ck
    (out/'config.json').write_text(json.dumps(config,indent=2))
    history=[];start_epoch=0
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert all(ck['config'][k]==v for k,v in config.items() if k!='gpu')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];start_epoch=ck['epoch'];del ck
    else:
        row={'epoch':0,'validation':evaluate(model,val),'gradients':probe(model,probes)}
        history.append(row);print(json.dumps(row),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2))
    for epoch in range(start_epoch,2):
        start=time.monotonic();model.train()
        if a.arm=='frozen':model.base.eval()
        # Independent seeded streams ensure identical ordering and q despite differing graph paths.
        order=torch.Generator().manual_seed(8000+epoch)
        qrng=torch.Generator().manual_seed(9000+epoch)
        loader=DataLoader(Subset(Pairs('train'),range(train_count)),batch_size=2,shuffle=True,generator=order,num_workers=2)
        total=0.
        for batch in loader:
            y,x=batch[:,0].cuda(),batch[:,1].cuda();q=torch.rand(len(y),generator=qrng).cuda()
            if a.sampling=='endpoints':q=torch.where(q<.5,q.new_tensor(LEVELS[0]),q.new_tensor(LEVELS[1]))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f,r=model.features(y)
                mse=(r.float()-x).square().mean()
                if a.arm=='mse':loss=mse
                else:
                    qloss=pinball(model.predict(y,f,q).float(),x,q)
                    loss=qloss if a.arm=='frozen' else mse+a.pinball_weight*qloss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);opt.step()
            total+=loss.item()*len(y)
        row={'epoch':epoch+1,'train_loss':total/train_count,'validation':evaluate(model,val),
             'gradients':probe(model,probes),'seconds':time.monotonic()-start}
        if a.arm=='frozen':
            assert digest(model.base)==base_hash,'Frozen backbone changed'
            assert abs(row['validation']['mse']-history[0]['validation']['mse'])<1e-10
        history.append(row)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'history':history,'config':config},out/'latest.tmp')
        (out/'latest.tmp').replace(out/'latest.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    (out/'complete.json').write_text(json.dumps({'config':config,'history':history},indent=2))

if __name__=='__main__':main()
