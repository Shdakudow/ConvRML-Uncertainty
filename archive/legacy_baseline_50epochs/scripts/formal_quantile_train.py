"""Independent real-RML quantile comparison, two-epoch resumable chunks."""
import sys,json,argparse,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML,pinball,LEVELS
from scripts.formal_rml_train import Pairs,DATA
from scripts.toy_grid_compare import digest
from scripts.training_records import manifest_digest, save_milestone
ROOT=Path('results/formal_quantile_comparison')

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['normal','conditioned','fixed']);p.add_argument('--epochs',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--lr',type=float);p.add_argument('--pinball-weight',type=float,default=1.)
    p.add_argument('--backbone-checkpoint',type=Path,help='Explicit normal checkpoint for a fresh fixed head')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--interval-alpha',type=float,default=.05,help='Native endpoint alpha; legacy default .05, use .10 for the new study')
    p.add_argument('--validation-mse',action='store_true');p.add_argument('--isolated-q',action='store_true');a=p.parse_args()
    ROOT=a.root
    if not 0<a.interval_alpha<1 or a.save_every<1:raise ValueError('Invalid interval/checkpoint configuration')
    levels=(a.interval_alpha/2,1-a.interval_alpha/2)
    manifest=manifest_digest(a.data)
    torch.set_num_threads(2);torch.manual_seed(20260925)
    out=ROOT/a.kind;out.mkdir(parents=True,exist_ok=True)
    base=ShamusRML()
    if a.kind=='fixed':
        state=torch.load(a.backbone_checkpoint or ROOT/'normal/best.pt',map_location='cpu',weights_only=False)
        if state.get('trained_stage',state.get('run_arguments',{}).get('kind'))!='normal':raise ValueError('Invalid normal checkpoint')
        if state.get('manifest_sha256')!=manifest:raise ValueError('Fixed backbone split mismatch')
        base.load_state_dict(state['model'],strict=True)
    model=(base if a.kind=='normal' else QuantileRML(base,a.kind,levels=levels)).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    qrng=torch.Generator(device='cuda').manual_seed(20260924)
    history=[];best=float('inf');start_epoch=0
    # Dataset identity also guards fixed-backbone initialization.
    if (out/'latest.pt').exists() and not a.smoke:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==manifest
        if ck.get('run_arguments',{}).get('interval_alpha',.05)!=a.interval_alpha:raise ValueError('Cannot change endpoint levels on resume')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];best=ck['best'];start_epoch=ck['epoch']
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
        if 'q_rng' in ck:qrng.set_state(ck['q_rng'])
        del ck
    if a.lr is not None:
        if a.lr<=0:raise ValueError('Learning rate must be positive')
        for group in opt.param_groups:group['lr']=a.lr
    if not a.smoke and not 0<=a.epochs-start_epoch<=2:raise ValueError('Each job must advance at most two epochs')
    frozen_hash=digest(base) if a.kind=='fixed' else None
    if not a.smoke and a.kind!='fixed':
        smoke=json.loads((ROOT/f'smoke_{a.kind}.json').read_text())
        if smoke.get('manifest_sha256',manifest)!=manifest or not smoke['passed'] or smoke['seconds_per_batch']*len(Pairs('train',a.data))*1.25>3*3600:
            raise RuntimeError('Two-epoch runtime estimate exceeds job allocation; review required')
    train=DataLoader(Pairs('train',a.data),batch_size=2,shuffle=True,num_workers=2,pin_memory=True)
    val=DataLoader(Pairs('val',a.data),batch_size=2,num_workers=2,pin_memory=True)
    def objective(b,validation=False):
        y,x=b[:,0].cuda(),b[:,1].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.kind=='normal':return (model(y).float()-x).square().mean()
            f,r=model.features(y)
            if validation and a.validation_mse:return (r.float()-x).square().mean()
            if a.kind=='fixed':
                pred=model.predict(y,f)
                qloss=sum(pinball(pred[:,i].float(),x,x.new_full((len(y),),q)) for i,q in enumerate(levels))/2
            elif validation:
                qloss=sum(pinball(model.predict(y,f,x.new_full((len(y),),q)).float(),x,x.new_full((len(y),),q)) for q in levels)/2
            else:
                q=torch.rand(len(y),device='cuda',generator=qrng if a.isolated_q else None);qloss=pinball(model.predict(y,f,q).float(),x,q)
            return qloss if a.kind=='fixed' or validation else a.pinball_weight*qloss+(r.float()-x).square().mean()
    if a.smoke:
        model.train();start=None
        for i,b in enumerate(train):
            opt.zero_grad(set_to_none=True);loss=objective(b);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if i==4:torch.cuda.synchronize();start=time.monotonic()
            if i==14:break
        torch.cuda.synchronize()
        report={'kind':a.kind,'seconds_per_batch':(time.monotonic()-start)/10,'peak_bytes':torch.cuda.max_memory_allocated(),'manifest_sha256':manifest,'interval_alpha':a.interval_alpha,'passed':True}
        (ROOT/f'smoke_{a.kind}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    for epoch in range(start_epoch,a.epochs):
        model.train();start=time.monotonic();total=0.
        for b in train:
            opt.zero_grad(set_to_none=True);loss=objective(b)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();total+=loss.item()*len(b)
        model.eval();vsum=0.
        with torch.inference_mode():
            for b in val:vsum+=objective(b,True).item()*len(b)
        vmean=vsum/len(val.dataset)
        if not __import__('math').isfinite(vmean):raise RuntimeError('Nonfinite validation loss')
        if frozen_hash:assert digest(base)==frozen_hash,'Frozen normal reconstruction changed'
        if vmean<best:
            best=vmean
            torch.save({'model':model.state_dict(),'epoch':epoch+1,'validation':best,'trained_stage':a.kind,'manifest_sha256':manifest,'levels':levels},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
        history.append({'epoch':epoch+1,'train_loss':total/len(train.dataset),'validation_loss':vmean,'seconds':time.monotonic()-start})
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'history':history,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'q_rng':qrng.get_state(),'manifest_sha256':manifest,'run_arguments':vars(a)},out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
        save_milestone(out,model,epoch+1,a.save_every,manifest_sha256=manifest,history=history,trained_stage=a.kind,levels=levels,run_arguments=vars(a))
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)

if __name__=='__main__':main()
