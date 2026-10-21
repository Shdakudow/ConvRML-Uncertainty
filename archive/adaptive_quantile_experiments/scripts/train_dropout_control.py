"""Corrected objectives and independent ensemble shuffles. Not submitted automatically."""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from baselines.formal_rml import FormalRML
from baselines.deep_ensemble import DeepEnsembleMember
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest
from scripts.corrected_runtime import select_runtime, transfer, restore_optimizer
ROOT=Path('results/dropout_control_20261015')
ARMS=['mc_000']
from models.corrected_uncertainty import FullDropoutRML, training_loss, experiment_seeds as reference_seeds
PROTOCOL='matched_p000_vs_p005_mse_control_v1'
def experiment_seeds(arm):
 if arm!='mc_000':raise ValueError('Only the no-dropout control belongs here')
 return reference_seeds('mc_005')
def check_resume(ck,arm,digest):
 if ck.get('protocol')!=PROTOCOL or ck.get('arm')!=arm or ck.get('manifest_sha256')!=digest or ck.get('seeds')!=experiment_seeds(arm):raise ValueError('Control resume mismatch')
PROBS={'mc_000':0.}

def paths(camera):
 name='optics_intervals_60k_5k_5k_20260928' if camera=='rml' else 'diffuser_intervals_60k_5k_5k_20260928'
 return Path('data')/name,Path('data/formal_rml_calibration95' if camera=='rml' else 'data/diffuser_intervals_calibration_20260928')

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)

def make_model(arm):
 torch.manual_seed(experiment_seeds(arm)['initialization'])
 # Same constructor and exact initialization as the trained p=.05 reference.
 # Setting p changes no parameters and consumes no random numbers.
 model=FullDropoutRML(.05)
 drops=[m for m in model.modules() if isinstance(m,nn.Dropout)]
 if len(model.dropout_locations)!=48 or sum(m.p==.05 for m in drops)!=48:raise ValueError('Expected 48 active dropout sites')
 for m in drops:m.p=0.
 return model

def load_model(camera,arm,epoch=50):
 ck=torch.load(ROOT/camera/arm/'checkpoints'/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=False)
 check_resume(ck,arm,manifest_digest(paths(camera)[0]))
 if ck['epoch']!=epoch:raise ValueError('Checkpoint epoch mismatch')
 model=make_model(arm);model.load_state_dict(ck['model'],strict=True)
 return model,ck

def loaders(camera):
 data,_=paths(camera)
 return Pairs('train',data),Pairs('val',data)

def step(model,arm,b,opt,qrng,runtime=None):
 y,x=transfer(b,runtime or {});opt.zero_grad(set_to_none=True)
 q=torch.rand(len(y),device='cuda',generator=qrng) if arm=='conditioned' else None
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=training_loss(model,arm,y,x,q)
 if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 return float(loss)

@torch.inference_mode()
def validate(model,arm,loader,limit=None,runtime=None):
 model.eval();sums=dict(psnr=0.,ssim=0.,mse=0.);count=0
 for i,b in enumerate(loader):
  if limit is not None and i>=limit:break
  y,x=transfer(b,runtime or {})
  with torch.autocast('cuda',dtype=torch.bfloat16):
   r=model.bounds(y)[0] if arm=='im2im' else model(y)
   if arm.startswith('ensemble_'):r=r[0]
  r=r.float().clamp(0,1);mse=(r-x).square().mean((1,2,3))
  sums['mse']+=float(mse.sum());sums['psnr']+=float((-10*mse.clamp_min(1e-12).log10()).sum())
  sums['ssim']+=float(ssim(r,x,data_range=1.,reduction='none').sum());count+=len(y)
 return {**{k:v/count for k,v in sums.items()},'count':count,'clipped':True,'reconstruction':'q=0.5' if arm=='conditioned' else 'center output' if arm=='im2im' else 'Gaussian mean'}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);p.add_argument('--camera',choices=['rml','diffuser']);p.add_argument('--arm',choices=ARMS);p.add_argument('--task',type=int);p.add_argument('--epochs',type=int,default=2);p.add_argument('--round',type=int);a=p.parse_args()
 if a.task is not None:a.camera=('rml','diffuser')[a.task//len(ARMS)];a.arm=ARMS[a.task%len(ARMS)]
 if not a.camera or not a.arm:raise ValueError('Missing camera/arm')
 if a.round is not None:
  a.epochs=min(50,2*a.round)
 if a.epochs not in range(1,51):raise ValueError('Only fifty epochs authorized')
 torch.set_num_threads(2);out=ROOT/a.camera/a.arm;out.mkdir(parents=True,exist_ok=True)
 data,_=paths(a.camera);digest=manifest_digest(data);trainset,valset=loaders(a.camera)
 model=make_model(a.arm)
 runtime=dict(version='baseline',channels_last=False,fused_adamw=False);torch.backends.cudnn.benchmark=runtime['channels_last']
 model.cuda()
 if runtime['channels_last']:model.to(memory_format=torch.channels_last)
 opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=1e-4,fused=runtime['fused_adamw'])
 print(json.dumps(dict(runtime=runtime,gpu=torch.cuda.get_device_name())),flush=True)
 # Independent per-member RNG streams; restore their exact states on resume.
 seeds=experiment_seeds(a.arm);torch.manual_seed(seeds['training']);torch.cuda.manual_seed_all(seeds['training'])
 qrng=torch.Generator(device='cuda').manual_seed(seeds['quantile']);drng=torch.Generator().manual_seed(seeds['data'])
 meta=dict(protocol=PROTOCOL,camera=a.camera,arm=a.arm,seeds=seeds,manifest_sha256=digest,
           initialization='same seeded initialization as mc_005; all 48 dropout probabilities set to zero',dropout_p=0.,reference_arm='mc_005',dropout_sites=48,test_count=5000,uncertainty='not defined for deterministic p=0 control',epochs=50,batch_size=2,lr=1e-4,weight_decay=1e-4,runtime=runtime,
           train_count=len(trainset),val_count=len(valset),levels=[.05,.95],
           objective='pinball only' if a.arm=='conditioned' else 'MSE + lower pinball + upper pinball' if a.arm=='im2im' else 'MSE' if a.arm.startswith('mc_') else 'Gaussian NLL')
 history=[];start=0
 if a.action=='train' and (out/'latest.pt').exists():
  ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  check_resume(ck,a.arm,digest)
  model.load_state_dict(ck['model']);restore_optimizer(opt,ck['optimizer'],runtime);history=ck['history'];start=ck['epoch']
  torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);qrng.set_state(ck['q_rng']);drng.set_state(ck['data_rng']);del ck
 if start and start%5==0 and not (out/'checkpoints'/f'epoch_{start:03d}.pt').exists():
  dest=out/'checkpoints'/f'epoch_{start:03d}.pt';dest.parent.mkdir(exist_ok=True);torch.save(dict(**meta,model=model.state_dict(),epoch=start,history=history),dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
 train=DataLoader(trainset,batch_size=2,shuffle=True,generator=drng,num_workers=2,pin_memory=True)
 val=DataLoader(valset,batch_size=2,num_workers=2,pin_memory=True)
 if a.action=='smoke':
  model.train();times=[]
  for i,b in enumerate(train):
   torch.cuda.synchronize();t=time.monotonic();step(model,a.arm,b,opt,qrng,runtime);torch.cuda.synchronize()
   if a.arm=='conditioned' and any(f.projection[-1].weight.grad is None for f in model.base.films.values()):raise ValueError('Disconnected FiLM')
   if i>=2:times.append(time.monotonic()-t)
   if i==6:break
  t=time.monotonic();v=validate(model,a.arm,val,limit=3,runtime=runtime);torch.cuda.synchronize();val_batch=(time.monotonic()-t)/3
  estimate=2*(sum(times)/len(times)*len(train)+val_batch*len(val))*1.3
  report=dict(**meta,passed=estimate<8*3600,estimated_two_epoch_hours=estimate/3600,gpu=torch.cuda.get_device_name(),validation=v)
  write(out/'smoke.json',report)
  if not report['passed']:raise RuntimeError('Two-epoch runtime exceeds eight-hour limit')
  print(json.dumps(report),flush=True);return
 if start>=a.epochs:
  print(json.dumps(dict(already_complete=True,epoch=start,target=a.epochs)));return
 if not 0<=a.epochs-start<=2:raise ValueError('Each job advances at most two epochs')
 smoke=json.loads((out/'smoke.json').read_text())
 if not smoke['passed'] or smoke['manifest_sha256']!=digest or smoke.get('protocol')!=PROTOCOL:raise ValueError('Smoke gate failed')
 write(out/'config.json',meta)
 if start==0 and not (out/'initial.pt').exists():torch.save(dict(**meta,model=model.state_dict(),epoch=0),out/'initial.pt')
 for epoch in range(start,a.epochs):
  model.train();begin=time.monotonic();total=0.
  for bi,b in enumerate(train):
   total+=step(model,a.arm,b,opt,qrng,runtime)*len(b)
   if (bi+1)%1000==0:write(out/'progress.json',dict(epoch=epoch+1,processed=(bi+1)*2,total=len(trainset)))
  validation=validate(model,a.arm,val,runtime=runtime)
  row=dict(epoch=epoch+1,loss=total/len(trainset),validation=validation,seconds=time.monotonic()-begin,gpu=torch.cuda.get_device_name(),runtime=runtime);history.append(row)
  ck=dict(**meta,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),q_rng=qrng.get_state(),data_rng=drng.get_state())
  torch.save(ck,out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
  if (epoch+1)%5==0:
   dest=out/'checkpoints'/f'epoch_{epoch+1:03d}.pt';dest.parent.mkdir(exist_ok=True)
   if dest.exists():raise FileExistsError(dest)
   torch.save(dict(**meta,model=model.state_dict(),epoch=epoch+1,history=history),dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
  write(out/'history.json',history);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
