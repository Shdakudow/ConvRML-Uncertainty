"""Fresh ten-epoch matched-data study; standalone outputs, two-epoch chunks."""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML, pinball
from models.film_everywhere import EverywhereQuantileRML
from baselines.formal_rml import FormalRML
from baselines.deep_ensemble import DeepEnsembleMember
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest
ROOT=Path('results/film10_both_cameras_20261004')
ARMS=[f'ensemble_{i:02d}' for i in range(10)]+['mc_005','mc_010','mc_020','normal','conditioned','everywhere']
PROBS={'mc_005':.05,'mc_010':.1,'mc_020':.2}

def paths(camera):
 name='optics_intervals_60k_5k_5k_20260928' if camera=='rml' else 'diffuser_intervals_60k_5k_5k_20260928'
 return Path('data')/name,Path('data/formal_rml_calibration95' if camera=='rml' else 'data/diffuser_intervals_calibration_20260928')

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)

def make_model(arm):
 # Matched backbone initialization for normal and both FiLM arms, and across dropout rates.
 seed=20261004+(int(arm.split('_')[1]) if arm.startswith('ensemble_') else 100 if arm.startswith('mc_') else 200)
 torch.manual_seed(seed)
 if arm.startswith('ensemble_'):model=DeepEnsembleMember(FormalRML('ensemble'))
 elif arm.startswith('mc_'):
  model=FormalRML('mc')
  for m in model.modules():
   if isinstance(m,nn.Dropout2d):m.p=PROBS[arm]
 else:
  base=ShamusRML()
  if arm=='normal':model=base
  elif arm=='everywhere':model=EverywhereQuantileRML(base,levels=(.05,.95))
  else:model=QuantileRML(base,arm,levels=(.05,.95))
 return model

def load_model(camera,arm,epoch=10):
 ck=torch.load(ROOT/camera/arm/'checkpoints'/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=False)
 if ck['epoch']!=epoch or ck['manifest_sha256']!=manifest_digest(paths(camera)[0]):raise ValueError('Checkpoint provenance mismatch')
 model=make_model(arm);model.load_state_dict(ck['model'],strict=True)
 return model,ck

def loaders(camera):
 data,_=paths(camera)
 return Pairs('train',data),Pairs('val',data)

def step(model,arm,b,opt,qrng):
 y,x=b[:,0].cuda(),b[:,1].cuda();opt.zero_grad(set_to_none=True)
 if arm in ('conditioned','everywhere'):
  # Sequential backward in both FiLM arms: same MSE + pinball objective.
  with torch.autocast('cuda',dtype=torch.bfloat16):mse=(model.base(y).float()-x).square().mean()
  mse.backward();q=torch.rand(len(y),device='cuda',generator=qrng)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   f=None if arm=='everywhere' else model.base.up2(model.base.features(y))
   loss=pinball(model.predict(y,f,q).float(),x,q)
  loss.backward();loss=loss.detach()+mse.detach()
 elif arm=='fixed':
  with torch.autocast('cuda',dtype=torch.bfloat16):
   f,_=model.features(y);pred=model.predict(y,f)
   loss=sum(pinball(pred[:,i].float(),x,x.new_full((len(y),),q)) for i,q in enumerate((.05,.95)))/2
  loss.backward()
 else:
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if arm.startswith('ensemble_'):
    mean,var=model(y);mean,var=mean.float(),var.float();loss=.5*(var.log()+(mean-x).square()/var).mean()
   else:loss=(model(y).float()-x).square().mean()
  loss.backward()
 if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 return float(loss)

@torch.inference_mode()
def validate(model,arm,loader,limit=None):
 model.eval();sums=dict(psnr=0.,ssim=0.,mse=0.);count=0
 for i,b in enumerate(loader):
  if limit is not None and i>=limit:break
  y,x=b[:,0].cuda(),b[:,1].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   r=model.base(y) if arm in ('fixed','conditioned','everywhere') else model(y)
   if arm.startswith('ensemble_'):r=r[0]
  r=r.float().clamp(0,1);mse=(r-x).square().mean((1,2,3))
  sums['mse']+=float(mse.sum());sums['psnr']+=float((-10*mse.clamp_min(1e-12).log10()).sum())
  sums['ssim']+=float(ssim(r,x,data_range=1.,reduction='none').sum());count+=len(y)
 return {**{k:v/count for k,v in sums.items()},'count':count,'clipped':True,'mc_mode':'deterministic during training; stochastic sweep after epoch 10'}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);p.add_argument('--camera',choices=['rml','diffuser']);p.add_argument('--arm',choices=ARMS+['fixed']);p.add_argument('--task',type=int);p.add_argument('--epochs',type=int,default=2);a=p.parse_args()
 if a.task is not None:a.camera=('rml','diffuser')[a.task//len(ARMS)];a.arm=ARMS[a.task%len(ARMS)]
 if not a.camera or not a.arm:raise ValueError('Missing camera/arm')
 if a.epochs not in range(1,11):raise ValueError('Only ten epochs authorized')
 torch.set_num_threads(2);out=ROOT/a.camera/a.arm;out.mkdir(parents=True,exist_ok=True)
 data,_=paths(a.camera);digest=manifest_digest(data);trainset,valset=loaders(a.camera)
 model=make_model(a.arm)
 if a.arm=='fixed':
  base,ck=load_model(a.camera,'normal');model.base.load_state_dict(base.state_dict())
  with torch.no_grad():
   model.head.weight.copy_(base.head.weight.repeat(2,1,1,1));model.head.bias.copy_(base.head.bias.repeat(2))
  del base,ck
 model.cuda();opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=1e-4)
 # Dedicated streams keep sample order and q draws matched despite extra FiLM layers.
 seed=20261004;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 qrng=torch.Generator(device='cuda').manual_seed(seed+1);drng=torch.Generator().manual_seed(seed+2)
 meta=dict(camera=a.camera,arm=a.arm,manifest_sha256=digest,initialization='fresh',epochs=10,batch_size=2,lr=1e-4,weight_decay=1e-4,train_count=len(trainset),val_count=len(valset),levels=[.05,.95],dropout_probability=PROBS.get(a.arm),film_locations=getattr(model,'film_locations',[]))
 history=[];start=0
 if a.action=='train' and (out/'latest.pt').exists():
  ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  if ck['manifest_sha256']!=digest:raise ValueError('Resume split mismatch')
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];start=ck['epoch']
  torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);qrng.set_state(ck['q_rng']);drng.set_state(ck['data_rng']);del ck
 if start and start%5==0 and not (out/'checkpoints'/f'epoch_{start:03d}.pt').exists():
  dest=out/'checkpoints'/f'epoch_{start:03d}.pt';dest.parent.mkdir(exist_ok=True);torch.save(dict(**meta,model=model.state_dict(),epoch=start,history=history),dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
 train=DataLoader(trainset,batch_size=2,shuffle=True,generator=drng,num_workers=2,pin_memory=True)
 val=DataLoader(valset,batch_size=2,num_workers=2,pin_memory=True)
 if a.action=='smoke':
  model.train();times=[]
  for i,b in enumerate(train):
   torch.cuda.synchronize();t=time.monotonic();step(model,a.arm,b,opt,qrng);torch.cuda.synchronize()
   if a.arm=='everywhere' and any(f.affine.weight.grad is None for f in model.deep_films.values()):raise ValueError('Disconnected FiLM')
   if i>=2:times.append(time.monotonic()-t)
   if i==6:break
  t=time.monotonic();v=validate(model,a.arm,val,limit=3);torch.cuda.synchronize();val_batch=(time.monotonic()-t)/3
  estimate=2*(sum(times)/len(times)*len(train)+val_batch*len(val))*1.3
  report=dict(**meta,passed=estimate<8*3600,estimated_two_epoch_hours=estimate/3600,gpu=torch.cuda.get_device_name(),validation=v)
  write(out/'smoke.json',report)
  if not report['passed']:raise RuntimeError('Two-epoch runtime exceeds eight-hour limit')
  print(json.dumps(report),flush=True);return
 if not 0<=a.epochs-start<=2:raise ValueError('Each job advances at most two epochs')
 smoke=json.loads((out/'smoke.json').read_text())
 if not smoke['passed'] or smoke['manifest_sha256']!=digest:raise ValueError('Smoke gate failed')
 write(out/'config.json',meta)
 for epoch in range(start,a.epochs):
  model.train();begin=time.monotonic();total=0.
  for b in train:total+=step(model,a.arm,b,opt,qrng)*len(b)
  validation=validate(model,a.arm,val)
  row=dict(epoch=epoch+1,loss=total/len(trainset),validation=validation,seconds=time.monotonic()-begin,gpu=torch.cuda.get_device_name());history.append(row)
  ck=dict(**meta,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),q_rng=qrng.get_state(),data_rng=drng.get_state())
  torch.save(ck,out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
  if (epoch+1)%5==0:
   dest=out/'checkpoints'/f'epoch_{epoch+1:03d}.pt';dest.parent.mkdir(exist_ok=True)
   if dest.exists():raise FileExistsError(dest)
   torch.save(dict(**meta,model=model.state_dict(),epoch=epoch+1,history=history),dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
  write(out/'history.json',history);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
