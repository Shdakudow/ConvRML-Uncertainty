"""Validation-only dropout selection, then frozen-policy held-out RCPS tests."""
import argparse,csv,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from baselines.mc_dropout import dropout_inference
from scripts.corrected_comparison_study import ROOT,PROBS,paths,load_model,write
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest
from scripts.asymmetric_rcps import select_asymmetric_factors
METHODS=['im2im','conditioned','ensemble5','ensemble10','mc_baseline30','mc_selected100']

@torch.inference_mode()
def tune(camera,arm):
 out=ROOT/camera/'dropout_validation'/f'{arm}.json'
 if out.exists():return
 torch.set_num_threads(2);torch.manual_seed(20261005);net,ck=load_model(camera,arm);net.cuda().eval()
 totals={n:dict(ssim=0.,psnr=0.,mae=0.,inference_seconds=0.) for n in (30,60,100)};count=0
 for b in DataLoader(Pairs('val',paths(camera)[0]),batch_size=2,num_workers=2):
  y,x=b[:,0].cuda(),b[:,1].cuda();mean=None;elapsed=0.
  with dropout_inference(net):
   for j in range(100):
    torch.cuda.synchronize();t=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):sample=net(y).float()
    mean=sample if mean is None else mean+(sample-mean)/(j+1)
    torch.cuda.synchronize();elapsed+=time.monotonic()-t
    if j+1 in totals:
     r=mean.clamp(0,1);e=r-x;mse=e.square().mean((1,2,3));v=totals[j+1]
     v['ssim']+=float(ssim(r,x,data_range=1.,reduction='none').sum());v['psnr']+=float((-10*mse.clamp_min(1e-12).log10()).sum());v['mae']+=float(e.abs().mean((1,2,3)).sum());v['inference_seconds']+=elapsed
  count+=len(y)
 results={n:{k:(v if k=='inference_seconds' else v/count) for k,v in row.items()} for n,row in totals.items()}
 write(out,dict(camera=camera,arm=arm,dropout_probability=PROBS[arm],epoch=50,validation_count=count,manifest_sha256=ck['manifest_sha256'],clipped=True,results=results,selection_split='val',test_used=False,gpu=torch.cuda.get_device_name()))

def select(camera):
 rows=[json.loads((ROOT/camera/'dropout_validation'/f'{arm}.json').read_text()) for arm in PROBS]
 if any(r['validation_count']!=5000 or r['test_used'] or r['manifest_sha256']!=manifest_digest(paths(camera)[0]) for r in rows):raise ValueError('Invalid selection evidence')
 # Predeclared: select p by full validation SSIM at 100 passes, tie-break lower p.
 best=max(rows,key=lambda r:(r['results']['100']['ssim'],-r['dropout_probability']))
 policy=dict(camera=camera,arm=best['arm'],dropout_probability=best['dropout_probability'],mc_passes=100,coverage_alpha=.1,selection='Highest clipped validation SSIM at 100 passes; ties choose lower p',validation_count=5000,test_used=False,locked=True,source_epoch=50,manifest_sha256=best['manifest_sha256'],trend=rows,note='Use this dropout probability during subsequent training and inference. 100 passes fixed in advance; improvements are not guaranteed.')
 out=ROOT/camera/'dropout_policy.json'
 if out.exists() and json.loads(out.read_text())!=policy:raise ValueError('Refusing to change locked policy')
 write(out,policy)
 trend_path=ROOT/camera/'dropout_validation'/'trend.csv'
 with trend_path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['dropout_probability','passes','ssim','psnr','mae','inference_seconds']);writer.writeheader()
  for row in rows:
   for passes,v in row['results'].items():writer.writerow(dict(dropout_probability=row['dropout_probability'],passes=passes,**v))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for passes in ('30','60','100'):
  for ax,metric,label in [(axes[0],'ssim','Validation SSIM'),(axes[1],'psnr','Validation PSNR (dB)')]:
   ax.plot([r['dropout_probability'] for r in rows],[r['results'][passes][metric] for r in rows],marker='o',label=passes+' passes');ax.set_xlabel('Dropout probability p');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
 fig.suptitle(camera+' · epoch 50 · 5,000 validation images · clipped metrics');fig.tight_layout();fig.savefig(trend_path.with_suffix('.png'),dpi=180);plt.close(fig)
 print(json.dumps(policy),flush=True)

def predictor(camera,method):
 arms=[f'ensemble_{i:02d}' for i in range(5 if method=='ensemble5' else 10)] if method.startswith('ensemble') else [method]
 samples=None
 if method.startswith('mc_'):
  if method=='mc_baseline30':arms=['mc_010'];samples=30
  else:
   policy=json.loads((ROOT/camera/'dropout_policy.json').read_text());arms=[policy['arm']];samples=policy['mc_passes']
 models=[];seconds=0.
 for arm in arms:
  net,ck=load_model(camera,arm);models.append(net.cuda().eval());seconds+=sum(r['seconds'] for r in ck['history'])
 def predict(y):
  if method.startswith('ensemble'):
   mu=None;second=None
   for net in models:
    r,v=net(y);r,v=r.float(),v.float();mu=r if mu is None else mu+r;second=v+r.square() if second is None else second+v+r.square()
   r=mu/len(models);variance=(second/len(models)-r.square()).clamp_min(0)
  elif method.startswith('mc_'):
   r=None;m2=None
   with dropout_inference(models[0]):
    for i in range(samples):
     sample=models[0](y).float()
     if r is None:r=sample;m2=torch.zeros_like(r)
     else:d=sample-r;r=r+d/(i+1);m2=m2+d*(sample-r)
   variance=m2/samples
  else:return tuple(t.float() for t in models[0].bounds(y))
  radius=1.6448536269514722*variance.sqrt();return r,r-radius,r+radius
 return predict,dict(arms=arms,mc_passes=samples,training_gpu_hours=seconds/3600)

@torch.inference_mode()
def evaluate(camera,method):
 out=ROOT/camera/'evaluation'/method;out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():return
 torch.set_num_threads(2);torch.manual_seed(20261006);predict,metadata=predictor(camera,method)
 data,caldir=paths(camera);cal=np.load(caldir/'pairs.npy',mmap_mode='r');begin=time.monotonic()
 def predict_batch(b):
  y,x=b[:,0].cuda(),b[:,1].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):r,lo,hi=predict(y)
  if not all(torch.isfinite(t).all() for t in (r,lo,hi)):raise ValueError('Nonfinite inference')
  return x,r,(r-lo).clamp_min(1e-6),(hi-r).clamp_min(1e-6)
 lower_path=out/'lower_scores.tmp.npy';upper_path=out/'upper_scores.tmp.npy'
 lower_scores=np.lib.format.open_memmap(lower_path,mode='w+',dtype=np.float32,shape=(len(cal),*cal.shape[2:]))
 upper_scores=np.lib.format.open_memmap(upper_path,mode='w+',dtype=np.float32,shape=(len(cal),*cal.shape[2:]))
 for i in range(0,len(cal),2):
  x,r,wl,wu=predict_batch(torch.from_numpy(np.array(cal[i:i+2],copy=True)).float()/255)
  lower_scores[i:i+len(x)]=((r-x)/wl).cpu().numpy();upper_scores[i:i+len(x)]=((x-r)/wu).cpu().numpy()
 lower_scores.flush();upper_scores.flush()
 factors,audit,losses=select_asymmetric_factors(lower_scores,upper_scores)
 del lower_scores,upper_scores;lower_path.unlink();upper_path.unlink();cal_seconds=time.monotonic()-begin
 np.savetxt(out/'calibration_losses.csv',np.stack([losses['empirical'],losses['rcps_hb']],1),delimiter=',',header='empirical,rcps_hb')
 rows=[];inference_seconds=0.
 for b in DataLoader(Pairs('test',data),batch_size=2,num_workers=2):
  torch.cuda.synchronize();t=time.monotonic();x,r,wl,wu=predict_batch(b);torch.cuda.synchronize();inference_seconds+=time.monotonic()-t
  clipped=r.clamp(0,1)
  for j in range(len(x)):
   e=clipped[j]-x[j];raw=r[j]-x[j]
   row=dict(index=len(rows),psnr=-10*math.log10(max(float(e.square().mean()),1e-12)),ssim=float(ssim(clipped[j:j+1],x[j:j+1],data_range=1.)),mae=float(e.abs().mean()),unclipped_psnr=-10*math.log10(max(float(raw.square().mean()),1e-12)))
   for key,factor in factors.items():
    lo=(r[j]-factor['lower']*wl[j]).clamp(0,1);hi=(r[j]+factor['upper']*wu[j]).clamp(0,1)
    row[key+'_coverage']=float(((x[j]>=lo)&(x[j]<=hi)).float().mean());row[key+'_width']=float((hi-lo).mean())
   rows.append(row)
 with (out/'test_per_image.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 write(out/'metrics.json',dict(camera=camera,method=method,backbone_epoch=50,head_epochs=None,training_protocol='end-to-end corrected protocol',calibration_scales='separate lower and upper',calibration_count=len(cal),test_count=len(rows),manifest_sha256=manifest_digest(data),calibration_sha256=manifest_digest(caldir),alpha=.1,delta=.05,factors=factors,calibration=audit,clipped=True,metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='index'},calibration_seconds=cal_seconds,inference_seconds=inference_seconds,total_seconds=time.monotonic()-begin,**metadata))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tune','select','evaluate']);p.add_argument('--camera',choices=['rml','diffuser']);p.add_argument('--task',type=int);a=p.parse_args()
 if a.action=='tune':tune(('rml','diffuser')[a.task//3],list(PROBS)[a.task%3])
 elif a.action=='select':select(a.camera)
 else:evaluate(('rml','diffuser')[a.task//len(METHODS)],METHODS[a.task%len(METHODS)])
