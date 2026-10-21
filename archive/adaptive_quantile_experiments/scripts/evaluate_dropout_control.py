"""Matched full-test reconstruction comparison at a saved training milestone."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from scripts.corrected_comparison_study import paths,load_model as load_reference,write
from scripts.train_dropout_control import load_model as load_control,ROOT
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest
from baselines.mc_dropout import dropout_inference

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',choices=['rml','diffuser'],required=True);p.add_argument('--epoch',type=int,choices=range(5,51,5),required=True);a=p.parse_args()
 out=ROOT/'evaluations'/a.camera/f'epoch_{a.epoch:03d}.json'
 if out.exists():print('Already complete',out);return
 torch.set_num_threads(2);torch.manual_seed(2026101505)
 data,_=paths(a.camera);digest=manifest_digest(data);dataset=Pairs('test',data);assert len(dataset)==5000
 zero,cz=load_control(a.camera,'mc_000',a.epoch);ref,cr=load_reference(a.camera,'mc_005',a.epoch)
 assert cz['seeds']==cr['seeds'] and cz['manifest_sha256']==cr['manifest_sha256']==digest
 zero.cuda().eval();ref.cuda().eval()
 names=['trained_p0','trained_p005_dropout_off','trained_p005_mc30'];values={k:[] for k in names};seconds={k:0. for k in names}
 for batch in DataLoader(dataset,batch_size=4,num_workers=2,pin_memory=True,shuffle=False):
  y,x=batch[:,0].cuda(),batch[:,1].cuda()
  for name in names:
   torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if name=='trained_p0':r=zero(y).float()
    elif name=='trained_p005_dropout_off':r=ref(y).float()
    else:
     r=None
     with dropout_inference(ref):
      for i in range(30):
       sample=ref(y).float();r=sample if r is None else r+(sample-r)/(i+1)
   torch.cuda.synchronize();seconds[name]+=time.monotonic()-start
   r=r.clamp(0,1);e=r-x
   if not torch.isfinite(r).all():raise ValueError('Nonfinite prediction')
   result=torch.stack([-10*e.square().mean((1,2,3)).clamp_min(1e-12).log10(),ssim(r,x,data_range=1.,reduction='none'),e.square().mean((1,2,3)),e.abs().mean((1,2,3))],dim=1)
   values[name].extend(result.cpu().tolist())
  if len(values[names[0]])%500==0:print(a.camera,a.epoch,len(values[names[0]]),'/5000',flush=True)
 metrics=['psnr','ssim','mse','mae'];arrays={k:np.asarray(v) for k,v in values.items()}
 assert all(v.shape==(5000,4) for v in arrays.values())
 aggregates={k:dict(zip(metrics,v.mean(0).tolist())) for k,v in arrays.items()}
 comparisons={}
 for label,left,right in [('training_effect_p005_minus_p0',names[1],names[0]),('inference_effect_mc30_minus_off',names[2],names[1])]:
  d=arrays[left]-arrays[right]
  comparisons[label]=dict(mean_difference=dict(zip(metrics,d.mean(0).tolist())),fraction_images_higher_ssim=float((d[:,1]>0).mean()))
 write(out,dict(camera=a.camera,epoch=a.epoch,test_count=5000,manifest_sha256=digest,seeds=cz['seeds'],clipped=True,
  aggregates=aggregates,paired_comparisons=comparisons,forward_seconds=seconds,seconds_per_image={k:v/5000 for k,v in seconds.items()},gpu=torch.cuda.get_device_name(),
  interval_metrics=None,notes='Point-reconstruction control. No intervals are assigned to deterministic p=0 or dropout-off inference. No calibration or test-based tuning. Timings measure model forward passes only, excluding data loading, metrics and synchronization overhead.',per_image_metric_order=metrics,per_image=values))
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
