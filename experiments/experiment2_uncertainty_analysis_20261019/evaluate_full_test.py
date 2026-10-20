"""Frozen epoch-50 full-test LPIPS and error/uncertainty correlations.

Calibrations are reused; no fitting on test data, no training, no region metrics.
Only per-image metrics and fixed first-three visualization maps are saved.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from metrics import analyze,describe,correlation
PROJECT=Path.home()/'ConvRML';ROOT=Path(__file__).resolve().parent
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--camera',choices=['rml','diffuser'],required=True);p.add_argument('--method',choices=['im2im','ensemble5','mc_005','qutcc'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(202610200+['im2im','ensemble5','mc_005','qutcc'].index(a.method)+(10 if a.camera=='diffuser' else 0))
 out=ROOT/'results'/a.camera/a.method;out.mkdir(parents=True,exist_ok=True)
 if not a.smoke and (out/'summary.json').exists():print('Already complete');return
 if a.method=='qutcc':
  paper=PROJECT/'experiments/qutcc_paper_style_20261018';sys.path.insert(0,str(paper))
  from model import Model
  from data import Pairs,identity
  folder=paper/'results'/a.camera/'qutcc';r=json.loads((folder/'evaluation/epoch_050/report.json').read_text());ckpath=folder/'checkpoints/epoch_050.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  assert ck['config']['dataset_identity']==identity(a.camera) and r['epoch']==50
  model=Model('qutcc');model.load_state_dict(ck['model'],strict=True);model=model.cuda().eval();del ck
  cal=r['calibration'];dataset=Pairs(a.camera,'test');records=dataset.rows
  def predict(y):return model(y),model(y,y.new_full((len(y),),cal['lower_q'])),model(y,y.new_full((len(y),),cal['upper_q']))
  provenance={'checkpoint':str(ckpath),'precision':'float32','calibration':cal}
 else:
  sys.path.insert(0,str(PROJECT))
  import scripts.corrected_comparison_study as study
  study.ROOT=ROOT/'weights'
  from scripts.export_corrected_milestone import predictor
  from scripts.formal_rml_train import Pairs
  predict_native,provenance=predictor(a.camera,a.method,50)
  folder=PROJECT/'results'/('ensemble_size_comparison_20261015' if a.method=='ensemble5' else 'corrected_demo_epoch50_20261011')/a.camera/a.method
  r=json.loads((folder/'report.json').read_text());factors=r['factors']['rcps_hb'];assert r['epoch']==50 and r['calibration_count']==2048
  dataset=Pairs('test',study.paths(a.camera)[0]);records=json.loads((study.paths(a.camera)[0]/'manifest.json').read_text())['test'];cal={'method':'rcps_hb_asymmetric','alpha':.1,'delta':.05,'factors':factors}
  def predict(y):
   with torch.autocast('cuda',dtype=torch.bfloat16):pred,lo,hi=predict_native(y)
   pred,lo,hi=[v.float() for v in (pred,lo,hi)]
   return pred,pred-factors['lower']*(pred-lo).clamp_min(1e-6),pred+factors['upper']*(hi-pred).clamp_min(1e-6)
 assert len(dataset)==5000
 lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval();n=4 if a.smoke else 5000
 rows=[];start=time.monotonic();maps_saved={};digest=hashlib.sha256()
 for i in range(0,n,4):
  b=torch.stack([dataset[j] for j in range(i,min(i+4,n))]);y,x=b[:,0].cuda(),b[:,1].cuda();pred,lo,hi=predict(y);pred,lo,hi=[v.clamp(0,1) for v in (pred,lo,hi)]
  assert all(torch.isfinite(v).all() for v in (pred,lo,hi))
  lpips.reset();scores=lpips(pred,x).cpu().numpy().reshape(-1);ssims=ssim(pred,x,data_range=1.,reduction='none').cpu().numpy().reshape(-1)
  pp,xx,ll,uu=[v.cpu().numpy() for v in (pred,x,lo,hi)]
  for j in range(len(pp)):
   index=i+j;identifier=records[index]['id'];digest.update((str(identifier)+'\n').encode());scales=analyze(pp[j],xx[j],ll[j],uu[j]);err=pp[j]-xx[j]
   row={'camera':a.camera,'method':a.method,'epoch':50,'id':identifier,'index':index,'lpips':float(scores[j]),'ssim':float(ssims[j]),'psnr':float(-10*np.log10(max(float(np.mean(err**2)),1e-12))),'coverage':float(np.mean((xx[j]>=ll[j])&(xx[j]<=uu[j]))),'width':float(np.mean(uu[j]-ll[j])),'crossing':float(np.mean(ll[j]>uu[j])),'scales':scales};rows.append(row)
   if index in [0,1,5]:
    for name,value in [('measurement',b[j,0].numpy()),('target',xx[j]),('prediction',pp[j]),('lower',ll[j]),('upper',uu[j])]:maps_saved[f'{index}_{name}']=value
  if len(rows)%100==0 or len(rows)==n:
   (out/'progress.json').write_text(json.dumps({'processed':len(rows),'total':n,'seconds':time.monotonic()-start}));print(a.camera,a.method,len(rows),'/',n,flush=True)
 if a.smoke:print('SMOKE PASS',len(rows),rows[0]['lpips'],flush=True);return
 (out/'per_image.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(out/'examples.npz',**maps_saved)
 summary={'camera':a.camera,'method':a.method,'epoch':50,'n_images':n,'split':'full held-out test','test_id_sequence_sha256':digest.hexdigest(),'lpips_variant':'AlexNet v0.1, normalize=True for [0,1] RGB inputs','smoothing':'Primary sigma=0; secondary 0.5,1,2,4 pixels, no region statistics','calibration':cal,'provenance':provenance,'metrics':{k:describe([r[k] for r in rows]) for k in ['lpips','ssim','psnr','coverage','width','crossing']},'scales':{s:{k:describe([r['scales'][s][k] for r in rows]) for k in ['pearson','spearman','mae','mean_width']} for s in ['0','0.5','1','2','4']},'image_level_width_vs_mae':correlation([r['width'] for r in rows],[r['scales']['0']['mae'] for r in rows]),'seconds':time.monotonic()-start,'gpu':torch.cuda.get_device_name(),'training':False}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
