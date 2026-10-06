"""Small final-epoch demo export; requires all four evaluation reports."""
import argparse,base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from scripts import large_interval_evaluate as evaluation
from scripts.training_records import split_bounds,manifest_digest

def enc(a):
 a=np.asarray(a,dtype='<f4')
 if not np.isfinite(a).all():raise ValueError('Nonfinite display data')
 return base64.b64encode(a.tobytes()).decode()

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--camera',choices=['rml','diffuser'],required=True);p.add_argument('--epoch',type=int,default=50);p.add_argument('--calibration',choices=['empirical','rcps'],default='rcps');p.add_argument('--output-root',type=Path);p.add_argument('--count',type=int,default=50);a=p.parse_args()
 if not 1<=a.count<=5000:raise ValueError('Invalid example count')
 if a.epoch not in range(5,51,5):raise ValueError('Invalid milestone')
 name='optics_intervals_60k_5k_5k_20260928' if a.camera=='rml' else 'diffuser_intervals_60k_5k_5k_20260928'
 evaluation.ROOT=Path('results')/name;evaluation.DATA=Path('data')/name
 out=(a.output_root or Path(f'results/error_width_demo_epoch{a.epoch}'))/a.camera/'data';out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2)
 manifest=json.loads((evaluation.DATA/'manifest.json').read_text())
 start,_,_=split_bounds(evaluation.DATA,'test');pairs=np.load(evaluation.DATA/'pairs.npy',mmap_mode='r')[start:start+a.count].copy()
 ids=[v['id'] for v in manifest['test'][:a.count]];objects=[]
 for i in range(a.count):objects.append(dict(id=ids[i],measurement=base64.b64encode(pairs[i,0].tobytes()).decode(),target=base64.b64encode(pairs[i,1].tobytes()).decode(),models={}))
 maps=[];aggregates={};factors={}
 for worker,method in enumerate(evaluation.METHODS):
  report=json.loads((evaluation.ROOT/'evaluation'/f'epoch_{a.epoch:03d}'/f'{method}.json').read_text())
  if report['manifest_sha256']!=manifest_digest(evaluation.DATA):raise ValueError('Dataset mismatch')
  factor=report['factor'];aggregates[method]=report['metrics']
  if a.calibration=='rcps':
   rcps=json.loads((Path(f'results/rcps_vs_empirical_epoch{a.epoch}')/a.camera/method/'comparison.json').read_text())
   if rcps['backbone_epoch']!=a.epoch or rcps['manifest_sha256']!=report['manifest_sha256'] or rcps['alpha']!=.1 or rcps['delta']!=.05:raise ValueError('RCPS provenance mismatch')
   factor=rcps['factors']['rcps_hb']
   if rcps['calibration']['hb_pvalues']['rcps_hb']>=.05:raise ValueError('RCPS certificate failed')
   v=rcps['metrics'];aggregates[method]={k:v[k] for k in ('psnr','ssim','mae')}
   aggregates[method].update(coverage=v['rcps_hb_coverage'],width=v['rcps_hb_width'])
  factors[method]=factor
  torch.manual_seed(20261002+worker)
  predict,_=evaluation.load_predictor(method,a.epoch)
  for i in range(0,a.count,2):
   b=torch.from_numpy(pairs[i:i+2].copy()).cuda().float()/255;y,x=b[:,0],b[:,1]
   with torch.autocast('cuda',dtype=torch.bfloat16):r,lo,hi=predict(y)
   r,lo,hi=r.float(),lo.float(),hi.float()
   lower=r-factor*(r-lo).clamp_min(1e-6);upper=r+factor*(hi-r).clamp_min(1e-6)
   r=r.clamp(0,1);lower=lower.clamp(0,1);upper=upper.clamp(0,1)
   for j in range(len(y)):
    width=upper[j]-lower[j];error=(r[j]-x[j]).abs()
    if not torch.isfinite(width).all() or not torch.isfinite(error).all():raise ValueError('Nonfinite prediction')
    maps.extend([width.mean(0).cpu().numpy(),error.mean(0).cpu().numpy()])
    objects[i+j]['models'][method]=dict(mean=enc(r[j].cpu().numpy()),width=enc(width.mean(0).cpu().numpy()),widthRGB=enc(width.cpu().numpy()),metrics=dict(psnr=float(-10*error.square().mean().clamp_min(1e-12).log10()),ssim=float(ssim(r[j:j+1],x[j:j+1],data_range=1.)),mae=float(error.mean()),coverage=float(((x[j]>=lower[j])&(x[j]<=upper[j])).float().mean()),width=float(width.mean())))
  del predict;torch.cuda.empty_cache();print(a.camera,method,'complete',flush=True)
 meta=dict(metricClipping='[0,1]',aggregateMetricClipping='unclipped',study='RML' if a.camera=='rml' else 'DiffuserCam',epoch=a.epoch,headEpochs=10,trainCount=60000,count=a.count,ids=ids,resolution=256,encoding='float32',mapMax=float(np.quantile(np.concatenate([m.ravel() for m in maps]),.995)),calibrationTarget=.9,calibrationCount=2048,calibrationMethod=('RCPS (Hoeffding–Bentkus)' if a.calibration=='rcps' else 'Empirical'),calibrationDelta=(.05 if a.calibration=='rcps' else None),aggregate=aggregates,factors=factors,manifest_sha256=manifest_digest(evaluation.DATA),notes=f'First {a.count} test images in manifest order; no selection by quality. 60,000 train / 5,000 validation / 5,000 test, with 2,048 separate calibration images. Exact epoch-{a.epoch} weights; fixed head trained for 10 additional epochs. {a.calibration.upper()} calibration, target 90% average coverage; measured coverage need not equal 90%. RCPS uses delta=0.05 and assumes independent calibration images; not a per-image coverage guarantee. Ensemble: five Gaussian-NLL members; MC dropout: 30 fresh reproducible draws, seed 20261002 + method index. Both maps share their pooled 99.5th-percentile scale. Example predictions and calibrated bounds clipped to [0,1] before calculating metrics and maps. Historical full-test aggregate metrics remain unclipped. Float32 arrays; full-precision metrics. Error is averaged after taking channelwise absolute differences. Width is upper minus lower, averaged over RGB. MC per-example metrics can differ from the original evaluation because these are fresh draws.')
 for i,obj in enumerate(objects):(out/f'example_{i:02d}.js').write_text('window.receiveExample('+str(i)+','+json.dumps(obj,separators=(',',':'))+');')
 (out/'manifest.js').write_text('window.DATASET='+json.dumps(meta,separators=(',',':'))+';')
 (out/'protocol.json').write_text(json.dumps(meta,indent=2));print('Export complete:',out,flush=True)

if __name__=='__main__':main()
