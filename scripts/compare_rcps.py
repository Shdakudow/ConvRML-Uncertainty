"""Paired empirical versus RCPS calibration; existing trained weights only."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from scripts import large_interval_evaluate as original
from scripts.rcps_bounds import select_factors
from scripts.training_records import manifest_digest, split_bounds

OUT=Path('results/rcps_vs_empirical_rml40_diffuser35_20261001')
METHODS=('ensemble','mc','fixed','conditioned')

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True);parser.add_argument('--epoch',type=int)
    a=parser.parse_args()
    if a.worker not in range(8):raise ValueError('worker must be 0..7')
    camera='rml' if a.worker<4 else 'diffuser'
    worker=a.worker%4;method=METHODS[worker];epoch=a.epoch or (40 if camera=='rml' else 35)
    if epoch not in range(5,51,5):raise ValueError('Invalid milestone')
    study='optics_intervals_60k_5k_5k_20260928' if camera=='rml' else 'diffuser_intervals_60k_5k_5k_20260928'
    original.ROOT=Path('results')/study;original.DATA=Path('data')/study
    caldir=Path('data/formal_rml_calibration95' if camera=='rml' else 'data/diffuser_intervals_calibration_20260928')
    out=(Path(f'results/rcps_vs_empirical_epoch{epoch}') if a.epoch else OUT)/camera/method;out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():
        print('Already complete:',out,flush=True);return
    torch.set_num_threads(2);torch.manual_seed(20260928+worker)
    begin=time.monotonic()
    predict,gpu_hours=original.load_predictor(method,epoch)
    cal=np.load(caldir/'pairs.npy',mmap_mode='r')
    test=np.load(original.DATA/'pairs.npy',mmap_mode='r')
    offset,count,_=split_bounds(original.DATA,'test')
    def batches(pairs,start,count):
        for i in range(0,count,2):
            b=torch.from_numpy(np.array(pairs[start+i:start+min(i+2,count)],copy=True)).cuda().float()/255
            with torch.autocast('cuda',dtype=torch.bfloat16):r,lo,hi=predict(b[:,0])
            r,lo,hi=r.float(),lo.float(),hi.float()
            if not all(torch.isfinite(v).all() for v in (r,lo,hi)):raise RuntimeError('Nonfinite predictions')
            yield i,b[:,1],r,(r-lo).clamp_min(1e-6),(hi-r).clamp_min(1e-6)
    path=out/'calibration_scores.tmp.npy'
    scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(cal),*cal.shape[2:]))
    for i,x,r,wl,wu in batches(cal,0,len(cal)):
        scores[i:i+len(x)]=torch.maximum((r-x)/wl,(x-r)/wu).cpu().numpy()
    scores.flush()
    factors,audit,losses=select_factors(scores)
    with (out/'calibration_per_image.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['calibration_index','empirical_loss','rcps_hb_loss'])
        writer.writerows((i,losses['empirical'][i],losses['rcps_hb'][i]) for i in range(len(cal)))
    del scores;path.unlink()
    calibration_seconds=time.monotonic()-begin
    print(json.dumps(dict(camera=camera,method=method,factors=factors,calibration=audit)),flush=True)
    rows=[]
    for start,x,r,wl,wu in batches(test,offset,count):
        intervals={k:(r-v*wl,r+v*wu) for k,v in factors.items()}
        for i in range(len(x)):
            error=(r[i]-x[i]).abs()
            row=dict(test_index=start+i,psnr=-10*math.log10(max(float(error.square().mean()),1e-12)),
                     ssim=float(ssim(r[i:i+1],x[i:i+1],data_range=1.)),mae=float(error.mean()))
            for name,(lo,hi) in intervals.items():
                row[name+'_coverage']=float(((x[i]>=lo[i])&(x[i]<=hi[i])).float().mean())
                row[name+'_width']=float((hi[i]-lo[i]).mean())
            rows.append(row)
    with (out/'test_per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k!='test_index'}
    if means['rcps_hb_coverage']+1e-9<means['empirical_coverage']:raise RuntimeError('Coverage nesting failed')
    if means['rcps_hb_width']+1e-9<means['empirical_width']:raise RuntimeError('Width nesting failed')
    previous=original.ROOT/'evaluation'/f'epoch_{epoch:03d}'/f'{method}.json'
    agreement=None
    if previous.exists():
        old=json.loads(previous.read_text())
        agreement={k:means[k]-old['metrics'][k] for k in ('psnr','ssim','mae')}
    result=dict(camera=camera,method=method,backbone_epoch=epoch,head_epochs=10 if method=='fixed' else None,
                test_count=count,calibration_count=len(cal),alpha=.1,delta=.05,factors=factors,
                calibration=audit,metrics=means,previous_reconstruction_metric_differences=agreement,
                manifest_sha256=manifest_digest(original.DATA),calibration_sha256=manifest_digest(caldir),
                training_gpu_hours_including_validation=gpu_hours,
                paired_predictions=True,no_retraining=True,gpu=torch.cuda.get_device_name(),
                calibration_seconds=calibration_seconds,total_seconds=time.monotonic()-begin,
                risk_unit='One image; loss is fraction of uncovered RGB channel values',
                guarantee='Per model/checkpoint: population mean image loss <= alpha with confidence 1-delta, assuming independent identically distributed held-out calibration images and a fixed predictor. Not simultaneous across methods; not per-image or all-pixel coverage.',
                caveat='Exploratory reused splits; acquisition-group independence has not been verified.',
                reference='https://arxiv.org/abs/2101.02703',bound='Hoeffding-Bentkus',lambda_grid_step=1e-6)
    tmp=out/'comparison.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'comparison.json')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
