"""Four models, fixed first 1000 test images, calibration alpha=.1."""
import sys,json,hashlib,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from scripts.calibrate_formal95 import predictions,moments
from scripts.formal_quantile_evaluate import arrays,widths

OUT=Path('results/four_models_1000_alpha10')
QR=Path('results/formal_quantile_comparison')
BR=Path('results/formal_rml_25k_epoch20')
BC=Path('results/formal_rml_calibration95')
TD=Path('data/formal_rml_25k');CD=Path('data/formal_rml_calibration95')
METHODS={'fixed':'im2im-style fixed head','conditioned':'Q-conditioned ConvRML','ensemble':'Deep ensemble','mc':'MC dropout'}

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    th=hashlib.sha256((TD/'manifest.json').read_bytes()).hexdigest()
    ch=hashlib.sha256((CD/'manifest.json').read_bytes()).hexdigest()
    qreport=json.loads((QR/'comparison.json').read_text());breport=json.loads((BR/'comparison.json').read_text())
    assert breport['manifest_sha256']==th
    for k in ('fixed','conditioned'):
        r=qreport['selected_checkpoints'][k]
        assert r['train_manifest_sha256']==th and r['calibration_manifest_sha256']==ch
    bf=json.loads((BC/'factors.json').read_text());assert bf['manifest_sha256']==ch
    factors={}
    for k in METHODS:
        # Earlier quantile selection permuted score arrays but preserved all values.
        path=QR/k/'calibration_scores.npy' if k in ('fixed','conditioned') else BC/f'{k}_scores.npy'
        scores=np.load(path,mmap_mode='r')
        assert scores.shape==(2048,3,256,256)
        if not np.isfinite(scores).all():raise ValueError('Invalid calibration scores')
        factors[k]=max(0.,float(np.quantile(scores,.90,method='higher')))
        print('FITTED',k,factors[k],flush=True)
        del scores
    (OUT/'calibration.json').write_text(json.dumps({'alpha':.1,'calibration_count':2048,'factors':factors,'calibration_manifest_sha256':ch},indent=2))
    pairs=np.load(TD/'pairs.npy',mmap_mode='r');manifest=json.loads((TD/'manifest.json').read_text());ids=[r['id'] for r in manifest['test'][:1000]]
    results={};records=[]
    for k,label in METHODS.items():
        pred=arrays(k,'test') if k in ('fixed','conditioned') else predictions(BR,k)
        per=[]
        for s in range(0,1000,8):
            e=min(s+8,1000);x=pairs[22500+s:22500+e,1].astype(np.float32)/255
            if k in ('fixed','conditioned'):
                r=pred['r'][s:e];wl,wu=widths(r,pred['lo'][s:e],pred['hi'][s:e]);lo=r-factors[k]*wl;hi=r+factors[k]*wu
            else:r,std=moments(pred,s,e);lo=r-factors[k]*std;hi=r+factors[k]*std
            mse=np.mean((r.astype(np.float64)-x)**2,axis=(1,2,3));psnr=-10*np.log10(mse)
            ssim=structural_similarity_index_measure(torch.from_numpy(np.array(r,copy=True)),torch.from_numpy(x),data_range=1.,reduction='none').numpy()
            cov=((lo<=x)&(x<=hi)).mean((1,2,3));wid=(hi-lo).mean((1,2,3))
            for j in range(e-s):
                row={'method':label,'id':ids[s+j],'ssim':float(ssim[j]),'psnr_db':float(psnr[j]),'coverage':float(cov[j]),'width':float(wid[j]),'mse':float(mse[j])}
                records.append(row);per.append(row)
            if s%200==0:print('EVALUATED',k,e,'/1000',flush=True)
        results[k]={'label':label,'mean_ssim':float(np.mean([v['ssim'] for v in per])),'mean_psnr_db':float(np.mean([v['psnr_db'] for v in per])),'coverage':float(np.mean([v['coverage'] for v in per])),'mean_width':float(np.mean([v['width'] for v in per])),'pooled_mse_psnr_db':float(-10*np.log10(np.mean([v['mse'] for v in per])))}
        print(json.dumps(results[k]),flush=True)
    report={'count':1000,'alpha':.1,'target_coverage':.9,'selection':'First 1000 entries of existing held-out test manifest, identical for every method','test_manifest_sha256':th,'image_ids':ids,'calibration_count':2048,'factors':factors,'metrics':results,'definitions':['SSIM: per-image TorchMetrics Gaussian SSIM, data_range=1; then mean.','PSNR: mean of per-image PSNR in dB, not PSNR of pooled MSE.','Coverage: inclusive pixel/channel interval coverage, averaged equally over images.','Unclipped RGB predictions and target range [0,1].','Existing .025/.975 quantile-head outputs are retained as width heuristics and rescaled to the new 90% target.','Independent empirical calibration; no test tuning or certified coverage.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    with (OUT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Four models: 1,000 test images, alpha = 0.1','', '| Model | Mean SSIM | Mean PSNR (dB) | Coverage | Mean width |','|---|---:|---:|---:|---:|']
    for r in results.values():lines.append(f'| {r["label"]} | {r["mean_ssim"]:.4f} | {r["mean_psnr_db"]:.2f} | {r["coverage"]:.2%} | {r["mean_width"]:.5f} |')
    lines+=['','90% calibration target fitted on 2,048 separate images. Same first 1,000 test images for all models.','PSNR is the average of per-image dB values; previous reports used pooled-MSE PSNR.','No retraining: the existing interval predictions were recalibrated to alpha=0.1.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
