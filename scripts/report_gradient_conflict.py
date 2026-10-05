"""Summarize validation-only gradient-conflict pilot without declaring causality."""
import json
from pathlib import Path
root=Path('results/gradient_conflict_20260924/pilot')
reports={arm:json.loads((root/arm/'complete.json').read_text()) for arm in ['frozen','joint','mse']}
assert len({r['config']['initial_model_sha256'] for r in reports.values()})==1
text=['# Shared-backbone gradient experiment','',
      'Two-epoch, single-seed pilot; all arms start from the same MSE checkpoint. Same 20,000 training images, batch ordering and sampled quantiles. Evaluation: first 256 validation images; no test images or calibration.', '',
      '| Arm | Initial PSNR | Final PSNR | Change (dB) | Initial SSIM | Final SSIM | Endpoint pinball |',
      '|---|---:|---:|---:|---:|---:|---:|']
for arm,r in reports.items():
    first=r['history'][0]['validation'];last=r['history'][-1]['validation']
    text.append(f"| {arm} | {first['psnr_db']:.4f} | {last['psnr_db']:.4f} | {last['psnr_db']-first['psnr_db']:+.4f} | {first['ssim']:.5f} | {last['ssim']:.5f} | {last['endpoint_pinball']:.6f} |")
text+=['','MSE-only arm quantile endpoints are untrained controls; do not interpret their pinball score as a competing uncertainty method.', '',
       '| Arm / epoch | Mean gradient cosine | Negative-cosine batches | Mean pinball/MSE norm ratio |',
       '|---|---:|---:|---:|']
for arm,r in reports.items():
    for row in r['history']:
        g=row['gradients'];n=len(g)
        text.append(f"| {arm} / {row['epoch']} | {sum(v['cosine'] for v in g)/n:.5f} | {sum(v['cosine']<0 for v in g)}/{n} | {sum(v['pinball_to_mse_norm'] for v in g)/n:.3f} |")
text+=['','Interpretation: negative cosine indicates opposing first-order directions on the probed shared parameters; a large norm ratio indicates unequal gradient scales despite equal loss coefficients. Neither alone proves reconstruction harm under AdamW. Compare joint versus MSE-only changes to isolate adding quantile supervision during fine-tuning. The frozen arm checks reconstruction preservation. A single seed, four diagnostic batches, 256 validation images and two epochs are preliminary; this does not establish why the prior from-scratch run differed.']
(root/'comparison.md').write_text('\n'.join(text)+'\n')
(root/'comparison.json').write_text(json.dumps(reports,indent=2))
print('\n'.join(text))
