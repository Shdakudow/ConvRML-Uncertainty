"""Compare new sampling/weight pilots with matched existing controls."""
import json
from pathlib import Path

old=Path('results/gradient_conflict_20260924/pilot')
root=Path('results/quantile_followups_20260924')
paths={'frozen uniform':old/'frozen/complete.json',
       'frozen endpoints':root/'endpoints/pilot/frozen/complete.json',
       'joint weight 1':old/'joint/complete.json',
       'MSE only':old/'mse/complete.json'}
for label,directory in [('joint weight 0.02','weight_002'),('joint weight 0.1','weight_01')]:
    path=root/directory/'pilot/joint/complete.json'
    if path.exists():paths[label]=path
reports={label:json.loads(path.read_text()) for label,path in paths.items()}
assert len({r['config']['initial_model_sha256'] for r in reports.values()})==1
for key in ['manifest_sha256','train_count','validation_count','epochs','seed','source_epoch']:
    assert len({r['config'][key] for r in reports.values()})==1,key
lines=['# Quantile sampling and objective-weight pilot','',
       'All runs: same MSE-trained checkpoint, same quantile-head initialization, two epochs on 20,000 training images, batch size 2, first 256 validation images. No test data or calibration. Existing matched controls are reused rather than retrained.', '',
       '| Arm | PSNR | Change from initialization | SSIM | Endpoint pinball | Raw coverage | Raw width |',
       '|---|---:|---:|---:|---:|---:|---:|']
for label,r in reports.items():
    first=r['history'][0]['validation'];v=r['history'][-1]['validation']
    lines.append(f"| {label} | {v['psnr_db']:.4f} | {v['psnr_db']-first['psnr_db']:+.4f} | {v['ssim']:.5f} | {v['endpoint_pinball']:.6f} | {v['raw_coverage']:.2%} | {v['raw_width']:.6f} |")
lines+=['','Primary sampling comparison: frozen uniform versus frozen endpoints. Both predict one sampled q per training image; endpoints are sampled with equal probability from {0.025,0.975}. Architecture, optimizer steps and sampling count match. This endpoint-conditioned model is not the older six-output fixed head.', '',
        'Primary loss-weight comparison: joint weights 0.02 and 0.1 versus weight 1 and MSE-only continuation. Loss is MSE + weight × pinball. Coefficients were specified before seeing the new validation results. They are empirical tests, not guaranteed gradient balancing under AdamW.', '',
        'Raw coverage/width are uncalibrated endpoint diagnostics; narrower is not better unless coverage is comparable. The MSE-only uncertainty branch is untrained, so its interval metrics are not a competing uncertainty result. Final epoch 2 is compared without best-epoch selection. Single-seed validation pilot; no general superiority or calibrated-coverage claims.']
(root/'comparison.md').write_text('\n'.join(lines)+'\n')
(root/'comparison.json').write_text(json.dumps(reports,indent=2))
print('\n'.join(lines))
