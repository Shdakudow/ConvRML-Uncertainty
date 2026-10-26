# Reconstruction and interval metrics

Every evaluation reports both **raw** model outputs and explicitly **clipped**
outputs. Ground truth is the prepared RGB target, scaled from uint8 to [0,1].
Calibration mode and endpoint convention are stored in the report.

| Quantity | Definition |
| --- | --- |
| Reconstruction | Im2im center; ensemble member-mean average; MC stochastic mean; QUTCC q=0.5 |
| Absolute error | `abs(target - reconstruction)` per RGB value |
| Uncertainty width | `upper - lower` per RGB value |
| Coverage | Fraction of target RGB values satisfying `lower <= target <= upper` |
| Mean width | Width averaged over channels and pixels, then over test images |
| MAE | Mean absolute reconstruction error |
| PSNR | `-10 log10(MSE)` with intensity range 1 and MSE floor 1e-12 |
| SSIM | TorchMetrics SSIM with `data_range=1.0`, measured per image |
| LPIPS | Frozen pretrained LPIPS-AlexNet; clipped RGB [0,1], internally normalized to [-1,1]; lower is better |
| Pearson | Per-image linear correlation of interval width with absolute reconstruction error |
| Spearman | Per-image rank correlation of width with absolute error, using average ranks for ties |
| Crossing | Fraction of RGB values where `lower > upper` |

Dataset aggregates are means of per-image metrics. Images in each benchmark have
equal dimensions. All per-image values and original ordered IDs are saved in
`report.json` and `per_image.csv` so aggregates can be checked independently.

Correlations flatten each RGB image into channel/pixel values, use its matching
raw or clipped width/error maps, and apply no smoothing or region segmentation.
No p-values assuming independent pixels are reported. Higher positive correlations
mean that larger widths tend to coincide with larger observed errors; they do not
establish calibrated coverage or recover a true conditional uncertainty distribution.
Constant maps give undefined correlations: JSON stores null, CSV leaves a blank,
and `metric_valid_counts` records the number of images entering each mean.

LPIPS is evaluated once per image on clipped predictions with the unchanged [0,1]
target. Unbounded raw predictions are outside its supported intensity range, so
raw LPIPS is null. The default AlexNet pretrained weights may be fetched on first
use into `TORCH_HOME`/the standard Torch cache. `--no-lpips` records it as disabled,
not zero. Images are not resized for LPIPS; use that flag for fixtures below 64x64.

## Clipped metrics

```text
clip(z) = min(1, max(0, z))
```

Clipped metrics use `clip(reconstruction)`, `clip(lower)` and `clip(upper)`.
The targets remain unchanged. Raw metrics use the original predictions/endpoints.
Bounds are never silently sorted; crossings remain measurable.

Mean width is in normalized intensity units: a width of 0.1 spans one tenth of
the represented [0,1] RGB range. It is not a sensor-bit count. Width and absolute
error use compatible intensity units but are different quantities; a 90% interval
width is not expected to equal a single observed absolute error.

## Coverage interpretation

Alpha=0.10 specifies a target of 90%. Calibration coverage is measured on the
calibration split; test coverage is measured on independent test pairs and can
vary. Baseline RCPS uses delta=0.05 under its stated assumptions; QUTCC uses
its Algorithm 1 finite-N quantile search. See [calibration](COVERAGE90.md).
Coverage/width assess intervals, not the accuracy of an entire probability density.

## Timing

Training history separates training and validation seconds and records their
combined epoch time. Evaluation separates checkpoint/LPIPS loading, calibration,
prediction and metric computation. `prediction_seconds_per_image` excludes LPIPS,
correlations and loading; CUDA timings synchronize before/after prediction.
`test_seconds` is the full test-loop wall time including data and analysis.
Hardware/software, ensemble size and MC pass count accompany the
report. Compare timings with the same batch sizes, hardware and interval protocol;
five ensemble members represent five independently trained networks.
