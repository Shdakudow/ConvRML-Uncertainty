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
| Crossing | Fraction of RGB values where `lower > upper` |

Dataset aggregates are means of per-image metrics. Images in each benchmark have
equal dimensions. All per-image values and original ordered IDs are saved in
`report.json` and `per_image.csv` so aggregates can be checked independently.

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
vary. Empirical fitting has no HB confidence certificate. Reference Exp2 RCPS
uses delta=0.05 under its stated assumptions. See [calibration](COVERAGE90.md).
Coverage/width assess intervals, not the accuracy of an entire probability density.

## Timing

Training history separates training and validation seconds and records their
combined epoch time. Evaluation separates checkpoint loading, calibration and
test inference. Hardware/software, ensemble size and MC pass count accompany the
report. Compare timings with the same batch sizes, hardware and interval protocol;
five ensemble members represent five independently trained networks.
