# Models, calibration, and metric definitions

This document defines the metrics and calibration conventions for [Experiment 2](../experiments/experiment2) and [Experiment 3](../experiments/experiment3). For the classes, heads, losses and code-reading order, see [model-code comparison](MODEL_COMPARISON.md). Exp2's four-model router is in [Exp2 source/](../experiments/experiment2); Exp3's raw-compatible source is in [clean/](../experiments/experiment3/clean).

Architecture, output heads and training losses are explained once in the [model-code comparison](MODEL_COMPARISON.md). This page defines calibration, clipping and evaluation metrics.

## Two calibration protocols

### Current QUTCC: quantile-space search

Freeze the checkpoint and search `q_lower` and `q_upper` on the separate calibration split. Lower-tail loss is the fraction of target RGB values below the lower image; upper-tail loss is the fraction above the upper image. `calibration.py` implements the reference Algorithm 1 budget:

```text
per_tail_target = (alpha - (1 - alpha) / N_calibration_images) / 2
```

At `alpha=0.10`, each tail targets about 0.05. Quantile search tolerance is 0.001. Feasibility at extreme queries and final tail risks are checked explicitly. Failed searches are recorded as failures. The search assumes quantile monotonicity, which shared weights do not strictly enforce; evaluators report crossings rather than silently sorting outputs.

This is the original-style adjusted empirical quantile search. It is **not** the earlier Hoeffding–Bentkus RCPS scale search and has no `delta` argument.

### Four-model baselines: independent lower/upper scales

Im2im, ensemble, and MC use separate scales fitted to frozen predictions:

```text
lower_distance = max(prediction - native_lower, 1e-6)
upper_distance = max(native_upper - prediction, 1e-6)
calibrated_lower = prediction - lower_scale * lower_distance
calibrated_upper = prediction + upper_scale * upper_distance
```

The corrected normal-camera evaluator reports empirical and RCPS-HB factors. RCPS uses `alpha=0.10`, `delta=0.05`, split into 0.05 risk and 0.025 confidence-failure budgets per tail. Calibration images are independent sampling units, with missed pixel-channel fraction as image loss. Fitted scales are then applied to held-out test predictions.

The **raw-paper Im2im** evaluator instead uses the reference adjusted empirical budget and a 0…6 scale grid. Its results must not be labeled RCPS-HB. Check each report's `procedure` field.

Neither procedure forces exactly 90% coverage on every test image. Test ground truth measures coverage; it does not select calibration parameters.

For the separate empirical evaluation targeting **90.0–90.1% calibration coverage**, see
[close-to-90% calibration](COVERAGE90.md). It preserves these reference protocols and
does not claim an RCPS confidence guarantee.

## Normalization, calibration, and clipping

Loaders divide existing RGB 8-bit values by 255. Ground truth and measurements therefore lie in `[0,1]`; this normalization is unrelated to calibration. The model handles its own encoder normalization; do not apply it again to dataset inputs.

The image head has no final sigmoid enforcing `[0,1]`. Raw predictions and bounds can be negative or exceed one, both before and after calibration. Uncalibrated outputs are learned estimates; MC randomness comes from dropout masks.

Clipping is a separate operation after predicting/calibrating an interval:

```text
clip(z) = min(1, max(0, z))
```

A calibrated raw interval `[-0.08,1.12]` becomes `[0,1]`; width falls from 1.20 to 1.00. It covers all normalized target values but is uninformatively broad.

Boundary behavior matters: raw bounds `[-0.10,-0.02]` miss a target of exactly zero, while independently clipping them gives `[0,0]`, which includes zero. Clipping can therefore **increase reported coverage**, especially for black/white pixels; it is not generally coverage-neutral. The current RML epoch-50 QUTCC report has raw coverage about 90.18% and clipped coverage about 91.63%. Clipping is not a new calibration pass.

For `[0,1]` targets, clipping reconstruction cannot increase squared error, though SSIM need not improve. For ordered bounds, clipping cannot increase interval width. Small quantile crossings are recorded without silent repair.

Compare models using the same raw/clipped convention. Keep the raw audit alongside clipped demo metrics.

## Metrics and aggregation

| Metric | Definition and unit |
| --- | --- |
| PSNR | Per-image `-10*log10(mean((prediction-target)**2))` with data range 1; then image average. Higher is better. |
| SSIM | TorchMetrics SSIM with `data_range=1`, averaged over images. Higher is better. |
| Coverage | Fraction of RGB pixel-channel targets inside inclusive `[lower,upper]`; then image average. |
| Mean interval width | Mean of `upper-lower`, in normalized intensity units. Width 0.02 spans 2% of the normalized intensity range. |
| Error map | `mean_RGB(abs(prediction-target))` at each spatial pixel. |
| Uncertainty map | `mean_RGB(upper-lower)` at each spatial pixel. |
| LPIPS | AlexNet LPIPS v0.1, `normalize=True` for `[0,1]` RGB input. Lower is better. |
| Pearson / Spearman | Correlate width/error across pixels within each image, then average image coefficients. Constant maps are undefined and counted separately. |

LPIPS/correlations use the same full 5,000 test IDs per normal camera/model at each evaluated milestone. Primary correlations are unsmoothed; epoch-50 smoothing is a separately labeled sensitivity analysis. Region Pearson/Spearman are omitted.

The first 50 demo images are a fixed preview, not full-test estimates. Curves use exact evaluated checkpoints and calibration at epochs 5, 10, …, 50. Missing milestones remain missing; results are never fabricated or interpolated.
