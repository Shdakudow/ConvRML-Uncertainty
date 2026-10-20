# Models, calibration, and metric definitions

This document describes the current comparison. The latest QUTCC uses `qutcc_paper_style_20261018`; the other three methods use `adjusted_comparison_50epochs`.

## Shared reconstruction architecture

The network uses a ConvNeXtV2 Base encoder and an FPN-style convolutional decoder, followed by upsampling and a residual image head. This substitutes a lensless ConvNeXt/FPN backbone for the QUTCC reference paper's attention U-Net. It is an adaptation of that method, not an exact reproduction of its architecture.

The latest raw-compatible model pads spatial dimensions to multiples of 32, then crops the output back to the original size. Padding pixels are excluded from losses and image metrics. Measurement and target must have the same supported RGB shape in this experiment; arbitrary unpaired raw sensor geometry is not handled by this loader.

| Method | Training objective | Reconstruction | Native interval |
| --- | --- | --- | --- |
| Im2im | Center MSE + lower pinball at 0.05 + upper pinball at 0.95, updating the entire model | Learned center image | Direct lower/upper head outputs |
| Deep ensemble | Gaussian NLL for each independently initialized and independently shuffled member | Mean of member means | Gaussian moment interval using predicted variance and variation of member means |
| MC dropout | MSE; elementwise dropout during training | Mean of 30 stochastic predictions in the current demo/metric protocol | Gaussian moment interval from variation of the 30 predictions |
| Current QUTCC | Pinball only; one independent uniform scalar q per image per training visit | Quantile image at q = 0.5 | Quantile images at calibration-selected global lower/upper q |

The main ensemble uses fixed members 00–04; the ten-member reference uses 00–09 and is calibrated separately. Each member trains on the complete training split. They are not disjoint data shards of one model.

MC dropout acts on the encoder MLP branch before residual addition: 36 encoder sites plus 12 decoder activation sites. The current demo uses `p=0.05` and 30 passes. The earlier validation study also compared 0.10/0.20 and 30/60/100 passes; those are distinct inference settings. Dropout remains stochastic at inference while other evaluation behavior stays fixed.

FiLM is exclusive to QUTCC: 36 encoder residual blocks and 12 decoder sites. A quantile embedding supplies feature-wise scale and shift there. The current model does not learn an image/pixel-specific endpoint selector. It learns `(measurement, q) → quantile image`, and calibration selects one pair of q values per camera/checkpoint.

## Losses in plain notation

For residual `e = target - prediction`, pinball loss is `maximum(q*e, (q-1)*e).mean()`. Im2im adds center MSE and two endpoint pinball losses. QUTCC uses only pinball at sampled q. MC dropout uses MSE. Each ensemble member uses `0.5 * (log(variance) + (mean-target)**2 / variance).mean()`.

Ensemble variance is `softplus(raw_variance) + 1e-6`. Total predictive variance is `mean(member_variance + member_mean**2) - ensemble_mean**2`. Native ensemble and MC intervals use approximately `mean ± 1.64485 * standard_deviation`, followed by separate calibration.

Width measures predicted uncertainty; it need not equal realized absolute error for one image. A useful uncertainty map can correlate with error without matching it numerically.

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
