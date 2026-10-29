# Model comparison

The [four model implementations](../../../models/) are shared. PLD is the
Parallel Lensless Dataset (ConvRML 100k); DLMD is the DiffuserCam Lensless Mirflickr
Dataset. Dataset loading is in [dataset.py](../../../dataset.py) and settings are
in [configs/](../../../configs/). Shared layers remain in [convrml/](../../../convrml/).

| Method | Training objective | Reconstruction | Native bounds | Added architecture |
| --- | --- | --- | --- | --- |
| Im2im | Center MSE + lower/upper pinball at 0.05/0.95 | Center head | Lower/upper heads | Nine-channel output convolution |
| Ensemble | Gaussian NLL per independent member | Mean of member means | Gaussian total-variance interval | RGB variance head per member |
| MC dropout | MSE | Mean of 30 stochastic outputs | Gaussian sample-spread interval | Dropout at 36 encoder + 12 decoder sites, p=0.05 |
| QUTCC | Pinball with q sampled uniformly per image | q=0.5 image | Images queried at lower/upper q | q embedding + FiLM at 36 encoder + 12 decoder sites |

All methods use ConvNeXtV2-Base/FPN. QUTCC conditions normalized features using
`features * (1 + scale(q)) + shift(q)`; its projections start at zero. No learned
q selector or additional MSE term is included in QUTCC. Im2im trains its entire
backbone and heads jointly. Ensemble members use independent initialization and
shuffle seeds while each sees the complete training split.

## Experiment-specific differences

| Setting | PLD | DLMD |
| --- | --- | --- |
| Measurements | RML / normalized DiffuserCam | Clean / fixed-corruption raw DiffuserCam |
| Image size | RGB 256×256 | RGB 270×480; internally pad to 288×480 and crop |
| Loader | `NumpyPairs`: uint8 memory-mapped cache | `ImagePairs`: paired RGB image files |
| Input conversion | Float32 / 255 | Float32 / 255 |
| Train / val / test / calibration | 60,000 / 5,000 / 5,000 / 2,048 | 17,999 / 2,000 / 2,000 / 3,000 |
| QUTCC implementation | Shared | Identical class |
| Im2im checkpoint names | New: `backbone.*`; legacy: `base.*` | New: `backbone.*`; legacy: `net.*` |
| Main clean training | 50 epochs | 50 epochs |
| Corruption adaptation | Not included | 30 clean epochs + 30 adaptation epochs |

Clean QUTCC in both experiments and raw Im2im use float32 Adam, effective batch
four, betas (0.9,0.99), no weight decay, 30% warmup from 1e-5 to 1e-4, then cosine
decay to 1e-9 over 50 epochs. Parallel lensless Im2im/ensemble/MC use the preserved
BF16 AdamW baseline pipeline, batch two. Adaptation uses constant LR 1e-5.
These historical runs are not an identical-optimizer ablation.

## Calibration and checkpoints

Model `.bounds()` returns native intervals without fitting calibration or clipping.
Calibration fits separate held-out pairs, after training. The shared
[implementation](../../../convrml/calibration.py) uses RCPS-HB for Im2im,
ensemble and MC dropout, and QUTCC Algorithm 1 to select quantile inputs.
`convrml.evaluate --calibration paper` is the default; `native` is an uncalibrated
control. Earlier empirical-matching and raw Im2im scale-grid reports remain
historical and must not be relabelled as RCPS results.

Im2im and QUTCC each have one implementation for both datasets, but trained
weights and dataset identities differ. Im2im now has one `backbone.*` namespace;
the checkpoint loader translates old `base.*`/`net.*` keys before strict loading.
Load the matching dataset checkpoint strictly. Source consolidation does not
make trained weights interchangeable or call for retraining.
Direct three-head construction preserves loaded predictions, but removes the old
discarded RGB head's random draws. Fresh Im2im head initialization therefore
differs from earlier constructors even with the same seed; use the historical
source snapshot to reproduce an old initialization exactly.

## Paper-to-code review

Reviewed against the primary papers on 2026-10-26. This is a source-level review
of the current implementation, not a replication of their reported results.

**QUTCC.** Uniform q sampling, using the same q in conditioning and pinball loss,
and reconstruction at q=0.5 follow Sections 3.1–3.2. Our 48 normalization-site
FiLM layers adapt the conditioning to ConvNeXt/FPN; the reference uses an
attention U-Net. Calibration chooses two global q inputs, not learned per-image
or per-pixel q selectors. The finite-N budget follows Algorithm 1 lines 1–2:
`(alpha - (1-alpha)/N)/2`. Adjacent prose gives a different correction; that
discrepancy is recorded rather than silently changing the implementation.
We report crossings and recheck final risks, but do not apply Appendix A.3.3's
optional endpoint repair/grid fallback. Binary search relies on monotonic
responses. [QUTCC v2](https://arxiv.org/html/2507.14760v2)

**Im2im.** Joint training of reconstruction and fixed endpoint quantiles follows
the quantile-regression approach in Sections 2.1.4 and 3.2. Our center MSE plus
two pinball terms has equal coefficients and uses a shared ConvNeXt/FPN. Direct
endpoint heads do not enforce lower ≤ center ≤ upper. Calibration floors each
center-to-endpoint distance at 1e-6 before scaling. The original RCPS procedure
uses one common scale and its experiments use alpha=delta=0.1. Our asymmetric
extension fits two scales, splits both risk/confidence budgets equally between
tails, and uses alpha=0.1, delta=0.05. It should be labelled as that extension.
[Im2im Sections 2–3](https://arxiv.org/html/2202.05265)

**Deep ensemble.** Gaussian NLL, softplus variance with a floor, independent
initialization/shuffling on the full training split, and mixture moment matching
follow Sections 2.2–2.4. The omitted NLL constant has no effect on gradients.
Adversarial training is optional in the reference and omitted here. Our
ConvNeXt/FPN architecture and added RCPS calibration are comparison choices;
moment-matched bounds are not exact mixture quantiles.
[Deep Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)

**MC dropout.** Repeated stochastic forwards and their mean follow Section 4.
Our elementwise dropout at 36 encoder and 12 decoder sites, p=0.05 and 30 passes
are experiment settings. Variance uses divisor T and measures prediction spread.
It omits the observation-noise term tau^-1 from the paper's predictive-variance
formula and uses the recorded AdamW setup rather than a derived prior/precision
relationship. Treat it as a dropout-spread baseline with added RCPS, not an exact
Bayesian predictive distribution.
[Gal and Ghahramani, Section 4](https://proceedings.mlr.press/v48/gal16.pdf)

**Comparison limits.** All four share the backbone family, but optimization,
parameter count and compute differ. QUTCC's median and the other methods' mean
estimates target different reconstruction summaries. Report these differences
alongside quality, width, coverage and time. Image-level RCPS concerns expected
pixel miss fraction under its sampling assumptions; it does not certify 90%
coverage for every image or each fixed pixel. Algorithm 1 calibration is not an
HB certificate. No calibration enforces exactly 90% on a future finite test set.

[Usage](RUNNING_EXPERIMENTS.md) · [Metrics](METHODS_AND_METRICS.md) ·
[Settings and seeds](../../../configs/experiment_settings.json)
