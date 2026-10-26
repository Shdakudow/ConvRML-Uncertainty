# Model comparison

[Experiment 2](../../../experiments/experiment2/models/) has four model definitions.
[Experiment 3](../../../experiments/experiment3/models/) has two. Shared layers are in
[convrml/](../../../convrml/); there are no copied backbone variants in the experiment folders.

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

| Setting | Experiment 2 | Experiment 3 |
| --- | --- | --- |
| Measurements | RML / normalized DiffuserCam | Clean / fixed-corruption raw DiffuserCam |
| Image size | RGB 256×256 | RGB 270×480; internally pad to 288×480 and crop |
| Loader | `NumpyPairs`: uint8 memory-mapped cache | `ImagePairs`: paired RGB image files |
| Input conversion | Float32 / 255 | Float32 / 255 |
| Train / val / test / calibration | 60,000 / 5,000 / 5,000 / 2,048 | 17,999 / 2,000 / 2,000 / 3,000 |
| QUTCC implementation | Shared | Identical to Experiment 2 |
| Im2im wrapper | `base.*` checkpoint names | `net.*` checkpoint names; internal padding/cropping |
| Main clean training | 50 epochs | 50 epochs |
| Corruption adaptation | Not included | 30 clean epochs + 30 adaptation epochs |

Clean QUTCC in both experiments and raw Im2im use float32 Adam, effective batch
four, betas (0.9,0.99), no weight decay, 30% warmup from 1e-5 to 1e-4, then cosine
decay to 1e-9 over 50 epochs. Experiment 2 Im2im/ensemble/MC use the preserved
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

QUTCC weights are structurally shared across the experiments, but the trained
weights and dataset identities differ. Im2im has different wrapper prefixes;
load the matching experiment checkpoint strictly. Repository organization does
not make checkpoints interchangeable or call for retraining.

[Usage](RUNNING_EXPERIMENTS.md) · [Metrics](METHODS_AND_METRICS.md) ·
[Settings and seeds](../../../convrml/experiment_settings.json)
