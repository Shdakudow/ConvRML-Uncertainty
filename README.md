# ConvRML-Uncertainty

Uncertainty quantification for lensless imaging reconstruction. Four methods built on a shared ConvNeXt-Tiny backbone (`ConvNeXt/backbone.py`).

## Models

| Model | Output | Training loss |
| --- | --- | --- |
| QUTCC | One RGB quantile image for the supplied `q` | Pinball loss at a randomly sampled `q` per image |
| Im2im | Lower, center, and upper RGB images | Center MSE plus lower/upper pinball losses |
| Deep ensemble | Ten independent RGB mean/variance predictors | Mean Gaussian NLL across the ten members |
| MC dropout | Mean and variance from 50 stochastic passes | MSE |

**QUTCC:** `q` is embedded and injected via FiLM after encoder block normalizations and decoder batch normalizations.

**Deep ensemble:** 10 members, each trained independently on the same shuffled batch order per epoch (seed 777 + member index). Loss is mean NLL averaged across members and images.

**MC dropout:** dropout probability 0.1. The native interval variance includes an observation-noise term `2 * dataset_size * weight_decay / (1 − dropout_p)` (length scale 1, weight decay 1e-4). Training uses one stochastic pass per batch; inference and calibration use the mean of 50 passes. BatchNorm stays in eval mode during MC sampling.

**Training (all models):** 50 epochs, AdamW, physical batch size 10, 9-epoch linear warmup 1e-5 → 1e-4, cosine decay to 1e-9. Weight decay cosine-annealed 1e-4 → 0. Checkpoint selection: cumulative best validation SSIM, saved every 5 epochs.

## Calibration

All methods target 90% coverage (α=0.1). Calibration and test splits are independent.

| Model | Method | Fitted parameter |
| --- | --- | --- |
| QUTCC | K-RCPS + HB (δ=0.1) | Lower and upper quantile inputs |
| Im2im | K-RCPS + HB (δ=0.1) | Spatial additive offset map (per pixel, shared across RGB) |
| Deep ensemble | Empirical grid search (1,000 steps, multiplier ∈ [0, 10]) | One Gaussian interval multiplier |
| MC dropout | Empirical grid search (2,000 steps, multiplier ∈ [0, 50]) | One Gaussian interval multiplier |

K-RCPS settings: 2 groups, 10 optimization images, 500 sampled spatial coordinates, initial expansion 0.5, step 0.002, 30-value gamma sweep [0.25, 0.75], 40 HB iterations.

<!-- calibration-results-20261118:start -->
### Results (MIRFLICKR, 3,000 calibration / 2,000 test images)

| Model | Selected epoch | Test SSIM | Calibration coverage | Test coverage | Mean width in [0,1] |
| --- | ---: | ---: | ---: | ---: | ---: |
| QUTCC | 49 | 0.8211 | 90.062% | 89.973% | 0.13013 |
| Im2im | 50 | 0.8215 | 91.070% | 90.895% | 0.11386 |
| Deep ensemble | 49 | 0.78541 | 90.272% | 90.062% | 0.13054 |
| MC dropout | 44 | 0.81662 | 90.056% | 90.076% | 0.10080 |

<!-- calibration-results-20261118:end -->

## Weights

Pretrained checkpoints and calibration parameters live in [`weights/`](weights/). Each model folder contains:

| File | Contents |
| --- | --- |
| `README.md` | Download link (GitHub Release), selected epoch, fitted calibration value, usage code snippet |
| `calibration.pt` | Fitted calibration parameters (`torch.load`); use with the downloaded checkpoint for inference |
| `calibration/parameters.pt` | Same parameters with full provenance (checkpoint + manifest SHA-256, run metadata) |
| `calibration/summary.json` | Complete calibration run record: procedure, fitted scale, grid size, coverage on calibration and test sets, HB bound, Slurm job ID |
| `results.json` | Test-set metrics (PSNR, SSIM, LPIPS, coverage, mean width) and checkpoint SHA-256 |
| `files/NN.json` | Download manifest: ordered tar parts and their file list with SHA-256 and byte count (ensemble weights are split across multiple tars) |
| `training-archive.md` | Training run notes and checkpoint history |

[`weights/README.md`](weights/README.md) has the full summary table with fitted parameters and coverage for all models on both datasets.

## Attribution

Backbone adapts [Meta's ConvNeXt](https://github.com/facebookresearch/ConvNeXt); decoder adapts [Lensless Imaging Transformer](https://github.com/BobPXX/Lensless_Imaging_Transformer).
