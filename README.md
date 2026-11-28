# ConvRML-Uncertainty

Uncertainty quantification for lensless imaging reconstruction. Four methods built on a shared ConvNeXt-Tiny backbone (`ConvNeXt/backbone.py`).

## Models

| Model | Output | Training loss |
| --- | --- | --- |
| QUTCC | One RGB quantile image for the supplied `q` | Pinball loss at a randomly sampled `q` per image |
| Im2im | Lower, center, and upper RGB images | Center MSE plus lower/upper pinball losses |
| Deep ensemble | Ten independent RGB mean/variance predictors | Mean Gaussian NLL across the ten members |
| MC dropout | Mean and variance from 50 stochastic passes | MSE |

Training: 50 epochs, AdamW, batch size 10, 9-epoch linear warmup 1e-5 → 1e-4, cosine decay to 1e-9.

## Calibration

All methods target 90% coverage (α=0.1). MC dropout and ensemble use empirical grid search; Im2im and QUTCC use K-RCPS with HB certificate (δ=0.1).

| Model | Fitted parameter |
| --- | --- |
| QUTCC | Lower and upper quantile inputs |
| Im2im | K-RCPS spatial offset map |
| Deep ensemble | One multiplier for the Gaussian interval from all members |
| MC dropout | One multiplier for the 50-pass Gaussian interval |

<!-- calibration-results-20261118:start -->
### Results (MIRFLICKR, 3,000 calibration / 2,000 test images)

| Model | Selected epoch | Test SSIM | Calibration coverage | Test coverage | Mean width in [0,1] |
| --- | ---: | ---: | ---: | ---: | ---: |
| QUTCC | 49 | 0.8211 | 90.062% | 89.973% | 0.13013 |
| Im2im | 50 | 0.8215 | 91.070% | 90.895% | 0.11386 |
| Deep ensemble | 49 | 0.78541 | 90.272% | 90.062% | 0.13054 |
| MC dropout | 44 | 0.81662 | 90.056% | 90.076% | 0.10080 |

<!-- calibration-results-20261118:end -->

## Attribution

Backbone adapts [Meta's ConvNeXt](https://github.com/facebookresearch/ConvNeXt); decoder adapts [Lensless Imaging Transformer](https://github.com/BobPXX/Lensless_Imaging_Transformer).
