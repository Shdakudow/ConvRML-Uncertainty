# Final model weights

Choose a model to download its selected checkpoint as a PyTorch file. Each model folder includes its calibration parameters and test results. Coverage counts RGB pixel-channel values.

| Dataset | Model | Selected epoch | Fitted parameter | Calibration coverage | Test coverage | Mean test width |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| MIRFLICKR | [QUTCC](mirflickr/qutcc/) | 49 | q lower = 0.041503906, q upper = 0.97680664 | 90.062% | 89.973% | 0.13018 |
| MIRFLICKR | [Im2im](mirflickr/im2im/) | 50 | Spatial offset (min 0, mean 0.00895018, max 0.0179004) | 91.070% | 90.895% | 0.11386 |
| MIRFLICKR | [Deep ensemble](mirflickr/ensemble/) | 49 | Shared λ = 0.89089084 | 90.272% | 90.062% | 0.13520 |
| MIRFLICKR | [MC dropout](mirflickr/mcdropout/) | 44 | Shared λ = 3.0015008 | 90.056% | 90.076% | 0.10788 |
| DiffuserCam | [QUTCC](diffusercam/qutcc/) | 48 | q lower = 0.098632812, q upper = 0.90820312 | 90.053% | 90.080% | 0.02764 |
| DiffuserCam | [Im2im](diffusercam/im2im/) | 49 | Spatial offset (min 0, mean 0, max 0) | 93.728% | 93.730% | 0.03365 |
| DiffuserCam | [Deep ensemble](diffusercam/ensemble/) | 50 | Shared λ = 0.62062061 | 90.129% | 90.146% | 0.03649 |
| DiffuserCam | MC dropout | Pending | — | — | — | — |

[Training archive](archive.md)
