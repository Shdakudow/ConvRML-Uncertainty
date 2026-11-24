# Weights

- [MIRFLICKR](mirflickr/) · Final QUTCC, Im2im, MC dropout and deep ensemble weights.
- [DiffuserCam](diffusercam/) · Ensemble snapshots included in the November 20 archive.

Choose a dataset and model to download weights from GitHub Releases. Numbered archives contain intermediate checkpoints. For a split tar, concatenate the parts in numerical order before extracting the checkpoint.

[Download the searchable browser](https://raw.githubusercontent.com/Shdakudow/ConvRML-Uncertainty/main/weights/browse.html) and open it locally, or use the [machine-readable catalog](catalog.json). Both cover 665 checkpoint paths from the November 15 run. [Earlier experiments](https://github.com/Shdakudow/global-archive/tree/main/ConvRML-Uncertainty/archive/weights/) have a separate directory.

## MIRFLICKR calibration

Coverage is the percentage of RGB channel values inside the predicted interval. All four models target 90% coverage and use the same 2,000 test images. Im2im uses 10 images to optimize its offset map and 2,990 to measure calibration coverage. The other methods measure calibration coverage on all 3,000 images.

| Model | Epoch | Fitted parameter | Calibration coverage | Test coverage | Mean test width | Parameters and results |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| [QUTCC](mirflickr/qutcc/) | 49 | Lower q = 0.04150390625, upper q = 0.976806640625 | 90.062% | 89.973% | 0.13018 | [Parameters](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc-parameters.pt) · [Results](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc.json) · [Release directory](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/tag/uq-weights-20261120) |
| [Im2im](mirflickr/im2im/) | 50 | Spatial offset map (min 0, mean 0.00895018, max 0.01790037) | 91.070% | 90.895% | 0.11386 | [Parameters](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im-parameters.pt) · [Results](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im.json) · [Release directory](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/tag/uq-weights-20261120) |
| [Deep ensemble](mirflickr/ensemble/) | 49 | Shared scale λ = 0.89089084 | 90.272% | 90.062% | 0.13520 | [Parameters](mirflickr/ensemble/calibration/parameters.pt) · [Results](mirflickr/ensemble/calibration/summary.json) · [Directory](mirflickr/ensemble/calibration/) |
| [MC dropout](mirflickr/mcdropout/) | 44 | Shared scale λ = 3.0015008 | 90.056% | 90.076% | 0.10788 | [Parameters](mirflickr/mcdropout/calibration/parameters.pt) · [Results](mirflickr/mcdropout/calibration/summary.json) · [Directory](mirflickr/mcdropout/calibration/) |

## DiffuserCam calibration

The [DiffuserCam weights directory](diffusercam/) lists the available checkpoints. Calibration parameters and coverage results for this dataset have not been published here.
