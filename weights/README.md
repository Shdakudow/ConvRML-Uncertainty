# Weights

- [MIRFLICKR](mirflickr/) · Final QUTCC, Im2im, MC dropout and deep ensemble weights.
- [DiffuserCam](diffusercam/) · Ensemble snapshots included in the November 20 archive.

Choose a dataset and model to download weights from GitHub Releases. Numbered archives contain intermediate checkpoints. For a split tar, concatenate the parts in numerical order before extracting the checkpoint.

[Download the searchable browser](https://raw.githubusercontent.com/Shdakudow/ConvRML-Uncertainty/main/weights/browse.html) and open it locally, or use the [machine-readable catalog](catalog.json). Both cover 665 checkpoint paths from the November 15 run. [Earlier experiments](https://github.com/Shdakudow/global-archive/tree/main/ConvRML-Uncertainty/archive/weights/) have a separate directory.

## Fitted calibration values

| Model | MIRFLICKR | DiffuserCam |
| --- | --- | --- |
| QUTCC | Lower quantile `q_lb = 0.04150390625`, upper quantile `q_ub = 0.976806640625` | No saved fit found |
| Im2im | Offset map min 0, mean 0.00895018, max 0.01790037 | No saved fit found |
| Deep ensemble | Epoch 49, shared multiplier `λ = 0.89089084`, calibration coverage 90.272%, test coverage 90.062%, mean test width 0.13520. [Parameters](mirflickr/ensemble/calibration/parameters.pt) · [Results](mirflickr/ensemble/calibration/summary.json) | Calibration job `826080` cancelled at user request |
| MC dropout | Epoch 44, shared multiplier `λ = 3.0015008`, calibration coverage 90.056%, test coverage 90.076%, mean test width 0.10788. [Parameters](mirflickr/mcdropout/calibration/parameters.pt) · [Results](mirflickr/mcdropout/calibration/summary.json) | Calibration deferred until training completes; job `916959` cancelled |

QUTCC uses two fitted quantiles. Im2im uses a spatial offset map. The previous K-RCPS fits for deep ensemble and MC dropout also used spatial offsets. Each pixel’s offset lowers the lower bound and raises the upper bound by the same amount across all three RGB channels. The MIRFLICKR maps are 270 × 480. The earlier ensemble K-RCPS fit had zero expansion.

MIRFLICKR downloads include the [QUTCC summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc.json), [fitted quantiles](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc-parameters.pt), [Im2im summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im.json), and [full Im2im offset map](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im-parameters.pt).

DiffuserCam calibration is deferred while its models finish training. Pending QUTCC (`786677`), Im2im (`788286`), MC dropout (`916959`), and ensemble (`826080`) calibration jobs have been cancelled at user request. Training jobs continue.
