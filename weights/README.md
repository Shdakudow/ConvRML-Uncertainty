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
| Deep ensemble | Fitted offset 0 at every pixel; K-RCPS passed | Calibration job `826080` cancelled at user request |
| MC dropout | Fitted offset 0 at every pixel; 100% test coverage, width 1.00000 (uninformative) | Calibration deferred until training completes; job `916959` cancelled |

QUTCC uses the two fitted quantiles as model inputs. Im2im, deep ensemble, and MC dropout use a spatial offset map. Each pixel’s offset lowers the lower bound and raises the upper bound by the same amount across all three RGB channels. The MIRFLICKR maps are 270 × 480. The ensemble’s original intervals passed the risk check with zero expansion.

MIRFLICKR downloads include the [QUTCC summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc.json), [fitted quantiles](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc-parameters.pt), [Im2im summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im.json), and [full Im2im offset map](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im-parameters.pt).

MC dropout calibration job `916957` completed in 33 minutes 59 seconds using
epoch 44 (best validation SSIM through epoch 50). The fitted lambda map has
minimum, mean, and maximum all equal to zero. Calibration coverage is 100%
and test coverage is 100%, with mean test width 1.00000 on 2,000 images.
The HB check passed at alpha=0.10 and delta=0.10, but the clipped intervals
span [0,1] everywhere and are uninformative. The added observation-noise
variance dominates the native intervals; nonnegative K-RCPS expansion cannot
shrink them. This is a measured fit, not an assumed zero offset.
Test PSNR: 26.13419 dB; SSIM: 0.81624; LPIPS: 0.25349.

DiffuserCam calibration is deferred while its models finish training. Pending QUTCC (`786677`), Im2im (`788286`), MC dropout (`916959`), and ensemble (`826080`) calibration jobs have been cancelled at user request. Training jobs continue.
