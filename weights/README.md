# Weights

- [MIRFLICKR](mirflickr/) · Final QUTCC, Im2im, MC dropout and deep ensemble weights.
- [DiffuserCam](diffusercam/) · Ensemble snapshots included in the November 20 archive.

Choose a dataset and model to download weights from GitHub Releases. Numbered archives contain intermediate checkpoints. For a split tar, concatenate the parts in numerical order before extracting the checkpoint.

[Download the searchable browser](https://raw.githubusercontent.com/Shdakudow/ConvRML-Uncertainty/main/weights/browse.html) and open it locally, or use the [machine-readable catalog](catalog.json). Both cover 665 checkpoint paths from the November 15 run. [Earlier experiments](../archive/weights/) have a separate directory.

## Fitted calibration values

| Model | MIRFLICKR | DiffuserCam |
| --- | --- | --- |
| QUTCC | Lower quantile `q_lb = 0.04150390625`, upper quantile `q_ub = 0.976806640625` | No saved fit found |
| Im2im | Offset map min 0, mean 0.00895018, max 0.01790037 | No saved fit found |
| Deep ensemble | Fitted offset 0 at every pixel; K-RCPS passed | Calibration job `826080` queued after ensemble training/evaluation |
| MC dropout | `lambda = 0` (uncalibrated; user-selected) | `lambda = 0` (uncalibrated; user-selected) |

QUTCC uses the two fitted quantiles as model inputs. Im2im, deep ensemble, and MC dropout use a spatial offset map. Each pixel’s offset lowers the lower bound and raises the upper bound by the same amount across all three RGB channels. The MIRFLICKR maps are 270 × 480. The ensemble’s original intervals passed the risk check with zero expansion.

MIRFLICKR downloads include the [QUTCC summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc.json), [fitted quantiles](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_qutcc-parameters.pt), [Im2im summary](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im.json), and [full Im2im offset map](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr_im2im-parameters.pt).

MC dropout uses no additional interval expansion under the selected reporting policy (`lambda = 0`). This is not a fitted K-RCPS result and carries no K-RCPS coverage certificate. Its native predictive variance remains unchanged. The DiffuserCam ensemble calibration job targets the best checkpoint through epoch 50 and depends on dispatcher job `886292` (queue checked 2026-11-23).
