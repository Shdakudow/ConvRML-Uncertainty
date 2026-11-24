# MIRFLICKR · MC dropout

[Download the selected weights](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-mcdropout.pt) · epoch 44, selected by highest **50-pass validation SSIM** among the available saved checkpoints through epoch 50.

- Validation: 2,000 images, 50 stochastic passes, seed 777, batch size 10.
- Raw validation SSIM: 0.81530726; PSNR: 26.08381 dB.
- Candidates: epochs 40, 44, 50. Separate weights for every training epoch were not available.
- Selection used validation data only. No retraining or test-based selection.

[Selection results](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-mcdropout-50passes-selection.json) · [Previous 30-pass selection: epoch 44](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-mcdropout.pt).

The published MC dropout calibration/test metrics belong to the previous **30-pass, epoch-44** evaluation. They have not been recomputed for this 50-pass selection; recalibration is required before reporting calibrated 50-pass test results. Existing checkpoints and results are preserved.
