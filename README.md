# ConvRML-Uncertainty

ConvNeXt-Tiny reconstruction with QUTCC, Im2im, deep ensemble, and MC dropout
uncertainty models. All four methods use the encoder and full-resolution
reconstruction decoder in `ConvNeXt/backbone.py`.

| Model | Output | Training loss |
| --- | --- | --- |
| QUTCC | One RGB quantile image for the supplied `q` | Pinball loss at a randomly sampled `q` per image |
| Im2im | Lower, center, and upper RGB images | Center MSE plus lower/upper pinball losses |
| Ensemble | Ten independent RGB mean/variance predictors | Mean Gaussian NLL across the ten members |
| MC dropout | Mean and variance from 30 stochastic passes | MSE |

`timesteps` is removed. QUTCC retains `q`, embedded and applied through FiLM
after the encoder block normalizations and decoder batch normalizations.
The ConvNeXt block owns its forward computation and accepts optional FiLM
conditioning. QUTCC calls the block directly instead of repeating its layers.
The other three methods receive only image tensors. The decoder's new
`out_channels` argument is needed for Im2im's 9 output channels and each
ensemble member's 6 output channels; the sensor input remains RGB.
Each epoch forms one set of training batches, then independently shuffles
their order for each ensemble member. Membership within each batch stays
the same across members, including the final partial batch. Each member
trains through its entire batch order before the next member starts.
Only batch indices are stored; images are loaded as needed.
Every member uses the same learning-rate and weight-decay schedule for that
epoch. AdamW updates only the active member, and gradients retain the mean-NLL
scaling across members. Training loss is averaged across members and images.
Seed 777 makes the batch orders reproducible across matching runs.

MC dropout keeps the draft's probability of 0.1 and its observation-noise
precision formula, with length scale 1 and weight decay 1e-4. Its interval
variance includes `2 * dataset_size * 1e-4 / 0.9` in addition to sample variance.
That term is inherited from the draft and can dominate the native width on a
large dataset. Calibration operates on those native intervals separately.
Training uses one stochastic pass per batch. Validation uses the mean of
30 passes, matching the reconstruction returned with the uncertainty bounds.
BatchNorm stays in evaluation mode during MC sampling while dropout is active.

## Data and training

Install dependencies with `python -m pip install -r requirements.txt`.
LPIPS downloads pretrained AlexNet metric weights on first use if they are
not already cached.

Use the existing JSON split format:

```json
{
  "roles": {
    "train": [{"id": "scene_1", "measurement": {"path": "inputs/1.png"}, "target": {"path": "targets/1.png"}}],
    "val": [{"id": "scene_2", "measurement": {"path": "inputs/2.png"}, "target": {"path": "targets/2.png"}}],
    "calibration": [{"id": "scene_3", "measurement": {"path": "inputs/3.png"}, "target": {"path": "targets/3.png"}}],
    "test": [{"id": "scene_4", "measurement": {"path": "inputs/4.png"}, "target": {"path": "targets/4.png"}}]
  }
}
```

Paths can be absolute or relative to the manifest. Images are converted to RGB
float tensors in [0,1]. The loader preserves image sizes; each batch uses scenes
with the same input/target sizes. The encoder pads to multiples of 32 and the
decoder produces the target size. The loader applies no additional corruption.

The manifest is a list of input/target pairs divided into `train`, `val`,
`calibration`, and `test` roles. It references the original files without copying
them. Keep each scene in one role to keep validation, calibration, and testing
independent of training.

For PNG/TIFF/JPEG exports, the loader converts RGB pixels from 0–255 to 0–1.
For original MIRFLICKR `.npy` files, it expects floating-point HWC arrays,
reverses BGR to RGB, flips vertically, divides measurements by 0.9, and clips
measurements and targets to [0,1]. Targets are not divided by 0.9. These are
the original dataset's preprocessing steps; exported images are not corrected
again. `.npy` values are not divided by 255 or quantized to 8-bit pixels.

```bash
python train.py --manifest /path/to/split.json --model qutcc --output runs/qutcc
```

The only command-line arguments select the data, model, and output folder.
Settings follow the draft and your updates: 50 epochs, AdamW, physical batch
size 10 without gradient accumulation, and a nine-epoch linear
warmup from 1e-5 to 1e-4, then cosine decay to 1e-9. Weight decay decreases
from 1e-4 to zero with a cosine schedule. All model weights start from scratch.

`history.json` records loss, validation PSNR/SSIM/LPIPS, learning rate, and
weight decay. PSNR, SSIM, and LPIPS come from TorchMetrics. The
`PeakSignalNoiseRatio` and `StructuralSimilarityIndexMeasure` classes accumulate
validation statistics within each epoch and start fresh for the next epoch.
PSNR is averaged per image, using `dim=(1,2,3)`. LPIPS uses the
pretrained AlexNet metric weights. PSNR and SSIM use raw predictions with
intensity range 1. LPIPS uses a copy of predictions clipped to [0,1], following
your selected evaluation policy and TorchMetrics' input range. Metrics are
averaged per image. LPIPS is an evaluation metric here, not a training loss.
Its lower values indicate closer
perceptual similarity; it is not a coverage probability or a percentage.
The fixed seed is your chosen value, 777. It supports repeatability rather than
improving accuracy by itself. `latest.pt` saves the current weights and optimizer
each epoch. Every 5 epochs, `best_through_epoch_005.pt`,
`best_through_epoch_010.pt`, and so on save the weights with the highest raw
validation SSIM seen from epoch 1 through that milestone. Selection is
cumulative and does not reset every 5 epochs. Each best checkpoint records
the selected model's actual `epoch`, `validation_ssim`, and
`selected_through_epoch`. It contains weights and model metadata, without
optimizer state. Tied SSIM values retain the earlier winner.
This script starts a new run; it does
not implement checkpoint resumption.

## Calibration

Use an explicit calibrator for each method:

```python
from calibration.calibration import KRCPS, QUTCCCalibration

calibrator = KRCPS(model)  # Im2im, ensemble, or MC dropout
parameters = calibrator.fit(calibration_loader, device)
center, lower, upper = calibrator.predict(test_images)
```

For QUTCC, construct `QUTCCCalibration(model)` instead. The fit/predict
interface is the same. Each object stores its fitted values in
`calibrator.parameters`; `fit` also returns that dictionary.
The two calibration classes are independent, with their own `fit` and
`predict` methods. Each fit uses evaluation mode and disables gradient tracking.
Both calibrated `predict` methods also use `torch.inference_mode()` so direct
prediction calls do not retain computation graphs.

An interval offset expands a native interval `[lower, upper]` to
`[lower - offset, upper + offset]`. K-RCPS stores an offset for each spatial
pixel, shared across its RGB channels. QUTCC instead selects quantile inputs.

QUTCC searches for two quantile levels using a finite-image correction.
`KRCPS` fits spatial additive interval offsets with alpha=0.1 (90% target
coverage) and delta=0.1 (90% confidence). The HB risk-threshold search uses
40 iterations. The remaining k-RCPS settings in `calibration/krcps_optimization.py` come from
your existing `code-for-quant/analysis_config.yaml`: 2 groups, 10 optimization
images, 500 sampled spatial coordinates, initial expansion 0.5, step 0.002,
and the original 30-value gamma sweep from 0.25 to 0.75.

The optimization images are excluded from the final HB check. Results record
the split indices, measured calibration coverage, and whether the certificate
passed. These are additive spatial offsets rather than the previous pair of
multiplicative scales. Final intervals are clipped to [0,1], following the
source k-RCPS implementation; PSNR/SSIM reconstruction metrics stay raw.
Save fitted parameters with `torch.save` because `lambda_map` is a tensor.
Training weights are unchanged. Test data never enters calibration.

`krcps_optimization.py` adapts the group-optimization routine from
[your repository](https://github.com/Shdakudow/code-for-quant/blob/0e164119fc81e1d0c9cb1a889d2a2f0f2525f081/evaluation/krcps/procedures/krcps.py),
based on [Sulam-Group/k-rcps](https://github.com/Sulam-Group/k-rcps).
The final calibration step uses HB with delta=0.1 instead of the source fork's
confidence-independent risk expression. CVXPY supplies the convex solver.

## Checks

```bash
python -m unittest discover -s tests -v
```

Runtime assertion and validation guards were removed after the first test pass.
Assertions remain in the tests. Numerical operations needed by the methods,
such as positive variance and normalization, remain in the implementation.
Tests use small synthetic images, reduced test-only backbone widths for
training, and the real LPIPS metric. They also exercise the original ConvNeXt
backbone. Full-dataset training is a separate run.

The loader checks include original MIRFLICKR orientation, channel order,
measurement normalization, clipping, float32 output, and a full 270×480 batch
of 10. To check GPU memory on the intended training GPU, run:

```bash
UQ_GPU_CHECK=1 python -m unittest discover -s tests -p test_code.py -k FullResolutionGPUTests -v
```

These optional checks use all four full-size models at both 270×480 and
300×480, physical batch size 10, and FP32. Each runs two training steps with
AdamW, then validation with PSNR, SSIM, and LPIPS. They print GPU identity,
peak allocated/reserved memory, and the result. Synthetic image values are
used to measure memory; this does not measure reconstruction quality or
replace a full-dataset run. Run individual checks in separate processes when
comparing memory or diagnosing an out-of-memory result, for example with
`-k FullResolutionGPUTests.test_qutcc_mirflickr`.

On 2026-11-11, all eight GPU checks passed on Unicorn's RTX 6000 Ada (48 GB)
with PyTorch 2.5.1+cu118. Peak allocated training memory ranged from 29.48 to
36.82 GiB; peak reserved memory reached 46.49 GiB. Those measurements used
five ensemble members and precede the independent batch-order update.
The current ten-member ensemble subsequently passed both full-resolution
checks on a 48 GB RTX A6000 with independent member batch orders. Peak
allocated training memory was 33.14 GiB at 270×480 and 36.34 GiB at 300×480.
Those checks used two training passes with AdamW and validation, with
physical batch size 10 and FP32.
All 23 regular checks also pass, including batch coverage, pairing, independent orders, seed
reproducibility, matched schedules, and AdamW update equivalence.
Detailed results are recorded in `tests/CHECKS.json`.

## Previous implementation and attribution

The previous implementation is preserved in
[`archive/previous_implementation_20261111/`](archive/previous_implementation_20261111/).
Earlier snapshots remain under [`archive/`](archive/).

The backbone adapts [Meta's ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
and the decoder adapts
[Lensless Imaging Transformer](https://github.com/BobPXX/Lensless_Imaging_Transformer).
License notices are available in the linked upstream repositories.
