# Shared model and runtime code

Every Python module here is used by the public models or the train/evaluate/download
commands. Optional plotting, provenance tools and GitHub Actions live in
[archive/publication_support/](../archive/publication_support/).

| File | Purpose |
| --- | --- |
| [backbone.py](backbone.py) | ConvNeXtV2-Base encoder, FPN decoder, RGB residual head |
| [film.py](film.py) | Quantile modulation at 36 encoder + 12 decoder normalization sites |
| [quantile_embedding.py](quantile_embedding.py) | Sinusoidal scalar-q embedding |
| [intervals.py](intervals.py) | Native Gaussian-moment bounds before calibration |
| [calibration.py](calibration.py) | Baseline RCPS-HB scales and QUTCC Algorithm 1 calibrated quantiles |
| [dataset.py](../dataset.py) | PLD NumPy-cache and DLMD RGB paired-file datasets |
| [../train.py](../train.py) / [../infer.py](../infer.py) | Training loop and calibration → test → export workflow |
| [metrics.py](metrics.py) | SSIM, PSNR, MAE, LPIPS, coverage, width, crossings and uncertainty/error correlations |
| [protocol.py](protocol.py) / [experiment_settings.json](../configs/experiment_settings.json) | Model/data selection, recorded settings, identity and resume checks |
| [checkpoints.py](checkpoints.py) | Atomic saving, legacy Im2im name translation, strict loading, resume and ensemble assembly |
| [download.py](download.py) / [checkpoint_index.json](checkpoint_index.json) | Verified downloads of published checkpoint files |

Training losses, including the pinball formula, are defined directly in each
[Im2im/QUTCC model file](../models/). PLD and DLMD use the same Im2im and QUTCC
classes; there are no dataset-specific copies of these losses or architectures.
The small `convrml/train.py` and `convrml/evaluate.py` modules forward the older
package commands to the root scripts; the workflows are implemented once.

## Following the backbone

For a 256×256 measurement, [backbone.py](backbone.py) follows this path:

```text
RGB measurement: 3×256×256
  encoder input normalization
  ConvNeXt stages: 128×64×64, 256×32×32, 512×16×16, 1024×8×8
  1×1 projections + coarse-to-fine FPN merging: 256×64×64
  up1: 256×128×128       ← features() returns here
  up2: 256×256×256
  RGB head + original measurement: 3×256×256
```

Im2im requests three RGB heads at construction. Ensemble members add a variance
head. QUTCC modulates normalized features with q; MC dropout inserts stochastic
masks. Those additions are in [models/](../models/). Encoder normalization and
decoder upsampling remain shared.
The sizes above illustrate the existing PLD data. All four default models accept
other RGB dimensions, pad internally to a multiple of 32 and crop their outputs
back; the data loader retains the original paired dimensions.

## Reading the data

Both loaders return float32 `(2, 3, H, W)`: measurement first, target second.
RGB uint8 values are divided by 255. Neither loader clips predictions, applies
augmentation, changes targets, or creates new train/test splits.

## Parallel Lensless Dataset (PLD): NumPy cache

```python
from dataset import NumpyPairs
train = NumpyPairs("pairs.npy", "manifest.json", "train")
calibration = NumpyPairs("calibration/pairs.npy", "calibration/manifest.json", "calibration")
```

The uint8 array shape is `(N, 2, 3, 256, 256)`. Normal cache order is train, val,
then test. The separate calibration cache has only calibration pairs. Manifests
retain the existing image IDs and order:

```json
{"train": [{"id": "scene_a"}], "val": [{"id": "scene_b"}], "test": [{"id": "scene_c"}]}
```

## DiffuserCam Lensless Mirflickr Dataset (DLMD): image-file manifest

```python
from dataset import ImagePairs
train = ImagePairs("split.json", "train")
```

```json
{
  "roles": {
    "train": [{
      "id": "scene_a",
      "measurement": {"path": "measurements/scene_a.png"},
      "target": {"path": "targets/scene_a.png"}
    }],
    "val": [], "test": [], "calibration": []
  }
}
```

Paths are relative to the manifest or absolute. Measurements and targets must
have matching RGB dimensions, normally 270×480. The model pads and crops;
the loader leaves dimensions intact. Prepared clean and fixed-corruption
manifests share this schema, with the original target paths/IDs retained.

[Portable commands](../archive/publication_support/docs/RUNNING_EXPERIMENTS.md) · [Recorded settings](../configs/experiment_settings.json)
