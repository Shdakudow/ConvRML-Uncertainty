# Shared model and runtime code

Every Python module here is used by the public models or the train/evaluate/download
commands. Optional plotting, provenance tools and GitHub Actions live in
[archive/publication_support/](../archive/publication_support/).

| File | Purpose |
| --- | --- |
| [backbone.py](backbone.py) | ConvNeXtV2-Base encoder, FPN decoder, RGB residual head |
| [film.py](film.py) | Quantile modulation at 36 encoder + 12 decoder normalization sites |
| [quantile_embedding.py](quantile_embedding.py) | Sinusoidal scalar-q embedding |
| [losses.py](losses.py) | Pixelwise pinball loss |
| [intervals.py](intervals.py) | Native Gaussian-moment bounds before calibration |
| [calibration.py](calibration.py) | Empirical endpoint scales and quantile search |
| [reference_calibration.py](reference_calibration.py) | Reference RCPS-HB and Algorithm 1 formulas |
| [data.py](data.py) | Portable NumPy-cache and RGB paired-file datasets |
| [train.py](train.py) / [evaluate.py](evaluate.py) | Training/resume and held-out calibrated evaluation |
| [metrics.py](metrics.py) | SSIM, PSNR, MAE, coverage, width and crossing metrics |
| [protocol.py](protocol.py) / [experiment_settings.json](experiment_settings.json) | Model/data selection, recorded settings, identity and resume checks |
| [checkpoints.py](checkpoints.py) | Strict loading and validation of saved weights |
| [download.py](download.py) / [checkpoint_index.json](checkpoint_index.json) | Verified downloads of published checkpoint files |

Both loaders return float32 `(2, 3, H, W)`: measurement first, target second.
RGB uint8 values are divided by 255. Neither loader clips predictions, applies
augmentation, changes targets, or creates new train/test splits.

## Experiment 2: NumPy cache

```python
from convrml.data import NumpyPairs
train = NumpyPairs("pairs.npy", "manifest.json", "train")
calibration = NumpyPairs("calibration/pairs.npy", "calibration/manifest.json", "calibration")
```

The uint8 array shape is `(N, 2, 3, 256, 256)`. Normal cache order is train, val,
then test. The separate calibration cache has only calibration pairs. Manifests
retain the existing image IDs and order:

```json
{"train": [{"id": "scene_a"}], "val": [{"id": "scene_b"}], "test": [{"id": "scene_c"}]}
```

## Experiment 3: image-file manifest

```python
from convrml.data import ImagePairs
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

[Portable commands](../archive/publication_support/docs/RUNNING_EXPERIMENTS.md) · [Recorded settings](experiment_settings.json)
