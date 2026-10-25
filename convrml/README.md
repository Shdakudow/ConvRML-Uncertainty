# Shared layers and paired data

| File | Purpose |
| --- | --- |
| [backbone.py](backbone.py) | ConvNeXtV2-Base encoder, FPN decoder, RGB residual head |
| [film.py](film.py) | Quantile modulation at 36 encoder + 12 decoder normalization sites |
| [quantile_embedding.py](quantile_embedding.py) | Sinusoidal scalar-q embedding |
| [losses.py](losses.py) | Pixelwise pinball loss |
| [intervals.py](intervals.py) | Native Gaussian-moment bounds before calibration |
| [data.py](data.py) | Portable NumPy-cache and RGB paired-file datasets |

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
