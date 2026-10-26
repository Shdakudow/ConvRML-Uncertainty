# Experiment 3: raw and corrupted raw DiffuserCam

Two models use the same ConvNeXt/FPN backbone as Experiment 2.
Open **[models/](models/)**; no separate architecture is introduced for corruption.

| Model | File | Difference from Experiment 2 |
| --- | --- | --- |
| Im2im | [im2im.py](models/im2im.py) | Same lower/center/upper heads and MSE + pinball loss; raw-size padding/cropping and preserved `net.*` checkpoint names |
| QUTCC | [qutcc.py](models/qutcc.py) | The exact same QUTCC implementation; only data and training protocol differ |

## Dimensions and data loading

- Measurement and target: **RGB 270×480**.
- Models replicate-pad right/bottom to **288×480**, then crop every output back. This makes encoder implementation easier.
- Padding is excluded from losses and metrics by cropping the predictions first.
- Loader: [`convrml.data.ImagePairs`](../../convrml/data.py), using a paired-file manifest.
- Read RGB images, convert to float32 and divide by 255. The loader does not resize.
- "Raw" identifies this dataset branch; the training loader reads its prepared RGB images, not native sensor-bit-depth arrays.
- Split: **17,999 training / 2,000 validation / 2,000 test / 3,000 calibration** pairs.

```python
from convrml.data import ImagePairs
from experiments.experiment3.models import Im2im

pairs = ImagePairs("split.json", split="train")
measurement, target = pairs[0]
model = Im2im()
loss = model.loss(measurement[None], target[None])
```

Clean and fixed-corruption manifests have the same loader interface. For corrupted
measurements, targets and split IDs remain unchanged; no new random corruption is
applied while loading. Use each dataset's own checkpoints and calibration results.

Clean models train from scratch for 50 epochs. Corruption adaptation starts from
each clean epoch-30 checkpoint and adds 30 adaptation epochs at LR 1e-5. This is a
training-protocol difference, not an additional model layer.

[Data format](../../convrml/README.md) · [Model comparison](../../docs/MODEL_COMPARISON.md) ·
[Portable training/evaluation](../../docs/RUNNING_EXPERIMENTS.md)
