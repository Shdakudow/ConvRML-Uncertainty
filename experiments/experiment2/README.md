# Experiment 2: RML and normalized DiffuserCam

Four uncertainty methods use the same ConvNeXtV2-Base/FPN reconstruction backbone.
Open **[models/](models/)**; each method has one annotated file.

| Model | File | Output and loss | Native uncertainty |
| --- | --- | --- | --- |
| Im2im | [im2im.py](models/im2im.py) | Lower/center/upper RGB; center MSE + two pinball terms, end to end | Two endpoint heads |
| Deep ensemble | [ensemble.py](models/ensemble.py) | Independent RGB mean/variance members; Gaussian NLL | Combined predicted variance and member disagreement |
| MC dropout | [mc_dropout.py](models/mc_dropout.py) | One RGB prediction; MSE; dropout at 48 sites | 30 stochastic passes, p=0.05 |
| QUTCC | [qutcc.py](models/qutcc.py) | RGB image at input q; sampled-q pinball only; 48 FiLM sites | Query separate lower and upper q values |

## Dimensions and data loading

- Measurement and target: **RGB 256×256**.
- Loader: [`convrml.data.NumpyPairs`](../../convrml/data.py).
- Disk cache: uint8 `(N, 2, 3, 256, 256)`, measurement first, target second.
- Loader output: float32 `(2, 3, 256, 256)` divided by 255; a batch is `(B, 2, 3, 256, 256)`.
- Existing split order and IDs are retained. Calibration uses its own cache/manifest.
- Per camera: **60,000 training / 5,000 validation / 5,000 test / 2,048 calibration** pairs.

```python
from convrml.data import NumpyPairs
from experiments.experiment2.models import QUTCC

pairs = NumpyPairs("pairs.npy", "manifest.json", split="train")
measurement, target = pairs[0]
model = QUTCC()
loss = model.loss(measurement[None], target[None])
```

These are the actual model definitions, with shared layers in [convrml/](../../convrml/).
Models return native outputs; clipping and fitted interval calibration are separate.
Ensemble members each train on the complete split with independent initialization
and shuffle seeds. Historical training settings differ between QUTCC and baselines;
see [the comparison](../../docs/MODEL_COMPARISON.md).

[Usage and checkpoints](../../docs/RUNNING_EXPERIMENTS.md) ·
[Original reproduction scripts](../../archive/reproduction/experiments/experiment2/)
