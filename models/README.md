# Four shared model implementations

Models are defined once. Im2im and QUTCC are shared by the Parallel Lensless
Dataset (PLD, ConvRML 100k) and the DiffuserCam Lensless Mirflickr Dataset (DLMD).
Ensemble and MC-dropout comparisons currently use PLD. Dataset-specific loading
and optimization settings are in [dataset.py](../dataset.py) and
[configs/](../configs/), not in separate model branches.

| Method | Implementation | Training objective | Uncertainty |
| --- | --- | --- | --- |
| Im2im | [im2im.py](im2im.py) | Center MSE + fixed endpoint pinball, end to end | Lower/upper output heads; RCPS after training |
| QUTCC | [qutcc.py](qutcc.py) | Uniform sampled-q pinball only | Quantile embedding/FiLM; calibrated q inputs |
| Deep ensemble | [ensemble.py](ensemble.py) | Gaussian NLL for each independent member | Predicted variance and member disagreement; RCPS |
| MC dropout | [mc_dropout.py](mc_dropout.py) | MSE with dropout at 48 sites | 30 stochastic passes, p=0.05; RCPS |

Each model file contains its loss. All methods use the shared ConvNeXtV2-Base/FPN
backbone in [convrml/](../convrml/). Existing PLD runs cover all four methods;
existing DLMD runs cover Im2im and QUTCC.

## Tensor and intensity conventions

`N` is batch size; `H,W` are image dimensions. Loaders supply float32 measurement
and target images in `[0,1]`, shaped `(N,3,H,W)` after batching. Outputs retain
normalized-intensity units but are unbounded; there is no sigmoid or automatic
clipping. Calibration does not change the target range or normalize predictions.

| Call | Output |
| --- | --- |
| `Im2im.forward(y)` | `(N,3,3,H,W)`: lower / center / upper heads, then RGB |
| `QUTCC.forward(y, q)` | `(N,3,H,W)`: one quantile image; `q` has shape `(N,)` |
| `EnsembleMember.forward(y)` | Tuple of mean and positive variance, each `(N,3,H,W)` |
| `DeepEnsemble.forward(y)` | `(N,9,H,W)`: concatenated lower / mean / upper RGB |
| `MCDropout.forward(y)` | `(N,3,H,W)`: one prediction, stochastic when dropout is active |
| Any method's `bounds(y)` | Tuple `(center, lower, upper)`, each `(N,3,H,W)` |
| Any trainable method's `loss(y, target)` | Scalar averaged over images, channels and pixels |

Variance has squared-intensity units. Interval width `upper-lower` and absolute
reconstruction error have intensity units. Crossing is reported rather than
silently sorting endpoints. Evaluation reports both raw and clipped metrics;
clipping is a separate metric/display choice. QUTCC's center is a median estimate,
whereas the other methods use mean estimates.

The [paper-to-code review](../archive/publication_support/docs/MODEL_COMPARISON.md#paper-to-code-review)
records which choices follow the references and which are lensless adaptations.

```python
from models import Im2im, QUTCC

im2im = Im2im()
qutcc = QUTCC()
```

All four default model implementations accept RGB `(N,3,H,W)` measurements.
They replicate-pad right/bottom to multiples of 32 and crop all outputs to the
original dimensions before computing losses or intervals. For example, 270×480
is padded to 288×480 internally. Spatial dimensions are not learned parameters.

The loader decodes and normalizes paired images; it does not resize them for the
network. Images in a normal stacked batch must share dimensions, and targets must
match the measurement dimensions. The recorded experiment profiles still enforce
their saved shapes, splits and available methods. Variable-size model support
does not create new trained comparisons or make weights transferable across
acquisition systems. Existing ensemble/MC-dropout benchmark runs remain PLD-only.

## Im2im checkpoint names

`Im2im()` always uses `backbone.*` weight names, for every dataset. It requests
three RGB output heads directly with `ConvNeXtFPN(output_heads=3)`; no temporary
three-channel head is created and replaced.

The common checkpoint loader automatically translates old `base.*` and `net.*`
names to `backbone.*` in memory before strict loading. It does not overwrite the
checkpoint file or alter weight values. For direct weight loading:

```python
from models import Im2im
from convrml.checkpoints import read_checkpoint, load_model_weights

model = Im2im()
checkpoint, _ = read_checkpoint("weights.pt")
load_model_weights(model, checkpoint["model"])
```

Translation depends on the saved key format, not the dataset or image dimensions.
Formal evaluation still checks checkpoint/data identity. Using an older model as
initialization for a new dataset is a separate transfer experiment; renaming
does not make its old calibration valid for that dataset.

Loaded checkpoints retain their predictions. Fresh Im2im initializations remain
seeded, but their head values differ from the old constructor: removing the
discarded layer removes its random draws. Encoder/decoder initialization and the
other methods are unchanged. Historical runs retain their original source snapshots.
