# Using the models

Install from the repository root after installing a suitable PyTorch build:

```bash
python -m pip install -e .
```

## Construct a model

```python
from experiments.experiment2.models import Im2im, QUTCC, MCDropout, EnsembleMember, DeepEnsemble
from experiments.experiment3.models import Im2im as RawIm2im

model = QUTCC()
raw_model = RawIm2im()
mc_model = MCDropout(p=0.05)
```

Construction initializes random weights; it does not download a trained model.
Use [checkpoint Releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases)
and the correct method/camera/epoch. Model dictionaries retain their historical
parameter names, including `base.*` (Exp2 Im2im), `net.*` (raw Im2im/QUTCC), and
`baseModel.*` (one ensemble member).

```python
import torch
checkpoint = torch.load("epoch_050.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

Dataset manifests and checkpoints are separate from code. Preserve their sample
IDs, order and camera/method identities. [Data schemas](../convrml/README.md) show
both loaders without machine-specific paths.

## Forward, loss and intervals

`y` is a normalized measurement batch `(B,3,H,W)` and `x` its ground truth.

```python
loss = model.loss(y, x)  # QUTCC samples one uniform scalar q per image.
loss.backward()

model.eval()
with torch.inference_mode():
    reconstruction, lower, upper = model.bounds(y)
    uncertainty = upper - lower
```

Im2im `.forward(y)` returns `(B,3 heads,3 RGB,H,W)` in lower/center/upper order.
QUTCC `.forward(y,q)` returns one RGB quantile image; `q` has shape `(B,)`.
MC dropout `.forward(y)` is one RGB pass; `.bounds(y)` uses 30 masked passes.
Each ensemble member `.forward(y)` returns `(RGB mean, RGB variance)` and
`.loss(y,x)` computes Gaussian NLL. Train members independently; do not reset
the same seed before each construction or share a dataloader shuffle generator.

```python
# Load each member's own checkpoint before aggregation.
members = [EnsembleMember() for _ in range(5)]
ensemble = DeepEnsemble(members).eval()
with torch.inference_mode():
    reconstruction, lower, upper = ensemble.bounds(y)
```

These are native bounds. Fit calibration only on a separate calibration split,
then evaluate frozen parameters on test pairs. QUTCC accepts calibrated global
levels via `model.bounds(y, lower_q=q_low, upper_q=q_high)`. Im2im/ensemble/MC use
independent lower/upper distance scales. Clipping to [0,1] is a separate display
or metric convention, not an automatic consequence of calibration.

## Reproduce the recorded runs

Model definitions are the publication interface. The exact historical optimizers,
source guards, split audits, RNG restoration and corruption trainers remain in
[archive/reproduction](../archive/reproduction/). Its experiment folders preserve
the previous import context and shared source links.

- [Exp2 trainer/evaluator commands](../archive/reproduction/docs/RUNNING_EXPERIMENTS.md#experiment-2)
- [Raw and corruption adaptation](../archive/reproduction/docs/RUNNING_EXPERIMENTS.md#experiment-3)
- [Matched-coverage evaluation](COVERAGE90.md)

For example, inspect the preserved router without running training:

```bash
python archive/reproduction/experiments/experiment2/run_model.py train \
  --method qutcc --camera rml --epoch 5 --dry-run
```

The original cluster data paths and complete recorded settings are in that
reproduction guide. The model-only examples above do not replace its optimizer
or checkpoint-resume protocols. No training or demo deployment is triggered by
importing the publication models.

## Verification

```bash
python -m unittest discover -s tests -v
python tools/verify_source_release.py
```

[Equivalence evidence](../archive/provenance/publication_model_equivalence_20261025.json)
records exact CPU matches against the models before this publication cleanup:
state keys/shapes, initialization, predictions, input/parameter gradients and RNG.
