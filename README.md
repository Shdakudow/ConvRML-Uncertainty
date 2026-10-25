# Lensless reconstruction and uncertainty

ConvNeXt/FPN implementations of **Im2im, deep ensemble, MC dropout and QUTCC**.
Each model predicts a reconstruction and native uncertainty bounds. Interval
width is `upper - lower`; calibration is fitted separately on held-out pairs.

| Experiment | Model code | Measurements |
| --- | --- | --- |
| **[Experiment 2](experiments/experiment2/)** | [Four models](experiments/experiment2/models/) | RML and normalized DiffuserCam, RGB 256×256 |
| **[Experiment 3](experiments/experiment3/)** | [Two models](experiments/experiment3/models/) | Clean/corrupted raw DiffuserCam, RGB 270×480 |

```text
experiments/
  experiment2/models/    im2im.py  ensemble.py  mc_dropout.py  qutcc.py
  experiment3/models/    im2im.py  qutcc.py
convrml/                 Shared backbone, FiLM, losses, data loaders and intervals
```

The experiment folders contain only their README and model definitions.
QUTCC uses the same implementation in both experiments. Raw Im2im preserves
its checkpoint-specific wrapper and adds padding/cropping for rectangular images.

## Getting started

Use a CUDA-compatible PyTorch installation for GPU execution. From a clone:

```bash
python -m pip install -e .
```

```python
from experiments.experiment2.models import Im2im, EnsembleMember, DeepEnsemble, MCDropout, QUTCC
from experiments.experiment3.models import Im2im as RawIm2im

model = QUTCC()  # Randomly initialized; load a trained checkpoint before evaluation.
```

Read **[model differences](docs/MODEL_COMPARISON.md)**,
**[data formats](convrml/README.md)** and **[usage](docs/RUNNING_EXPERIMENTS.md)**.
The class APIs do not automatically calibrate or clip outputs.

Datasets are not included. Trained checkpoints are stored in
[Releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases).
Original training/evaluation engines and run settings are preserved under
[archive/reproduction](archive/reproduction/), separate from the model interface.
The [equivalence report](archive/provenance/publication_model_equivalence_20261025.json)
checks unchanged state keys, initialized weights, predictions, gradients and RNG.

[Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) ·
[ConvRML](https://github.com/lakabuli/ConvRML) ·
[QUTCC reference](https://github.com/Shdakudow/code-for-quant) · [License](LICENSE)
