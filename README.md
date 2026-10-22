# Lensless reconstruction and uncertainty

Compare reconstructions and uncertainty intervals from ConvNeXt/FPN models.
Uncertainty is `upper - lower`.

| Open | Contents |
| --- | --- |
| **[shared/](shared/)** | One implementation of the backbone, FiLM, quantile model and common pipeline |
| **[Experiment 2](experiments/experiment2/)** | RML / normalized DiffuserCam: Im2im, ensemble, MC dropout and QUTCC |
| **[Experiment 3](experiments/experiment3/)** | Raw / corrupted raw DiffuserCam: Im2im and QUTCC |
| [tools/](tools/) | Plot curves, export saved predictions and check source/equivalence |
| [tests/](tests/) | Shared layout, routing and report checks |
| [archive/](archive/) | Bundled historical code, run records and reproducibility evidence |

**Start:** [run commands](docs/RUNNING_EXPERIMENTS.md) · [model differences](docs/MODEL_COMPARISON.md) · [metrics/calibration](docs/METHODS_AND_METRICS.md).

## Setup

Use Python 3.11 and install CUDA-compatible PyTorch/torchvision first:

```bash
python -m pip install -r requirements.txt
python experiments/experiment2/run_model.py --help
```

Data and weights are separate from source. [Checkpoint releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases) contain archived weights; the workflow guide explains existing cluster data paths.

Common files are linked from experiment entry points. Preserve Git symlinks when copying the checkout. Experiment-specific settings and output directories stay separate. The [equivalence report](archive/provenance/shared_code_equivalence_20261021.json) records matching fixed-seed CPU predictions and gradients.

[Interactive demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) · [upstream ConvRML](https://github.com/lakabuli/ConvRML) · [QUTCC reference](https://github.com/Shdakudow/code-for-quant) · [license](LICENSE)
