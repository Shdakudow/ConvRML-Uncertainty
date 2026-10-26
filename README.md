# ConvRML-Uncertainty

A ConvNeXt/FPN comparison of lensless reconstruction and uncertainty:
**Im2im, deep ensemble, MC dropout and QUTCC**.

| Experiment | Models | Measurements |
| --- | --- | --- |
| [Experiment 2](experiments/experiment2/) | [Four models](experiments/experiment2/models/) | RML / normalized DiffuserCam, RGB 256×256 |
| [Experiment 3](experiments/experiment3/) | [Two models](experiments/experiment3/models/) | Clean / corrupted raw DiffuserCam, RGB 270×480 |

```text
experiments/       Four / two annotated model definitions
convrml/           Shared layers, data loaders, training, calibration and evaluation
reproducibility/   Settings, ordered split IDs, checkpoint index and tested environment
tools/             Plotting and optional source-verification tools
```

## Install

Linux/macOS, Python 3.9 or newer. Install a suitable PyTorch/torchvision build for
your CPU or CUDA device, then:

```bash
python -m pip install -e . -c reproducibility/constraints-tested.txt
python -m convrml.train --help
python -m convrml.evaluate --help
python -m convrml.download --help
```

[Quick start](docs/RUNNING_EXPERIMENTS.md) · [Data formats](convrml/README.md) ·
[Model differences](docs/MODEL_COMPARISON.md) · [Reproducibility](reproducibility/README.md)

## Weights and evaluation

```bash
python -m convrml.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
```

The [checkpoint index](reproducibility/checkpoints.json) identifies every saved
five-epoch milestone, including five-/ten-member ensemble references, with file
size and SHA-256. Downloads verify both before accepting a file. GitHub CLI
access is needed for private release assets; public assets can download without it.
Prepared image data are external; [data availability](reproducibility/DATA.md)
links the original datasets and specifies the exact input formats and split IDs.

`python -m convrml.evaluate` loads frozen weights, fits on the separate calibration
split and reports held-out SSIM, PSNR, MAE, coverage, width and crossings. It saves
raw and clipped metrics, per-image CSV and optional reconstruction/bound arrays.
[Calibration modes](docs/COVERAGE90.md) distinguish empirical matching, reference
RCPS/Algorithm 1 and native intervals. Calibration does not force test coverage.

## Verification and attribution

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

The public workflow needs no archived trainers or cluster configuration.
[Verification evidence](reproducibility/verification.json) records the checks;
[model equivalence](reproducibility/model-equivalence.json)
records unchanged original model predictions and gradients.

[Citation](CITATION.cff) · [Upstream methods and credit](THIRD_PARTY.md) ·
[License](LICENSE) · [Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)

Original run engines, cluster operations and historical source records remain
in [archive/](archive/). Existing trained weights retain their parameter names.
