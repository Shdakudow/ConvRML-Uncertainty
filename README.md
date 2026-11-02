# ConvRML-Uncertainty

Four methods for lensless reconstruction and uncertainty, sharing a ConvNeXt/FPN
backbone: **Im2im, deep ensemble, MC dropout and QUTCC**.
The models are defined once in [models/](models/). Dataset selection changes the
loader and recorded training settings; each dataset has its own trained weights.

| Dataset | `--dataset` | Available comparisons | Prepared RGB size |
| --- | --- | --- | --- |
| [Parallel Lensless Dataset (PLD)](https://lakabuli.github.io/ConvRML/) | `rml`, `diffuser` | All four methods | 256 × 256 |
| [DiffuserCam Lensless Mirflickr (DLMD)](https://waller-lab.github.io/LenslessLearning/dataset.html) | `raw_clean`, `raw_corrupted` | Im2im and QUTCC | 270 × 480 |

## Install

Python 3.9+, Linux/macOS. Install a suitable PyTorch/torchvision build for your
CPU or CUDA device first, then:

```bash
python -m pip install --upgrade pip
python -m pip install -e . -c reproducibility/constraints-tested.txt
python train.py --help
python calibration.py --help
```

Dependencies and packaging are defined in [pyproject.toml](pyproject.toml).
Optional tools use `python -m pip install -e '.[dev]'` for Black or
`python -m pip install -e '.[plot]'` for archived plotting scripts. Install both
with `'.[dev,plot]'`. The constraints file records the tested numerical versions;
it is not a second dependency list.

## Train and evaluate

Prepare paired measurements and targets using the documented
[data formats](convrml/README.md#reading-the-data) and
[split IDs](reproducibility/README.md#dataset-splits). Image data are external;
see [data availability](reproducibility/DATA.md).

```bash
python train.py --dataset rml --method qutcc \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --output runs/rml/qutcc --epochs 50 --device cuda

python calibration.py --dataset rml --method qutcc \
  --checkpoints /path/to/checkpoint.pt \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --calibration-manifest /path/to/rml/calibration/manifest.json \
  --calibration-pairs /path/to/rml/calibration/pairs.npy \
  --output outputs/rml/qutcc/epoch_050 --device cuda
```

For DLMD, supply its image-file manifest and omit the NumPy `--pairs` arguments.
Models pad inputs internally to multiples of 32 and crop outputs to input size.

Evaluation fits calibration on its separate split, then measures held-out test
SSIM, PSNR, MAE, LPIPS-AlexNet, coverage, mean interval width, crossings and
Pearson/Spearman correlations between uncertainty and absolute error. It exports
per-image CSV, aggregate metrics and optional prediction/bound arrays.
Raw and clipped metrics are separate; LPIPS uses clipped RGB. `--no-lpips` skips
LPIPS for offline fixtures. LPIPS weights otherwise use the standard Torch cache.

The default calibration is RCPS-HB for the baselines and Algorithm 1 quantile
search for QUTCC. `--calibration native` provides an uncalibrated control.
Calibration does not force exact test coverage or make QUTCC outputs monotone.
See [calibration details](archive/publication_support/docs/COVERAGE90.md).

## Saved weights

```bash
python -m convrml.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
```

The single [checkpoint index](convrml/checkpoint_index.json) records release URLs,
sizes and SHA-256 hashes. Downloads verify files before accepting them. Private
release assets require an authenticated GitHub CLI. Milestone availability and
withdrawn artifacts are documented in [reproducibility/](reproducibility/).
Legacy Im2im checkpoint names are translated during strict loading; stored
weights are not rewritten.

## Code map

| Location | Purpose |
| --- | --- |
| [models/](models/) | Four model classes: architecture, training loss and prediction bounds |
| [dataset.py](dataset.py) | PLD NumPy-cache and DLMD image-file loaders |
| [train.py](train.py) | Training, validation, resume and checkpoints |
| [calibration.py](calibration.py) | Frozen weights → calibration → test metrics → exports |
| [convrml/](convrml/) | Shared backbone, FiLM, calibration algorithms, metrics and checkpoint utilities |
| [configs/](configs/) | Recorded optimizer settings, seeds and dataset identities |
| [input_modifications/](input_modifications/) | Measurement corruptions and dataset preparation, preserving targets |
| [experiments/](experiments/) | Optional controlled training studies and cluster launchers |
| [reproducibility/](reproducibility/) | Split IDs, environment and verification records |
| [tests/](tests/) | CPU correctness and workflow checks |
| [archive/](archive/) | Historical source, extended guides and optional tools |

Start with a model's `forward`, `loss` and `bounds`, then follow the loader and
root training/calibration scripts. In `train.py`, read `run_training` for the
complete workflow, `train_one_epoch` for gradient updates, and `create_optimizer`
for explicit Adam/AdamW selection. The older `python -m convrml.train` and
`python -m convrml.evaluate` commands remain small compatibility entry points.
The main workflow does not require a cluster account or archived code.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use small CPU fixtures; they do not retrain the published benchmarks.
[Recorded verification](reproducibility/README.md),
[model details and adaptations](models/README.md), and the
[extended running guide](archive/publication_support/docs/RUNNING_EXPERIMENTS.md)
explain reproducibility and practical limits. Historical optimizer profiles differ
between methods; report those differences alongside accuracy and runtime.

[Citation](CITATION.cff) · [Attribution](THIRD_PARTY.md) · [License](LICENSE) ·
[Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)
