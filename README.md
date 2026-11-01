# ConvRML-Uncertainty

A ConvNeXt/FPN comparison of lensless reconstruction and uncertainty:
**Im2im, deep ensemble, MC dropout and QUTCC**.

The four methods are implemented once in [models/](models/). Dataset selection
changes the loader and recorded settings, not the model implementation.

| Dataset | Command selector | Existing comparisons | Prepared measurements |
| --- | --- | --- | --- |
| [Parallel Lensless Dataset (PLD)](https://lakabuli.github.io/ConvRML/) | `rml`, `diffuser` | All four methods | RML / normalized diffuser, RGB 256×256 |
| [DiffuserCam Lensless Mirflickr Dataset (DLMD)](https://waller-lab.github.io/LenslessLearning/dataset.html) | `raw_clean`, `raw_corrupted` | Im2im and QUTCC | Clean / fixed-corruption RGB 270×480 |

```text
models/           im2im.py, qutcc.py, ensemble.py, mc_dropout.py — shared across datasets
dataset.py        PLD NumPy-cache and DLMD paired-image loaders
input_modifications/  One file per raw-input modification and dataset preparation
configs/          Recorded settings, seeds, split sizes and dataset descriptions
train.py          Training loop, validation and epoch checkpoints
calibration.py          Calibration, test predictions, metrics and export
convrml/          Shared backbone, FiLM, runtime, calibration and metrics
reproducibility/   Settings, ordered split IDs, checkpoint index and tested environment
tests/             Local correctness checks
archive/           Optional guides, plotting, CI configuration and historical code
```

## Install

Linux/macOS, Python 3.9 or newer. Install a suitable PyTorch/torchvision build for
your CPU or CUDA device, then:

```bash
python -m pip install -e . -c reproducibility/constraints-tested.txt
python train.py --help
python calibration.py --help
python -m convrml.download --help
```

[Quick start](archive/publication_support/docs/RUNNING_EXPERIMENTS.md) · [Data formats](convrml/README.md) ·
[Model differences](archive/publication_support/docs/MODEL_COMPARISON.md) · [Reproducibility](reproducibility/README.md)

## Read the code

Start with one file in [models/](models/): `forward` defines the network,
`loss` defines its training objective, and `bounds` produces the center/lower/upper
images. Then read [dataset.py](dataset.py) to see how paired images reach the model.

In [train.py](train.py), `train_one_epoch` contains the forward/backward loop;
`run_training` handles setup, validation and checkpoints. The
[calibration entry point](calibration.py) validates the request, loads frozen
weights, then runs `calibrate_checkpoint` → `evaluate_test_set` → `save_results`.
[Calibration algorithms](convrml/calibration.py) and
[test metrics/exports](convrml/evaluation.py) live in separate modules;
[checkpoints.py](convrml/checkpoints.py) handles saving, strict loading and resume.
Recorded experiment settings stay in [configs/](configs/).
See [tensor shapes and intensity conventions](models/README.md#tensor-and-intensity-conventions)
and the [paper-to-code review](archive/publication_support/docs/MODEL_COMPARISON.md#paper-to-code-review)
for the model interfaces and documented adaptations.
The [backbone walkthrough](convrml/README.md#following-the-backbone) traces feature
sizes; the [output-file guide](archive/publication_support/docs/RUNNING_EXPERIMENTS.md#outputs)
explains which saved files to open.

## Weights and evaluation

```bash
python -m convrml.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
```

With prepared data at `/path/to/rml`, train or evaluate using:

```bash
python train.py --dataset rml --method qutcc \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --output runs/rml/qutcc --epochs 50 --device cuda

python calibration.py --dataset rml --method qutcc \
  --checkpoints weights/rml/qutcc/exp2__rml__qutcc__epoch_050.pt \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --calibration-manifest /path/to/rml/calibration/manifest.json \
  --calibration-pairs /path/to/rml/calibration/pairs.npy \
  --calibration paper --output outputs/paper/rml/qutcc/epoch_050 --device cuda
```

`--dataset` selects PLD or DLMD data handling and settings automatically. Each
dataset has its own trained weights and calibration; shared code does not mean
interchangeable weights. Historical checkpoint filenames/metadata retain their
original IDs. The package commands `python -m convrml.train` and
`python -m convrml.evaluate` remain equivalent to these entry points.

The layout uses `models/`, `configs/`, `dataset.py`, `train.py`, and the renamed
`calibration.py` entry point. The model/data organization follows
[upstream ConvRML](https://github.com/lakabuli/ConvRML). Our recorded
prepared-data formats, preprocessing and independent splits remain unchanged.

The [checkpoint index](reproducibility/checkpoints.json) identifies every saved
five-epoch milestone, including five-/ten-member ensemble references, with file
size and SHA-256. Downloads verify both before accepting a file. GitHub CLI
access is needed for private release assets; public assets can download without it.
Prepared image data are external; [data availability](reproducibility/DATA.md)
links the original datasets and specifies the exact input formats and split IDs.

[Raw-input modifications](input_modifications/README.md) provides separate modules
for Gaussian, Poisson, combined noise, exposure, dead pixels, fixed-pattern noise,
Gaussian blur, resolution loss with factor 64, and JPEG compression. The preparation
command creates separate MIRFLICKR measurement datasets while preserving the
original targets and split identities. It does not submit model training.

`python -m convrml.evaluate` loads frozen weights, fits on the separate calibration
split and reports held-out SSIM, PSNR, MAE, LPIPS-AlexNet, coverage, width,
crossings and Pearson/Spearman correlations of width with absolute error. It saves
per-image CSV, dataset means/valid counts and optional reconstruction/bound arrays.
LPIPS uses clipped RGB; the other metrics retain separate raw/clipped results.
Pretrained LPIPS weights use the standard Torch cache; `--no-lpips` explicitly
omits that metric for tiny/offline fixtures. Prediction and metric time are separate.
[Calibration](archive/publication_support/docs/COVERAGE90.md) defaults to the paper
procedures: RCPS-HB for Im2im, ensemble and MC dropout; Algorithm 1 calibrated
quantiles for QUTCC. `--calibration native` is an uncalibrated control. Calibration
does not force exact test coverage. Superseded empirical matching is archived.

## Verification and attribution

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Training and evaluation need no archived code or cluster configuration.
[Release verification](reproducibility/verification.json) records the v0.1.0 checks;
[model equivalence](reproducibility/model-equivalence.json)
records unchanged original model predictions and gradients.
[Cleanup verification](reproducibility/cleanup_verification.json) records the
post-release archive move and unused-code removal checks.

[Citation](CITATION.cff) · [Upstream methods and credit](THIRD_PARTY.md) ·
[License](LICENSE) · [Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)

Original run engines, cluster operations and historical source records remain
in [archive/](archive/). Extended guides, plotting/provenance tools and inactive
GitHub Actions configuration are in [publication support](archive/publication_support/).
Existing checkpoint files remain unchanged. The loader translates legacy Im2im
names to the common `backbone.*` namespace; new models need no dataset-specific
naming option. See [checkpoint compatibility](models/README.md#im2im-checkpoint-names).
