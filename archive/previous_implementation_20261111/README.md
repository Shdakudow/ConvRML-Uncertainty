# ConvRML-Uncertainty

Four methods for lensless image reconstruction and uncertainty: **Im2im, QUTCC,
deep ensemble, and MC dropout**, sharing a ConvNeXtV2/FPN backbone.

**FPN means Feature Pyramid Network** in this implementation: its decoder combines
features at multiple scales. The original ConvRML paper calls its decoder a
**Feedforward Neural Network (FNN)**. Our `ConvNeXtFPN` name describes this
implementation and is not the paper’s terminology; the architecture is an
adaptation, not an exact reproduction of its ConvNeXt-T/FNN model.
See the [paper’s reconstruction architecture](https://arxiv.org/html/2602.04834v2#S4).

Generalized-input and corruption-specialist studies are maintained separately in
[ConvRML-Generalized-Specialists](https://github.com/Shdakudow/ConvRML-Generalized-Specialists).
[Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)
show previously trained experiments; changing this code does not retrain those models.

## Code layout

| Location | What it does |
| --- | --- |
| [train.py](train.py) | Arguments, setup, epoch coordination, validation and saving |
| [config.json](config.json) | Defaults for datasets, optimizers, schedules, seeds and calibration |
| [dataset.py](dataset.py) | Load paired measurements and ground truths |
| [models/](models/) | Four model definitions, their losses and prediction methods |
| [convrml/backbone.py](convrml/backbone.py) | Shared ConvNeXt/FPN architecture |
| [convrml/film.py](convrml/film.py) | QUTCC feature modulation by quantile embeddings |
| [training.py](training.py) | Settings/profile loading, AdamW, learning-rate schedule and batch/epoch updates |
| [convrml/calibration.py](convrml/calibration.py) | Baseline RCPS and QUTCC quantile calibration algorithms |
| [evaluation/](evaluation/) | Evaluation entry point, metrics, reports and downloads |
| [tests/](tests/) | Small automated correctness checks; not the image test dataset |
| [archive/](archive/) | Historical code, split records, environments and checkpoint catalog |

Start with a model's `forward`, `loss` and `bounds`, then `train_batch` in
`training.py`. The main workflow is `run_training` in `train.py`.
The training workflow has only two Python files: `train.py` coordinates it and
`training.py` implements optimization. Edit numerical settings in `config.json`.
Model construction lives in `models/__init__.py`; loader selection and split
checks live in `dataset.py`. There is no separate `protocol.py` layer.
Loss formulas live in model classes; the shared loop calls `model.loss(...)`.
Only QUTCC uses the optional quantile generator passed to that method.

## Methods

| Method | Training loss | Uncertainty at inference |
| --- | --- | --- |
| Im2im | Center MSE + fixed lower/upper pinball losses, end to end | Endpoint heads, then asymmetric RCPS scaling |
| QUTCC | Pinball with independently sampled uniform q per image | Quantile-conditioned predictions using FiLM; calibrated q endpoints |
| Deep ensemble | Gaussian NLL for each independent member | Member mean/variance aggregation, then RCPS |
| MC dropout | MSE with dropout | Repeated stochastic predictions, then RCPS |

Current defaults: target miss fraction `alpha=0.10`, RCPS failure budget
`delta=0.05`, dropout probability `p=0.05`, and 30 MC inference passes.
Calibration uses separate scenes and does not update model weights or force
exactly 90% test coverage. It does not normalize images or guarantee noncrossing
QUTCC endpoints. See [model details](models/README.md).

## Setup

Run commands from this repository's root. Use a Python environment with a
CPU/CUDA-compatible PyTorch and torchvision installation, then install the
remaining dependencies:

```bash
python -m pip install timm==1.0.22 numpy Pillow torchmetrics==1.8.2 scipy
python train.py --help
python -m evaluation.run --help
```

A [historical tested environment](archive/reproducibility/environment-tested.json)
and [version constraints](archive/reproducibility/constraints-tested.txt) are
retained for reference. This checkout runs directly; it does not require
`pip install -e .`. LPIPS evaluation may download pretrained AlexNet weights;
`--no-lpips` skips that metric for offline checks.

## Data

| Dataset | `--dataset` | Models configured | Prepared RGB dimensions |
| --- | --- | --- | --- |
| Parallel Lensless Dataset: RML | `rml` | All four | 256 × 256 |
| Parallel Lensless Dataset: normalized DiffuserCam | `diffuser` | All four | 256 × 256 |
| DiffuserCam Lensless MIRFLICKR | `raw_clean` | Im2im, QUTCC | 270 × 480 |
| Prepared corrupted MIRFLICKR | `raw_corrupted` | Im2im, QUTCC adaptation | 270 × 480 |

Images are external. See [data availability](archive/reproducibility/DATA.md).
These are prepared uint8 RGB pairs, converted to float32 by dividing by 255.
The loader does not resize or corrupt images. Models pad internally and crop
outputs to the input dimensions. Predictions themselves are not restricted to [0,1].

A **manifest** is a JSON file listing fixed split membership, not a new random
split. PLD uses a NumPy cache shaped `(N, 2, 3, H, W)` and a split file:

```json
{"train": [{"id": "a"}], "val": [{"id": "b"}], "test": [{"id": "c"}]}
```

The cache concatenates train, val, then test pairs; each pair stores measurement
first and target second. The calibration cache has its own manifest containing
only `calibration` rows.

MIRFLICKR uses image-file paths instead:

```json
{"roles": {"train": [{"id": "a", "measurement": {"path": "measurements/a.png"}, "target": {"path": "targets/a.png"}}], "val": [], "test": [], "calibration": []}}
```

Fill the required split lists with actual pairs. Paths may be absolute or relative
to the manifest. Split IDs must be unique and disjoint. The model code is shared
across datasets; each dataset requires its own trained weights.

## Train

```bash
python train.py --dataset rml --method qutcc \
  --manifest data/rml/manifest.json --pairs data/rml/pairs.npy \
  --output runs/rml/qutcc --epochs 50 --device cuda
```

For MIRFLICKR, use `--dataset raw_clean --manifest data/split.json` and omit
`--pairs`. Select another method with `--method im2im`, `mc_dropout`, or
`ensemble`. Ensemble training requires `--member 0` through `--member 9`, with a
separate run folder for each member. Member seeds control both initialization
and data order independently.

All **new runs use AdamW**. Current profiles still differ:

| Profile | Used for | LR schedule | Betas | Weight decay | Effective batch |
| --- | --- | --- | --- | --- | --- |
| `baseline` | PLD Im2im, ensemble, MC dropout | Constant 1e-4, 50 epochs | (0.9, 0.999) | 1e-4 | 2 |
| `paper` | QUTCC and clean MIRFLICKR Im2im | 30% warmup 1e-5 → 1e-4, then cosine toward 1e-9, 50 epochs | (0.9, 0.99) | 0 | 4 |
| `adaptation` | Corrupted MIRFLICKR | Constant 1e-5, 30 epochs | (0.9, 0.99) | 0 | 4 |

`baseline` uses BF16 and gradient clipping at norm 1; the other profiles use FP32
without gradient clipping. Image-file datasets use microbatch size 1. Adaptation
requires `--initialize /path/to/same_method_clean_epoch_030.pt` or `--resume`.
Historical Adam runs retain their recorded optimizer; resume them with their
original source. Standardizing the optimizer name does not make every profile identical.

Each run saves its effective `config.json`, per-epoch `history.json`, a recovery
checkpoint `latest.pt`, and weight milestones every five epochs. Resume with the
same arguments plus `--resume`. `--epochs` is an absolute stopping epoch; it does
not reset the configured learning-rate horizon. Resume checks settings and split
counts, not image-ID hashes, so retain the same split file and data ordering.
Use `--smoke` and a separate output folder for a two-batch check.

## Calibrate and evaluate

```bash
python -m evaluation.run --dataset rml --method qutcc \
  --checkpoints runs/rml/qutcc/checkpoints/epoch_050.pt \
  --manifest data/rml/manifest.json --pairs data/rml/pairs.npy \
  --calibration-manifest data/rml/calibration/manifest.json \
  --calibration-pairs data/rml/calibration/pairs.npy \
  --output outputs/rml/qutcc_050 --examples 10 --device cuda
```

For image-file data, omit both NumPy arguments and pass a manifest containing
the appropriate test/calibration splits to each manifest option. For ensembles,
pass the ordered member checkpoints after `--checkpoints`. `--calibration native`
provides an uncalibrated control without opening calibration images.

Results include `calibration.json`, `report.json`, `per_image.csv`, and optional
`examples.npz`. Metrics include PSNR, SSIM, MAE, LPIPS, coverage, mean interval
width, crossed-bound frequency, and Pearson/Spearman uncertainty–error correlation.
Raw and clipped metrics are reported separately; LPIPS uses clipped RGB.
Uncertainty is `upper - lower`, in normalized-intensity units.

Published weights can be downloaded using:

```bash
python -m evaluation.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
```

The [archived catalog](archive/checkpoints/checkpoint_index.json) records available
artifacts and verifies their sizes/hashes. Private assets require GitHub access.
Weights are not included in the source repository.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/` contains small CPU fixtures that check code correctness: pairing and
normalization, model outputs, losses, calibration, metrics, export formats,
checkpoint handling, and resumed versus uninterrupted training. It contains no
held-out benchmark images and does not run full model training. The checks do
not establish accuracy on real data or deterministic equality across GPU types.
