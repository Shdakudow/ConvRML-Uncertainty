# Shared model and runtime code

FPN expands to **Feature Pyramid Network**, our implementation's multiscale
decoder name. The original ConvRML paper uses **Feedforward Neural Network
(FNN)** for its decoder; these names should not imply identical architectures.

Every Python module here is used by the public models or the train/evaluate/download
commands. Optional plotting, provenance tools and GitHub Actions live in
[archive/publication_support/](../archive/publication_support).

| File                                                                                         | Purpose                                                                                                      |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [backbone.py](backbone.py)                                                                   | ConvNeXtV2-Base encoder, FPN decoder, RGB residual head                                                      |
| [film.py](film.py)                                                                           | Quantile modulation at 36 encoder + 12 decoder normalization sites                                           |
| [quantile_embedding.py](quantile_embedding.py)                                               | Sinusoidal scalar-q embedding                                                                                |
| [intervals.py](intervals.py)                                                                 | Native Gaussian-moment bounds before calibration                                                             |
| [calibration.py](calibration.py)                                                             | Baseline RCPS-HB scales and QUTCC Algorithm 1 calibrated quantiles                                           |
| [reporting.py](../evaluation/reporting.py)                                                               | Held-out test loop, metrics, report assembly, CSV and example exports                                        |
| [dataset.py](../dataset.py)                                                                  | PLD NumPy-cache and DLMD RGB paired-file datasets                                                            |
| [../train_setup.py](../train_setup.py) / [evaluation/run.py](../evaluation/run.py)                          | Training loop and calibration → test → export workflow                                                       |
| [metrics.py](../evaluation/metrics.py)                                                                     | SSIM, PSNR, MAE, LPIPS, coverage, width, crossings and uncertainty/error correlations                        |
| [protocol.py](protocol.py) / [config.json](../config.json) | Model/data selection, recorded settings, identity and resume checks                                          |
| [checkpoints.py](checkpoints.py)                                                             | Training-state/milestone saving, legacy Im2im name translation, strict loading, resume and ensemble assembly |
| [download.py](../evaluation/download.py) / [checkpoint_index.json](../archive/checkpoints/checkpoint_index.json)                  | Verified downloads of published checkpoint files                                                             |

Training losses, including the pinball formula, are defined directly in each
[Im2im/QUTCC model file](../models). PLD and DLMD use the same Im2im and QUTCC
classes; there are no dataset-specific copies of these losses or architectures.
Use the root `train_setup.py` and `python -m evaluation.run` commands; each workflow has one entry point.

## Read it as a PyTorch experiment

Start with [dataset.py](../dataset.py), then a model in [models/](../models),
then `train_batch` in [training.py](../training.py). These are the data, prediction/loss,
and optimizer update. Configuration and checkpoint bookkeeping can wait.

A **manifest** is only a saved list of images. `read_split_file` uses ordinary
`json.load` and returns a dictionary such as `splits["train"]`. The MIRFLICKR
files wrap that dictionary in a `roles` key; the reader removes that wrapper.
It never makes a new random split. Keeping these saved lists prevents accidental
changes to which scenes are used for training, calibration and testing.

```python
from torch.utils.data import DataLoader
from dataset import ImagePairs

# JSON lists paired filenames; ImagePairs opens the actual images lazily.
training = ImagePairs("split.json", "train")
loader = DataLoader(training, batch_size=4, shuffle=True)
pairs = next(iter(loader))                # [B, 2, 3, H, W]
measurement = pairs[:, 0]                 # [B, 3, H, W]
ground_truth = pairs[:, 1]                # [B, 3, H, W]
```

This snippet illustrates the data API, not the full reproducible training command:
`train_setup.py` additionally sets the recorded seeds and shuffle policy. The loader
converts prepared uint8 images to float32 by dividing by 255; it does not resize
or corrupt them. Image paths are resolved once when constructing the dataset.
For PLD, `NumpyPairs(pairs_path, manifest_path, "train")` returns the same tensors
from a memory-mapped NumPy array instead of opening image files.

The training update is conventional PyTorch: clear gradients, predict and compute
`model.loss(measurement, ground_truth)`, call `loss.backward()`, then
`optimizer.step()`. QUTCC additionally receives its seeded quantile generator.
Microbatching performs several weighted backward calls before one optimizer step
to fit the effective batch in GPU memory. Validation uses `model.eval()` and does
not update weights. Calibration is a separate program using separate scenes.

## Tracing one experiment

| Step                 | Follow this code                                        | What to inspect                                                                          |
| -------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Recorded settings    | `protocol.profile` → `config.json` | Optimizer, schedule, precision, batch/microbatch sizes and four seed streams             |
| Image pairs          | `protocol.dataset_for` → `dataset.py`                   | Manifest IDs/order, measurement/target paths, uint8 / 255 conversion                     |
| Learned mapping      | `protocol.make_model` → `models/` → `backbone.py`       | `forward`, `loss`, and `bounds`; q conditioning is in `film.py`                          |
| Weight updates       | `train_setup.py`: `run_training` → `training.py`: `train_one_epoch`     | Learning rate, loss, backward, accumulation and AdamW step                          |
| Saved training state | `checkpoints.save_training_progress`                    | `latest.pt` has optimizer/RNG state; milestone files retain model/data identity          |
| Interval fitting     | Root `python -m evaluation.run` → `calibration.fit_calibration`   | RCPS tail scales or QUTCC q inputs, fitted on calibration pairs only                     |
| Held-out scores      | `evaluation.evaluate_test_set` → `metrics.py`           | `report.json`, `per_image.csv`, optional `examples.npz`; raw/clipped views stay separate |

`run` and `config` are ordinary dictionaries of recorded settings. Resume checks
that those settings and split counts still match, then restores weights,
Adam moments and random-generator states. Checkpoint hashes identify loaded files.
The current code saves complete epochs; historical source/software guards and
mid-epoch recovery are in the [snapshot](../archive/research_prototype_20261102).

Input corruption and dataset-preparation code is maintained in
[ConvRML-Generalized-Specialists/input_modifications](https://github.com/Shdakudow/ConvRML-Generalized-Specialists/tree/main/input_modifications).
The baseline loads prepared pairs and does not depend on that package.

## Following the backbone

For a 256×256 measurement, [backbone.py](backbone.py) follows this path:

```text
RGB measurement: 3×256×256
  encoder input normalization
  ConvNeXt stages: 128×64×64, 256×32×32, 512×16×16, 1024×8×8
  1×1 projections + coarse-to-fine FPN merging: 256×64×64
  up1: 256×128×128       ← features() returns here
  up2: 256×256×256
  RGB head + original measurement: 3×256×256
```

Im2im requests three RGB heads at construction. Ensemble members add a variance
head. QUTCC modulates normalized features with q; MC dropout inserts stochastic
masks. Those additions are in [models/](../models). Encoder normalization and
decoder upsampling remain shared.
The sizes above illustrate the existing PLD data. All four default models accept
other RGB dimensions, pad internally to a multiple of 32 and crop their outputs
back; the data loader retains the original paired dimensions.

## Reading the data

Both loaders return float32 `(2, 3, H, W)`: measurement first, target second.
RGB uint8 values are divided by 255. Neither loader clips predictions, applies
augmentation, changes targets, or creates new train/test splits.

## Parallel Lensless Dataset (PLD): NumPy cache

```python
from dataset import NumpyPairs
train = NumpyPairs("pairs.npy", "manifest.json", "train")
calibration = NumpyPairs("calibration/pairs.npy", "calibration/manifest.json", "calibration")
```

The uint8 array shape is `(N, 2, 3, 256, 256)`. Normal cache order is train, val,
then test. The separate calibration cache has only calibration pairs. Manifests
retain the existing image IDs and order:

```json
{
  "train": [{ "id": "scene_a" }],
  "val": [{ "id": "scene_b" }],
  "test": [{ "id": "scene_c" }]
}
```

## DiffuserCam Lensless Mirflickr Dataset (DLMD): image-file manifest

```python
from dataset import ImagePairs
train = ImagePairs("split.json", "train")
```

```json
{
  "roles": {
    "train": [
      {
        "id": "scene_a",
        "measurement": { "path": "measurements/scene_a.png" },
        "target": { "path": "targets/scene_a.png" }
      }
    ],
    "val": [
      {
        "id": "scene_b",
        "measurement": { "path": "measurements/scene_b.png" },
        "target": { "path": "targets/scene_b.png" }
      }
    ]
  }
}
```

List the other roles (`test`, `calibration`) with the same row structure, or omit
them when not needed. Included splits must be nonempty with disjoint IDs.
Paths are relative to the manifest or absolute. Measurements and targets must
have matching RGB dimensions, normally 270×480. The model pads and crops;
the loader leaves dimensions intact. Prepared clean and fixed-corruption
manifests share this schema, with the original target paths/IDs retained.

[Portable commands](../archive/publication_support/docs/RUNNING_EXPERIMENTS.md) · [Recorded settings](../config.json)

Dataset setup checks nonempty, unique, disjoint image IDs but no longer computes
manifest or ID hashes. Resume checks split counts, not exact image membership or
ordering; keep the same split file when resuming. Published historical hashes
remain in the reproducibility records. Checkpoint download integrity checks are
separate and remain enabled.

The main `train_setup.py` coordinates setup, epochs, validation and saving.
Root `training.py` owns `train_one_epoch` and `train_batch`. Every model has
`loss(measurement, target, generator=None)`, so the shared loop calls it directly
without a method-specific loss wrapper. Only QUTCC consumes that generator;
MC dropout continues to use the usual PyTorch random stream for its masks.

## Current configuration

Edit the root [config.json](../config.json) for new runs. `datasets` selects the
loader and profile; `profiles` defines AdamW, batch sizes, precision, epochs and
learning-rate schedules. `seed_policy` controls random streams. `alpha` is the
interval miss target, and `calibration.delta` is the RCPS failure budget.
Training still saves the effective configuration beside each run's checkpoints.

Historical split IDs, counts, hashes, environment records and the pre-cleanup
settings are in [archive/reproducibility/](../archive/reproducibility/). They are
not required to configure a new run. No numerical training or calibration
setting changed in this reorganization.
