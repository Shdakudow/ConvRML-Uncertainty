# Adjusted comparison: MC dropout, QUTCC, and ensemble

**How well can a lensless reconstruction model recover an image—and tell us where its reconstruction is uncertain?**

This repository compares four approaches on **RML** and **DiffuserCam** measurements. Each method produces a reconstructed RGB image and a lower/upper interval for every pixel channel. We compare image quality, interval coverage, interval width, and computational cost.

[Experiment settings](configs/parameters.json) · [Training code](scripts/selective_film50_study.py) · [Evaluation code](scripts/selective_film50_evaluate.py) · [Interactive demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)

> **Current experiment:** a 50-epoch comparison with FiLM conditioning in QUTCC, ten ensemble members, and validation-based dropout selection. This folder contains the code and configuration, **not datasets or trained weights**. The separate demo shows earlier completed experiments; it does not establish results for this new comparison.

## What do the four models do?

All four use a **ConvNeXtV2-Base encoder with a feature pyramid decoder**. They differ in how they learn or estimate uncertainty.

| Method | Reconstruction and uncertainty | Training loss | Current experiment |
| --- | --- | --- | --- |
| **Im2im / fixed head** | A reconstruction backbone predicts the image. A separate, fixed-quantile head predicts lower and upper bounds from its frozen features. | MSE for the backbone; endpoint pinball loss for the uncertainty head. | Reuse the existing 50-epoch backbone and its trained 10-epoch head. **No retraining.** |
| **Deep ensemble** | Independently trained models each predict a mean and variance. Their predictions are combined into a mean and total variance, then converted to Gaussian intervals. | Gaussian negative log-likelihood (NLL). | Keep the original five members; finish five additional members to epoch 50. Compare five versus ten members. |
| **MC dropout** | Keep dropout active during repeated inference passes. Their mean gives the reconstruction; their variation gives an uncertainty estimate. | MSE. | Train dropout variants to epoch 50, then select the dropout probability using validation data. |
| **Q-conditioned ConvRML (QUTCC)** | A reconstruction branch predicts the image. A quantile-conditioned branch predicts an intensity at a requested quantile, such as `q = 0.05` or `q = 0.95`. | Reconstruction MSE + pinball loss, with equal coefficients. | Train the revised FiLM-conditioned architecture from scratch to epoch 50. |

**FiLM is used only in QUTCC.** It turns the requested quantile into learned scale and shift values that modify normalized features. There are **48 conditioning sites: 36 in the encoder and 12 in the decoder**. The im2im, ensemble, and MC-dropout models do not receive these FiLM layers.

The current comparison uses the losses listed above. Historical CRPS and toy-experiment scripts are in [the exploratory archive](../../archive/exploratory/).

## Experiment at a glance

The same split sizes and preprocessing are used for each method within each camera dataset.

| Setting | Value |
| --- | --- |
| Datasets | RML and DiffuserCam, trained separately |
| Image size | RGB, `256 × 256`, intensities scaled to `[0, 1]` |
| Training / validation / test | 60,000 / 5,000 / 5,000 images per dataset |
| Separate calibration set | 2,048 images per dataset |
| Backbone training target | **50 total epochs**, including any completed pilot epochs |
| Optimizer | AdamW; learning rate `1e-4`; weight decay `1e-4` |
| Batch size / precision | 2 / BF16 autocast |
| Saved checkpoints | Every 5 epochs; resumable state saved after every epoch |
| Cluster scheduling | At most 6 GPUs concurrently; training advances at most 2 epochs per job |
| MC-dropout candidates | Dropout probabilities `0.05`, `0.10`, and `0.20` |
| MC validation passes | 30, 60, and 100; choose the dropout probability by validation SSIM at 100 passes |
| QUTCC training quantiles | One uniformly sampled `q` in `(0, 1)` per image |
| Native quantile endpoints | `q = 0.05` and `q = 0.95` |
| Calibration target | RCPS: `alpha = 0.10`, `delta = 0.05` |

See [the full parameter file](configs/parameters.json) for architecture details, seeds, and resume settings. **This JSON documents the settings; the scripts do not load it as a configuration input.** Editing the JSON alone does not change training.

## How to read the outputs

For a ground-truth image `x`, reconstructed image `prediction`, and calibrated bounds `LB` and `UB`:

```text
Absolute reconstruction error = |x - prediction|
Uncertainty width             = UB - LB
Coverage                      = fraction of ground-truth channel values inside [LB, UB]
```

- **PSNR and SSIM:** higher values indicate better reconstruction quality.
- **Mean absolute error:** lower values indicate smaller reconstruction errors.
- **Mean interval width:** measures how broad the uncertainty intervals are. Smaller is preferable **when coverage is adequate**.
- **Coverage:** checks whether the intervals contain the observed ground truth. A wide interval can achieve high coverage without being informative.

Error and uncertainty maps answer different questions: error measures the mistake on a known test image, while uncertainty expresses the model's uncertainty before seeing its ground truth. They need not match pixel by pixel.

The evaluator reports clipped image-quality and interval metrics on `[0, 1]`, with an additional unclipped PSNR. Width is measured in these normalized intensity units.

### What does calibration change?

Calibration adjusts interval widths using a held-out calibration set; it does not retrain the network or change its reconstructed image. The evaluator compares empirical calibration with Hoeffding–Bentkus RCPS calibration.

RCPS targets an expected uncovered fraction of at most `0.10`, with confidence controlled by `delta = 0.05` under the calibration assumptions. An image is the calibration unit, and its loss is its uncovered RGB fraction. **This does not require exactly 90% coverage on every image or on a finite test set.**

For QUTCC, this evaluator keeps the native `0.05` and `0.95` quantile queries and scales the resulting intervals around the reconstruction. It does not search for new calibrated quantile values.

## Where to start in the code

| File | Purpose |
| --- | --- |
| [configs/parameters.json](configs/parameters.json) | Full settings for the current comparison |
| [models/shamus_rml.py](models/shamus_rml.py) | Shared reconstruction backbone and decoder |
| [models/formal_quantile.py](models/formal_quantile.py) | Fixed endpoint head and earlier quantile-conditioned model |
| [models/convnext_norm_film.py](models/convnext_norm_film.py) | Current QUTCC model with FiLM throughout the encoder and decoder |
| [baselines/formal_rml.py](baselines/formal_rml.py) | Ensemble and dropout backbone variants |
| [scripts/selective_film50_study.py](scripts/selective_film50_study.py) | Training, validation, checkpointing, and model loading |
| [scripts/submit_selective_film50.py](scripts/submit_selective_film50.py) | Submit the dependent Slurm workflow |
| [scripts/selective_film50_evaluate.py](scripts/selective_film50_evaluate.py) | Dropout selection, calibration, and held-out evaluation |
| [scripts/rcps_bounds.py](scripts/rcps_bounds.py) | RCPS interval calibration |

The included `film10_study.py` is a model-construction dependency that preserves the pilot initialization. It is not an additional run to submit. Older experiment entry points are in [the archive](../../archive/exploratory/).

## Setup and running

### 1. Set up the environment

Use Python 3.11 and a PyTorch/torchvision installation compatible with your CUDA environment. Then install the remaining dependencies from this experiment directory:

```bash
python -m pip install -r requirements.txt
```

The training and evaluation scripts require an NVIDIA GPU. You can check the FiLM implementation without submitting cluster jobs:

```bash
python scripts/test_convnext_norm_film.py
```

### 2. Provide the data and checkpoints

All `data/`, `results/`, and `logs/` paths are relative to **this folder**. To reuse the existing cluster experiment, link its data and result directories here or keep using the original cluster checkout. Moving the GitHub source does not relocate those files.

The experiment expects paired measurement/ground-truth caches (`pairs.npy`) and split manifests (`manifest.json`) at these locations:

```text
data/
├── optics_intervals_60k_5k_5k_20260928/       # RML train / validation / test
├── diffuser_intervals_60k_5k_5k_20260928/     # DiffuserCam train / validation / test
├── formal_rml_calibration95/                # RML calibration
└── diffuser_intervals_calibration_20260928/  # DiffuserCam calibration
```

Each paired image cache contains unsigned-byte arrays with shape `[N, 2, 3, 256, 256]`: measurement first, ground truth second. Preserve the original manifests so every method uses the same split.

**The submitted workflow resumes an existing experiment.** It also requires the original im2im and five-member ensemble checkpoints, plus the additional ensemble and MC-dropout pilot states. These pilot states resume at epoch 8 for RML and epoch 6 for DiffuserCam, including optimizer and random-number states. Their destination is:

```text
results/selective_film50_both_cameras_20261005/<camera>/<arm>/latest.pt
```

Here, `<camera>` is `rml` or `diffuser`; resumable arms are `ensemble_05` through `ensemble_09`, `mc_005`, `mc_010`, and `mc_020`. The new `conditioned` arm starts fresh. Original baseline checkpoint paths are specified in `load_model()` in the training script.

A fresh clone contains neither these weights nor the data, so it cannot reproduce the resume workflow by itself.

### 3. Adapt the cluster template and submit

Review [the Slurm template](scripts/selective_film50_study.sbatch), including its working directory, Python environment, GPU constraints, partition, and excluded nodes. The templates run from `SLURM_SUBMIT_DIR`, so submit from this experiment directory. The Python environment and GPU settings remain specific to Unicorn.

Once the required files and cluster settings are ready:

```bash
mkdir -p logs
python scripts/submit_selective_film50.py
```

This command allocates cluster jobs. The workflow runs:

```text
GPU smoke tests
    → training in two-epoch stages
    → MC-dropout validation sweep
    → lock the selected dropout probability
    → calibration and held-out test evaluation
```

The submission script records job IDs in `jobs.json` and refuses to submit again if that record already exists.

### 4. Find the results

Outputs are written under:

```text
results/selective_film50_both_cameras_20261005/
├── jobs.json
├── rml/
│   ├── <arm>/                   # Checkpoints, training history, and configuration
│   ├── dropout_validation/      # Validation metrics and trend plots
│   ├── dropout_policy.json      # Selected dropout setting
│   └── evaluation/<method>/     # Aggregate metrics and per-image CSVs
└── diffuser/                    # Same structure
```

Evaluation includes the fixed head, FiLM QUTCC, five- and ten-member ensembles, the original 30-pass MC baseline, and the validation-selected 100-pass MC model.

## Related repositories and attribution

- **This code:** [Shdakudow/ConvRML-Uncertainty](https://github.com/Shdakudow/ConvRML-Uncertainty)
- **Demo code:** [Shdakudow/ConvRML-Uncertainty-Demo](https://github.com/Shdakudow/ConvRML-Uncertainty-Demo)
- **Interactive demo:** [Open the image comparison viewer](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)
- **Upstream ConvRML:** [lakabuli/ConvRML](https://github.com/lakabuli/ConvRML)
- **Original QUTCC source used here:** [Shdakudow/code-for-quant](https://github.com/Shdakudow/code-for-quant) (private)

Upstream attribution is retained in [LICENSE](../../LICENSE), [CITATION.cff](../../CITATION.cff), and [UPSTREAM_README.md](../../archive/exploratory/UPSTREAM_README.md). The imported QUTCC source is recorded in [its provenance file](models/qutcc_original/provenance.json).

The demo is maintained separately. Updating this repository does not deploy the website.
