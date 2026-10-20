# Four-model ConvRML comparison

This folder contains the **current QUTCC, end-to-end Im2im, deep ensemble, and MC dropout implementations**. Start with `run_model.py`, which selects the correct training and calibration code for each method.

[Complete workflow](../../docs/RUNNING_EXPERIMENTS.md) · [losses and metrics](../../docs/METHODS_AND_METRICS.md) · [current parameters](configs/four_model_comparison.json) · [historical protocol notes](LEGACY_PROTOCOL.md)

## Models in this folder

| Method | Implementation | Training | Calibration |
| --- | --- | --- | --- |
| **QUTCC** | [`qutcc/model.py`](qutcc/model.py), [`qutcc/train.py`](qutcc/train.py) | Current original-style quantile embedding; 48 FiLM sites; one independent uniform q per image; pinball only | [`qutcc/calibration.py`](qutcc/calibration.py): separate Algorithm 1 search for global lower/upper q; no RCPS scale factors |
| **Im2im** | [`models/corrected_uncertainty.py`](models/corrected_uncertainty.py): `EndToEndIm2Im` | Entire backbone and RGB lower/center/upper head trained jointly; center MSE + endpoint pinball | Independent lower/upper RCPS-HB scales |
| **Deep ensemble** | [`baselines/deep_ensemble.py`](baselines/deep_ensemble.py), [`scripts/corrected_comparison_study.py`](scripts/corrected_comparison_study.py) | Gaussian NLL; separate initialization and shuffle sequence per member | Independent lower/upper RCPS-HB scales; calibrate five- and ten-member aggregates separately |
| **MC dropout** | [`models/corrected_uncertainty.py`](models/corrected_uncertainty.py): `FullDropoutRML` | MSE; dropout at 36 encoder and 12 decoder sites | Independent lower/upper RCPS-HB scales; current default p=0.05, 30 stochastic passes |

All use a ConvNeXtV2-Base/FPN reconstruction architecture. QUTCC retains its trained protocol: float32, Adam, effective batch four, 30% warmup followed by cosine decay. The three baseline trainers retain BF16, AdamW and batch two. This integration consolidates source and commands; it does not claim the optimizer/precision settings are identical or retrain completed models.

For each normal camera: 60,000 training, 5,000 validation, 5,000 test and 2,048 separate calibration pairs. Current QUTCC reconstructs at q=0.5; calibration selects its endpoint queries. QUTCC does not use the historical adaptive endpoint selector.

## Setup and running

Use Python 3.11, CUDA-compatible PyTorch/torchvision, and the repository-root requirements. Run the commands below **from this folder**. `--project-root` identifies existing data and baseline results; it defaults to `CONVRML_PROJECT`, otherwise `$HOME/ConvRML`.

```bash
python run_model.py --help
# Inspect commands first; these do not import models or run a GPU job:
python run_model.py train --method qutcc --camera rml --epoch 5 --dry-run
python run_model.py evaluate --method ensemble --camera diffuser --epoch 50 --dry-run
```

**GPU:** QUTCC smoke test and five-epoch training segments:

```bash
python run_model.py smoke --method qutcc --camera rml
python run_model.py train --method qutcc --camera rml --epoch 5
python run_model.py train --method qutcc --camera rml --epoch 10
python run_model.py evaluate --method qutcc --camera rml --epoch 10
```

For Im2im and MC, the existing trainer advances at most **two new epochs per call**. `--epoch` is the absolute target, not the number to add:

```bash
python run_model.py smoke --method im2im --camera rml
python run_model.py train --method im2im --camera rml --epoch 2
python run_model.py train --method im2im --camera rml --epoch 4
python run_model.py train --method im2im --camera rml --epoch 5
python run_model.py evaluate --method im2im --camera rml --epoch 5

python run_model.py smoke --method mc_dropout --camera diffuser --dropout 0.05
python run_model.py train --method mc_dropout --camera diffuser --dropout 0.05 --epoch 2
```

An ensemble trains each member independently on the complete training split:

```bash
python run_model.py smoke --method ensemble --member 0 --camera rml
python run_model.py train --method ensemble --member 0 --camera rml --epoch 2
# Repeat independent chains for members 0..4 (or 0..9 for the ten-member reference).
python run_model.py evaluate --method ensemble --members 5 --camera rml --epoch 50
```

Evaluation needs every selected member's requested checkpoint. Repeat evaluation at saved epochs **5,10,…,50** and for both cameras. Smoke tests do not save trained model weights. Train calls run immediately inside an existing GPU allocation; `run_model.py` does not submit Slurm jobs. Review the recorded `.sbatch` files if scheduling remotely, and never run two training writers for the same model/output folder.

## Where data, checkpoints, and results go

| Method | Dataset lookup | Output / checkpoint lookup |
| --- | --- | --- |
| Current QUTCC | `--project-root/data/…`, using `qutcc/data.py` | This folder's `qutcc/results/<camera>/qutcc/` |
| Other three methods | `--project-root/data/…` | `--project-root/results/corrected_comparison_20261006/<camera>/<arm>/` |
| Baseline interval evaluation | Same independent calibration and test caches | `--project-root/results/current_interval_curves_20261015/reports/<camera>/<method>/epoch_NNN.json` |

QUTCC evaluation writes `qutcc/results/<camera>/qutcc/evaluation/epoch_NNN/{calibration.json,report.json,examples.npz}`. Restore the desired compact checkpoint under `qutcc/results/<camera>/qutcc/checkpoints/epoch_NNN.pt` for frozen inference. Existing historical result/weight directories are not automatically moved or overwritten by this integration.

The included QUTCC files are exact copies of the current formatted [source snapshot](../qutcc_paper_style_20261018/); [provenance.json](qutcc/provenance.json) lists their hashes. Saved state-dictionary keys are unchanged. Historical full training resume still requires matching source/config/data identities; do not weaken those checks. Fresh training uses a new smoke report for this directory.

## Curves, width, and clipping

Use [`tools/plot_learning_curves.py`](../../tools/plot_learning_curves.py) with the QUTCC evaluation directory and each baseline report directory to export SSIM, PSNR, actual coverage and mean width at evaluated epochs. The [workflow guide](../../docs/RUNNING_EXPERIMENTS.md#7-learning-curves-reconstruction-coverage-and-width) has the command.

Uncertainty is `upper - lower`. Calibration is fitted on separate data, never test targets. Ground truth is normalized to [0,1]. Calibrated bounds remain unconstrained until explicit clipping; QUTCC reports raw and clipped metrics, and the demo uses clipped metrics. Alpha=0.10 targets coverage, not exactly 90% on every image.

## Historical compatibility

`models.corrected_uncertainty.PinballQUTCC`, the legacy `conditioned` arm, adaptive-selector dependencies and historical scheduling scripts remain for old checkpoints. They are **not** the current QUTCC path. `run_model.py --method qutcc` always selects `qutcc/train.py` / `qutcc/evaluate.py`. Historical parameter JSON files and [legacy notes](LEGACY_PROTOCOL.md) document earlier protocols; use [four_model_comparison.json](configs/four_model_comparison.json) for this consolidated layout.

## Checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

From this folder with the ML environment installed:

```bash
python qutcc/test_protocol.py
python -m unittest scripts.test_corrected_comparison scripts.test_bayesian_baselines scripts.test_rcps_bounds
```

These cover current-QUTCC command routing/source identity, pinball training, optimizer/RNG resume, separate ensemble seeds, end-to-end head gradients, dropout behavior and calibration budgets. They do not rerun the full GPU benchmark.
