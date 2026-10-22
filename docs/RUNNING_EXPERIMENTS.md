# Training, calibration, inference and curves

Use [Exp2](../experiments/experiment2) for the four-model normalized-camera comparison and [Exp3](../experiments/experiment3) for raw/corrupted QUTCC and Im2im. [Model differences](MODEL_COMPARISON.md) identifies the implementation behind each command.

Identical implementations are stored once in [shared/](../shared) and reached
through relative links at the documented experiment paths. Preserve Git symlinks
when copying the checkout. Run commands through those experiment entry points,
which retain separate outputs/source guards; do not invoke shared engines directly.

## Setup and existing data

```bash
git clone https://github.com/Shdakudow/ConvRML-Uncertainty.git
cd ConvRML-Uncertainty
python3.11 -m venv .venv
source .venv/bin/activate
# Install CUDA-compatible torch/torchvision for your system first.
python -m pip install -r requirements.txt
export CONVRML_PROJECT="$HOME/ConvRML"
```

The source repository is private. Use your authorized GitHub login. `CONVRML_PROJECT` points to the existing data/checkpoint project, which is separate from this reorganized source checkout. Historical source contains absolute or project-relative cluster paths; the repository moves do not move those external files.

| Data | Historical location under `$CONVRML_PROJECT` |
| --- | --- |
| Exp2 RML pairs/manifest | `data/optics_intervals_60k_5k_5k_20260928/` |
| Exp2 normalized DiffuserCam pairs/manifest | `data/diffuser_intervals_60k_5k_5k_20260928/` |
| RML calibration | `data/formal_rml_calibration95/` |
| Normalized DiffuserCam calibration | `data/diffuser_intervals_calibration_20260928/` |
| Raw paired split | `experiments/dlmd_raw_adaptive_size_20261016/data/split.json` |
| Fixed corrupted split | `experiments/raw_diffuser_corruptions_20261018/dataset/manifest.json` |

Normal caches contain uint8 `(N,2,3,H,W)` pairs, measurement first and target second; loaders divide by 255. Raw manifests reference paired image paths. Data, split files and weight binaries are not included in a clone. Restore needed weights from [Releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases) or their existing cluster outputs and verify recorded hashes.

Validation selects settings, calibration fits intervals, and test evaluates the frozen result. Do not substitute test pairs into calibration or regenerate split IDs for an existing checkpoint.

## Experiment 2

From the repository root:

```bash
cd experiments/experiment2
python run_model.py --help
# Inspect routing without importing models, training or submitting a job:
python run_model.py train --method qutcc --camera rml --epoch 5 --dry-run
python run_model.py evaluate --method ensemble --camera diffuser --epoch 50 --dry-run
```

**Inside a GPU allocation**, fresh QUTCC training and evaluation:

```bash
python run_model.py smoke --method qutcc --camera rml
python run_model.py train --method qutcc --camera rml --epoch 5
python run_model.py train --method qutcc --camera rml --epoch 10
python run_model.py evaluate --method qutcc --camera rml --epoch 10
```

`--epoch` is the absolute target, not extra epochs. The current QUTCC router selects `qutcc/train.py` and its own calibration evaluator, not the historical adaptive-selector arm. It writes `qutcc/results/<camera>/qutcc/` beside the source.

The preserved Im2im, MC and ensemble trainers advance at most **two new epochs per call**. For example:

```bash
python run_model.py smoke --method im2im --camera rml
python run_model.py train --method im2im --camera rml --epoch 2
python run_model.py train --method im2im --camera rml --epoch 4
python run_model.py train --method im2im --camera rml --epoch 5
python run_model.py evaluate --method im2im --camera rml --epoch 5

python run_model.py smoke --method mc_dropout --camera diffuser --dropout 0.05
python run_model.py train --method mc_dropout --camera diffuser --epoch 2

python run_model.py smoke --method ensemble --camera rml --member 0
python run_model.py train --method ensemble --camera rml --member 0 --epoch 2
# Train independent chains for members 0..4, or 0..9 for the ten-member reference.
python run_model.py evaluate --method ensemble --members 5 --camera rml --epoch 50
```

Each ensemble member sees the complete training split. Baseline data/results are resolved from `--project-root` (default `CONVRML_PROJECT`); baseline checkpoints remain under `results/corrected_comparison_20261006/<camera>/<arm>/`. Interval reports go to `results/current_interval_curves_20261015/reports/<camera>/<method>/epoch_NNN.json`.

Evaluate saved milestones 5,10,…,50 independently. Each calibration uses the separate calibration split for that checkpoint. Im2im/ensemble/MC use asymmetric RCPS-HB; QUTCC uses Algorithm 1 quantile search. An epoch-50 calibration must not silently replace an earlier checkpoint's calibration.

See [the four-model README](../experiments/experiment2/README.md) for the full router options and export scripts. The saved dropout sweep compares p=0.05/0.10/0.20 at 30/60/100 passes; main demo inference uses p=0.05 and 30 passes.

## Experiment 3

From the repository root, **inside a GPU allocation**:

```bash
cd experiments/experiment3/clean
python test_protocol.py
python train.py --task 2 --smoke
python train.py --task 2 --target-epoch 5
python train.py --task 2 --target-epoch 10
python evaluate.py --task 2 --epoch 10
```

Use **task 2 for raw QUTCC**, **task 3 for raw Im2im**. The same preserved source also contains historical normalized-camera tasks 0/1; the Exp2 router above is the clearer normalized-camera entry point.

Both clean raw models train for 50 epochs using float32 Adam, effective batch four, and the recorded warmup/cosine schedule. QUTCC learns with sampled-q pinball only; Im2im learns center MSE plus two endpoint pinball losses. The raw wrapper pads/crops internally. Outputs go beside the source under `results/raw/<method>/`.

### Fixed corruption adaptation

Review [corrupted/](../experiments/experiment3/corrupted) for dataset generation. The historical loader reads the existing reviewed manifest from the project paths in the table above. Ground truth remains unchanged; one fixed operation is assigned per measurement. Preview controls include additional stress tests that are not all in the training mixture.

The corrupted pilot loads each clean epoch-30 checkpoint, starts a fresh adaptation optimizer and trains for ten epochs. A `TRAINING_APPROVED` marker gates that preserved pilot trainer; review/approval of a newly generated dataset is separate from historical authorization.

The [continuation](../experiments/experiment3/corrupted_continuation) retains model, optimizer and RNG state through adaptation epoch 30. Its task IDs are **0 QUTCC / 1 Im2im**:

```bash
# From experiments/experiment3/corrupted_continuation, GPU required for evaluation:
python test_continuation.py
python evaluate.py --task 0 --epoch 30
python evaluate.py --task 1 --epoch 30
```

Final corrupted checkpoints are 30 clean pretraining + 30 adaptation epochs. The stopped clean 50→70 follow-up is preserved in the [history bundle](../archive/README.md).

## Checkpoint and report files

| File | Meaning |
| --- | --- |
| `config.json` / `protocol.json` | Hyperparameters and dataset/source identity; not model weights |
| `latest.pt` | Full-state resumable checkpoint, including optimizer/RNG/progress |
| `checkpoints/epoch_005.pt` | Compact milestone weights/history for frozen evaluation |
| `history.json` | Training and validation history |
| `evaluation/epoch_NNN/calibration.json` | Fitted quantile levels or endpoint scales |
| `evaluation/epoch_NNN/report.json` | Full-test aggregates, per-image metrics and calibration provenance |
| `evaluation/epoch_NNN/examples.npz` | Saved measurement, target, prediction and endpoint arrays |

Formatting/publication and historical resume have different source byte hashes. [The original archive](../archive/provenance/README.md) and migration map preserve that distinction. Resume checks must retain the original source/config/data identities. Fresh formatted-source training needs its own smoke report; a historical smoke report is not transferable.

Training commands above execute within an allocation; they do not submit Slurm jobs. Historical cluster wrappers and submission records are in the [archive](../archive/README.md). Review their paths and resources before reuse. Never run concurrent training writers to the same method/output directory.

## Inference and learning curves

`evaluate.py` freezes a checkpoint, calibrates on the separate calibration split, then evaluates the test split. Existing complete reports may be reused. A failed early calibration is recorded in `failure.json`; it is not a zero-valued metric.

QUTCC/raw reports distinguish raw and clipped metrics. Calibration changes q or interval scales; display clipping to [0,1] is a separate operation. See [metric definitions](METHODS_AND_METRICS.md).

Collect completed five-epoch reports without retraining:

```bash
# From the repository root; replace paths with your completed evaluation folders.
python tools/plot_learning_curves.py \
  --series QUTCC=/path/to/qutcc/evaluation \
  --series Im2im=/path/to/im2im/reports \
  --metric-space clipped --output /path/to/curves
```

Add ensemble/MC series for Exp2. Outputs are CSV and a PNG containing PSNR, SSIM, width and actual coverage. Missing evaluated milestones remain gaps. Compare methods within the same camera/split and metric space.

[Exp2 analysis](../experiments/experiment2/analysis) adds LPIPS and within-image pixel-wise width/error Pearson/Spearman at five-epoch milestones. Those scripts retain their original external project/checkpoint lookup paths. [Exports](../tools/export) contains frozen-checkpoint image packing helpers for the separate demo repository; source publication alone does not deploy a website.

## Source checks

```bash
python tools/verify_source_release.py
python -m unittest discover -s tests -v
python experiments/experiment2/analysis/test_metrics.py
```

The verifier resolves original source paths through [the migration map](../archive/provenance/layout_migration_20261021.json), checks original archive hashes and compares published Python ASTs. It confirms model behavior was preserved by this organization; it does not replace a full CUDA/data smoke test.
