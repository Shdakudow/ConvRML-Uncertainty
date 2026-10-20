# Run training, calibration, inference, and learning curves

These commands document the current experiment without submitting or restarting existing jobs. Read the folder/task table before choosing an entry point. Commands marked **GPU** need a GPU allocation; a login node is only for inspection and Slurm submission.

## 1. Install and locate the project

```bash
git clone https://github.com/Shdakudow/ConvRML-Uncertainty.git
cd ConvRML-Uncertainty
python3.11 -m venv .venv
source .venv/bin/activate
# Install CUDA-compatible torch and torchvision for your machine first.
python -m pip install -r requirements.txt
```

The code repository is private; use your own authorized GitHub login. No credentials are included. PyTorch/torchvision are platform-specific and deliberately not pinned to an incompatible CUDA wheel in this source package. The cluster uses Python 3.11; LPIPS additionally downloads AlexNet weights into the Torch cache on first use.

Latest dataset loaders default to `$HOME/ConvRML`. Set `CONVRML_PROJECT` when reading existing datasets from another project root:

```bash
export CONVRML_PROJECT="$HOME/ConvRML"
```

That variable changes data/parent-checkpoint lookup, not every output path: latest trainers write beside their own source folder, while older baseline scripts use paths relative to the working directory. Follow each command's `cd` instruction. Saved manifests contain absolute source paths and are not portable merely by changing the environment variable.

## 2. Restore data and checkpoints separately

Normal-camera datasets contain `pairs.npy` and `manifest.json` at:

```text
$CONVRML_PROJECT/data/optics_intervals_60k_5k_5k_20260928/
$CONVRML_PROJECT/data/diffuser_intervals_60k_5k_5k_20260928/
$CONVRML_PROJECT/data/formal_rml_calibration95/
$CONVRML_PROJECT/data/diffuser_intervals_calibration_20260928/
```

`pairs.npy` is uint8 with shape `(N,2,3,H,W)`: measurement first, target second. A normal manifest has `train`, `val`, and `test` lists of records containing IDs. Its cache concatenates those splits in that order. Calibration has its own cache and `calibration` list. Loaders divide both images by 255.

Raw splits are listed in:

```text
$CONVRML_PROJECT/experiments/dlmd_raw_adaptive_size_20261016/data/split.json
```

This manifest has `roles.train`, `roles.val`, `roles.test`, and `roles.calibration`. Each record includes an ID and `measurement.path` / `target.path`. The corrupted loader uses the separate reviewed `raw_diffuser_corruptions_20261018/dataset/manifest.json`. It reads fixed corrupted measurements and unchanged targets; it does not generate new corruption each epoch.

The manifests/images are not bundled. The audit rejects missing pairs, split overlap, or wrong counts. Dataset SHA-256 identities are saved with checkpoints. Do not regenerate a split or replace targets to make an old checkpoint pass an audit.

Baseline checkpoints archived in [Releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases) are not downloaded by `git clone`. Restore only the needed epoch/member and verify its archived SHA-256. Current QUTCC/raw checkpoints remain separate cluster outputs. A parameter JSON documents settings; it is not a weight file or an automatically loaded configuration.

## 3. Latest QUTCC and clean raw Im2im training

From the repository root:

```bash
cd experiments/qutcc_paper_style_20261018
python test_protocol.py
```

| Task | Dataset | Model |
| ---: | --- | --- |
| 0 | RML | QUTCC |
| 1 | Normalized DiffuserCam | QUTCC |
| 2 | Raw DiffuserCam | QUTCC |
| 3 | Raw DiffuserCam | Im2im |

**GPU, fresh experiment only:** run a smoke test before the first real segment. The smoke pass checks gradients, validation, and runtime; its weights are not the trained checkpoint.

```bash
python train.py --task 0 --smoke
python train.py --task 0 --target-epoch 5
python train.py --task 0 --target-epoch 10
```

`--target-epoch 10` means **continue until total epoch 10**, not add ten more epochs. The same experiment resumes `latest.pt`, with optimizer moments, quantile RNG, CPU/CUDA RNG, offsets, and learning-rate position. It does not reset learning rate at each job boundary. The original schedule spans 50 epochs; effective batch is four via microbatch accumulation. Checkpoints are saved every five epochs.

| Output under `results/<camera>/<method>/` | Meaning |
| --- | --- |
| `config.json` | Actual hyperparameters and dataset/source identities |
| `smoke.json` | Passed smoke/runtime gate for this exact source |
| `latest.pt` | Resumable model + optimizer + RNG + partial-epoch progress |
| `checkpoints/epoch_005.pt` | Compact five-epoch model snapshot for evaluation; not a full optimizer resume |
| `history.json` | Training loss, validation metrics, LR, elapsed segment information |
| `progress.json` | Last reported training/validation progress |

Formatting the published files changes the source hash. Fresh formatted-source training is supported after its own smoke check. To continue historical training, restore [the exact source](../provenance/README.md) and the original full-state checkpoint/data. Do not delete the resume assertions or reuse a smoke report from different source.

### Slurm execution

Review environment paths, partition access, walltime, and GPU constraints in `run.sbatch`. The historical script excludes Monakhova and prefers shared fast GPUs. Later live jobs were explicitly moved into the authorized owner partition; that does not change this recorded script. A direct owner-partition override for **one authorized task** is:

```bash
mkdir -p logs
sbatch --partition=monakhova --constraint=a6000 --prefer= \
  --exclude=nikola-compute-15,bhattacharjee-compute-02 \
  --time=04:00:00 run.sbatch train.py --task 0 --target-epoch 5
```

**This submits a GPU job.** It overrides the historical script's partition/constraint/preference and removes its Monakhova exclusion. Our live scheduling policy permits at most two project GPUs in that owner partition. Do not copy this partition choice to an account without access.

`submit.py` and job JSON files preserve the original two-epoch schedule. `reschedule_five_epochs.sh` is a one-time historical migration, not a fresh generic launcher. For a new five-epoch chain, submit targets 5,10,…,50 with `afterok` dependencies and run each milestone's evaluation only after its training checkpoint exists. Never submit multiple training writers to the same `results/<camera>/<method>` concurrently.

## 4. Separate calibration and full-test inference

**GPU**, from the latest QUTCC folder after the checkpoint exists:

```bash
python evaluate.py --task 0 --epoch 5
python evaluate.py --task 0 --epoch 50
```

This freezes the selected checkpoint, fits interval parameters using **only calibration pairs**, then evaluates the complete test split. QUTCC fits lower/upper q; raw Im2im fits independent endpoint scales. It writes:

```text
results/<camera>/<method>/evaluation/epoch_050/
    calibration.json       # fitted q levels or endpoint scales
    report.json            # full-test metrics, per-image rows, provenance
    examples.npz           # fixed first 10 test image arrays
```

`report.json` separates `metrics.raw` and `metrics.clipped`. Do not interpret clipping as part of the calibration algorithm. Calibration can fail at an early checkpoint; check `failure.json` and the process log. A missing report is not a zero-valued result. Existing reports are reused rather than overwritten.

For example, inspect final clipped metrics without a GPU:

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('results/rml/qutcc/evaluation/epoch_050/report.json').read_text())
print(report['metrics']['clipped'])
print(report['calibration'])
PY
```

## 5. Im2im, ensemble, and MC baseline code

From the repository root:

```bash
cd experiments/adjusted_comparison_50epochs
python scripts/test_corrected_comparison.py
python scripts/corrected_comparison_study.py --help
python scripts/corrected_comparison_evaluate.py --help
```

This directory is self-contained for its model/script imports; data and results are external. Fresh training uses `corrected_comparison_study.py` and the guarded submission workflow in its [README](../experiments/adjusted_comparison_50epochs/README.md). Each ensemble member has independent initialization and shuffle seeds, but uses the same train/validation/test split as the other methods. Legacy `conditioned` and adaptive-selector modules remain available for historical checkpoints; current QUTCC is trained in the newer folder in step 3.

**GPU**, evaluate independently calibrated baseline curves at a saved milestone:

```bash
python scripts/evaluate_interval_curves.py --camera rml --method im2im --epoch 5
python scripts/evaluate_interval_curves.py --camera rml --method ensemble5 --epoch 5
python scripts/evaluate_interval_curves.py --camera rml --method mc_005 --epoch 5
```

Repeat at 10,15,…,50 and for `--camera diffuser`. These write `results/current_interval_curves_20261015/reports/<camera>/<method>/epoch_005.json`. Each checkpoint is calibrated separately; using epoch-50 scales for all earlier epochs would change the experiment. The legacy `adaptive_qutcc` option in this script is not current QUTCC; use the latest `evaluate.py` for that model.

For fixed-first-50 image exports with saved reconstructions/bounds:

```bash
python scripts/export_corrected_milestone.py --task 1 --epoch 50 --count 50
python scripts/export_corrected_milestone.py --task 3 --method mc_005 --epoch 50 --count 50
```

Task `1` is RML Im2im. Task `3` with the explicit override is RML MC `p=0.05`; tasks `5`/`7` are the corresponding DiffuserCam cases. Ensemble exports use `export_ensemble_size_comparison.py`: tasks 0/1 are RML five/ten members; 2/3 are DiffuserCam five/ten. These scripts **fit calibration again** on the calibration split; they do not train model weights. Their results are separate from the newer frozen-calibration analysis.

## 6. Raw and corrupted continuations

`raw_diffuser_continue80_20261019` continues clean raw epoch-50 full states. It reached epoch 70 and was stopped by request. Do not run its historical submission script to restart the canceled 75/80-epoch jobs.

`raw_diffuser_corruptions_20261018` builds the reviewed fixed corruption dataset. `build_dataset.py --preview-only` prepares a review; full materialization must use the same approved protocol/manifest. Its ten-epoch pilot starts from clean **epoch-30 model weights**, with a new adaptation optimizer. The later `raw_corrupted_continue30_20261019` retains that pilot's **full optimizer/RNG state** through adaptation epoch 30.

From the corrupted continuation directory, task 0 is QUTCC and task 1 is Im2im:

```bash
python test_continuation.py
# GPU; evaluate existing completed checkpoints without training:
python evaluate.py --task 0 --epoch 30
python evaluate.py --task 1 --epoch 30
```

The final checkpoint is 30 clean pretraining epochs + 30 corruption-adaptation epochs. Parent checkpoints, reviewed manifests and their exact identities are required even when inspecting this source snapshot. Training is complete; `submit.py` is a historical launcher, not a progress-check command.

## 7. Learning curves: reconstruction, coverage, and width

Collect already evaluated reports into CSV/PNG without loading images or retraining:

```bash
cd /path/to/ConvRML-Uncertainty
python tools/plot_learning_curves.py \
  --series QUTCC=/path/to/qutcc/results/rml/qutcc/evaluation \
  --series Im2im=/path/to/current_interval_curves/reports/rml/im2im \
  --metric-space clipped --output /path/to/curve_export
```

Add ensemble/MC series similarly. For corrupted raw, point both series at their `results/raw/<method>/evaluation` directories. Outputs are `learning_curves.csv` and `learning_curves.png`, with SSIM, PSNR, mean width and actual coverage. The script uses only completed reports, distinguishes raw/clipped data, and leaves missing five-epoch milestones as gaps. Labels/dataset choice are supplied by the caller: never compare different cameras or calibration protocols as though they were the same experiment.

Full-test LPIPS and width/error Pearson/Spearman use `experiment2_uncertainty_analysis_20261019/evaluate_full_test.py --epoch 5` and subsequent five-epoch milestones. The evaluator expects the original `$HOME/ConvRML` project layout, exact checkpoint reports, and baseline weights under its temporary `weights/` tree. The Mac-only `milestone_pipeline.py` restores archived baseline checkpoints with hash checks and submits eight camera/model evaluations per milestone. It needs the existing release manifests, `gh` authentication, and SSH access; it is not a standalone dataset downloader.

`publish_curves.py` is the Mac/demo integration helper, not model training. It requires the separate demo checkout and local asset directories documented in the analysis README. The current website, raw exports, and code publication remain separate processes.

## 8. Validate a source-only change

```bash
python tools/verify_source_release.py
python -m unittest discover -s tests -v
python experiments/experiment2_uncertainty_analysis_20261019/test_metrics.py
# Development-only formatter:
python -m pip install -r requirements-dev.txt
python -m black --check experiments/adjusted_comparison_50epochs \
  experiments/qutcc_paper_style_20261018 \
  experiments/raw_diffuser_continue80_20261019 \
  experiments/raw_diffuser_corruptions_20261018 \
  experiments/raw_corrupted_continue30_20261019 \
  experiments/raw_diffuser_demo_20261019 \
  experiments/experiment2_uncertainty_analysis_20261019 tools tests
```

The source verifier checks archive hashes, Python syntax, and AST equivalence against the exact captured cluster source. This validates formatting equivalence; it does not replace an end-to-end CUDA/data smoke test. This publication does not rerun training or alter live experiment/checkpoint directories.
