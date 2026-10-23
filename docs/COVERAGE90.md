# Coverage close to 90%

`tools/evaluate_coverage90.py` recalibrates **existing frozen checkpoints**, without
retraining. Its target is **90.0–90.1% calibration coverage** (a 0.1 percentage point
numerical tolerance). This is an empirical matched-coverage comparison, **not an
RCPS certificate**. Reference calibration/results remain unchanged.

## Read the code in this order

| File | Responsibility |
| --- | --- |
| [`shared/calibration/coverage90.py`](../shared/calibration/coverage90.py) | Fit independent endpoint scales or search global q values |
| [`tools/evaluate_coverage90.py`](../tools/evaluate_coverage90.py) | Load a frozen checkpoint, fit on calibration, then measure the full test set |
| [`tools/run_coverage90_milestone.py`](../tools/run_coverage90_milestone.py) | Run one task; keep calibration failures separate from execution errors |
| [`tools/dispatch_coverage90_milestones.py`](../tools/dispatch_coverage90_milestones.py) | Restore verified weights and advance the two evaluation streams |
| [`tools/coverage90_cluster.py`](../tools/coverage90_cluster.py) | Readable cluster commands, report checks, submission receipts and safe cleanup |

For the backbone, output heads and losses, start with the separate
[model comparison](MODEL_COMPARISON.md). Scheduling and publishing do not alter
model weights, losses or calibration mathematics.

- Im2im, deep ensemble and MC dropout: independent lower and upper scales are
  selected by calibration score order statistics, with 5% misses per tail. This
  avoids a coarse scale grid and RCPS confidence margins. A few floating-point
  ULPs protect selected boundary values during endpoint multiplication.
- QUTCC: refine the two **global quantile inputs**, retaining native quantile
  images. Start from the existing calibrated pair when available. Each tail is
  measured on the full calibration split and refined toward 5% misses. No extra
  interval scale or prediction head is introduced. Observed nonmonotone q responses
  fail explicitly; the method does not assume a guaranteed monotone network.
- Calibration uses **clipped [0,1] endpoints**, matching displayed metrics. Raw
  and clipped test metrics are both reported. Test targets never select parameters.
- Quantized/tied outputs can make the requested tolerance unattainable. Every
  report records actual calibration coverage and `within_tolerance`; a failed
  tolerance is never reported as exactly 90%. Test coverage is allowed to vary.

## Experiment 2

Run from the repository using the model environment and existing dataset/checkpoints:

```bash
python experiments/experiment2/run_model.py evaluate \
  --method im2im --camera rml --epoch 50 \
  --project-root /home/jy2228/ConvRML --calibration tight90

# Original cluster QUTCC result location, if not copied beside the published code:
python experiments/experiment2/run_model.py evaluate \
  --method qutcc --camera diffuser --epoch 50 \
  --project-root /home/jy2228/ConvRML --calibration tight90 \
  --calibration-source /home/jy2228/ConvRML/experiments/qutcc_paper_style_20261018
```

The baseline router defaults to the published Exp2 source folder and reads weights
under `PROJECT/results/corrected_comparison_20261006`. If those weights were moved
to a GitHub Release, restore the exact milestone first; the evaluator does not
silently substitute another model or epoch. Use `--method ensemble --members 5`
(or 10), and `--method mc_dropout --dropout 0.05`, for the other baselines.
The standalone evaluator also accepts `--weights-root` for an existing restored
baseline checkpoint tree (camera/arm/checkpoints/epoch_XXX.pt).

## Experiment 3

Select the clean or corrupted source/results folder explicitly:

```bash
python tools/evaluate_coverage90.py \
  --project-root /home/jy2228/ConvRML \
  --source-folder /home/jy2228/ConvRML/experiments/raw_corrupted_continue30_20261019 \
  --family paper --camera raw --method qutcc --epoch 30 \
  --output /home/jy2228/ConvRML/results/tight_coverage90/corrupted/qutcc/epoch_030
```

For clean raw use `qutcc_paper_style_20261018`, epoch 50. Replace `qutcc` with
`im2im` to evaluate its independent endpoint scales. Use separate output folders.

The evaluator saves `calibration.json` **before opening the test dataset**, then
`report.json` with calibration coverage, held-out test coverage/width/SSIM/PSNR/MAE,
per-image metrics, split identity, source provenance and GPU timing. `examples.npz`
contains the first 50 fixed test examples with clipped reconstruction and bounds.
Temporary calibration scores are deleted after fitting. Existing reports are
reused only for identical requests. No demo is changed automatically by this tool.

## Every five epochs

The milestone run evaluates epochs **5, 10, …, 50** for Exp2 and clean raw Exp3.
Corrupted raw has only **5, 10, …, 30 adaptation epochs**; no 35–50 checkpoints are
invented. This makes 112 model/camera/checkpoint evaluations, including the twelve
latest evaluations already completed. The same calibration/test IDs are retained
at every epoch. Each checkpoint fits its own interval parameters.

Earlier milestones use `--examples 0`: full held-out PSNR, SSIM, MAE, width and
coverage are saved without redundant image caches. The fifty latest-checkpoint
demo examples remain unchanged. `tools/run_coverage90_milestone.py` runs an entry
from the prepared task list and records genuine calibration infeasibility as a
gap, distinct from interrupted jobs or other execution failures.
Failures are explicit JSON records with an error code. A matching phrase in a log
cannot classify an unrelated execution error as a scientific calibration failure.

The project dispatcher, `tools/dispatch_coverage90_milestones.py`, uses the existing
SHA-256 release manifests under `~/.local/share/convrml-checkpoint-archive` and
the task list under `~/.local/share/convrml-coverage90-20261022`. It restores only
one baseline epoch at a time, verifies all five ensemble members for both cameras,
and removes only its disposable weight copies/links after their evaluations.
Original snapshots and GitHub Releases remain intact. One paper-model array and
one baseline array run at most two GPU tasks in total. Failed execution is retried
up to three submissions; unresolved failures are recorded for attention.

`milestone_dispatch.json` records restoration, submission, progress and retry
state. Stable copies of the operations scripts run from that same local work
folder, avoiding dependencies on experimental-source folder rearrangements.
Cluster calls pass JSON to a normal Python module rather than embedding Python
source in SSH commands. Submission receipts and a lock per stream prevent a
lost connection from creating duplicate GPU jobs. Reports are checked against
the requested camera, epoch, full test count, distinct ensemble members and
initialization/data-order seeds before being accepted. The curve collector also
recomputes aggregate metrics from all saved per-image metrics and checks matching
dataset split identities.

Cleanup requires valid results for all six baseline tasks at that epoch. It
removes only this pipeline's disposable copies; the original archived weights
remain available. These conditions use explicit exceptions, so Python's `-O`
option cannot disable them.

During code maintenance, the deployed evaluator is versioned and source hashes
are captured at job startup. A running job retains its original evaluator;
future tasks use the reviewed version. Already completed reports are audited
and retained rather than rewritten with new provenance.
The demo's separate `publish_coverage90_curves.py` publishes completed metrics
every five minutes. Pending checkpoints and infeasible calibrations remain
explicit; curve values are never fabricated or interpolated.
