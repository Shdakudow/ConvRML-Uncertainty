# Experiment 4: robustness of clean and generalized models

Compare the **original clean-trained QUTCC and Im2im at epoch 50**, already shown
in Exp 3, with the **from-scratch generalized QUTCC and Im2im at epoch 100**.
This measures the robustness of the complete training setups. Training time and
input distribution both differ, so this is not an equal-budget ablation of
augmentation alone. Gaussian specialists are not part of this comparison.

`conditions.py` defines clean input and ten displayed modifications: nine training
families, with exposure split into global gain 0.50 and 1.50. The existing separate
`input_modifications/` files perform the changes. Targets stay unchanged. Every
test input is generated once and passed to all four models, without intermediate
PNG rounding. Per-image seeds do not depend on model, batch, or worker order.

Calibration uses the original **3,000 separate calibration scenes**. Exactly 300
are clean; 300 each use Gaussian, Poisson, mixed noise, dead pixels, fixed pattern,
blur, resolution, and JPEG; 150 each use the two exposure gains. These weights
match the ten training families. Each model gets one global fit on the same
mixture. All fits are saved before the test images are loaded.

The 90% target is empirical overall RGB-value coverage, not a population guarantee
or a per-category guarantee. QUTCC searches global quantile endpoints; Im2im fits
asymmetric order-statistic scales. `empirical.py` preserves the current demo's
Near90 implementation (also archived under `archive/reproduction/shared/`).
Observed nonmonotonicity or unattainable QUTCC tail targets are explicit failures;
we then display the extreme q=0/1 intervals with a failure label. No endpoint
sorting, fabricated coverage, or test-driven parameter adjustment is allowed.

All **2,000 original test scenes** are evaluated under every condition. The first
50 in manifest order are browsable. PSNR, SSIM, MAE, interval coverage, width and
crossing use full-precision clipped outputs, with intensity range 1. Width images
use one fixed [0,1] RGB-average color scale; crossed bounds are pink. PNGs are
only display exports. Original clean reconstructions must reproduce the current
demo's first 50 PSNR/SSIM/MAE values before publication is permitted.

The pipeline checks the original split hash and disjoint IDs, exact baseline
checkpoint hashes, generalized 100-epoch completion/history, scratch initialization
hashes, training protocol and source commit. It rehashes weights after evaluation.
Requeued jobs reuse completed calibrations and conditions only when the complete
request identity matches. Temporary Im2im score arrays (~9.3 GB per fit) are
removed; no permanent ninefold corrupted dataset is created. Display exports are
5,000 RGB PNGs, plus public metric/provenance JSON.

Run from a frozen repository snapshot on the Cornell GPU cluster:

```sh
python experiments/experiment4_demo/evaluate.py \
  --manifest /path/to/original/split.json \
  --generalized-root /path/to/raw_online_scratch_30epochs_20261031_v2 \
  --clean-root /path/to/qutcc_paper_style_20261018 \
  --baseline-reference /path/to/baseline_reference.json \
  --output /path/to/experiment4/output
```

Use `run.sbatch` with `--dependency=afterok:<QUTCC-100-job>`; it forwards the
checkpoint signal and requeues on exit 75. Both 100-epoch completion markers are
still required. Signals are checked between inference batches. A cancelled job
stops rather than restarting itself. A running condition can be recomputed on
requeue; completed condition reports are atomic.

`publish.py` is a short local LaunchAgent task, run every five minutes for up to
seven days. It waits for `READY.json`, downloads only the public exports, validates
every file hash and image shape, pushes an immutable asset branch, then publishes
`web/` as `experiment4/` and adds shared navigation. Isolated Git checkouts protect
other local edits. It never publishes a pending page or uses placeholder scores.
Status is in `~/.local/share/convrml-exp4-20261101/status.json`. A sleeping/offline
Mac resumes checking when it is available; expiry or errors remain visible there.
The main demo is changed only after the completed export passes verification.
