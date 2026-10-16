# Controlled comparison: training without dropout

This experiment changes the 48 active dropout locations from p=0.05 to p=0 while keeping the ConvNeXt/FPN architecture, MSE loss, initial weights, data-order seeds, optimizer, batch size, dataset splits and 50-epoch schedule matched. Existing p=0.05 checkpoints are reused; only the p=0 control is trained anew. RML and DiffuserCam have independent job chains and result directories.

The tests in `scripts/test_dropout_control.py` verified identical initial state tensors, matched shuffled sample sequences across two epochs, identical dropout-off predictions before training, deterministic p=0 training-mode predictions, and stochastic p=0.05 predictions. The initial-state hash and test outcome are in `parameters/dropout_control_matching_audit.json`.

## Comparisons

| Comparison | Question |
|---|---|
| Newly trained p=0 versus trained p=0.05 with dropout off | Does dropout during training improve reconstruction? |
| p=0.05 dropout off versus p=0.05 mean of 30 stochastic passes | What is the effect of stochastic averaging at inference? |

All three predictions are evaluated on the same 5,000 held-out images at epochs 5, 10, …, 50. Reports contain mean PSNR, SSIM, MSE and MAE; paired per-image differences; and forward-pass runtime on the same GPU. Runtime excludes data loading and metric computation. These are one matched training-seed pair per camera, not a multi-seed statistical study.

The deterministic p=0 and dropout-off models are **not assigned uncertainty intervals**. A zero MC variance is not evidence of accurate uncertainty. The existing p=0.05 demo separately uses 30-pass MC intervals and RCPS calibration; this control report evaluates point reconstruction only.

## Run

From the experiment root, with the same caches and CUDA environment as the main experiment:

```bash
mkdir -p logs
python scripts/test_dropout_control.py
python scripts/submit_dropout_control.py
python scripts/submit_dropout_control_evaluation.py
```

The first command is a CPU matching test. Submission creates one GPU smoke job per camera, then 25 dependent two-epoch training jobs per camera. Training starts only after its smoke test passes. Evaluation waits for each five-epoch checkpoint to be saved; one evaluation lane per camera limits concurrency.

Results: `results/dropout_control_20261015/{rml,diffuser}/mc_000/`. Paired reports: `results/dropout_control_20261015/evaluations/{rml,diffuser}/epoch_NNN.json`.

Latest checkpoints include optimizer and random-generator states for resuming after the last completed epoch. Milestone checkpoints are saved every five epochs. No existing QUTCC or baseline checkpoint is modified.
