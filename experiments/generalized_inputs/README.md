# Generalized input training: 50 epochs, then a lower-rate phase to 100

One QUTCC and one Im2im start from **fresh random weights, with no pretrained or
clean-checkpoint weights**, with a **100-epoch total budget**. The active jobs
finish their existing 50 epochs before the second phase starts. Each epoch visits the original 17,999
training pairs once. The [online sampler](../../input_modifications/online.py)
chooses clean or one of nine modifications with 10% probability each per image.
Global exposure splits its share between gains 0.50 and 1.50. All other strengths
remain at the reviewed settings, including resolution factor 64 and JPEG 50.
Targets stay original, and no modified image datasets are saved.

The models are blind: modification labels are logged but never supplied to the
network. Both architectures retain their existing losses. Fresh Adam moments,
effective batch four, microbatch one, and float32 are recorded in
[protocol.json](protocol.json). The scratch-training schedule linearly warms up
from 1e-5 to 1e-4 during the first nine epochs, then decays by cosine toward
1e-9 at epoch 50. The nine-epoch warmup preserves the schedule already used by
the initial 30-epoch run. Epochs 51–100 begin a separate low-rate cosine phase,
from **1e-5 to 1e-6**, without warmup or optimizer reset. Its starting rate is
one-tenth of the original 1e-4 peak, and deliberately higher than the near-zero
rate at the end of phase one. This preserves all rates used in epochs 1–50.

A fixed panel of 100 validation scenes is evaluated under all ten conditions
at initialization and after every epoch (1,000 predictions per validation).
The scene indices and per-condition random draws stay fixed across epochs and
models. This is a validation panel, not the full 2,000-scene validation split.
Per-condition PSNR, SSIM, MAE, and training condition counts are logged. Calibration
and test images are not loaded. Interval calibration is a separate post-training
step; clean calibration parameters are not claimed to calibrate the new models.

`submit.py` creates a short GPU smoke job for each method, then one dependent
100-epoch job. Before any smoke updates, freshly randomized weights are saved as
`initial_random.pt` with epoch 0, zero trained steps, and `pretrained=False`.
Smoke updates are discarded; actual training loads only that run's untrained
epoch-zero state and verifies the complete weight hash. This keeps initialization
identical across CPU types. There is no argument for loading an external trained
checkpoint. Recovery loads only the matching scratch run's own resume state.
`run.sbatch` requests a checkpoint before allocation expiry and requeues
the same job. Cancellation does not requeue it. Full resume state includes Adam,
augmentation/quantile RNGs, epoch permutation offset, and accumulated epoch counts.
Training is checkpointed every 500 steps and at epoch boundaries. Compact model
weights are retained every five epochs, including epochs 30, 50, and 100.

Use a frozen copy of the repository containing `source_commit.txt`:

```bash
python experiments/generalized_inputs/submit.py \
  --run-root /path/to/new/generalized-input-run \
  --project /home/jy2228/ConvRML
```

Outputs include `jobs.json`, GPU logs, and `results/{qutcc,im2im}/` containing
configuration, smoke checks, baseline validation, progress, history, resume state,
five-epoch checkpoints, and `COMPLETE.json` after all 100 epochs finish.

## Automatic continuation after epoch 50

For the active runs, use `schedule_after50.py --run-root /path/to/existing/run`.
It submits one dependent GPU job per model with `afterok` on that model's current
50-epoch job. Current jobs and their sources stay untouched. The continuation
also verifies the successful completion marker, epoch 50 checkpoint, next epoch
51, zero sample offset, all 225,000 optimizer steps, and all 50 validation records.
It cannot begin the new phase early.

The wrapper `continue_after50.py` archives the original configuration, full
checkpoint, and completion marker. Model weights, Adam moments, random-generator
states, sample position, and training history are restored. The epoch-zero
provenance still points to the original scratch initialization. Pretrained clean
weights are never introduced. `COMPLETE_50epochs.json` remains available while
the new phase runs; `COMPLETE.json` is written for epoch 100 at final completion.
The wrapper replaces itself with the trainer so checkpoint/requeue signals reach
the same child PID used by Slurm.

The [queued 100-epoch plan](../../reproducibility/generalized_inputs_100epochs_after50_20261031.json)
records the dependency job IDs, learning rates, and verification checks. The
continuations wait for their respective models to finish epoch 50 successfully.

## Historical 30-to-50 extension

The existing scratch runs continue from their own full checkpoints, preserving
weights, Adam moments, RNG states, sample offsets, counts, and history. They do not
restart or load trained clean-model weights. Stop the old jobs gracefully, then
run `submit_extension.py --run-root /path/to/existing/run` from the archived 50-epoch
source snapshot. It archives each original config and checkpoint before submitting
the continuations. That historical utility is separate from the new after-50 scheduler.
Model, dataset, loss, and input-modification code must match
the original snapshot. The explicit config transition permits only the epoch
budget, schedule, protocol version, and source provenance to change. The runner
checks that the handoff occurs during the original warmup and saves
`extension_applied.json` with its preserved step and offset.

The [50-epoch continuation record](../../reproducibility/generalized_inputs_50epochs_extension_20261031.json)
records both handoff positions, replacement job IDs, and verified 50-epoch progress.

The earlier [fine-tuning launch](../../reproducibility/generalized_inputs_30epochs_launch_20261031.json)
was cancelled because training must start from scratch. Its outputs must not be
used to initialize this run. New scratch runs use a separate experiment directory.

The [current scratch launch](../../reproducibility/generalized_inputs_scratch30_launch_20261031.json)
records the corrected source and job IDs. Passing smoke checks or submitting
jobs does not mean the training budget has finished.
