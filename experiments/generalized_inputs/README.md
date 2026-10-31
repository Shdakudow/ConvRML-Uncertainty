# Thirty-epoch generalized input fine-tuning

One QUTCC and one Im2im start from their verified clean raw epoch-50 checkpoints
and train for **30 additional epochs**. Each epoch visits the original 17,999
training pairs once. The [online sampler](../../input_modifications/online.py)
chooses clean or one of nine modifications with 10% probability each per image.
Global exposure splits its share between gains 0.50 and 1.50. All other strengths
remain at the reviewed settings, including resolution factor 64 and JPEG 50.
Targets stay original, and no modified image datasets are saved.

The models are blind: modification labels are logged but never supplied to the
network. Both architectures retain their existing losses. Fresh Adam moments,
constant learning rate 1e-5, effective batch four, microbatch one, and float32 are
recorded in [protocol.json](protocol.json). Epoch numbers 1–30 refer to this new
fine-tuning run, independently of the clean parent's 50 epochs.

A fixed panel of 100 validation scenes is evaluated under all ten conditions
at initialization and after every epoch (1,000 predictions per validation).
The scene indices and per-condition random draws stay fixed across epochs and
models. This is a validation panel, not the full 2,000-scene validation split.
Per-condition PSNR, SSIM, MAE, and training condition counts are logged. Calibration
and test images are not loaded. Interval calibration is a separate post-training
step; clean calibration parameters are not claimed to calibrate the new models.

`submit.py` creates a short GPU smoke job for each method, then one dependent
30-epoch job. Smoke updates are discarded; actual training reloads the clean
parent. `run.sbatch` requests a checkpoint before allocation expiry and requeues
the same job. Cancellation does not requeue it. Full resume state includes Adam,
augmentation/quantile RNGs, epoch permutation offset, and accumulated epoch counts.
Training is checkpointed every 500 steps and at epoch boundaries. Compact model
weights are retained every five epochs, including epoch 30.

Use a frozen copy of the repository containing `source_commit.txt`:

```bash
python experiments/generalized_inputs/submit.py \
  --run-root /path/to/new/generalized-input-run \
  --project /home/jy2228/ConvRML
```

Outputs include `jobs.json`, GPU logs, and `results/{qutcc,im2im}/` containing
configuration, smoke checks, baseline validation, progress, history, resume state,
five-epoch checkpoints, and `COMPLETE.json` after all 30 epochs finish.

The [launch record](../../reproducibility/generalized_inputs_30epochs_launch_20261031.json)
records the frozen source commit, job IDs, and GPU smoke timings. Submission and
passing smoke checks do not mean the 30-epoch runs have finished.
