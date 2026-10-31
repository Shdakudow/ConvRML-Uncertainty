# Thirty-epoch generalized input training from scratch

One QUTCC and one Im2im start from **fresh random weights, with no pretrained or
clean-checkpoint weights**, and train for **30 epochs**. Each epoch visits the original 17,999
training pairs once. The [online sampler](../../input_modifications/online.py)
chooses clean or one of nine modifications with 10% probability each per image.
Global exposure splits its share between gains 0.50 and 1.50. All other strengths
remain at the reviewed settings, including resolution factor 64 and JPEG 50.
Targets stay original, and no modified image datasets are saved.

The models are blind: modification labels are logged but never supplied to the
network. Both architectures retain their existing losses. Fresh Adam moments,
effective batch four, microbatch one, and float32 are recorded in
[protocol.json](protocol.json). The scratch-training schedule linearly warms up
from 1e-5 to 1e-4 during the first 30% of the run, then decays by cosine toward
1e-9. Epoch numbers 1–30 start with this new random initialization.

A fixed panel of 100 validation scenes is evaluated under all ten conditions
at initialization and after every epoch (1,000 predictions per validation).
The scene indices and per-condition random draws stay fixed across epochs and
models. This is a validation panel, not the full 2,000-scene validation split.
Per-condition PSNR, SSIM, MAE, and training condition counts are logged. Calibration
and test images are not loaded. Interval calibration is a separate post-training
step; clean calibration parameters are not claimed to calibrate the new models.

`submit.py` creates a short GPU smoke job for each method, then one dependent
30-epoch job. Before any smoke updates, freshly randomized weights are saved as
`initial_random.pt` with epoch 0, zero trained steps, and `pretrained=False`.
Smoke updates are discarded; actual training loads only that run's untrained
epoch-zero state and verifies the complete weight hash. This keeps initialization
identical across CPU types. There is no argument for loading an external trained
checkpoint. Recovery loads only the matching scratch run's own resume state.
`run.sbatch` requests a checkpoint before allocation expiry and requeues
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

The earlier [fine-tuning launch](../../reproducibility/generalized_inputs_30epochs_launch_20261031.json)
was cancelled because training must start from scratch. Its outputs must not be
used to initialize this run. New scratch runs use a separate experiment directory.

The [current scratch launch](../../reproducibility/generalized_inputs_scratch30_launch_20261031.json)
records the corrected source and job IDs. Passing smoke checks or submitting
jobs does not mean the 30 epochs have finished.
