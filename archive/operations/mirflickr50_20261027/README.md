# Fresh clean Mirflickr comparison, 2026-10-27

This deployment uses the public, shared `models/im2im.py`, `models/qutcc.py`
and root `train.py` from an immutable source snapshot. It adds no dataset-specific
network implementation and starts both production models from random weights.
Earlier runs and results are retained.

Data: the existing clean DiffuserCam Lensless Mirflickr prepared RGB exports,
270×480; 17,999 train / 2,000 validation / 2,000 test / 3,000 calibration pairs.
The original split and targets are unchanged. A CPU audit rechecks every file
and decoded pixel hash before a GPU job is allowed to run.

Both models use the recorded clean DLMD FP32 Adam profile: effective batch 4,
microbatch 1, betas (0.9,0.99), no weight decay; LR warms from 1e-5 to 1e-4 over
30% of the fixed 50-epoch horizon, then follows cosine decay to 1e-9. Im2im jointly
trains center MSE plus 0.05/0.95 endpoint pinball. QUTCC uses uniform sampled-q
pinball only, with 36 encoder + 12 decoder FiLM sites.

`preflight.py` first runs the public test suite and data audit. GPU preflights use
actual training pairs to verify full-size cropped outputs, finite losses and
gradients reaching the encoder, decoder and heads. QUTCC additionally verifies
FiLM/embedding gradients and that q changes predictions after updates. A temporary
production-trainer smoke run tests Adam, microbatching and checkpoint saving;
its weights are discarded and never initialize the actual experiment.

`submit.py` schedules one CPU audit, two one-GPU preflights, then ten dependent
five-epoch jobs per method. At most two GPUs can be used simultaneously. Public
GPU allocations prefer H100/H200/6000 Ada, with A6000 fallback, through the user's
permitted account and normal QOS. No higher QOS is assumed. Requeued jobs resume
from their last complete epoch; failures cancel dependent jobs rather than
allowing unverified training. No previously submitted job is cancelled.

On Unicorn, with the source snapshot extracted at STUDY/source:

```bash
python source/archive/operations/mirflickr50_20261027/submit.py \
  --study /path/to/STUDY --manifest /path/to/existing/split.json
# Review the dry-run commands, then add --submit.
```

The source remains fixed for all 50 epochs. `results/<method>/latest.pt` retains
full optimizer/RNG state; `checkpoints/epoch_005.pt` through `epoch_050.pt` preserve
the inference milestones. Validation is separate from calibration/test. Calibration
is performed after training, using asymmetric RCPS for Im2im and paper Algorithm 1
quantile calibration for QUTCC, with alpha=0.1; it is not fitted on test images.
