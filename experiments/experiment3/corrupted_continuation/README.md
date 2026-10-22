# Corruption adaptation epochs 11–30

Continues [the ten-epoch pilot](../corrupted/) on the same reviewed inputs,
retaining model, Adam moments, RNG and training counters. LR stays 1e-5.
Final checkpoints represent **30 clean pretraining + 30 adaptation epochs**.

Task **0 = QUTCC / 1 = Im2im**. Inside a GPU allocation:

```bash
python test_continuation.py
python train.py --task 0 --target-epoch 15
python evaluate.py --task 0 --epoch 30
```

`continuation.py` verifies parent state/counters. Common model and evaluation
files link to [shared/](../../../shared/); data/continuation training remains
specific to this phase. The original full-state parents and reviewed manifests
must be restored at their recorded external paths.

[Setup, calibration and outputs](../../../docs/RUNNING_EXPERIMENTS.md#experiment-3).
