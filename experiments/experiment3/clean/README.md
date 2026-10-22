# Clean raw training

QUTCC and end-to-end Im2im, 50 epochs. Source files here link to the common
implementations in [shared/](../../../shared/). Outputs stay in this folder's
`results/raw/<method>/`.

Inside a GPU allocation:

```bash
python train.py --task 2 --smoke
python train.py --task 2 --target-epoch 5
python evaluate.py --task 2 --epoch 5
```

**Task 2 = QUTCC; task 3 = Im2im.** `--target-epoch` is the total target epoch.
Both use float32 Adam, effective batch four and the recorded warmup/cosine
schedule. Raw task outputs are cropped back to 270×480 after internal padding.
Historical normalized-camera task IDs 0/1 remain in the shared engine; use
[Exp2](../../experiment2/) for those comparisons.

[Data/checkpoint setup](../../../docs/RUNNING_EXPERIMENTS.md) ·
[model and calibration differences](../../../docs/MODEL_COMPARISON.md).
