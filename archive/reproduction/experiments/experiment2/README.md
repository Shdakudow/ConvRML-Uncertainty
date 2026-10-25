# Experiment 2 · RML and normalized DiffuserCam

Four methods, 50 epochs, saved five-epoch milestones. Per camera: **60,000 train /
5,000 validation / 5,000 test / 2,048 separate calibration pairs**.

| Open | Role |
| --- | --- |
| **[run_model.py](run_model.py)** | Main smoke/train/evaluate command for all four methods |
| [models/](models/) | End-to-end Im2im and full-network MC dropout; imported historical compatibility code |
| [baselines/](baselines/) | Gaussian ensemble members, aggregation and MC inference |
| [qutcc/](qutcc/) | Current QUTCC entry points linking to shared code |
| [scripts/](scripts/) | Baseline training, calibration, interval evaluation and image exports |
| [analysis/](analysis/) | LPIPS and width/error correlation metrics/figures |
| [tests/](tests/) | Grouped model, runtime and checkpoint tests |
| [configs/four_model_comparison.json](configs/four_model_comparison.json) | Method/settings map |

```bash
python run_model.py --help
python run_model.py train --method qutcc --camera rml --epoch 5 --dry-run
```

Main inference uses five ensemble members and MC dropout p=0.05 with 30 passes.
Ten members and other dropout rates are validation comparisons. QUTCC uses
paper Algorithm 1 calibration; the three other methods use asymmetric RCPS-HB.

[Run instructions](../../docs/RUNNING_EXPERIMENTS.md#experiment-2) ·
[model differences](../../docs/MODEL_COMPARISON.md) · [shared source](../../shared/).
Compatibility classes are retained only where current imports/checkpoints need
them; current QUTCC routes through `qutcc/`, not the adaptive-selector trainers.
