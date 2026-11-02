# Verification

```bash
python -m unittest discover -s tests -v
```

Small CPU fixtures check shared model/data APIs, real loss methods, calibration
numerics, split isolation, metrics, exports, milestone weights and uninterrupted
versus resumed training. Resume checks cover both shuffle policies and the
online mixture/Gaussian-only workflows, including Adam and RNG states.

Full benchmark training and CUDA equality across hardware are outside this
suite. The pre-simplification tests for source/version enforcement, Slurm
submission and historical migrations remain with their
[original source snapshot](../archive/research_prototype_20261102/).
Existing verification records describe their recorded revision, not later edits.
