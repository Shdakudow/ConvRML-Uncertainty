# Original large-data comparison — 50 epochs

This folder contains the earlier four-method comparison on **RML and DiffuserCam**. Each dataset uses **60,000 training images**, 5,000 validation images, 5,000 test images, and 2,048 separate calibration images.

[Parameters](configs/parameters.json) · [Current adjusted comparison](../adjusted_comparison_50epochs/) · [Repository overview](../../README.md)

## Models

| Method | Original setting |
| --- | --- |
| Im2im / fixed head | MSE reconstruction backbone trained to 50 epochs; fixed endpoint head trained for 10 epochs on frozen backbone features at each evaluated milestone. |
| Deep ensemble | Five independently trained Gaussian-NLL members, each trained to 50 epochs. |
| MC dropout | MSE training with decoder dropout probability `0.10`; 30 stochastic inference passes. |
| QUTCC + ConvRML | Original quantile conditioner on the final decoder feature map; reconstruction MSE plus pinball loss. No all-block FiLM. |

Backbone checkpoints are saved at epochs 5, 10, …, 50. Evaluation compares reconstruction quality, coverage, and interval width. Empirical and RCPS calibration use held-out calibration images and target `alpha = 0.10`; RCPS uses `delta = 0.05`.

The current adjusted experiment reuses the final im2im checkpoint and these original five ensemble members. It does not retrain them.

## Files to start with

| File | Role |
| --- | --- |
| [scripts/large_interval_study.py](scripts/large_interval_study.py) | RML training and evaluation orchestration |
| [scripts/diffuser_interval_study.py](scripts/diffuser_interval_study.py) | Same protocol with DiffuserCam data and result paths |
| [scripts/formal_rml_train.py](scripts/formal_rml_train.py) | Ensemble and MC-dropout training |
| [scripts/formal_quantile_train.py](scripts/formal_quantile_train.py) | MSE backbone, quantile-conditioned model, and fixed-head training |
| [scripts/large_interval_evaluate.py](scripts/large_interval_evaluate.py) | Reconstruction and empirical interval evaluation |
| [scripts/compare_rcps.py](scripts/compare_rcps.py) | RCPS versus empirical calibration |
| [configs/parameters.json](configs/parameters.json) | Documented experiment settings |

A few historically named helpers, including `toy_grid_compare.py`, remain because the original trainers import their utility functions. They are dependencies, not additional experiments to run.

## Environment and execution

Use Python 3.11 and a CUDA-compatible PyTorch/torchvision installation, then run from **this folder**:

```bash
python -m pip install -r requirements.txt
```

Data and result paths remain relative to the experiment working directory. The scripts expect the original `pairs.npy` caches and `manifest.json` split files under:

```text
data/optics_intervals_60k_5k_5k_20260928/
data/diffuser_intervals_60k_5k_5k_20260928/
data/formal_rml_calibration95/
data/diffuser_intervals_calibration_20260928/
```

The corresponding original result roots are:

```text
results/optics_intervals_60k_5k_5k_20260928/
results/diffuser_intervals_60k_5k_5k_20260928/
```

Data and weights are not included in GitHub. Existing completed results can be made available at those paths without retraining.

The historical submission entry points are `scripts/submit_large_interval_study.py` and `scripts/submit_diffuser_interval_study.py`. They allocate GPU jobs; they are not needed simply to inspect this completed comparison. Review their schedule and cluster settings before launching a new replication. Slurm templates now use the experiment submission directory; their Python environments and resource constraints still reflect Unicorn.

## Relation to the archive

The [exploratory archive](../../archive/exploratory/) contains earlier toy, small-data, and superseded studies. They are not part of this large-data comparison. This folder preserves its required code dependencies separately.
