# Experiment 2 · four-model comparison

Compare Im2im, five-member deep ensemble, MC dropout and QUTCC on **RML** and **normalized Cornell DiffuserCam**. Each camera has 60,000 training, 5,000 validation, 5,000 test and 2,048 separate calibration pairs. Main checkpoints span epochs 5, 10, …, 50; ten-member ensembles and dropout-rate sweeps belong to this experiment.

| Open this | Purpose |
| --- | --- |
| **[Four-model source](four_models/)** | Common command router, baseline networks and current QUTCC |
| **[Model-code comparison](../../docs/MODEL_COMPARISON.md)** | Architecture, output heads, objectives and calibration differences |
| [Analysis](analysis/) | LPIPS, width/error Pearson and Spearman, every-five-epoch evaluations |
| [Commands](../../docs/RUNNING_EXPERIMENTS.md#experiment-2) | Fresh training, checkpoint evaluation and curves |

## Short code-reading route

1. [Shared ConvNeXt/FPN backbone](four_models/models/shamus_rml.py).
2. [Im2im and full-network dropout](four_models/models/corrected_uncertainty.py): `EndToEndIm2Im`, `FullDropoutRML`, `training_loss`.
3. [Ensemble backbone](four_models/baselines/formal_rml.py), [Gaussian member/aggregation](four_models/baselines/deep_ensemble.py).
4. [Current QUTCC model/loss](four_models/qutcc/model.py), [48 FiLM sites](four_models/qutcc/models/convnext_norm_film.py).
5. [QUTCC quantile calibration](four_models/qutcc/calibration.py), [baseline asymmetric RCPS](four_models/scripts/asymmetric_rcps.py).

Historical adaptive classes remain inside compatibility modules used for old checkpoints. The current router selects **`four_models/qutcc/`**. Its model source is identical to the clean Exp3 implementation; task and dataset distinguish the runs.

Main inference uses MC p=0.05, 30 passes and ensemble members 00–04. Other dropout rates and the ten-member aggregate have separate validation/calibration results. Each ensemble member trains on the full training split with different initialization and shuffle seeds.
