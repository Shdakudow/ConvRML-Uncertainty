# Exploratory and historical experiments

This archive keeps the earlier investigations separate from the two main comparisons.

- Toy Fourier/motion experiments and smaller real-image pilots.
- CRPS experiments on smaller datasets.
- Residual-distribution, neighborhood, and momentum experiments.
- Gradient-conflict and frozen-backbone controls.
- Superseded FiLM pilots and the original-QUTCC architecture trial.
- Historical visualization, reporting, and cluster utilities.

These experiments used different datasets, splits, training budgets, or objectives. Their results should not be treated as one matched comparison.

For the main studies, use:

- [Original 50-epoch large-data comparison](../../experiments/baseline_50epochs/)
- [Current adjusted MC-dropout / QUTCC / ensemble comparison](../../experiments/adjusted_comparison_50epochs/)

The files here preserve historical assumptions, local paths, and dependencies. Some reference scripts now grouped with the main comparisons. They are archived research code, not a single runnable pipeline; consult the relevant script before attempting to reproduce an old run. They do not contain datasets or checkpoint weights.

Shared helper copies are retained where historical scripts import them. The original upstream README is available as [UPSTREAM_README.md](UPSTREAM_README.md).
