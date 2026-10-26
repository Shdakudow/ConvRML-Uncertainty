# Superseded calibration modes

These sources retain the previous empirical-matching and combined "reference"
evaluation paths. The active package no longer imports them.

- `empirical.py`: the removed close-to-90% scale/quantile matching routines.
- `reference_calibration.py`: the previous combined implementation, retained for
  numerical comparison with the RCPS and Algorithm 1 routines now in `convrml/calibration.py`.
- `evaluate_snapshot.py` and `settings_snapshot.json`: original source snapshots,
  not standalone entry points. Use the earlier commit to run that whole pipeline.
- `tests/`: checks for the archived empirical formulas.

From the checkout root:

```bash
python -m unittest discover -s archive/publication_support/calibration_legacy/tests -v
```

Historical reports retain their original calibration labels and numbers. In
particular, old raw Im2im "reference" reports used a finite-N scale grid, not
RCPS-HB; the active evaluator now uses RCPS-HB for Im2im in both experiments.
No old report, demo asset or trained checkpoint is rewritten by this code change.
