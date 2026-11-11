# Evaluation

Run these commands from the repository root:

```bash
python -m evaluation.run --help
python -m evaluation.download --help
```

| File | Responsibility |
| --- | --- |
| `run.py` | Load frozen weights, fit calibration, evaluate test images and save results |
| `metrics.py` | PSNR, SSIM, MAE, LPIPS, coverage, width and error correlations |
| `reporting.py` | Test iteration, aggregation, CSV and example exports |
| `download.py` | Download published weights using the archived checkpoint catalog |

Example (PLD NumPy caches):

```bash
python -m evaluation.run --dataset rml --method qutcc \
  --checkpoints weights/qutcc.pt \
  --manifest data/manifest.json --pairs data/pairs.npy \
  --calibration-manifest data/calibration/manifest.json \
  --calibration-pairs data/calibration/pairs.npy \
  --output outputs/qutcc --device cuda
```

For image-file data, omit the NumPy cache arguments. Calibration and test scenes
remain separate. This package never trains model weights. Calibration algorithms live in
[`convrml/calibration.py`](../convrml/calibration.py); `run.py` calls them before
evaluating the test split. Training reuses
`metrics.py` for validation. Model/checkpoint loading remains in
`convrml/checkpoints.py` because both training and evaluation need it.

The former root `calibration.py` command is now `python -m evaluation.run`.
Downloads use `python -m evaluation.download` instead of `python -m convrml.download`.
Archived experiment source retains its original imports and deployed environment.
