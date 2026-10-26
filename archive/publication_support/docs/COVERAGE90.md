# Paper calibration and 90% coverage

Calibration freezes model weights and uses a **separate calibration split**.
The active evaluator defaults to `--calibration paper`; it chooses the procedure
from the model, with no additional empirical-matching mode.

| Model | Calibration | Parameters selected |
| --- | --- | --- |
| Im2im, ensemble, MC dropout | Image-level RCPS-HB | Separate lower/upper scales |
| QUTCC | Paper Algorithm 1 | Lower/upper quantile inputs |

`--calibration native` is an optional uncalibrated control: baseline endpoints
are unscaled, and QUTCC uses q=0.05/0.95. Neither calibrated procedure forces
every test image or the finite test set to exactly 90% coverage.

## Baselines: RCPS

For center `mu` and native bounds `L0, U0`:

```text
lower_distance = max(mu - L0, 1e-6)
upper_distance = max(U0 - mu, 1e-6)
L = mu - s_lower × lower_distance
U = mu + s_upper × upper_distance
```

The implementation uses Hoeffding–Bentkus RCPS with **alpha=0.10, delta=0.05**,
allocating alpha/2 and delta/2 to each tail. A statistical sample is one whole
image, whose loss is its uncovered RGB fraction. The certificate relies on the
bounded-loss/nested-interval and i.i.d. image assumptions; it does not assume
independent pixels or guarantee a coverage fraction for every individual image.

For MC dropout, the calibration audit replays the fit-time random masks and
restores the post-fit RNG state before evaluating the held-out test split.

## QUTCC: calibrated quantiles

Algorithm 1 selects two scalar q values shared across future images. It searches
for the largest feasible lower q and smallest feasible upper q, using per-tail
target `(alpha - (1-alpha)/N_images)/2` and q tolerance 1e-3.

Each selected q produces a spatially varying RGB bound. Calibration does not
retrain the network, add a selector or multiply its endpoint distances. The
binary search retains the paper's assumption of monotone quantile responses;
shared weights do not strictly enforce that assumption. This finite-N procedure
is distinct from the baseline HB confidence certificate.

If the extreme q predictions cannot satisfy a tail budget, evaluation writes
`failure.json` and stops before loading test pixels. A failed checkpoint remains
a gap, not a fabricated metric or a substituted asymmetric search.

## Outputs and previous results

Calibration uses raw endpoints. Evaluation separately reports raw and clipped
test metrics; example images use clipped predictions/bounds. Clipping is not
calibration. See [metric definitions](METHODS_AND_METRICS.md).

`calibration.json` records the procedure, selected scales or q values, sample
count, measured calibration coverage and HB certificate when applicable.
`report.json` records measured test coverage/width, checkpoint and split identity,
seed, software, hardware and timing. Calibration parameters are frozen before
test pixels are loaded.

Superseded empirical matching and the old raw Im2im finite-N scale grid are in
[calibration_legacy/](../calibration_legacy/). The active evaluator now uses RCPS
for raw Im2im as well as parallel lensless baselines. Existing reports and demo assets retain
their original procedures and numbers; they are not relabelled as new RCPS results.
