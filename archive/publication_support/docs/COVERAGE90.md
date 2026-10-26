# Calibration and 90% coverage

Calibration freezes trained weights and uses a **separate calibration split**.
The test split measures the selected intervals without choosing their parameters.

| `--calibration` | Baselines | QUTCC | Interpretation |
| --- | --- | --- | --- |
| `empirical` (default) | Separate lower/upper score order statistics | Global lower/upper q search | Target calibration coverage 90.0–90.1%; no RCPS certificate |
| `reference` | Exp2: asymmetric RCPS-HB; raw Im2im: finite-N adjusted scale grid | Paper Algorithm 1 with finite-N correction | Reproduce the recorded reference procedure; actual coverage may exceed 90% |
| `native` | Unscaled head / Gaussian-moment endpoints | q=0.05 / 0.95 | Uncalibrated control |

Use the same portable [evaluation command](RUNNING_EXPERIMENTS.md) and add
`--calibration reference` or `--calibration native` as needed. Use separate output roots such as `outputs/empirical/` and `outputs/reference/`
so learning curves do not mix protocols. All reports retain the selected mode. Neither calibration method forces each test image to 90%.

## Independent endpoint scales

For center `mu` and native bounds `L0, U0`:

```text
lower_distance = max(mu - L0, 1e-6)
upper_distance = max(U0 - mu, 1e-6)
L = mu - s_lower × lower_distance
U = mu + s_upper × upper_distance
```

Empirical fitting selects the smallest nonnegative scale meeting a 5% miss
budget per tail, with numerical protection at the selected boundary. It uses
clipped [0,1] endpoints and accounts for ground-truth boundary values. Ties can
prevent exact coverage; `within_tolerance` records the measured outcome.
For MC dropout, the calibration audit replays the same random masks, then restores
the post-fit RNG state before test inference. Test samples remain held out.

Reference Exp2 calibration uses image-level Hoeffding–Bentkus RCPS with
**alpha=0.10, delta=0.05**, splitting both budgets between tails. A calibration
unit is one whole image, whose loss is its uncovered RGB fraction. The reported
certificate relies on the reference bounded-loss/nested-interval and i.i.d.
image assumptions. It does not assume pixels are independent or certify a fixed
coverage fraction on every individual test image.

Reference raw Im2im uses a 1,000-point scale grid from 0 to 6 with per-tail target
`(alpha - (1-alpha)/N_images)/2`. This adjusted empirical procedure is distinct
from the HB certificate. Reference calibration uses raw endpoints; raw and clipped
test metrics are reported separately in every mode.

## QUTCC

Calibration changes two scalar q inputs shared across future images. It does not
add a selector, retrain the network, multiply interval scales or sort crossings.
Each queried q still produces a spatially varying RGB image.

Empirical matching targets 5% misses per tail. Algorithm 1 instead uses the
finite-image correction above. The model is not structurally monotone: empirical
search checks observed q responses and explicitly rejects violations. The original
Algorithm 1 retains its documented monotonicity assumption.

A checkpoint can fail the **equal-tail requirement** even if an asymmetric
interval with 90% total coverage might exist. This occurred for clean-raw QUTCC at
epoch 20: its lower-tail miss fraction at q=0 was about 7.82%, exceeding 5%.
The workflow preserves that failure rather than replacing it with fabricated
coverage, another checkpoint or an unlabelled calibration method.

## Audit fields

`calibration.json` records the procedure, fitted values, sample count, clipping,
measured calibration coverage and confidence status. `report.json` records actual
test coverage/width plus checkpoint hashes, split identities, seed, hardware,
software and timing. Clipping is an explicit metric/display operation, separate
from calibration. See [metric definitions](METHODS_AND_METRICS.md).
