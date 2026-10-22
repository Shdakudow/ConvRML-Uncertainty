"""Tight empirical coverage matching, separate from RCPS and paper calibration.

Only calibration targets may be used by these routines. A numerical tolerance
is an audit criterion, not a population or held-out-test coverage guarantee.
"""

import math

import numpy as np


def fit_tail_scale(scores, alpha=0.05, *, overwrite=False):
    """Smallest nonnegative scale covering at least 1-alpha of these scores.

    Scores are (center-target)/lower_distance or its upper-tail counterpart.
    Negative infinity denotes a target always covered after endpoint clipping.
    In-place selection is intended for disposable disk-backed calibration data;
    it destroys image ordering, so this function makes no image-level certificate.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    values = np.asanyarray(scores).reshape(-1)
    if not len(values):
        raise ValueError("Empty calibration scores")
    for start in range(0, len(values), 1_000_000):
        part = values[start : start + 1_000_000]
        if np.isnan(part).any() or np.isposinf(part).any():
            raise ValueError("NaN or positive-infinite calibration scores")
    rank = math.ceil((1 - alpha) * len(values)) - 1
    working = values if overwrite else values.copy()
    working.partition(rank)
    scale = max(0.0, float(working[rank]))
    # Division to form scores and multiplication to apply endpoints need not
    # round identically. A few ULPs avoid excluding the selected boundary value.
    if scale > 0:
        dtype = working.dtype if working.dtype.kind == "f" else np.dtype("float64")
        scale = float(scale + 8 * float(np.finfo(dtype).eps) * max(1.0, scale))
    missed = sum(
        int(np.count_nonzero(working[i : i + 1_000_000] > scale))
        for i in range(0, len(working), 1_000_000)
    )
    return dict(scale=scale, risk=missed / len(values), count=len(values))


def search_quantiles(
    risk,
    *,
    alpha=0.1,
    coverage_tolerance=0.001,
    initial=(0.05, 0.95),
    q_tolerance=1e-7,
    max_iterations=25,
):
    """Refine global q endpoints to equal empirical tail risks of alpha/2.

    ``risk(lo,hi)`` returns the two calibration miss fractions. It should cache
    unchanged endpoints. Q monotonicity is checked along evaluated points;
    observed violations fail explicitly rather than claim a successful search.
    """
    if not 0 < alpha < 1 or not 0 < coverage_tolerance < alpha:
        raise ValueError("Invalid target/tolerance")
    if not 0 <= initial[0] <= initial[1] <= 1:
        raise ValueError("Invalid initial quantile pair")
    target = alpha / 2
    trace = []
    samples = [{}, {}]

    def check(lo, hi):
        risks = tuple(float(v) for v in risk(lo, hi))
        if len(risks) != 2 or not all(np.isfinite(v) and 0 <= v <= 1 for v in risks):
            raise ValueError("Invalid calibration risks")
        trace.append(
            dict(lower_q=lo, upper_q=hi, lower_risk=risks[0], upper_risk=risks[1])
        )
        for side, q in enumerate((lo, hi)):
            samples[side][q] = risks[side]
            ordered = sorted(samples[side].items())
            vals = [v for _, v in ordered]
            changes = np.diff(vals) * (1 if side == 0 else -1)
            if np.any(changes < -1e-6):
                raise ValueError(
                    "Observed nonmonotone q response; calibration not certified"
                )
        return risks

    valid = [None, None]
    probe_step = [0.001, 0.001]
    pair = list(initial)
    for _ in range(max_iterations):
        risks = check(*pair)
        done = []
        for side, value in enumerate(risks):
            q = pair[side]
            if value <= target and (valid[side] is None or value >= valid[side][1]):
                valid[side] = (q, value)
            done.append(value <= target and target - value <= coverage_tolerance / 2)
        if all(done):
            break
        for side in range(2):
            if not done[side]:
                # Express both tails in an increasing-risk coordinate. Probe
                # locally first: saved q pairs are already close to the target,
                # and each full-calibration forward is expensive.
                points = [
                    (q if side == 0 else 1 - q, v) for q, v in samples[side].items()
                ]
                feasible = [t for t, v in points if v <= target]
                infeasible = [t for t, v in points if v > target]
                if feasible and infeasible:
                    left, right = max(feasible), min(infeasible)
                    t = (left + right) / 2
                    if right - left < q_tolerance:
                        t = left
                else:
                    t = pair[side] if side == 0 else 1 - pair[side]
                    t = float(
                        np.clip(
                            t + (probe_step[side] if feasible else -probe_step[side]),
                            0,
                            1,
                        )
                    )
                    probe_step[side] *= 2
                pair[side] = t if side == 0 else 1 - t
    # Evaluate selected feasible endpoints again; no unmeasured interpolation.
    if any(v is None for v in valid):
        extreme_risks = check(0.0, 1.0)
        if any(v > target for v in extreme_risks):
            raise ValueError("No feasible extreme quantile endpoints")
        for side in range(2):
            if valid[side] is None:
                valid[side] = ((0.0, 1.0)[side], extreme_risks[side])
    lo, hi = (v[0] for v in valid)
    if lo > hi:
        raise ValueError("Quantile endpoints reversed")
    lower_risk, upper_risk = check(lo, hi)
    if max(lower_risk, upper_risk) > target:
        raise ValueError("Final calibration recheck failed")
    coverage_bound = 1 - lower_risk - upper_risk
    return dict(
        lower_q=lo,
        upper_q=hi,
        lower_risk=lower_risk,
        upper_risk=upper_risk,
        coverage_lower_bound=coverage_bound,
        target_coverage=1 - alpha,
        coverage_tolerance=coverage_tolerance,
        within_tolerance=coverage_bound <= 1 - alpha + coverage_tolerance,
        trace=trace,
        calibrated_on="calibration split only",
        procedure="tight empirical equal-tail q search; not RCPS",
    )
