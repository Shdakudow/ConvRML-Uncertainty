"""Reference calibration formulas used by the recorded comparisons.

RCPS: image-level Hoeffding--Bentkus bounds, independent lower/upper scales,
alpha=0.10, delta=0.05, half budgets per tail. QUTCC uses its paper Algorithm 1
instead; that adjusted empirical search is not an HB certificate. Formula bodies
are preserved from the captured experiment source.
"""

import math
import numpy as np
from scipy.stats import binom
from scipy.special import xlogy


def hb_pvalue(risk, n, alpha):
    if not 0 <= risk <= 1 or n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid risk, sample count or target")
    if risk >= alpha:
        return 1.0
    kl = xlogy(risk, risk / alpha) + xlogy(1 - risk, (1 - risk) / (1 - alpha))
    hoeffding = math.exp(-n * float(kl))
    bentkus = math.e * float(binom.cdf(math.ceil(n * risk), n, alpha))
    return min(1.0, hoeffding, bentkus)


def risk_threshold(n, alpha=0.1, delta=0.05):
    if not 0 < delta < 1:
        raise ValueError("Invalid confidence failure probability")
    if hb_pvalue(0.0, n, alpha) >= delta:
        raise ValueError("Insufficient calibration images for a finite HB certificate")
    lo, hi = 0.0, alpha
    for _ in range(80):
        mid = (lo + hi) / 2
        if hb_pvalue(mid, n, alpha) < delta:
            lo = mid
        else:
            hi = mid
    return max(0.0, lo - 1e-10)


def select_asymmetric_factors(lower_scores, upper_scores, alpha=0.1, delta=0.05):
    if lower_scores.shape != upper_scores.shape or lower_scores.ndim < 2:
        raise ValueError("Expected matching image-by-pixel lower/upper scores")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("Invalid risk or confidence budget")
    n = len(lower_scores)
    tail_alpha, tail_delta = alpha / 2, delta / 2
    threshold = risk_threshold(n, tail_alpha, tail_delta)
    factors = {"empirical": {}, "rcps_hb": {}}
    tail_losses = {}
    for side, scores in [("lower", lower_scores), ("upper", upper_scores)]:
        # Avoid a full-sized Boolean temporary for disk-backed score arrays.
        if not all(np.isfinite(image).all() for image in scores):
            raise ValueError("Nonfinite calibration scores")
        empirical, certified = np.quantile(
            scores, [1 - tail_alpha, 1 - threshold], method="higher"
        )
        factors["empirical"][side] = max(0.0, float(empirical))
        factors["rcps_hb"][side] = math.ceil(max(0.0, float(certified)) * 1e6) / 1e6
        for method in factors:
            values = np.array(
                [np.mean(image > factors[method][side]) for image in scores]
            )
            tail_losses[method + "_" + side] = values
            if (
                method == "rcps_hb"
                and hb_pvalue(float(values.mean()), n, tail_alpha) >= tail_delta
            ):
                raise RuntimeError("Failed " + side + " RCPS certificate")
    losses = {
        method: np.array(
            [
                np.mean(
                    (lo > factors[method]["lower"]) | (hi > factors[method]["upper"])
                )
                for lo, hi in zip(lower_scores, upper_scores)
            ]
        )
        for method in factors
    }
    audit = dict(
        n_images=n,
        alpha=alpha,
        delta=delta,
        tail_alpha=tail_alpha,
        tail_delta=tail_delta,
        maximum_accepted_tail_empirical_risk=threshold,
        tail_risks={k: float(v.mean()) for k, v in tail_losses.items()},
        total_risks={k: float(v.mean()) for k, v in losses.items()},
        confidence_policy="Bonferroni: delta/2 per tail, joint failure <= delta",
        tail_hb_pvalues={
            side: hb_pvalue(float(tail_losses["rcps_hb_" + side].mean()), n, tail_alpha)
            for side in ("lower", "upper")
        },
    )
    return factors, audit, losses


class PaperCalibrationFailure(RuntimeError):
    """A frozen checkpoint cannot meet the requested empirical tail budgets."""

    pass


def tail_budget(alpha, n):
    budget = (alpha - (1 - alpha) / n) / 2
    if not 0 < budget < 0.5:
        raise ValueError("Calibration set too small for alpha")
    return budget


def paper_quantiles(risk, n, alpha=0.1, tol=1e-3, max_iterations=20):
    target = tail_budget(alpha, n)
    trace = []

    def check(lo, hi):
        low, high = risk(lo, hi)
        if not 0 <= low <= 1 or not 0 <= high <= 1:
            raise ValueError("Invalid calibration risk")
        trace.append(dict(lower_q=lo, upper_q=hi, lower_risk=low, upper_risk=high))
        return low, high

    # Explicitly check feasibility; never invent a passing quantile.
    low, high = check(0.0, 1.0)
    if low > target or high > target:
        raise PaperCalibrationFailure(
            "No feasible extreme quantile bounds; inspect model fit/crossing"
        )
    lmin, lmax, umin, umax = 0.0, 1.0, 0.0, 1.0
    validlo, validhi = 0.0, 1.0
    for _ in range(max_iterations):
        lo, hi = (lmin + lmax) / 2, (umin + umax) / 2
        rl, ru = check(lo, hi)
        if rl > target:
            lmax = lo
        else:
            lmin = lo
            validlo = lo
        if ru > target:
            umin = hi
        else:
            umax = hi
            validhi = hi
        if lmax - lmin < tol and umax - umin < tol:
            break
    if validlo > validhi:
        raise PaperCalibrationFailure("Calibrated quantile levels reversed")
    rl, ru = check(validlo, validhi)
    if rl > target or ru > target:
        raise PaperCalibrationFailure("Final bounds failed calibration recheck")
    return dict(
        lower_q=validlo,
        upper_q=validhi,
        alpha=alpha,
        per_tail_target=target,
        n_images=n,
        tolerance=tol,
        trace=trace,
        procedure="paper Algorithm 1 adjusted empirical tail-risk binary search",
        assumption="Quantile response should be monotone; shared weights do not strictly enforce it",
        calibrated_on="calibration split only",
    )
