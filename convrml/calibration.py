"""Paper-based calibration of frozen models on a separate held-out split.

Baselines use image-level Hoeffding--Bentkus RCPS with separate endpoint scales.
QUTCC uses Algorithm 1 to select lower/upper quantile inputs. Neither procedure
updates model weights or forces an exact coverage fraction on future test images.
"""

import math

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom


class CalibrationFailure(ValueError):
    """A frozen model or calibration set cannot satisfy the requested procedure."""


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
        raise CalibrationFailure(
            "Insufficient calibration images for a finite HB certificate"
        )
    lo, hi = 0.0, alpha
    for _ in range(80):
        mid = (lo + hi) / 2
        if hb_pvalue(mid, n, alpha) < delta:
            lo = mid
        else:
            hi = mid
    return max(0.0, lo - 1e-10)


def rcps_scales(lower_scores, upper_scores, alpha=0.1, delta=0.05):
    """Fit separate lower/upper scales with image-level RCPS-HB certificates.

    Each row is one image's pixel scores. The statistical sample count is the
    number of images, not pixels. Alpha and delta are split equally between tails.
    """
    if lower_scores.shape != upper_scores.shape or lower_scores.ndim < 2:
        raise ValueError("Expected matching image-by-pixel lower/upper scores")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("Invalid risk or confidence budget")
    n = len(lower_scores)
    tail_alpha, tail_delta = alpha / 2, delta / 2
    threshold = risk_threshold(n, tail_alpha, tail_delta)
    scales, losses = {}, {}
    for side, scores in (("lower", lower_scores), ("upper", upper_scores)):
        if not all(np.isfinite(image).all() for image in scores):
            raise ValueError("Nonfinite calibration scores")
        selected = np.quantile(scores, 1 - threshold, method="higher")
        scales[side] = math.ceil(max(0.0, float(selected)) * 1e6) / 1e6
        losses[side] = np.array([np.mean(image > scales[side]) for image in scores])
        if hb_pvalue(float(losses[side].mean()), n, tail_alpha) >= tail_delta:
            raise CalibrationFailure("Failed " + side + " RCPS certificate")
    total_risk = float(
        np.mean(
            [
                np.mean((lo > scales["lower"]) | (hi > scales["upper"]))
                for lo, hi in zip(lower_scores, upper_scores)
            ]
        )
    )
    certificate = dict(
        n_images=n,
        alpha=alpha,
        delta=delta,
        tail_alpha=tail_alpha,
        tail_delta=tail_delta,
        maximum_accepted_tail_empirical_risk=threshold,
        tail_risks={side: float(value.mean()) for side, value in losses.items()},
        total_risk=total_risk,
        confidence_policy="Bonferroni: delta/2 per tail, joint failure <= delta",
        tail_hb_pvalues={
            side: hb_pvalue(float(value.mean()), n, tail_alpha)
            for side, value in losses.items()
        },
    )
    return dict(
        scales=scales,
        certificate=certificate,
        delta=delta,
        procedure="asymmetric RCPS-HB",
    )


def tail_budget(alpha, n):
    if n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid calibration size or alpha")
    budget = (alpha - (1 - alpha) / n) / 2
    if not 0 < budget < 0.5:
        raise ValueError("Calibration set too small for alpha")
    return budget


def paper_quantiles(risk, n, alpha=0.1, tol=1e-3, max_iterations=20):
    """QUTCC Algorithm 1: binary-search global q endpoints with finite-N correction.

    ``risk(lo, hi)`` returns lower/upper miss fractions over calibration pixels.
    The search assumes monotone quantile responses, as in the original algorithm.
    """
    if tol <= 0 or max_iterations < 1:
        raise ValueError("Quantile tolerance and iteration count must be positive")
    target = tail_budget(alpha, n)
    trace = []

    def check(lo, hi):
        low, high = risk(lo, hi)
        if not 0 <= low <= 1 or not 0 <= high <= 1:
            raise ValueError("Invalid calibration risk")
        trace.append(dict(lower_q=lo, upper_q=hi, lower_risk=low, upper_risk=high))
        return low, high

    low, high = check(0.0, 1.0)
    if low > target or high > target:
        raise CalibrationFailure(
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
        raise CalibrationFailure("Calibrated quantile levels reversed")
    rl, ru = check(validlo, validhi)
    if rl > target or ru > target:
        raise CalibrationFailure("Final bounds failed calibration recheck")
    return dict(
        lower_q=validlo,
        upper_q=validhi,
        alpha=alpha,
        per_tail_target=target,
        n_images=n,
        tolerance=tol,
        trace=trace,
        procedure="QUTCC paper Algorithm 1 calibrated quantile search",
        assumption="Quantile response should be monotone; shared weights do not strictly enforce it",
        calibrated_on="calibration split only",
    )
