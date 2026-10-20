"""Image-wise interval-width/error association, without region metrics.

RGB error is mean_c(abs(pred_c-target_c)), width is mean_c(UB_c-LB_c).
Shapes must agree; misaligned data are rejected rather than center-cropped.
No smoothing in the primary analysis. Optional Gaussian scales are secondary.
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter


def correlation(u, e):
    u = np.asarray(u, dtype=np.float64).ravel()
    e = np.asarray(e, dtype=np.float64).ravel()
    if u.shape != e.shape:
        raise ValueError("Correlation maps have different shapes")
    good = np.isfinite(u) & np.isfinite(e)
    u, e = u[good], e[good]
    if len(u) < 2 or np.ptp(u) == 0 or np.ptp(e) == 0:
        return dict(pearson=None, spearman=None, n_valid_pixels=int(len(u)))
    return dict(
        pearson=float(stats.pearsonr(u, e).statistic),
        spearman=float(stats.spearmanr(u, e).statistic),
        n_valid_pixels=int(len(u)),
    )


def maps(pred, target, lower, upper):
    arrays = [np.asarray(x, dtype=np.float32) for x in [pred, target, lower, upper]]
    if (
        any(x.shape != arrays[0].shape for x in arrays)
        or arrays[0].ndim != 3
        or arrays[0].shape[0] != 3
    ):
        raise ValueError("Expected matching RGB CHW images")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite predictions")
    p, t, lo, hi = arrays
    return (hi - lo).mean(0), np.abs(p - t).mean(0)


def analyze(pred, target, lower, upper, sigmas=(0, 0.5, 1, 2, 4)):
    u, e = maps(pred, target, lower, upper)
    result = {}
    for sigma in sigmas:
        us, es = (
            (u, e)
            if sigma == 0
            else (gaussian_filter(u, sigma=sigma), gaussian_filter(e, sigma=sigma))
        )
        result[str(sigma)] = {
            **correlation(us, es),
            "mean_width": float(us.mean()),
            "mae": float(es.mean()),
        }
    return result


def describe(values, seed=20261019):
    a = np.array(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    if not len(a):
        return dict(mean=None, std=None, median=None, ci95=None, n_valid=0)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(1)
    return dict(
        mean=float(a.mean()),
        std=float(a.std()),
        median=float(np.median(a)),
        ci95=np.quantile(means, [0.025, 0.975]).tolist(),
        n_valid=len(a),
    )
