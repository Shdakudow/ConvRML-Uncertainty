"""Image-level Hoeffding--Bentkus RCPS for nested scalar interval scaling.

Bates et al., arXiv:2101.02703, Propositions 3--4 and UCB calibration.
A calibration unit is a whole image, with loss its uncovered RGB fraction.
"""
import math
import numpy as np
from scipy.stats import binom
from scipy.special import xlogy


def hb_pvalue(risk, n, alpha):
    if not 0 <= risk <= 1 or n < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid risk, sample count or target')
    if risk >= alpha:
        return 1.
    kl = xlogy(risk, risk/alpha) + xlogy(1-risk, (1-risk)/(1-alpha))
    hoeffding = math.exp(-n*float(kl))
    bentkus = math.e*float(binom.cdf(math.ceil(n*risk), n, alpha))
    return min(1., hoeffding, bentkus)


def risk_threshold(n, alpha=.1, delta=.05):
    if not 0 < delta < 1:
        raise ValueError('Invalid confidence failure probability')
    if hb_pvalue(0., n, alpha) >= delta:
        raise ValueError('Insufficient calibration images for a finite HB certificate')
    lo, hi = 0., alpha
    for _ in range(80):
        mid = (lo+hi)/2
        if hb_pvalue(mid, n, alpha) < delta:
            lo = mid
        else:
            hi = mid
    return max(0., lo-1e-10)  # Strict rejection, including binomial step boundaries.


def select_factors(scores, alpha=.1, delta=.05):
    """Equal-size images only. Preserve scores for image-level loss auditing.

    A fixed 1e-6 lambda lattice and upward rounding implement a conservative
    discrete monotone selection. No independence assumption across pixels.
    """
    if scores.ndim < 2 or len(scores) < 2:
        raise ValueError('Expected image-by-pixel scores')
    n = len(scores)
    threshold = risk_threshold(n, alpha, delta)
    empirical, rcps = np.quantile(scores, [1-alpha, 1-threshold], method='higher')
    factors = dict(empirical=max(0., float(empirical)),
                   rcps_hb=math.ceil(max(0., float(rcps))*1e6)/1e6)
    risks = {}
    losses = {}
    for name, factor in factors.items():
        losses[name] = np.array([np.mean(image > factor) for image in scores], dtype=np.float64)
        risks[name] = float(losses[name].mean())
    if hb_pvalue(risks['rcps_hb'], n, alpha) >= delta:
        raise RuntimeError('Selected interval fails HB certificate')
    if factors['rcps_hb'] < factors['empirical']:
        raise RuntimeError('RCPS unexpectedly narrower than empirical calibration')
    return factors, dict(n_images=n, alpha=alpha, delta=delta,
                         maximum_accepted_empirical_risk=threshold,
                         calibration_risks=risks,
                         hb_pvalues={k: hb_pvalue(v, n, alpha) for k,v in risks.items()}), losses
