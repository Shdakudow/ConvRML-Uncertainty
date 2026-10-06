"""Independent lower/upper scales with a joint two-tail RCPS confidence budget."""
import math
import numpy as np
from scripts.rcps_bounds import risk_threshold, hb_pvalue


def select_asymmetric_factors(lower_scores, upper_scores, alpha=.1, delta=.05):
    if lower_scores.shape != upper_scores.shape or lower_scores.ndim < 2:
        raise ValueError('Expected matching image-by-pixel lower/upper scores')
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('Invalid risk or confidence budget')
    n = len(lower_scores)
    tail_alpha, tail_delta = alpha/2, delta/2
    threshold = risk_threshold(n, tail_alpha, tail_delta)
    factors = {'empirical': {}, 'rcps_hb': {}}
    tail_losses = {}
    for side, scores in [('lower', lower_scores), ('upper', upper_scores)]:
        # Avoid a full-sized Boolean temporary for disk-backed score arrays.
        if not all(np.isfinite(image).all() for image in scores):
            raise ValueError('Nonfinite calibration scores')
        empirical, certified = np.quantile(scores, [1-tail_alpha, 1-threshold], method='higher')
        factors['empirical'][side] = max(0., float(empirical))
        factors['rcps_hb'][side] = math.ceil(max(0., float(certified))*1e6)/1e6
        for method in factors:
            values = np.array([np.mean(image > factors[method][side]) for image in scores])
            tail_losses[method+'_'+side] = values
            if method == 'rcps_hb' and hb_pvalue(float(values.mean()), n, tail_alpha) >= tail_delta:
                raise RuntimeError('Failed '+side+' RCPS certificate')
    losses = {method: np.array([
        np.mean((lo > factors[method]['lower']) | (hi > factors[method]['upper']))
        for lo, hi in zip(lower_scores, upper_scores)]) for method in factors}
    audit = dict(n_images=n, alpha=alpha, delta=delta, tail_alpha=tail_alpha,
                 tail_delta=tail_delta, maximum_accepted_tail_empirical_risk=threshold,
                 tail_risks={k:float(v.mean()) for k,v in tail_losses.items()},
                 total_risks={k:float(v.mean()) for k,v in losses.items()},
                 confidence_policy='Bonferroni: delta/2 per tail, joint failure <= delta',
                 tail_hb_pvalues={side:hb_pvalue(float(tail_losses['rcps_hb_'+side].mean()),n,tail_alpha)
                                  for side in ('lower','upper')})
    return factors, audit, losses
