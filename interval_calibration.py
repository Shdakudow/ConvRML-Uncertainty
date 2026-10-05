"""Empirical calibration of image intervals. Pixels are not independent samples.

Factors target mean image pixel/channel miscoverage, with equal-sized images.
No finite-sample population guarantee is claimed by this empirical fitter.
"""
import numpy as np


def components(r, lo, hi, epsilon=1e-3):
    return np.maximum(r-lo, epsilon), np.maximum(hi-r, epsilon)


def fit(x, r, lo, hi, alpha=0.1, epsilon=1e-3):
    if not 0 < alpha < 1 or epsilon <= 0:
        raise ValueError('Invalid alpha or epsilon')
    if not (x.shape == r.shape == lo.shape == hi.shape) or x.ndim != 4:
        raise ValueError('Expected matching NCHW arrays')
    if len(x) == 0 or not all(np.isfinite(a).all() for a in (x,r,lo,hi)):
        raise ValueError('Empty or nonfinite calibration data')
    wl, wu = components(r,lo,hi,epsilon)
    def threshold(score, probability):
        return max(0., float(np.quantile(score, probability, method='higher')))
    sl, su = (r-x)/wl, (x-r)/wu
    return {'alpha':alpha, 'epsilon':epsilon, 'calibration_images':len(x),
            'scope':'empirical mean image pixel/channel coverage; no population certificate',
            'additive':threshold(np.maximum(lo-x,x-hi),1-alpha),
            'shared':threshold(np.maximum(sl,su),1-alpha),
            'lower':threshold(sl,1-alpha/2),
            'upper':threshold(su,1-alpha/2)}


def apply(r, lo, hi, parameters, method):
    if method == 'raw': return lo,hi
    if method == 'additive':
        t=parameters['additive']; return lo-t,hi+t
    wl,wu=components(r,lo,hi,parameters['epsilon'])
    if method == 'repaired': a=b=1.
    elif method == 'shared': a=b=parameters['shared']
    elif method == 'asymmetric': a,b=parameters['lower'],parameters['upper']
    else: raise ValueError(method)
    return r-a*wl,r+b*wu


def metrics(x, lo, hi):
    covered=((lo<=x)&(x<=hi)).mean(axis=(1,2,3))
    return {'coverage':float(covered.mean()),
            'lower_tail':float((x<lo).mean()),'upper_tail':float((x>hi).mean()),
            'mean_width':float(np.maximum(hi-lo,0).mean()),
            'crossing':float((lo>hi).mean()),
            'image_coverage_p10':float(np.quantile(covered,.1)),
            'image_coverage_median':float(np.median(covered)),
            'image_coverage_se':float(covered.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None}
