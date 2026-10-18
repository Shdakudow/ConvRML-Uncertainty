"""QUTCC Algorithm 1: calibrate shared quantile inputs, not output scale factors.

Follow Algorithm 1 and reference evaluation/calibration.py for finite-size
correction: (alpha - (1-alpha)/N_images)/2. The paragraph before Algorithm 1
uses a different expression; that discrepancy is recorded in the protocol.
No Hoeffding-Bentkus UCB or delta is substituted into this paper algorithm.
"""

class CalibrationFailure(RuntimeError):
    """A frozen checkpoint cannot meet the requested empirical tail budgets."""
    pass

def tail_budget(alpha,n):
    budget=(alpha-(1-alpha)/n)/2
    if not 0<budget<.5:raise ValueError('Calibration set too small for alpha')
    return budget

def search_quantiles(risk,n,alpha=.1,tol=1e-3,max_iterations=20):
    target=tail_budget(alpha,n);trace=[]
    def check(lo,hi):
        low,high=risk(lo,hi)
        if not 0<=low<=1 or not 0<=high<=1:raise ValueError('Invalid calibration risk')
        trace.append(dict(lower_q=lo,upper_q=hi,lower_risk=low,upper_risk=high));return low,high
    # Explicitly check feasibility; never invent a passing quantile.
    low,high=check(0.,1.)
    if low>target or high>target:raise CalibrationFailure('No feasible extreme quantile bounds; inspect model fit/crossing')
    lmin,lmax,umin,umax=0.,1.,0.,1.;validlo,validhi=0.,1.
    for _ in range(max_iterations):
        lo,hi=(lmin+lmax)/2,(umin+umax)/2;rl,ru=check(lo,hi)
        if rl>target:lmax=lo
        else:lmin=lo;validlo=lo
        if ru>target:umin=hi
        else:umax=hi;validhi=hi
        if lmax-lmin<tol and umax-umin<tol:break
    if validlo>validhi:raise CalibrationFailure('Calibrated quantile levels reversed')
    rl,ru=check(validlo,validhi)
    if rl>target or ru>target:raise CalibrationFailure('Final bounds failed calibration recheck')
    return dict(lower_q=validlo,upper_q=validhi,alpha=alpha,per_tail_target=target,n_images=n,tolerance=tol,trace=trace,
        procedure='paper Algorithm 1 adjusted empirical tail-risk binary search',assumption='Quantile response should be monotone; shared weights do not strictly enforce it',calibrated_on='calibration split only')
