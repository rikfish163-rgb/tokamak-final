"""Convex diagnostic of the existing linear risk head, not a new score module."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

RIDGE = .01


def objective(theta, x, y, weights, origin):
    z = x @ theta[:-1] + theta[-1]
    residual = weights * (expit(z) - y)
    delta = theta - origin
    value = weights @ (np.logaddexp(0., z) - y*z) + .5*RIDGE*(delta @ delta)
    gradient = np.r_[x.T @ residual, residual.sum()] + RIDGE*delta
    return float(value), gradient


def fit_readout(x, y, weights, origin):
    x, y, weights, origin = [np.asarray(a, dtype=np.float64) for a in (x,y,weights,origin)]
    if (x.ndim != 2 or not len(x) or y.shape != (len(x),)
            or weights.shape != y.shape or origin.shape != (x.shape[1]+1,)
            or not all(np.isfinite(a).all() for a in (x,y,weights,origin))
            or set(np.unique(y)) != {0.,1.} or not (weights > 0).all()
            or not np.isclose(weights.sum(),1.,atol=1e-12,rtol=0)):
        raise ValueError('finite binary two-class features and normalized positive weights required')
    result = minimize(objective, origin.copy(), args=(x,y,weights,origin), jac=True,
                      method='L-BFGS-B', options=dict(maxiter=500, gtol=1e-9, ftol=1e-14, maxls=50))
    loss, gradient = objective(result.x,x,y,weights,origin)
    norm = float(np.abs(gradient).max())
    if not np.isfinite(loss) or norm >= 1e-6:
        raise RuntimeError(f'linear probe did not converge: {result.message}; gradient={norm}')
    return result.x, dict(converged=True, scipy_success=bool(result.success),
                         message=str(result.message), iterations=int(result.nit),
                         objective_before=objective(origin,x,y,weights,origin)[0],
                         objective_after=loss, gradient_inf=norm,
                         parameter_delta_norm=float(np.linalg.norm(result.x-origin)), ridge=RIDGE)


def endpoint_manifest(shots):
    groups = {label:[] for label in (0,1)}
    if len({s['shot'] for s in shots}) != len(shots):
        raise ValueError('distinct preparation identities required')
    for s in shots:
        for label in groups:
            eligible = np.flatnonzero(s['valid'] & ((s['y'] > .5) == bool(label)))
            if len(eligible):
                ends = eligible[np.linspace(0,len(eligible)-1,min(64,len(eligible)),dtype=int)]
                groups[label].append((int(s['shot']),ends))
    if not all(groups.values()):
        raise ValueError('both actually observed endpoint classes required for this diagnostic')
    return [dict(shot=sid,end=int(e),label=label,weight=.5/len(group)/len(ends))
            for label,group in groups.items() for sid,ends in group for e in ends]
