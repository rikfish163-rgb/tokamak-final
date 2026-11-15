"""Conditional direct forecasts, never a risk-output correction.

Squared Mahalanobis errors are composite scores, NOT calibrated likelihoods.
No classifier parameters are trained in this feasibility experiment.
"""
import numpy as np
import torch


def shot_weights(ids, labels):
    ids, labels = np.asarray(ids), np.asarray(labels)
    if ids.ndim != 1 or labels.shape != ids.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError('two classes with matching shot identities required')
    weights = np.zeros(len(ids), float)
    for c in (0, 1):
        shots = np.unique(ids[labels == c])
        for sid in shots:
            m = (ids == sid) & (labels == c)
            weights[m] = .5 / len(shots) / m.sum()
    return weights


def misplaced_labels(ids, labels):
    result = np.asarray(labels).copy()
    for sid in np.unique(ids):
        m = np.asarray(ids) == sid
        result[m] = np.roll(result[m], int(m.sum())//2)
    return result


def ridge_fit(x, y, weights, penalty, *, device):
    x, y, weights = [np.asarray(a, dtype=np.float64) for a in (x, y, weights)]
    if (x.ndim != 2 or y.ndim != 2 or len(y) != len(x) or weights.shape != (len(x),)
            or not all(np.isfinite(a).all() for a in (x, y, weights))
            or (weights < 0).any() or weights.sum() <= 0 or penalty <= 0):
        raise ValueError('finite paired matrices, nonnegative weights and positive penalty required')
    weights = weights / weights.sum()
    xx = torch.as_tensor(np.column_stack((x, np.ones(len(x)))), device=device)
    yy = torch.as_tensor(y, device=device); w = torch.as_tensor(weights, device=device)
    gram = xx.T @ (w[:, None]*xx)
    reg = torch.eye(gram.shape[0], device=device, dtype=gram.dtype)*penalty
    reg[-1, -1] = 0.
    b = torch.linalg.solve(gram+reg, xx.T @ (w[:, None]*yy))
    if not torch.isfinite(b).all():
        raise FloatingPointError('nonfinite conditional fit')
    return b.cpu().numpy()


def ridge_predict(x, coefficients):
    return np.column_stack((x, np.ones(len(x)))) @ coefficients


def direct_targets(signal, ends, center, scale, horizon):
    if set(signal) != {'values', 'available'}:
        raise ValueError('only observed signal fields allowed')
    x, a = np.asarray(signal['values']), np.asarray(signal['available'])
    ends = np.asarray(ends, dtype=int)
    if (x.ndim != 2 or x.shape[1] != 23 or a.shape != x.shape or a.dtype != bool
            or ends.ndim != 1 or not len(ends) or ends.min() < 0 or horizon < 1
            or ends.max()+horizon >= len(x)):
        raise ValueError('paired, fully arrived direct targets required')
    a = a & np.isfinite(x)
    q = np.arcsinh((np.where(a, x, center).astype('float64')-center)/scale)
    indices = ends[:, None]+np.arange(1, horizon+1)
    return q[indices]-q[ends, None], a[indices] & a[ends, None]


def fit_metric(target, valid, weights):
    """Common target covariance, 10% diagonal shrinkage, no imputed targets."""
    if target.shape != valid.shape or target.ndim != 3:
        raise ValueError('matching batch/time/channel observations required')
    w = np.asarray(weights, float); w = w/w.sum()
    active = (valid & np.isfinite(target)).all((0, 1))
    mean = np.einsum('n,ntc->c', w, np.where(valid, target, 0.))/target.shape[1]
    centered = np.where(valid, target-mean, 0.)
    cov = np.einsum('n,ntc,ntd->cd', w, centered, centered)/target.shape[1]
    # Exclusion is a floating-point resolution test, not a sensor noise floor.
    active &= np.diag(cov) > 64*np.finfo(float).eps
    if active.sum() < 2:
        raise ValueError('fewer than two complete nonconstant continuous channels')
    cov = .9*cov + .1*np.diag(np.diag(cov))
    cov[~active] = 0.; cov[:, ~active] = 0.
    return cov, active


def maturity_error(prediction, observed, valid, covariance, active):
    """Only observed.shape[1] has arrived; never uses later prediction/target."""
    if (prediction.ndim != 3 or observed.ndim != 3 or observed.shape != valid.shape
            or observed.shape[0] != prediction.shape[0] or observed.shape[2] != prediction.shape[2]
            or not 1 <= observed.shape[1] <= prediction.shape[1] or valid.dtype != bool):
        raise ValueError('matching partial observations inside the forecast horizon required')
    mask = valid & np.isfinite(observed) & np.asarray(active)[None, None]
    residual = prediction[:, :observed.shape[1]] - np.where(mask, observed, 0.)
    # A submatrix inverse correctly marginalizes absent channels; precision
    # slicing would incorrectly condition them to an unobserved value.
    flat_mask = mask.reshape(-1, mask.shape[-1]); flat_r = residual.reshape(flat_mask.shape)
    patterns, inverse = np.unique(flat_mask, axis=0, return_inverse=True)
    error = np.full(len(flat_mask), np.nan)
    for index, use in enumerate(patterns):
        if not use.any():
            continue  # No measured target: absent score, not zero error.
        rows = np.flatnonzero(inverse == index)
        r = flat_r[rows][:, use]; cov = covariance[np.ix_(use, use)]
        error[rows] = np.einsum('ni,ni->n', r, np.linalg.solve(cov, r.T).T)/use.sum()
    error = error.reshape(mask.shape[:2]); count = np.isfinite(error).sum(1)
    result = np.full(len(error), np.nan)
    np.divide(np.nansum(error, axis=1), count, out=result, where=count > 0)
    return result
