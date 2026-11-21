"""Signal-only comparability checks, not a physical equivalence certificate.

Current drift describes the observed response. It is NOT a planned ramp phase
or a safety label. No feature is normalized away by these diagnostics.
"""
import numpy as np

from ..io23 import RAW23


def qa_geometry_check(signal, *, radius, major_radius):
    """Compare stored qa to 2*pi*a^2*Bt/(R*Ip_A), without changing inputs.

    Nominal geometry is from paper_131 sections 3.1/4.1, not per-shot
    equilibrium reconstruction. An algebraic match does not establish q95,
    instantaneous plasma shape, or the unknown extraction implementation.
    """
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only values and available required')
    if not 0 < radius < major_radius or not np.isfinite([radius,major_radius]).all():
        raise ValueError('finite positive nominal geometry required')
    x, mask = signal['values'], signal['available']
    c = [RAW23.index(k) for k in ('ip','Bt','qa_proxy')]
    values = np.asarray(x[:,c], float)
    valid = mask[:,c].all(1) & np.isfinite(values).all(1) & (values[:,0]!=0) & (values[:,1]!=0)
    coefficient = 2*np.pi*radius**2/major_radius
    v = values[valid]
    result = dict(valid_points=len(v),excluded_points=len(x)-len(v),coefficient=coefficient,
        interpretation='empirical_formula_consistency_not_q95_or_extractor_certification')
    if not len(v):
        result['relative_error_quantiles_50_95_99_100'] = None
        return result
    expected = coefficient*v[:,1]/v[:,0]
    error = np.abs(v[:,2]/expected-1)
    result['relative_error_quantiles_50_95_99_100'] = np.quantile(error,[.5,.95,.99,1]).tolist()
    return result


def describe_window(signal, end):
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only values and available required')
    x, mask = signal['values'], signal['available']
    if (x.ndim != 2 or x.shape[1] != 23 or mask.shape != x.shape
            or mask.dtype != bool or type(end) is not int or not 127 <= end < len(x)):
        raise ValueError('full W128 window and matching boolean availability required')
    x, mask = x[end-127:end+1].astype(np.float64), mask[end-127:end+1]
    mask = mask & np.isfinite(x)
    i, b, q = [RAW23.index(k) for k in ('ip', 'Bt', 'qa_proxy')]
    if min(int(mask[:, i].sum()), int(mask[:, b].sum())) < 64:
        return dict(excluded='insufficient_current_or_field')
    if min(int(mask[:32, i].sum()), int(mask[-32:, i].sum())) < 16:
        return dict(excluded='insufficient_endpoint_current')
    ip = float(np.median(np.abs(x[mask[:, i], i])))
    bt = float(np.median(np.abs(x[mask[:, b], b])))
    if min(ip, bt) <= 0:
        return dict(excluded='nonpositive_magnitude')
    early = np.median(np.abs(x[:32, i][mask[:32, i]]))
    late = np.median(np.abs(x[-32:, i][mask[-32:, i]]))
    qa = float(np.median(x[mask[:, q], q])) if mask[:, q].sum() >= 64 else None
    return dict(ip_A=ip, Bt_magnitude=bt,
                relative_current_change=float((late-early)/ip),
                current_sign=float(np.sign(np.median(x[mask[:, i], i]))),
                field_sign=float(np.sign(np.median(x[mask[:, b], b]))),
                qa_raw=qa, qa_Ip_over_Bt=None if qa is None else qa*ip/bt)


def caliper_match(source, target, relative_tolerance, *, drift_tolerance):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3
            or not len(source) or not np.isfinite(source).all() or not np.isfinite(target).all()
            or not (source[:, :2] > 0).all() or not (target[:, :2] > 0).all()
            or not 0 < relative_tolerance < 1
            or (drift_tolerance is not None and not 0 < drift_tolerance < 1)):
        raise ValueError('positive current/field, finite descriptors and valid calipers required')
    counts, nearest = [], []
    for t in target:
        eligible = (np.abs(source[:, :2]/t[:2]-1) <= relative_tolerance).all(1)
        if drift_tolerance is not None:
            eligible &= np.abs(source[:, 2]-t[2]) <= drift_tolerance
        candidates = np.flatnonzero(eligible)
        counts.append(len(candidates))
        if len(candidates):
            distance = np.square(np.log(source[candidates, :2]/t[:2])).sum(1)
            if drift_tolerance is not None:
                distance += np.square((source[candidates, 2]-t[2])/drift_tolerance)
            nearest.append(int(candidates[np.argmin(distance)]))
        else:
            nearest.append(-1)
    return np.asarray(counts, int), np.asarray(nearest, int)


def weighted_covariance(h, weights):
    h, weights = np.asarray(h, float), np.asarray(weights, float)
    if (h.ndim != 2 or len(h) != len(weights) or weights.ndim != 1
            or not np.isfinite(h).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('finite matrix and nonnegative nonzero weights required')
    weights = weights/weights.sum()
    centered = h-weights@h
    return (centered*weights[:, None]).T@centered


def covariance_decomposition(h, weights, groups):
    """Population law Cov(h)=E[Cov(h|g)]+Cov(E[h|g]); exact weighted identity."""
    h, weights, groups = np.asarray(h, float), np.asarray(weights, float), np.asarray(groups)
    total = weighted_covariance(h, weights)
    if groups.shape != (len(h),):
        raise ValueError('one group per state required')
    weights = weights/weights.sum()
    within, between = np.zeros_like(total), np.zeros_like(total)
    mean = weights@h
    for group in np.unique(groups):
        ix = groups == group
        mass = weights[ix].sum()
        if mass == 0:
            continue
        mu = weights[ix]@h[ix]/mass
        within += mass*weighted_covariance(h[ix], weights[ix])
        between += mass*np.outer(mu-mean, mu-mean)
    return total, within, between
