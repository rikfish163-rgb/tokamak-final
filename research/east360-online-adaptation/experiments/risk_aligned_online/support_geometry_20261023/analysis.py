"""Descriptive support geometry; no detector, label inference, or online gate."""
import numpy as np


def descriptor(x, mask, center, scale):
    x=np.asarray(x);mask=np.asarray(mask)&np.isfinite(x)
    if x.shape!=mask.shape or x.shape[1:]!=(23,) or np.any(np.asarray(scale)<=0):
        raise ValueError('observed T x23 and positive source scales required')
    z=np.arcsinh((np.where(mask,x,center)-center)/scale)
    median=np.zeros(23);spread=np.zeros(23)
    for c in range(23):
        if mask[:,c].any():
            q=np.quantile(z[mask[:,c],c],[.25,.5,.75]);median[c]=q[1];spread[c]=q[2]-q[0]
    return np.r_[median,spread,mask.mean(0)]


def covariance(x, weights):
    x=np.asarray(x,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or w.shape!=(len(x),) or len(x)<2 or not np.isfinite(x).all() or np.any(w<=0):
        raise ValueError('finite reference rows and positive shot-balanced weights required')
    w=w/w.sum();mean=(x*w[:,None]).sum(0);r=x-mean
    cov=(r*w[:,None]).T@r
    # Fixed descriptive regularizer, not a tuned detection threshold.
    # 1e-12 makes a constant reference invertible in float64; exact degeneracy
    # is unit-tested and never interpreted as a calibrated probability.
    isotropic=max(float(np.trace(cov)/x.shape[1]),1e-12)
    return mean,.5*cov+.5*isotropic*np.eye(x.shape[1])


def nearest_sq(query, reference, cov):
    chol=np.linalg.cholesky(cov)
    q=np.linalg.solve(chol,np.asarray(query,dtype=np.float64).T).T
    r=np.linalg.solve(chol,np.asarray(reference,dtype=np.float64).T).T
    return ((q[:,None]-r[None])**2).sum(-1).min(-1)


def boundary_shifts(x, mask, low, high, center, scale):
    x=np.asarray(x);mask=np.asarray(mask)&np.isfinite(x)
    d=descriptor(x,mask,center,scale)[:23];shift=np.zeros(23)
    known=np.isfinite(low)&np.isfinite(high)&mask.any(0)
    shift[known]=np.clip(d[known],np.asarray(low)[known],np.asarray(high)[known])-d[known]
    z=np.arcsinh((np.where(mask,x,center)-center)/scale)
    probes=np.repeat(x[None],24,axis=0)
    for c in range(23):
        if shift[c]:
            probes[c,mask[:,c],c]=center[c]+scale[c]*np.sinh(z[mask[:,c],c]+shift[c])
            probes[-1,mask[:,c],c]=probes[c,mask[:,c],c]
    if not np.isfinite(probes).all():raise FloatingPointError('nonfinite diagnostic intervention')
    return probes,shift
