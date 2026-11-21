"""Fast ridge state learned from already observed excursion/return pairs."""

import numpy as np
import torch


FIRST, STRIDE, HISTORY, RIDGE, FLOOR, CAP = 135, 32, 16, .50, .02, .35


def rolling_block(level, observed, stride):
    n, d = level.shape
    start = np.maximum(np.arange(n)+1-stride, 0)
    sum_x = np.vstack((np.zeros((1,d)), np.cumsum(np.where(observed,level,0.),axis=0)))
    sum_a = np.vstack((np.zeros((1,d),dtype=np.int64), np.cumsum(observed,axis=0)))
    count = sum_a[1:] - sum_a[start]
    return ((sum_x[1:] - sum_x[start]) / np.maximum(count,1),
            (count >= (stride+1)//2) & (np.arange(n)[:,None] >= stride-1))


def recovery_field(level, available, output, *, first=FIRST, stride=STRIDE):
    """Return recent normalized-value writes, without ever using outcome data."""
    level = np.asarray(level,dtype=np.float64)
    n,d = level.shape
    observed = available & np.isfinite(level)
    avg, valid = rolling_block(level,observed,stride)
    closes = np.arange(first%stride,n,stride)
    closes = closes[closes>=stride-1]
    b,e,r = (avg[k] for k in (closes[:-2],closes[1:-1],closes[2:]))
    common = valid[closes[:-2]] & valid[closes[1:-1]] & valid[closes[2:]]
    u,v = np.where(common,e-b,0.),np.where(common,r-b,0.)
    energy = np.linalg.norm(u,axis=1)
    useful = energy > 0.
    ratio = np.divide(np.linalg.norm(v,axis=1),energy,out=np.ones_like(energy),where=useful)
    recovered = np.clip(1.-ratio,0.,1.)
    x = (u / np.maximum(energy,FLOOR)[:,None])[useful]
    y = x * recovered[useful,None] * output
    when = closes[2:][useful]
    state = RIDGE * np.eye(d)
    actual,shuffled = (np.zeros((n,d)) for _ in range(2))
    predictions = []

    def estimate(idx, coef):
        support = valid[idx] & valid[idx-stride]
        step = np.where(support,avg[idx]-avg[idx-stride],0.)
        sign = np.sign(step)
        return tuple(sign * np.clip(sign*z,0.,np.abs(step)) * support * output
                     for z in np.split(step@coef,2,axis=1))

    for m in closes[(closes>=first) & (closes+1<n)]:
        hi = np.searchsorted(when,m,side="right")
        if hi < 2:
            continue
        xx,yy = (z[max(0,hi-HISTORY):hi] for z in (x,y))
        if not yy.any():
            continue
        coef = np.linalg.solve(state + xx.T@xx, xx.T@np.concatenate((yy,np.roll(yy,1,axis=0)),axis=1))
        now = np.arange(m+1,min(n,m+stride+1))
        a,w = estimate(now,coef)
        actual[now],shuffled[now] = -a,-w
        predictions.append((int(m),*estimate(np.array([m]),coef)))
    # Check future observations only against predictions already computed above.
    errors = []
    for m,a,w in predictions:
        j = np.searchsorted(closes[2:],m+stride)
        if j == len(u) or not useful[j]:
            continue
        keep = common[j] & output
        if keep.any():
            target = u[j,keep] * recovered[j]
            errors.append([float(np.square(z-target).mean()) for z in (a[0,keep],w[0,keep],0.)])
    return actual,shuffled,dict(supported_triples=len(when),
        recovery_triples=int((recovered[useful]>0.).sum()), release_fits=len(predictions),
        forecast_blocks=len(errors), forecast_mse=np.mean(errors,axis=0).tolist() if errors else None)


def value_bias(weight, observed, shift):
    """Actual embedding delta on the latest four patches, from observed values."""
    n,d = shift.shape
    change = (observed * shift[:,None,:]).reshape(n,4,8,d)
    return torch.einsum("bpqc,kqc->bpk",change,weight)


def matched_bias(actual, wrong, cap=CAP):
    a,b = (x.norm(dim=(1,2)) for x in (actual,wrong))
    target = torch.where((a>1e-9) & (b>1e-9),a.clamp(max=cap),0.)
    return (actual*(target/a.clamp_min(1e-9))[:,None,None],
            wrong*(target/b.clamp_min(1e-9))[:,None,None])
