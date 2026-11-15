"""Historical supervised prediction of signal-gradient utility, not risk scores."""
from collections import Counter

import numpy as np

from .learning import solve_weights


def fit(rows, allowed_ids, *, shuffle=False):
    if (not rows or len(allowed_ids)!=len(set(allowed_ids))
            or set(allowed_ids)!={r['shot'] for r in rows}):
        raise ValueError('all and only declared training identities required')
    labels={}; counts=Counter(r['shot'] for r in rows)
    for r in rows:
        if type(r['is_disrupt']) is not bool or (r['shot'] in labels and labels[r['shot']]!=r['is_disrupt']):
            raise ValueError('consistent boolean shot outcomes required')
        labels[r['shot']]=r['is_disrupt']
    groups=Counter(labels.values())
    if set(groups)!={True,False}:
        raise ValueError('two historical outcome groups required')
    mass=np.array([.5/groups[r['is_disrupt']]/counts[r['shot']] for r in rows])
    x=np.array([r['context'] for r in rows],dtype=np.float64)
    y=np.array([r['risk_alignment'] for r in rows],dtype=np.float64)
    if x.shape!=(len(rows),185) or y.shape!=(len(rows),23) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('185 causal context and23 finite teacher coordinates required')
    order=np.random.default_rng(131).permutation(len(rows)) if shuffle else np.arange(len(rows))
    y=y[order]
    mean=mass@x; scale=np.maximum(np.sqrt(mass@((x-mean)**2)),1e-4)
    z=np.column_stack(((x-mean)/scale,np.ones(len(rows))))
    penalty=np.eye(186);penalty[-1,-1]=0.
    lhs=z.T@(mass[:,None]*z)+penalty;rhs=z.T@(mass[:,None]*y)
    coef=np.linalg.solve(lhs,rhs)
    residual=float(np.max(np.abs(lhs@coef-rhs)))
    if residual>1e-8 or not np.isfinite(coef).all():
        raise AssertionError('ridge normal equation failed')
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficients=coef.tolist(),
                weights=mass.tolist(),allowed_ids=list(allowed_ids),target_mean=(mass@y).tolist(),
                shuffle_order=order.tolist(),ridge=1.,normal_equation_residual=residual)


def predicted_alignment(policy, context):
    x=np.asarray(context,dtype=np.float64)
    if x.shape!=(185,) or not np.isfinite(x).all():
        raise ValueError('one finite causal context required')
    return np.r_[(x-policy['mean'])/policy['scale'],1.]@np.asarray(policy['coefficients'])


def decide(predicted, signal_alignment):
    a,b=np.asarray(predicted),np.asarray(signal_alignment)
    w,receipt=solve_weights(a[None],b[None])
    abstain=receipt['margin']<=1e-8
    # Abstention follows an estimated nonpositive benefit, not a hidden
    # future-label guard. It is recorded as zero update, never a fallback.
    if abstain: w=np.zeros_like(w)
    return w,dict(receipt,abstain=abstain)
