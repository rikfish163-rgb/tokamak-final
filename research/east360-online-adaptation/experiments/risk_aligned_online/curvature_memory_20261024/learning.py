"""Historical Bernoulli GGN preconditions encoder updates; no output add-on.

The curvature is model-expected, not the observed-label empirical Fisher.
It is estimated only on the authorized preparation inputs. Current signals
route historical directions; current outcomes are never accepted here.
"""
import math

import numpy as np
import torch

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder
from ..history_tangent_20261023.learning import logit
from ..trajectory_memory_20261024.learning import (
    observed_window, describe, distances, select_weights, combine_step,
)

ARMS = ('raw', 'full', 'diagonal', 'level', 'reverse', 'uniform', 'raw_kl')


def bernoulli_rows(j, p, q):
    if (j.ndim != 2 or min(j.shape) < 1 or p.shape != (len(j),) or q.shape != p.shape
            or not all(torch.isfinite(v).all() for v in (j,p,q))
            or (p<0).any() or (p>1).any() or (q<0).any()
            or abs(float(q.sum())-1.) > 1e-10):
        raise ValueError('finite Jacobians, Bernoulli probabilities and normalized input weights required')
    return (q*p*(1-p)).sqrt()[:,None]*j


def solve_directions(g, u):
    if (g.ndim != 2 or u.ndim != 2 or min(g.shape)<1 or min(u.shape)<1
            or g.shape[1] != u.shape[1] or g.device != u.device
            or not torch.isfinite(g).all() or not torch.isfinite(u).all()):
        raise ValueError('matching finite historical gradients and curvature rows required')
    g,u=g.double(),u.double()
    gram=u@u.T
    largest=float(torch.linalg.eigvalsh(gram)[-1])
    if largest<=0:raise ValueError('zero curvature has no identifiable damping scale')
    damping=.01*largest
    system=gram+damping*torch.eye(len(u),device=u.device,dtype=u.dtype)
    full=(g-torch.linalg.solve(system,u@g.T).T@u)/damping
    diagonal=g/(u.square().sum(0)+damping)
    residual=full@u.T@u+damping*full-g
    relative=float(residual.norm()/g.norm()) if float(g.norm()) else 0.
    if relative>1e-10:raise ArithmeticError(f'dual solve residual {relative}')
    return full,diagonal,dict(damping=damping,largest_eigenvalue=largest,
        relative_residual=relative,regularized_condition_number_bound=101.)


def functional_match(raw, full, u, length):
    if (raw.ndim!=1 or full.shape!=raw.shape or u.ndim!=2 or u.shape[1]!=len(raw)
            or not all(torch.isfinite(v).all() for v in (raw,full,u))
            or not math.isfinite(length) or length<0):
        raise ValueError('finite matching directions, curvature and nonnegative length required')
    rn=float(raw.double().norm());fn=float(full.double().norm())
    if rn==0 and fn==0:return 0.
    if rn==0 or fn==0:raise ValueError('one direction is zero: functional match unidentified')
    rq=float((u@(raw/rn)).double().square().sum())
    fq=float((u@(full/fn)).double().square().sum())
    if rq==0:raise ValueError('zero raw functional sensitivity: matched length unidentified')
    return length*math.sqrt(fq/rq)


def route(query, bank, mode):
    if mode=='uniform':
        d=np.full(len(bank['shots']),np.inf)
        for sid in np.unique(bank['shots']):
            ids=np.flatnonzero(bank['shots']==sid);d[ids[len(ids)//2]]=0.
    else:d=distances(query,bank,mode)
    selected,w=select_weights(d,bank['shots'],bank['labels'],bank['temperature'])
    return selected,w,d[selected]


def update_all(model, signal, count, bank, device='cuda'):
    x,m=observed_window(signal,count)
    query=describe(x,m,bank['center'],bank['scale'],bank['spread'])
    routed={a:route(query,bank,a) for a in ('trajectory','level','reverse','uniform')}
    parent=initial_encoder(model);states={};receipts={};length=0.
    for arm in ARMS:
        route_name=arm if arm in ('level','reverse','uniform') else 'trajectory'
        ix,w,d=routed[route_name]
        if not len(ix):
            states[arm]=initial_encoder(model)
            r=dict(gradient_norm=0.,requested_step_norm=0.,step_norm=0.,reason='insufficient_joint_observations')
        else:
            kind='gradients' if arm in ('raw','raw_kl') else ('diagonal' if arm=='diagonal' else 'full')
            weights=torch.as_tensor(w,device=device)
            wanted=None if arm=='raw' else length
            if arm=='raw_kl':
                raw=weights.to(bank['gradients'])@bank['gradients'][ix]
                full=weights.to(bank['full'])@bank['full'][ix]
                wanted=functional_match(raw,full,bank['u'],length)
            states[arm],r=combine_step(parent,bank[kind][ix],weights,learning_rate=.01,length=wanted)
            if arm=='raw':length=r['requested_step_norm']
        receipts[arm]=dict(r,selected=ix.tolist(),weights=w.tolist(),distances=d.tolist(),
            last_observation=count-1,first_affected_prediction=count)
    return states,receipts


def predict(model, signal, bank, device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or signal['available'].shape!=signal['values'].shape
            or signal['available'].dtype!=bool or not len(signal['values'])):
        raise ValueError('nonempty signal-only flat23 stream required')
    states={a:initial_encoder(model) for a in ('off',)+ARMS}
    scores={a:[] for a in states};receipts=[];n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            ws=[simple_data.window(signal,t,128) for t in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*ws)]
            with torch.no_grad():
                for arm,state in states.items():scores[arm].append(logit(model,x,m,state).sigmoid().cpu().numpy())
        if hi>=128 and hi<n:
            changed,r=update_all(model,{k:v[:hi] for k,v in signal.items()},hi,bank,device)
            states.update(changed);receipts.append(r)
    curves={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite stream prediction')
    return curves,receipts
