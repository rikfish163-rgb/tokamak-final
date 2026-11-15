"""Same full candidate set, with mean versus exact peak risk aggregation."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data


def aggregation_for(arm,disruptive):
    if arm not in ('mean','safe_peak','peak'):raise ValueError('unknown preparation arm')
    return 'mean' if arm=='mean' or (arm=='safe_peak' and disruptive) else 'peak'


def candidate_ends(shot):
    t=np.asarray(shot['times']);valid=np.asarray(shot['valid']);y=np.asarray(shot['y'])
    if (shot['device']!='EAST' or shot['horizon']!=125. or t.ndim!=1 or not len(t)
            or valid.shape!=t.shape or valid.dtype!=bool or y.shape!=t.shape
            or not np.isfinite(t).all() or np.any(np.diff(t)<0) or not np.isin(y,[0,1]).all()):
        raise ValueError('chronological EAST records and unchanged fixed125 labels required')
    if shot['is_disrupt']:
        if shot['down'] is None or not np.isfinite(shot['down']):raise ValueError('finite revealed disruption time required')
        lead=np.round((shot['down']-t)*1000.,9)
        eligible=valid & (y==1) & (lead>30) & (lead<=125)
    else:
        if shot['down'] is not None or np.any(y) or not valid.all():raise ValueError('safe training shot must have all valid zero labels')
        eligible=valid
    ends=np.flatnonzero(eligible)
    if not len(ends):raise ValueError('no eligible training endpoints; no silent shot exclusion')
    return ends


def bag_loss(logits,disruptive,boundary,mode):
    if (logits.ndim!=1 or not len(logits) or not torch.isfinite(logits).all()
            or mode not in ('mean','peak') or not math.isfinite(boundary)):
        raise ValueError('finite nonempty logits, boundary and explicit aggregation required')
    if mode=='peak':logits=logits[logits.detach().argmax()].reshape(1)
    return F.softplus(boundary-logits if disruptive else logits-boundary).mean()


def logits_at(model,signal,ends,*,device):
    if set(signal)!={'values','available'}:raise ValueError('signal-only windows required')
    windows=[simple_data.window(signal,int(end),model.cfg.window) for end in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    h=model(x,m,mode='risk_state')['risk_state']
    return model.base.risk_head(h).squeeze(-1)


def backward_bag(model,signal,ends,disruptive,boundary,mode,*,weight,device,batch_size=32):
    ends=np.asarray(ends)
    if (ends.ndim!=1 or not len(ends) or np.any(np.diff(ends)<=0)
            or mode not in ('mean','peak') or type(batch_size) is not int or batch_size<1
            or not math.isfinite(weight) or weight<=0):
        raise ValueError('nonempty sorted distinct endpoints and positive finite weight/batch required')
    peak_logit=-math.inf;peak_end=None;loss_value=0.
    for start in range(0,len(ends),batch_size):
        chunk=ends[start:start+batch_size]
        # Keep grad-enabled math dispatch identical to differentiable replay.
        # Peak scan graphs are discarded each chunk, never retained for the shot.
        with torch.enable_grad():
            z=logits_at(model,signal,chunk,device=device)
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite training scan')
            index=int(z.detach().argmax());value=float(z[index].detach())
            if value>peak_logit:peak_logit=value;peak_end=int(chunk[index])
            if mode=='mean':
                loss=bag_loss(z,disruptive,boundary,mode)*len(chunk)/len(ends)
                (weight*loss).backward();loss_value+=float(loss.detach())
    if mode=='peak':
        z=logits_at(model,signal,[peak_end],device=device)
        if abs(float(z.detach()[0])-peak_logit)>2e-5:
            raise AssertionError('peak replay differs from full scan; model must be deterministic')
        loss=bag_loss(z,disruptive,boundary,mode)
        (weight*loss).backward();loss_value=float(loss.detach())
    return dict(candidates=len(ends),peak_end=peak_end,peak_logit=peak_logit,loss=loss_value,aggregation=mode)
