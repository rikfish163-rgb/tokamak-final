"""Signal-trajectory routing of historical gradients, not an output correction.

No current outcome enters this module. A local encoder is rebuilt from the
fixed parent at each update; cached gradients are exact first-order gradients
at that parent. Hard DTW is a routing metric, not a differentiable objective.
"""
import math

import numpy as np
import torch

from .. import simple_data
from ..history_tangent_20261023.learning import flat, apply_step, logit
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('trajectory', 'level', 'reverse', 'uniform')


def observed_window(signal, count):
    if (set(signal) != {'values','available'} or type(count) is not int
            or signal['values'].ndim != 2 or signal['values'].shape[1] != 23
            or signal['available'].shape != signal['values'].shape
            or signal['available'].dtype != bool or not 128 <= count <= len(signal['values'])):
        raise ValueError('signal-only observed prefix and at least128 points required')
    return signal['values'][count-128:count], signal['available'][count-128:count]


def describe(values, available, center, scale, spread=None):
    if (values.shape != (128,23) or available.shape != values.shape or available.dtype != bool
            or np.shape(center) != (23,) or np.shape(scale) != (23,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('128x23 masked observations and finite positive source scale required')
    if spread is None: spread = np.ones(23)
    if np.shape(spread) != (23,) or not np.isfinite(spread).all() or np.any(spread<=0):
        raise ValueError('finite positive historical spread required')
    valid = available & np.isfinite(values)
    x = np.arcsinh((np.where(valid, values, center).astype('float64')-center)/scale)/spread
    x = np.where(valid,x,0.)
    count = valid.sum(0)
    mean = x.sum(0)/np.maximum(count,1)
    v = valid.reshape(8,16,23).sum(1)
    path = x.reshape(8,16,23).sum(1)/np.maximum(v,1)-mean
    return dict(path=np.where(v>0,path,0.),valid=v>0,mean=mean,mean_valid=count>0)


def distances(query, bank, mode):
    if mode not in ('trajectory','level','reverse'):
        raise ValueError('explicit trajectory, level or reverse metric required')
    common = query['mean_valid'][None] & bank['mean_valid']
    counts = common.sum(-1)
    level = ((query['mean'][None]-bank['mean'])**2*common).sum(-1)/np.maximum(counts,1)
    level[counts<8] = np.inf
    if mode == 'level': return .25*level
    q, qm = query['path'], query['valid']
    if mode == 'reverse': q,qm = q[::-1],qm[::-1]
    n = len(bank['path'])
    dp = np.full((n,9,9),np.inf); dp[:,0,0] = 0.
    for i in range(8):
        for j in range(max(0,i-1),min(8,i+2)):
            valid = qm[i][None] & bank['valid'][:,j]
            c = valid.sum(-1)
            cost = ((q[i][None]-bank['path'][:,j])**2*valid).sum(-1)/np.maximum(c,1)
            cost[c<8] = np.inf
            dp[:,i+1,j+1] = cost + np.minimum(np.minimum(dp[:,i,j+1],dp[:,i+1,j]),dp[:,i,j])
    return dp[:,8,8]/8 + .25*level


def select_weights(distance, shots, labels, temperature, exclude=None):
    if (distance.shape != shots.shape or labels.shape != shots.shape or distance.ndim != 1
            or not np.isin(labels,[0,1]).all() or np.isnan(distance).any()
            or np.any(distance<0) or not math.isfinite(temperature) or temperature<=0):
        raise ValueError('matching distances/identities/binary labels and positive temperature required')
    selected = []
    for sid in np.unique(shots):
        if sid == exclude: continue
        ids = np.flatnonzero((shots==sid)&np.isfinite(distance))
        if len(ids): selected.append(int(ids[np.argmin(distance[ids])]))
    selected = np.asarray(selected,dtype=int)
    if not len(selected): return selected, np.empty(0)
    yy = labels[selected]
    # Equal class prior, not equal per-window prevalence. Within each class
    # each historical shot has equal prior mass before signal conditioning.
    prior = np.array([1./np.sum(yy==y) for y in yy])
    logits = -distance[selected]/temperature + np.log(prior)
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    return selected, weights


def combine_step(parent, gradients, weights, *, learning_rate, length=None):
    if (gradients.ndim != 2 or gradients.shape[1] != sum(v.numel() for v in parent.values())
            or weights.shape != (len(gradients),) or not torch.isfinite(gradients).all()
            or not torch.isfinite(weights).all() or (weights<0).any()
            or abs(float(weights.sum())-1.)>1e-5 or not math.isfinite(learning_rate) or learning_rate<=0
            or (length is not None and (not math.isfinite(length) or length<0))):
        raise ValueError('finite matching gradients, normalized weights and step bounds required')
    g = weights.to(gradients) @ gradients
    norm = float(g.double().norm())
    wanted = learning_rate*norm/max(1.,norm) if length is None else length
    # A genuinely zero historical gradient has no direction. Do not inject an
    # unrelated gradient merely to satisfy the matched-step control.
    step = g*(wanted/norm) if norm>0 else torch.zeros_like(g)
    out = apply_step(parent,step)
    actual = float(flat([out[k].detach()-v.detach() for k,v in parent.items()]).double().norm())
    return out, dict(gradient_norm=norm,requested_step_norm=wanted,step_norm=actual)


def make_bank(model, shots, device='cuda'):
    center=model.base.center.detach().cpu().double().numpy()
    scale=model.base.scale.detach().cpu().double().numpy()
    windows=[];masks=[];rows=[]
    for s in shots:
        eligible=s['valid'] & (np.arange(len(s['times']))>=127)
        if s['is_disrupt']:
            lead=np.round((s['down']-s['times'])*1000,9)
            eligible &= (lead>30)&(lead<=125)
        ends=np.flatnonzero(eligible)
        if not len(ends):raise ValueError(f"no authorized full historical window: {s['shot']}")
        ends=ends[np.unique(np.rint(np.linspace(0,len(ends)-1,16)).astype(int))]
        for end in ends:
            x,m=simple_data.window({k:s[k] for k in ('values','available')},int(end),128)
            windows.append(x);masks.append(m)
            rows.append(dict(shot=int(s['shot']),end=int(end),label=int(s['is_disrupt'])))
    x=np.stack(windows); mask=np.stack(masks)&np.isfinite(x)
    level=np.arcsinh((np.where(mask,x,center).astype('float64')-center)/scale)
    count=mask.sum((0,1)); mean=np.where(mask,level,0.).sum((0,1))/np.maximum(count,1)
    var=np.where(mask,(level-mean)**2,0.).sum((0,1))/np.maximum(count,1)
    spread=np.maximum(np.sqrt(var),.1)
    desc=[describe(a,b,center,scale,spread) for a,b in zip(x,mask)]
    bank={k:np.stack([d[k] for d in desc]) for k in desc[0]}
    bank.update(shots=np.array([r['shot'] for r in rows]),labels=np.array([r['label'] for r in rows]),
                rows=rows,center=center,scale=scale,spread=spread)
    nearest=[]
    for i,d in enumerate(desc):
        distance=distances(d,bank,'trajectory');distance[bank['shots']==bank['shots'][i]]=np.inf
        nearest.append(float(distance.min()))
    temperature=float(np.median(nearest))
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('historical routing temperature not identifiable')
    bank['temperature']=temperature
    fast=initial_encoder(model);gradients=[];losses=[]
    for a,b,row in zip(x,mask,rows):
        xx=torch.as_tensor(a[None],device=device);mm=torch.as_tensor(b[None],device=device)
        z=logit(model,xx,mm,fast)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.full_like(z,float(row['label'])))
        gradients.append(flat(torch.autograd.grad(loss,tuple(fast.values()))).detach())
        losses.append(float(loss.detach()))
    bank.update(gradients=torch.stack(gradients),losses=np.array(losses),x=x,mask=mask)
    if not torch.isfinite(bank['gradients']).all():raise FloatingPointError('nonfinite historical gradient bank')
    return bank


def update_all(model, signal, count, bank, device='cuda'):
    x,m=observed_window(signal,count)
    query=describe(x,m,bank['center'],bank['scale'],bank['spread'])
    states={};receipts={};length=None;parent=initial_encoder(model)
    for arm in ARMS:
        if arm=='uniform':
            d=np.full(len(bank['shots']),np.inf)
            for sid in np.unique(bank['shots']):
                ids=np.flatnonzero(bank['shots']==sid);d[ids[len(ids)//2]]=0.
        else:d=distances(query,bank,arm)
        selected,w=select_weights(d,bank['shots'],bank['labels'],bank['temperature'])
        if not len(selected):
            states[arm]=initial_encoder(model)
            r=dict(gradient_norm=0.,requested_step_norm=0.,step_norm=0.,reason='insufficient_joint_observations')
            if arm=='trajectory':length=0.
        else:
            states[arm],r=combine_step(parent,bank['gradients'][selected],torch.as_tensor(w,device=device),
                                      learning_rate=.01,length=length)
            if arm=='trajectory':length=r['requested_step_norm']
        receipts[arm]=dict(r,selected=selected.tolist(),weights=w.tolist(),
                           distances=d[selected].tolist(),last_observation=count-1,first_affected_prediction=count)
    return states,receipts


def predict(model, signal, bank, device='cuda', *, arms=ARMS, updater=update_all):
    # Validate even short shots without inventing observations for an update.
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or signal['available'].shape!=signal['values'].shape
            or signal['available'].dtype!=bool or not len(signal['values'])):
        raise ValueError('nonempty signal-only flat23 stream required')
    states={a:initial_encoder(model) for a in ('off',)+arms}
    scores={a:[] for a in states};receipts=[];n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            ws=[simple_data.window(signal,t,128) for t in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*ws)]
            with torch.no_grad():
                for arm,state in states.items():scores[arm].append(logit(model,x,m,state).sigmoid().cpu().numpy())
        if hi>=128 and hi<n:
            changed,r=updater(model,{k:v[:hi] for k,v in signal.items()},hi,bank,device)
            states.update(changed);receipts.append(r)
    curves={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite stream prediction')
    return curves,receipts
