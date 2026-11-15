"""Small historical loss-weight policy, matched to the deployment update clock.

Future risk gradients are used ONLY to train weights on authorized preparation
shots. Deployed predict() takes signal-only input and immutable23 weights.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..feedback_cone_feasibility_20261024.learning import solve_weights
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import flat, logit, apply_step, matched_step
from ..paired_feedback_20261022.model import initial_encoder

GROUPS=('safe','early','danger')
ARMS=('block','point','uniform','permuted','hold')


def future_groups(times,count,flag,down):
    times=np.asarray(times)
    if times.ndim!=1 or not np.isfinite(times).all() or not 0<=count<len(times):
        raise ValueError('finite times and actual update count required')
    ends=np.arange(count,min(count+128,len(times)))
    if not flag:return {'safe':ends}
    if down is None or not np.isfinite(down):raise ValueError('historical disruption time required')
    lead=np.round((down-times[ends])*1000.,9)
    groups=dict(early=ends[lead>125.],danger=ends[(lead>30.)&(lead<=125.)])
    return {g:e for g,e in groups.items() if len(e)}


def group_means(rows,*,key='a'):
    means=[]; signals=[]
    for group in GROUPS:
        take=[r for r in rows if r['group']==group]
        ids=sorted({r['shot'] for r in take})
        if not ids:raise ValueError(f'missing historical group: {group}')
        means.append(np.mean([np.mean([r[key] for r in take if r['shot']==sid],axis=0) for sid in ids],axis=0))
        signals.append(np.mean([np.mean([r['b'] for r in take if r['shot']==sid],axis=0) for sid in ids],axis=0))
    a,b=np.asarray(means),np.asarray(signals)
    if a.shape!=(3,23) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite group-by23 geometry required')
    return a,b


def weights_array(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(23,) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1.)>1e-8:
        raise ValueError('23 nonnegative simplex weights required')
    return w


def bounded_step(parts,weights):
    w=weights_array(weights)
    if parts.ndim!=2 or parts.shape[0]!=23 or not torch.isfinite(parts).all():
        raise ValueError('finite channel gradient matrix required')
    scale=parts.double().norm(dim=1).max()
    if scale==0:return torch.zeros_like(parts[0])
    return (.01*(torch.as_tensor(w,device=parts.device)@parts.double())/scale).to(parts)


def fit_policy(rows):
    result={}; receipts={}
    for name,key in (('block','a'),('point','point_a')):
        a,b=group_means(rows,key=key)
        w,receipt=solve_weights(a,b)
        weights_array(w)
        result[name]=w.tolist()
        receipts[name]=dict(receipt,group_risk_alignment=(a@w).tolist(),group_signal_alignment=(b@w).tolist(),
                            risk_coefficients=a.tolist(),signal_coefficients=b.tolist())
    result['permuted']=np.random.default_rng(131).permutation(result['block']).tolist()
    return result,receipts


def _risk_gradient(model,signal,ends,fast,label,*,device):
    # Detached argmax is exactly the subgradient of max away from ties. Scan
    # using the same grad-enabled math backend as the single-endpoint backward.
    scores=[]
    for lo in range(0,len(ends),32):
        windows=[simple_data.window(signal,int(e),128) for e in ends[lo:lo+32]]
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        scores.extend(logit(model,x,m,fast).detach().cpu().tolist())
    end=int(ends[int(np.argmax(scores))])
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,end,128)]
    z=logit(model,x,m,fast)
    loss=F.binary_cross_entropy_with_logits(z,torch.full_like(z,float(label)))
    gradient=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach()
    return gradient,dict(end=end,logit=float(z.detach()[0]),loss=float(loss.detach()),candidate_count=len(ends))


def collect_shot(model,signal,times,sid,flag,down,*,device='cuda'):
    rows=[];events=[]
    for count in range(256,len(times),128):
        groups=future_groups(times,count,flag,down)
        if not groups:continue  # No >30ms future target, as specified before run.
        fast=initial_encoder(model)
        parts,receipt=signal_parts(model,signal,count,fast,hold=False,device=device)
        receipt=dict(receipt,shot=sid)
        if parts is None:
            events.append(dict(receipt,status='no_measured_forecast_targets'));continue
        pd=parts.double();scale=pd.norm(dim=1).max();g=pd.sum(0)
        if scale==0:
            events.append(dict(receipt,status='all_channel_gradients_zero'));continue
        # If the total gradient cancels, its first-order descent is identically
        # zero; keep this exact neutral constraint, do not invent a direction.
        b=pd@g/(scale*g.norm()) if g.norm()>0 else torch.zeros(23,device=device,dtype=torch.float64)
        for group,ends in groups.items():
            label=int(group=='danger')
            r,rr=_risk_gradient(model,signal,ends,fast,label,device=device)
            point=int(ends[np.argmin(np.abs(ends-(count+24)))])
            rp,pr=_risk_gradient(model,signal,np.array([point]),fast,label,device=device)
            rows.append(dict(shot=sid,count=count,group=group,label=label,
                last_observation=count-1,block_end=int(ends[-1]),risk=rr,point_risk=pr,
                scale=float(scale),a=(pd@r.double()/scale).cpu().tolist(),
                point_a=(pd@rp.double()/scale).cpu().tolist(),b=b.cpu().tolist(),
                observations=receipt['observed'],forecast_origins=receipt['origins']))
        events.append(dict(receipt,status='used',groups=list(groups)))
    return rows,events


def predict(model,signal,policy,*,device='cuda'):
    if set(signal)!={'values','available'} or signal['values'].shape!=signal['available'].shape:
        raise ValueError('signal-only matching values and masks required')
    if set(policy)!={'block','point','permuted'}:raise ValueError('fixed declared policy keys required')
    weights={k:weights_array(w) for k,w in policy.items()}
    states={a:initial_encoder(model) for a in ('off',)+ARMS}
    curves={a:[] for a in states};receipts={a:[] for a in ARMS};n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(hi,start+32))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for arm in states:curves[arm].append(logit(model,x,m,states[arm]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        fast=initial_encoder(model)
        real,rr=signal_parts(model,signal,hi,fast,hold=False,device=device)
        fake,fr=signal_parts(model,signal,hi,fast,hold=True,device=device)
        zero=torch.zeros(sum(v.numel() for v in fast.values()),device=device)
        main=zero if real is None else bounded_step(real,weights['block'])
        length=float(main.double().norm())
        for arm in ARMS:
            if arm=='block':step=main
            elif arm=='point':step=zero if real is None else bounded_step(real,weights['point'])
            elif arm=='uniform':step=zero if real is None else matched_step(real.sum(0),length)
            elif arm=='permuted':
                step=zero if real is None else matched_step(torch.as_tensor(weights['permuted'],device=device,dtype=real.dtype)@real,length)
            else:
                step=zero if fake is None else matched_step(torch.as_tensor(weights['block'],device=device,dtype=fake.dtype)@fake,length)
            states[arm]=apply_step(fast,step)
            actual=flat([states[arm][k].detach()-v.detach() for k,v in fast.items()])
            receipt=dict(fr if arm=='hold' else rr)
            receipt.update(actual_step_norm=float(actual.double().norm()),requested_step_norm=float(step.double().norm()),
                main_requested_norm=length,gradient_steps=int(bool(step.any())),reset='parent',
                control_zero_direction=bool(not step.any() and length>0))
            receipts[arm].append(receipt)
    return {a:np.concatenate(v) for a,v in curves.items()},receipts


def eligible(results):
    main,off=results['block'],results['off']
    return (main['tp']>=off['tp'] and main['fp']<=off['fp']
        and main['danger_loss']<=off['danger_loss'] and main['safe_loss']<=off['safe_loss']
        and all(main['balanced_loss']<results[a]['balanced_loss']-1e-6
                for a in ('off','point','uniform','permuted','hold')))
