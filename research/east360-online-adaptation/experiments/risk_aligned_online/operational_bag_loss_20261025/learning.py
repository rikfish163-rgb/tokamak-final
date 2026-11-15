"""Whole-shot operational outer loss; signal-only, parent-reset deployment."""
import math
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..finite_conditional_loss_20261025.learning import (
    LossPolicy, load_policy, feedback_context, policy_step, outer_gradient,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import flat, logit, apply_step, matched_step
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('stage_old','bag_global','bag_context','uniform','permuted','hold')


def candidate_ends(times,flag,down):
    t=np.asarray(times)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or (np.diff(t)<=0).any():
        raise ValueError('nonempty finite strictly ordered times required')
    if flag:
        if down is None or not np.isfinite(down):raise ValueError('historical outcome required')
        ends=np.flatnonzero(np.round((down-t)*1000.,9)>30.)
        if not len(ends):raise ValueError('no timely candidate endpoint')
        return ends
    return np.arange(len(t))


def bag_loss(peak,flag,threshold):
    if not 0<threshold<1:raise ValueError('fixed probability threshold in(0,1) required')
    margin=peak-math.log(threshold/(1-threshold))
    return F.softplus(-margin if flag else margin)


def aggregate(rows):
    if len({r['shot'] for r in rows})!=len(rows):raise ValueError('one whole-bag row per shot required')
    total=0.;gradient=None;detail=[]
    for flag in (False,True):
        take=[r for r in rows if bool(r['flag'])==flag]
        if not take:raise ValueError('both shot classes required')
        for r in take:
            delta=r['delta'];g=r['gradient'];factor=2. if delta>0 else 1.
            if not np.isfinite(delta) or not np.isfinite(g).all():raise FloatingPointError('nonfinite objective')
            total+=(delta+max(delta,0))/(2*len(take))
            term=factor*g/(2*len(take));gradient=term if gradient is None else gradient+term
            detail.append({k:v for k,v in r.items() if k!='gradient'})
    return total,gradient,detail


def historical_gate(rows,current,baseline):
    d=[r['delta'] for r in rows if r['flag']];s=[r['delta'] for r in rows if not r['flag']]
    return bool(d and s and max(d)<=0 and np.mean(s)<=0 and current['tp']>=baseline['tp']
                and current['fp']<=baseline['fp'] and current['very_early_over3000']<=baseline['very_early_over3000'])


def _batch(signal,ends,device):
    windows=[simple_data.window(signal,int(e),128) for e in ends]
    return [torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]


def _state(model,policy,item,device):
    fast=initial_encoder(model)
    if policy is None or item is None:return fast
    parts=item['parts'].to(device)
    coeff=policy(item['context'].to(device),item['observed'].to(device)).detach()
    return apply_step(fast,policy_step(parts,coeff))


def scan_bag(model,policy,shot,threshold,*,device='cuda',with_gradient=True):
    """Only authorized ended preparation shots enter this label-aware teacher."""
    ends=candidate_ends(shot['times'],shot['flag'],shot['down'])
    b=math.log(threshold/(1-threshold));best=-math.inf;winner=None;first=None
    for count in range(0,len(shot['times']),128):
        take=ends[(ends>=count)&(ends<count+128)]
        if not len(take):continue
        item=None if count<256 else shot['cache'][count]
        state=_state(model,policy,item,device)
        for lo in range(0,len(take),32):
            chunk=take[lo:lo+32];x,m=_batch(shot['signal'],chunk,device)
            z=logit(model,x,m,state).detach()
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite historical risk')
            j=int(z.argmax())
            if float(z[j])>best:
                best=float(z[j]);winner=dict(count=count,chunk=chunk.tolist(),position=j,end=int(chunk[j]))
            alarms=torch.nonzero(z>=b).flatten()
            if first is None and len(alarms):first=int(chunk[int(alarms[0])])
    assert winner is not None
    state=_state(model,policy,None if winner['count']<256 else shot['cache'][winner['count']],device)
    x,m=_batch(shot['signal'],winner['chunk'],device)
    z=logit(model,x,m,state)[winner['position']]
    if float(z.detach())!=best:raise AssertionError('same-chunk winning logit replay differs')
    loss=bag_loss(z,shot['flag'],threshold)
    result=dict(winner,loss=float(loss.detach()),logit=best,first_alarm=first,
                first_lead_ms=None if first is None or not shot['flag'] else float(round((shot['down']-shot['times'][first])*1000,9)),
                candidate_count=len(ends),static_prefix_winner=winner['count']<256)
    gradient=None
    if policy is not None and with_gradient:
        gradient=np.zeros(sum(p.numel() for p in policy.parameters()),dtype=np.float64)
        if winner['count']>=256:
            item=shot['cache'][winner['count']]
            risk=flat(torch.autograd.grad(loss,tuple(state.values()))).detach()
            gradient=outer_gradient(policy,item['context'].to(device),item['observed'].to(device),
                item['parts'].to(device),item['scale'],risk).cpu().numpy()
    return result,gradient


def _counts(records):
    return dict(tp=sum(r['flag'] and r['first_alarm'] is not None for r in records),
        fp=sum(not r['flag'] and r['first_alarm'] is not None for r in records),
        very_early_over3000=sum(r['flag'] and r['first_lead_ms'] is not None and r['first_lead_ms']>3000 for r in records))


def evaluate_policy(model,policy,shots,threshold,*,device='cuda'):
    rows=[];predictions=[];base=[]
    for sid,shot in shots.items():
        r,g=scan_bag(model,policy,shot,threshold,device=device)
        rows.append(dict(shot=sid,flag=shot['flag'],delta=r['loss']-shot['baseline']['loss'],
                         gradient=g,**r))
        predictions.append(dict(flag=shot['flag'],**r));base.append(dict(flag=shot['flag'],**shot['baseline']))
    loss,gradient,detail=aggregate(rows)
    current,baseline=_counts(predictions),_counts(base)
    return loss,gradient,detail,dict(current=current,baseline=baseline,passed=historical_gate(rows,current,baseline))


def train_policy(model,kind,shots,threshold,*,device='cuda',progress=None):
    policy=LossPolicy(kind,device=device);optimizer=torch.optim.Adam(policy.parameters(),lr=.05)
    logs=[];best=None
    for epoch in range(11):
        loss,gradient,detail,gate=evaluate_policy(model,policy,shots,threshold,device=device)
        r=dict(epoch=epoch,objective=loss,per_shot=detail,history_gate=gate,
               strength=float(policy.strength.detach().sigmoid()),gradient_norm=float(np.linalg.norm(gradient)),policy=policy.export())
        logs.append(r)
        if best is None or loss<best['objective']:best=r
        if progress is not None:progress(kind,r)
        if epoch==10:break
        offset=0
        for p in policy.parameters():
            p.grad=torch.as_tensor(gradient[offset:offset+p.numel()],device=device,dtype=p.dtype).reshape_as(p).clone()
            offset+=p.numel()
        assert offset==len(gradient)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    return best,logs


def predict(model,signal,records,stage_record,*,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signal-only input required')
    policies={k:load_policy(v,device=device) for k,v in records.items()}
    if set(policies)!={'global','context'}:raise ValueError('both policies required')
    old=load_policy(stage_record,device=device)
    states={a:initial_encoder(model) for a in ('off',)+ARMS}
    curves={a:[] for a in states};receipts={a:[] for a in ARMS};n=len(signal['values'])
    perm=torch.as_tensor(np.random.default_rng(131).permutation(23),device=device)
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            x,m=_batch(signal,range(start,min(hi,start+32)),device)
            with torch.no_grad():
                for arm in states:curves[arm].append(logit(model,x,m,states[arm]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        fast=initial_encoder(model)
        real,rr=signal_parts(model,signal,hi,fast,hold=False,device=device)
        fake,fr=signal_parts(model,signal,hi,fast,hold=True,device=device)
        c,mask=feedback_context(model,signal,hi,fast,hold=False,device=device)
        fc,fmask=feedback_context(model,signal,hi,fast,hold=True,device=device)
        with torch.no_grad():
            coeff={k:p(c,mask) for k,p in policies.items()}
            hold=policies['context'](fc,fmask);stage=old(c,mask)
        zero=torch.zeros(sum(v.numel() for v in fast.values()),device=device)
        main=zero if real is None else policy_step(real,coeff['context']);length=float(main.double().norm())
        for arm in ARMS:
            if arm=='bag_context':step=main
            elif arm=='bag_global':step=zero if real is None else policy_step(real,coeff['global'])
            elif arm=='stage_old':step=zero if real is None else policy_step(real,stage)
            elif arm=='uniform':step=zero if real is None else matched_step(real.sum(0),length)
            elif arm=='permuted':step=zero if real is None else matched_step((coeff['context'][perm]@real.double()).to(real),length)
            else:step=zero if fake is None else matched_step((hold@fake.double()).to(fake),length)
            states[arm]=apply_step(fast,step)
            actual=flat([states[arm][k].detach()-v.detach() for k,v in fast.items()])
            r=dict(fr if arm=='hold' else rr)
            r.update(actual_step_norm=float(actual.double().norm()),requested_step_norm=float(step.double().norm()),
                main_requested_norm=length,gradient_steps=int(bool(step.any())),reset='parent',
                control_zero_direction=bool(not step.any() and length>0))
            if arm in ('bag_global','bag_context'):r['coefficients']=coeff[arm.removeprefix('bag_')].cpu().tolist()
            receipts[arm].append(r)
    return {a:np.concatenate(v) for a,v in curves.items()},receipts


def operational_losses(rows,threshold):
    groups={False:[],True:[]};detail=[];b=math.log(threshold/(1-threshold))
    for r in rows:
        ends=candidate_ends(r['times'],r['is_disrupt'],r['down'])
        peak=float(np.max(r['scores'][ends]))
        p=np.clip(peak,1e-12,1-1e-12);z=np.log(p)-np.log1p(-p)
        loss=float(np.logaddexp(0,b-z if r['is_disrupt'] else z-b))
        groups[bool(r['is_disrupt'])].append(loss)
        detail.append(dict(shot=r['shot'],flag=bool(r['is_disrupt']),peak=peak,loss=loss))
    if not all(groups.values()):raise ValueError('both evaluation classes required')
    d,s=float(np.mean(groups[True])),float(np.mean(groups[False]))
    return dict(bag_disrupt_loss=d,bag_safe_loss=s,bag_balanced_loss=(d+s)/2),detail


def eligible(results,history_passed):
    main,off=results['bag_context'],results['off']
    return bool(history_passed and main['tp']>=off['tp'] and main['fp']<=off['fp']
        and main['very_early_over3000']<=off['very_early_over3000']
        and main['bag_disrupt_loss']<=off['bag_disrupt_loss'] and main['bag_safe_loss']<=off['bag_safe_loss']
        and all(main['bag_balanced_loss']<results[a]['bag_balanced_loss']-1e-6
                for a in ('off','stage_old','bag_global','uniform','permuted','hold')))
