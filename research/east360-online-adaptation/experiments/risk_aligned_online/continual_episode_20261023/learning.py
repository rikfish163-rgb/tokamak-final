"""Ended-shot ordered experience; temporary signal adaptation is never copied back.

No current-shot event status, outcome, or pseudo negative is needed by predict().
Outer supervised training is legal only after the controller seals and reveals.
"""
import re

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import target_mask, update
from ..matured_negative_20261023.learning import tensors
from ..paired_feedback_20261022.model import initial_encoder
from ..persistence_control_20261023.control import persistence_target


class Access:
    """Sequential reveal authority with one exact saved prediction digest."""
    def __init__(self, order, preparation):
        if len(order)!=len(set(order)) or set(order)&set(preparation):
            raise ValueError('distinct stream identities disjoint from preparation required')
        self.order=list(order);self.allowed=set(preparation);self.index=0
        self.current=None;self.digest=None;self.revealed=False

    def begin(self, sid):
        if self.current is not None or self.index>=len(self.order) or sid!=self.order[self.index]:
            raise ValueError('exact next shot required; previous shot must be finished')
        self.current=sid;self.digest=None;self.revealed=False

    def seal(self, digest):
        if self.current is None or self.digest is not None or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('one real SHA256 prediction seal required')
        self.digest=digest

    def reveal(self, reader):
        if self.current is None or self.digest is None or self.revealed:
            raise ValueError('reveal requires an unrevealed sealed current shot')
        outcome=reader(self.current)
        self.allowed.add(self.current);self.revealed=True
        return outcome

    def finish(self):
        if not self.revealed:raise ValueError('cannot finish before outcome reveal')
        self.current=None;self.index+=1


def episode_indices(s):
    needed={'shot','values','available','times','y','valid','is_disrupt','down'}
    if not needed<=set(s):raise ValueError('ended labeled shot required')
    n=len(s['times'])
    if (s['values'].shape!=(n,23) or s['available'].shape!=(n,23)
            or s['available'].dtype!=bool or len(s['y'])!=n or len(s['valid'])!=n):
        raise ValueError('aligned ended-shot arrays required')
    indices=np.arange(n);counts=(indices//128)*128
    eligible=s['valid']&(counts>=256)
    leads=np.round((s['down']-s['times'])*1000,9) if s['is_disrupt'] else np.full(n,np.inf)
    # Positives after the operational deadline are not useful anticipation
    # teaching endpoints. Negative targets retain the established y=0 rule.
    eligible&=(s['y']==0)|((leads>30)&(leads<=125))
    result=[]
    for label in (0,1):
        ends=indices[eligible&(s['y']==label)]
        if len(ends)>8:ends=ends[np.linspace(0,len(ends)-1,8).astype(int)]
        for e in ends:
            result.append(dict(end=int(e),count=int(counts[e]),label=label))
    return result


def select_pair(memory, current, rng):
    """Current + one prior shot; no window-count preference for long shots."""
    valid=[sid for sid,(s,es) in memory.items() if es]
    if not valid:return []
    ids=[current] if current in valid else []
    past=[sid for sid in valid if sid!=current]
    if past:ids.append(int(rng.choice(sorted(past))))
    result=[]
    for sid in ids:
        s,es=memory[sid];labels=sorted({e['label'] for e in es})
        label=int(rng.choice(labels));options=[e for e in es if e['label']==label]
        result.append((s,options[int(rng.integers(len(options)))]))
    return result


def trim_memory(memory, cap=32):
    if type(cap) is not int or cap<1:raise ValueError('positive per-class shot cap required')
    for disruptive in (False,True):
        ids=[sid for sid,(s,_) in memory.items() if bool(s['is_disrupt'])==disruptive]
        for sid in ids[:-cap]:del memory[sid]


def meta_loss(model, episodes, *, device, adapt):
    if not episodes:raise ValueError('at least one authorized historical episode required')
    fast=initial_encoder(model,track_initial=True)
    dtype=next(iter(fast.values())).dtype
    before=[];after=[];details=[]
    for s,e in episodes:
        count,end=e['count'],e['end']
        if not 256<=count<=end<count+128 or end>=len(s['values']) or float(s['y'][end])!=e['label']:
            raise ValueError('ordered support/query episode with authoritative historical target required')
        x,m=tensors(s,[end],device);x=x.to(dtype)
        y=torch.tensor([e['label']],dtype=dtype,device=device)
        b=F.binary_cross_entropy_with_logits(logit(model,x,m,fast),y)
        before.append(b)
        if adapt:
            prefix={k:s[k][:count] for k in ('values','available')}
            sx,sm,truth,tm,origins=forecast_examples(prefix,count,model.cfg,device)
            sx,truth=sx.to(dtype),truth.to(dtype)
            mask=target_mask(sx,sm,truth,tm)
            if not mask.any():raise ValueError('historical episode has no measured feedback')
            loss=model.base.forecast_loss(model.predict_fast(sx,sm,fast)['forecast'],truth,mask)
            gradients=torch.autograd.grad(loss,tuple(fast.values()),create_graph=True)
            # Differentiate the SAME clipped inner step, including its norm;
            # no detached first-order surrogate masquerades as a meta-gradient.
            norm=sum(g.square().sum() for g in gradients).clamp_min(1e-30).sqrt()
            factor=norm.clamp_min(1).reciprocal()
            adapted={k:v-.001*factor*g for (k,v),g in zip(fast.items(),gradients)}
            a=F.binary_cross_entropy_with_logits(logit(model,x,m,adapted),y)
            details.append(dict(shot=int(s['shot']),**e,forecast_origins=origins,
                last_feedback_observation=count-1,inner_loss=float(loss.detach())))
        else:
            a=b;details.append(dict(shot=int(s['shot']),**e))
        after.append(a)
    b=torch.stack(before).mean();a=torch.stack(after).mean()
    return .5*(a+b),dict(before_bce=float(b.detach()),after_bce=float(a.detach())),details


def after_shot_step(model, optimizer, episodes, *, adapt, device):
    names=set(model.encoder_names())
    for k,p in model.named_parameters():p.requires_grad_(k in names)
    params=[p for k,p in model.named_parameters() if k in names]
    old=[p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    try:
        loss,parts,details=meta_loss(model,episodes,device=device,adapt=adapt)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite post-shot outer objective')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite persistent state')
        distance=sum((p.detach()-q).double().square().sum() for p,q in zip(params,old)).sqrt()
        return dict(loss=float(loss.detach()),**parts,episodes=details,gradient_norm=float(norm),
                    parameter_delta_norm=float(distance),adaptation_aware=adapt)
    finally:
        model.set_stage('frozen')


def hold_update(model,prefix,count,fast,*,device):
    # Same input/mask/clock as real feedback; only the numerical target is flat.
    x,m,target,tm,ends=persistence_target(forecast_examples(prefix,count,model.cfg,device))
    mask=target_mask(x,m,target,tm)
    receipt=dict(last_observation=count-1,first_affected_prediction=count,
        forecast_origins=ends,observed_targets=int(mask.sum()),gradient_steps=0,parameter_delta_norm=0.)
    if not mask.any():return fast,receipt
    loss=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],target,mask)
    grads=torch.autograd.grad(loss,tuple(fast.values()))
    norm=sum(g.square().sum() for g in grads).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite flat control')
    new={k:(v-.001*g/norm.clamp_min(1)).detach().requires_grad_(True) for (k,v),g in zip(fast.items(),grads)}
    distance=sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    receipt.update(gradient_steps=1,gradient_norm=float(norm),parameter_delta_norm=float(distance),loss_before=float(loss.detach()))
    return new,receipt


def predict(meta,inter,signal,*,device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2 or signal['values'].shape[1]!=23
            or signal['available'].shape!=signal['values'].shape or signal['available'].dtype!=bool):
        raise ValueError('signal-only flat23 arrays required')
    states={a:initial_encoder(meta) for a in ('meta_static','meta_intra','meta_hold')}
    states['inter_only']=initial_encoder(inter)
    scores={a:[] for a in states};receipts={a:[] for a in ('meta_intra','meta_hold')}
    n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            x,m=tensors(signal,range(start,min(start+32,hi)),device)
            with torch.no_grad():
                for arm,state in states.items():
                    net=inter if arm=='inter_only' else meta
                    scores[arm].append(logit(net,x,m,state).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        prefix={k:v[:hi] for k,v in signal.items()}
        states['meta_intra'],a=update(meta,prefix,hi,states['meta_intra'],device=device)
        states['meta_hold'],b=hold_update(meta,prefix,hi,states['meta_hold'],device=device)
        receipts['meta_intra'].append(a);receipts['meta_hold'].append(b)
    curves={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite current-shot risk')
    for a in ('meta_intra','meta_hold'):np.testing.assert_array_equal(curves[a][:256],curves['meta_static'][:256])
    return curves,receipts
