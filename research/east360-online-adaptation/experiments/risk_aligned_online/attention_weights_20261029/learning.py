"""Ended-shot operational supervision for a small QK-feedback loss policy."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..attention_lowrank_20261029.channel_direction import channel_losses
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS=('cold','uniform_reset','learned_reset','permuted_reset','reverse_reset')


def validate_fit_ids(ids,order):
    if len(order)!=360 or len(set(order))!=360 or not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(order[:20]):
        raise ValueError('policy fitting only on unique ended cold20 prefix IDs')


def weighted_state(parts,log_weights):
    if parts.ndim!=3 or parts.shape[1]!=23 or log_weights.shape!=(23,) or not torch.isfinite(parts).all() or not torch.isfinite(log_weights).all():
        raise ValueError('finite batch/channel/parameter gradients and23 log weights required')
    direction=torch.einsum('c,bcp->bp',log_weights.softmax(0).to(parts),parts)
    return -.05*direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-12)


def operational_objective(peaks,baseline_peaks,flags,threshold):
    if (peaks.ndim!=1 or peaks.shape!=baseline_peaks.shape or flags.shape!=peaks.shape or flags.dtype!=torch.bool
            or not flags.any() or flags.all() or not 0<threshold<1):
        raise ValueError('paired shot peaks with both outcome classes and fixed threshold required')
    sign=torch.where(flags,-1.,1.);boundary=math.log(threshold/(1-threshold))
    delta=F.softplus(sign*(peaks-boundary))-F.softplus(sign*(baseline_peaks-boundary))
    harm=delta+delta.clamp_min(0)
    return .5*(harm[flags].mean()+harm[~flags].mean()),delta


def feedback_terms(net,prefix,slow,*,reverse=False):
    if set(prefix)!={'values','available'}:raise ValueError('signal-only arrived prefix required')
    count=len(prefix['values'])
    if count<256:raise ValueError('no matured scheduled feedback before256 observations')
    x,a,truth,mask,origins=forecast_examples(prefix,count,net.base.cfg,net.base.center.device)
    mask=mask & torch.isfinite(truth) & a[:,-1:] & torch.isfinite(x[:,-1:]);mask[:,:,10]=False
    if reverse:truth=truth.flip(1);mask=mask.flip(1)
    state=x.new_zeros(1,512,requires_grad=True)
    prediction=net(x,a,state.expand(len(x),-1),slow,'qk')['forecast']
    terms=channel_losses(net.base,prediction,truth,mask)
    return terms,state,dict(count=count,origins=origins,observed=int(mask.sum()))


def feedback_parts(net,prefix,slow):
    terms,state,info=feedback_terms(net,prefix,slow)
    parts=torch.stack([torch.autograd.grad(t,state,retain_graph=i<22)[0][0] for i,t in enumerate(terms)])
    if not torch.isfinite(parts).all():raise FloatingPointError('invalid channel gradients')
    return parts.detach(),info


def feedback_states(net,prefix,slow,log_weights):
    if log_weights.shape!=(23,) or not torch.isfinite(log_weights).all():raise ValueError('frozen23 log weights required')
    weights=log_weights.to(net.base.center).softmax(0)
    perm=torch.as_tensor(np.random.default_rng(131).permutation(23),device=weights.device)
    terms,state,info=feedback_terms(net,prefix,slow)
    coefficients=(torch.ones_like(weights)/23,weights,weights[perm]);states={};norms={}
    for i,(arm,coeff) in enumerate(zip(ARMS[1:4],coefficients)):
        g,=torch.autograd.grad((terms*coeff).sum(),state,retain_graph=i<2)
        if not torch.isfinite(g).all():raise FloatingPointError('invalid feedback gradient')
        length=g.norm();states[arm]=(-.05*g/length.clamp_min(1e-12)).detach()[0];norms[arm]=float(length)
    reverse,rs,_=feedback_terms(net,prefix,slow,reverse=True)
    g,=torch.autograd.grad((reverse*weights).sum(),rs)
    if not torch.isfinite(g).all():raise FloatingPointError('invalid reverse gradient')
    length=g.norm();states['reverse_reset']=(-.05*g/length.clamp_min(1e-12)).detach()[0];norms['reverse_reset']=float(length)
    info.update(gradient_norm=norms)
    return states,info


def predict(net,signal,slow,log_weights):
    if set(signal)!={'values','available'}:raise ValueError('no outcomes in online prediction inputs')
    state={k:net.base.center.new_zeros(512) for k in ARMS};scores={k:[] for k in ARMS}
    info={k:dict(updates=0,opportunities=0,observed_targets=0,max_state_norm=0.,last_observations=[]) for k in ARMS[1:]}
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,a=batch(signal,range(lo,hi),net.base.center.device)
        states=torch.cat([state[k].expand(len(x),-1) for k in ARMS])
        with torch.no_grad():
            z=net(x.repeat(len(ARMS),1,1),a.repeat(len(ARMS),1,1),states,slow,'qk',forecast=False)['logit']
            z=z.sigmoid().reshape(len(ARMS),len(x)).cpu().numpy()
        if not np.isfinite(z).all():raise FloatingPointError('invalid risk')
        for k,v in zip(ARMS,z):scores[k].append(v)
        if hi>=256 and hi%128==0 and hi<len(signal['values']):
            new,receipt=feedback_states(net,{k:v[:hi] for k,v in signal.items()},slow,log_weights)
            for k in ARMS[1:]:
                state[k]=new[k];q=info[k];q['updates']+=int(receipt['gradient_norm'][k]>0)
                q['opportunities']+=1;q['observed_targets']+=receipt['observed'];q['last_observations'].append(hi-1)
                q['max_state_norm']=max(q['max_state_norm'],float(state[k].norm()))
    return {k:np.concatenate(v) for k,v in scores.items()},info
