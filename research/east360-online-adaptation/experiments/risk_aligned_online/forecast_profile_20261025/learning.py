"""Separate constant forecast-error offsets from arrived temporal changes.

No input centering, risk head, learned event label or probability correction is
added. The decomposition is algebraic; it is not a physical nuisance detector.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import flat,logit,matched_step,apply_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('raw','full','level','shape','hold_shape','shuffle_shape')


def decompose(prediction,target,mask):
    if (prediction.ndim!=3 or prediction.shape!=target.shape or mask.shape!=target.shape
            or mask.dtype!=torch.bool or not torch.isfinite(prediction[mask]).all()
            or not torch.isfinite(target[mask]).all()):
        raise ValueError('matching finite observed forecast targets required')
    r=prediction.masked_fill(~mask,0)-target.masked_fill(~mask,0)
    count=mask.sum((0,1),keepdim=True)
    mean=r.sum((0,1),keepdim=True)/count.clamp_min(1)
    total=mask.sum().clamp_min(1)
    shape=(r-mean).masked_fill(~mask,0)
    return dict(full=r.square().sum()/(2*total),
                level=(mean.square()*count).sum()/(2*total),
                shape=shape.square().sum()/(2*total))


def shuffle_targets(target,mask):
    if target.ndim!=3 or target.shape!=mask.shape or mask.dtype!=torch.bool:
        raise ValueError('three-dimensional target and matching mask required')
    # Fixed randomization of already observed values only. Missing positions
    # are not filled with another channel or interpreted as measured zero.
    result=torch.zeros_like(target);rng=np.random.default_rng(131)
    for c in range(target.shape[-1]):
        observed=target[:,:,c][mask[:,:,c]]
        order=torch.as_tensor(rng.permutation(len(observed)),device=target.device)
        result[:,:,c][mask[:,:,c]]=observed[order]
    return result


def losses_at(model,signal,count,fast,*,device):
    if set(signal)!={'values','available'} or type(count) is not int or not 256<=count<=len(signal['values']):
        raise ValueError('signal-only arrived prefix, at least256 points required')
    x,m,truth,tm,origins=forecast_examples({k:v[:count] for k,v in signal.items()},count,model.cfg,device)
    mask=target_mask(x,m,truth,tm)
    def standard(values):
        return torch.asinh((torch.where(mask,values,model.base.center)-model.base.center)/model.base.scale)
    y=standard(truth).detach();hold=standard(x[:,-1:].expand_as(truth)).detach()
    prediction=model.predict_fast(x,m,fast)['forecast']
    losses=decompose(prediction,y,mask)
    losses['raw']=F.smooth_l1_loss(prediction,y,reduction='none').masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)
    losses['hold_shape']=decompose(prediction,hold,mask)['shape']
    losses['shuffle_shape']=decompose(prediction,shuffle_targets(y,mask),mask)['shape']
    return losses,dict(last_observation=count-1,first_affected_prediction=count,
                        origins=origins,observed=int(mask.sum()))


def feedback(model,signal,count,fast,*,device):
    losses,receipt=losses_at(model,signal,count,fast,device=device)
    gradients={k:flat(torch.autograd.grad(losses[k],tuple(fast.values()),retain_graph=True)).detach() for k in ARMS}
    if not all(torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('nonfinite forecast component gradient')
    error=(gradients['full']-gradients['level']-gradients['shape']).double().norm()
    scale=gradients['full'].double().norm()+gradients['level'].double().norm()+gradients['shape'].double().norm()
    relative=float(error/scale) if scale>0 else 0.
    if relative>3e-5:raise AssertionError('square-loss gradient decomposition inconsistent')
    def cosine(a,b):
        den=gradients[a].double().norm()*gradients[b].double().norm()
        return float(gradients[a].double()@gradients[b].double()/den) if den>0 else None
    receipt.update(losses={k:float(v.detach()) for k,v in losses.items()},
        gradient_norms={k:float(g.double().norm()) for k,g in gradients.items()},
        gradient_decomposition_relative_error=relative,
        raw_level_cosine=cosine('raw','level'),raw_shape_cosine=cosine('raw','shape'),
        shape_hold_cosine=cosine('shape','hold_shape'),shape_shuffle_cosine=cosine('shape','shuffle_shape'))
    return gradients,receipt


def predict(model,signal,*,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signal-only current shot required')
    states={a:initial_encoder(model) for a in ('off',)+ARMS}
    curves={a:[] for a in states};updates={a:[] for a in ARMS};n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(hi,start+32))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for a in states:curves[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        fast=initial_encoder(model);gradients,receipt=feedback(model,signal,hi,fast,device=device)
        for a in ARMS:
            step=matched_step(gradients[a],.001);states[a]=apply_step(fast,step)
            # Keep the same gradient-enabled math backend as the before loss.
            after,_=losses_at(model,signal,hi,states[a],device=device)
            actual=flat([states[a][k].detach()-v.detach() for k,v in fast.items()])
            updates[a].append(dict(receipt,requested_step_norm=float(step.double().norm()),
                actual_step_norm=float(actual.double().norm()),gradient_steps=int(bool(step.any())),
                loss_before=receipt['losses'][a],loss_after=float(after[a].detach()),reset='parent'))
    return {a:np.concatenate(v) for a,v in curves.items()},updates


def eligible(results):
    main,off=results['shape'],results['off']
    return (main['tp']>=off['tp'] and main['fp']<=off['fp']
        and main['danger_loss']<=off['danger_loss'] and main['safe_loss']<=off['safe_loss']
        and all(main['balanced_loss']<results[a]['balanced_loss']-1e-6 for a in ('off','raw','full','level','hold_shape','shuffle_shape')))
