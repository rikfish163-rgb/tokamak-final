"""Maturity-clock ablation, not a new classifier or oracle update policy.

The clock is fixed before each shot and accepts signals only. All arms start
after observation 255. Increasing update frequency scales the learning rate
by interval/128: this matches a per-time clipped-step upper bound, NOT total
compute, target reuse, actual parameter travel or optimization trajectories.
"""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples,adapt_arrived
from ..paired_feedback_20261022.model import initial_encoder


def update_counts(length,interval):
    if type(length) is not int or length<1 or type(interval) is not int or interval<1:
        raise ValueError('positive integer record length and update interval required')
    return list(range(256,length,interval))


def feedback_step(model,signal,count,state,*,targets,interval,device):
    if type(targets) is not int or targets not in (1,4):
        raise ValueError('one latest or four matured forecast targets required')
    update_counts(len(signal['values']),interval)
    if model.cfg.window!=128 or model.cfg.horizon!=25:
        raise ValueError('fixed W128 H25 comparison required')
    rate=.001*interval/128
    if targets==4:
        fast,receipt=adapt_arrived(model,signal,count,state,device=device,learning_rate=rate)
    else:
        x,m,target,tm,origins=forecast_examples(signal,count,model.cfg,device)
        x,m,target,tm=[v[:1] for v in (x,m,target,tm)];origins=origins[:1]
        observed=int(tm.sum())
        receipt=dict(last_observation=count-1,first_affected_prediction=count,
            forecast_origins=origins,observed_targets=observed,gradient_steps=0,parameter_delta_norm=0.)
        fast=state
        if observed:
            prediction=model.predict_fast(x,m,state)['forecast']
            loss=model.base.forecast_loss(prediction,target,tm)
            grads=torch.autograd.grad(loss,tuple(state.values()),allow_unused=False)
            norm=sum(g.square().sum() for g in grads).sqrt()
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('nonfinite freshest-matured feedback')
            factor=(1./norm.clamp_min(1.)).detach()
            fast={k:(v-rate*factor*g).detach().requires_grad_(True)
                  for (k,v),g in zip(state.items(),grads)}
            delta=sum((fast[k].detach()-v.detach()).double().square().sum() for k,v in state.items()).sqrt()
            with torch.no_grad():after=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],target,tm)
            receipt.update(gradient_steps=1,gradient_norm=float(norm),parameter_delta_norm=float(delta),
                loss_before=float(loss.detach()),loss_after=float(after))
    return fast,dict(receipt,learning_rate=rate,target_segments=targets,clock_interval=interval)


def predict_clock(model,signal,*,interval,device='cuda'):
    if set(signal)!={'values','available'}:
        raise ValueError('signals only, no outcome/event fields')
    length=len(signal['values']);counts=update_counts(length,interval)
    boundaries=[0,*counts,length];state={a:initial_encoder(model) for a in ('four','latest')}
    prediction={a:[] for a in state};updates={a:[] for a in state}
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,end,model.cfg.window) for end in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            # Preserve the reference's grad-enabled attention execution path;
            # torch.no_grad selects a different fused CPU path (regression
            # measured ~9e-8 differences). We detach predictions immediately.
            for arm in state:
                h=model.predict_fast(x,m,state[arm],mode='risk_state')['risk_state']
                p=model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy()
                prediction[arm].append(p)
        if hi<length:
            for arm,targets in (('four',4),('latest',1)):
                state[arm],receipt=feedback_step(model,signal,hi,state[arm],targets=targets,
                    interval=interval,device=device)
                updates[arm].append(receipt)
    prediction={k:np.concatenate(v) for k,v in prediction.items()}
    if not all(len(v)==length and np.isfinite(v).all() for v in prediction.values()):
        raise FloatingPointError('incomplete/nonfinite stream predictions')
    return prediction,updates
