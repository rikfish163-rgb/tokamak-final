"""Single-shot signal-only updates, consumed once after H25 observations arrive."""
import math
import time

import numpy as np
import torch

from .. import simple_data
from ..fast_update import normalized_step
from ..redesign_20261004.intra import _batch_windows
from .loss import context_features, signal_components, weighted_signal_loss


def signal_step(model, delta, x, available, truth, observed, *, step, radius,
                rule=None, parent_forecast=None, create_graph=False):
    """Shared source-training/target-replay update. Caller owns arrival timing."""
    out = model(x, available, delta)
    parts = signal_components(out['forecast'], truth, observed, model.center, model.scale)
    weights = None
    if rule is not None:
        if parent_forecast is None:
            parent_forecast = model(x, available, model.initial_delta())['forecast'].detach()
        weights = rule(context_features(parent_forecast, truth, observed, model.center, model.scale))
    loss = weighted_signal_loss(parts, weights).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite mature signal loss')
    gradient, = torch.autograd.grad(loss, delta, create_graph=create_graph)
    updated = normalized_step(delta[None], torch.zeros_like(delta)[None], gradient[None],
                              step=step, radius=radius)[0]
    if not torch.isfinite(updated).all():
        raise FloatingPointError('nonfinite temporary FFN update')
    return updated, loss, gradient


def predict_shot(model, shot, intra=True, device='cpu', stride=26, step=.03,
                 radius=.3, rule=None, data_api=simple_data, offline_source=False):
    if (type(stride) is not int or stride <= 25
            or any(not math.isfinite(v) or v <= 0 for v in (step,radius))):
        raise ValueError('nonoverlapping H25 origins and positive finite budgets required')
    if shot.get('device','EAST') != ('J-TEXT' if offline_source else 'EAST'):
        raise ValueError('online is EAST only; source replay must explicitly be offline auxiliary development')
    signals = {k:np.asarray(shot[k]) for k in ('values','available')}
    times=np.asarray(shot['times'])
    n=len(times)
    if (not n or times.shape != (n,) or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or signals['values'].shape!=(n,23)
            or signals['available'].shape!=(n,23) or signals['available'].dtype!=bool):
        raise ValueError('finite increasing timestamps and matching flat23 observations required')
    scores=np.empty(n,np.float32)
    events=[]
    delta=model.initial_delta()
    next_origin=0
    pending=None
    start=0
    started=time.perf_counter()
    while start<n:
        stop=min(n,start+256)
        if intra:
            stop=min(stop,next_origin+1)
        if pending is not None:
            stop=min(stop,pending['origin']+26)
        x,a=_batch_windows(data_api,signals,range(start,stop),model,device)
        with torch.enable_grad():
            out=model(x,a,delta)
            if not torch.isfinite(out['logit']).all():
                raise FloatingPointError('nonfinite emitted risk')
            scores[start:stop]=out['logit'].detach().sigmoid().cpu().numpy()
            end=stop-1  # These predictions are irrevocably emitted before update.
            if intra and end==next_origin:
                pending=dict(origin=end,x=x[-1:].detach(),a=a[-1:].detach(),
                    issued=out['forecast'][-1:].detach())
                if rule is not None:
                    pending['parent']=model(pending['x'],pending['a'],model.initial_delta())['forecast'].detach()
                next_origin+=stride
            if pending is not None and end==pending['origin']+25:
                origin=pending['origin']
                truth=torch.as_tensor(signals['values'][origin+1:end+1],
                    device=device,dtype=model.center.dtype)[None]
                observed=torch.as_tensor(signals['available'][origin+1:end+1],device=device)[None]
                observed=observed & torch.isfinite(truth)
                record=dict(origin_index=origin,maturity_index=end,effective_index=end+1 if end+1<n else None,
                    valid_targets=int(observed.sum()),accepted=False,reason='no_valid_targets',
                    delta_norm=float(delta.detach().double().norm()),loss_before=None,loss_after=None)
                if observed.any():
                    before=time.perf_counter()
                    candidate,loss,gradient=signal_step(model,delta,pending['x'],pending['a'],truth,observed,
                        step=step,radius=radius,rule=rule,parent_forecast=pending.get('parent'))
                    delta=candidate.detach().requires_grad_()
                    # This is a diagnostic, NOT acceptance by risk or future labels.
                    after=model(pending['x'],pending['a'],delta)['forecast']
                    parts=signal_components(after,truth,observed,model.center,model.scale)
                    weights=(rule(context_features(pending['parent'],truth,observed,model.center,model.scale))
                             if rule is not None else None)
                    loss_after=weighted_signal_loss(parts,weights).mean().detach()
                    record.update(accepted=bool(gradient.detach().norm()>0),reason='finite_bounded_signal_step',
                        gradient_norm=float(gradient.detach().double().norm()),
                        delta_norm=float(delta.detach().double().norm()),loss_before=float(loss.detach()),
                        loss_after=float(loss_after),update_wall_seconds=time.perf_counter()-before)
                events.append(record)
                pending=None
        start=stop
    return scores,dict(events=events,accepted_updates=sum(r['accepted'] for r in events),
        intra=bool(intra),rule='learned' if rule is not None else 'uniform',step=step,radius=radius,stride=stride,
        temporary_reset=True,parameter_count=model.delta_count,parameter_names=model.parameter_names,
        outcome_access=False,alarm_adaptation=False,elapsed_seconds=time.perf_counter()-started,
        latency_scope='batched offline replay wall time, NOT demonstrated real-time latency')
