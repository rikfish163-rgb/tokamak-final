"""Direct signal-change prediction; the original risk function is unchanged."""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..future_forecast_20261023.diagnostic import issue
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder


class IncrementFeedback(PairedFeedback):
    def __init__(self, base, forecast_mode):
        if forecast_mode not in ('absolute', 'increment'):
            raise ValueError('explicit absolute or increment forecast mode required')
        super().__init__(base, np.ones(23))
        self.forecast_mode = forecast_mode

    def forward(self, values, available, *, mode='risk'):
        result = super().forward(values, available, mode=mode)
        if mode == 'risk' and self.forecast_mode == 'increment':
            valid = available[:, -1:] & torch.isfinite(values[:, -1:])
            last = torch.where(valid, values[:, -1:], self.base.center)
            last = torch.asinh((last - self.base.center) / self.base.scale)
            result['forecast'] = result['forecast'] + last
        return result


def prepare_head(model):
    model.set_stage('frozen')
    with torch.no_grad():
        model.base.forecast_head.weight.zero_()
        model.base.forecast_head.bias.zero_()


def target_mask(x, available, truth, truth_mask):
    return truth_mask & torch.isfinite(truth) & available[:, -1:] & torch.isfinite(x[:, -1:])


def feedback_loss(model, prediction, x, available, truth, truth_mask):
    return model.base.forecast_loss(prediction, truth, target_mask(x, available, truth, truth_mask))


def train_head_step(model, optimizer, x, available, truth, truth_mask):
    if not target_mask(x, available, truth, truth_mask).any():
        raise ValueError('no measured paired target in forecast training batch')
    model.set_stage('frozen');model.base.forecast_head.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    try:
        loss = feedback_loss(model, model(x, available)['forecast'], x, available, truth, truth_mask)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite head training')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    finally:
        model.set_stage('frozen')
    return float(loss.detach())


def update(model, prefix, count, fast, *, device):
    if set(prefix) != {'values', 'available'} or len(prefix['values']) != count:
        raise ValueError('exact observed signal prefix required')
    x, m, target, tm, ends = forecast_examples(prefix, count, model.cfg, device)
    mask = target_mask(x, m, target, tm)
    receipt = dict(last_observation=count-1, first_affected_prediction=count, forecast_origins=ends,
        observed_targets=int(mask.sum()), omitted_unobserved_last=int((tm & ~mask).sum()),
        gradient_steps=0, parameter_delta_norm=0.)
    if not mask.any():
        return fast, receipt  # No truth, no gradient; record the non-update.
    loss = model.base.forecast_loss(model.predict_fast(x, m, fast)['forecast'], target, mask)
    gradients = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=False)
    norm = sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite feedback')
    factor = (1. / norm.clamp_min(1.)).detach()
    new = {k: (v - .001*factor*g).detach().requires_grad_(True) for (k,v),g in zip(fast.items(),gradients)}
    distance = sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    with torch.no_grad():after = model.base.forecast_loss(model.predict_fast(x,m,new)['forecast'],target,mask)
    receipt.update(gradient_steps=1, parameter_delta_norm=float(distance), gradient_norm=float(norm),
                   loss_before=float(loss.detach()), loss_after=float(after))
    return new, receipt


def predict(model, signal, *, device='cuda'):
    if (set(signal) != {'values','available'} or signal['values'].ndim != 2
            or signal['values'].shape[1] != 23 or signal['available'].shape != signal['values'].shape
            or signal['available'].dtype != bool or not len(signal['values'])):
        raise ValueError('nonempty signal-only flat23 input required')
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('W128/H25 experiment required')
    fast, original = initial_encoder(model), initial_encoder(model)
    scores = dict(static=[], intra=[]);updates=[]
    packets = {k:[] for k in ('counts','static','intra','last_value','last_available')}
    n = len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for name,state in (('static',original),('intra',fast)):
                    scores[name].append(logit(model,x,m,state).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        prefix={k:v[:hi] for k,v in signal.items()}
        fast,r=update(model,prefix,hi,fast,device=device);updates.append(r)
        if hi+25<=n:
            packets['counts'].append(hi)
            packets['static'].append(issue(model,prefix,original,device=device))
            packets['intra'].append(issue(model,prefix,fast,device=device))
            packets['last_value'].append(prefix['values'][-1].copy())
            packets['last_available'].append(prefix['available'][-1].copy())
    scores={k:np.concatenate(v) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in scores.values()):raise FloatingPointError('nonfinite risk')
    return scores, updates, {k:np.asarray(v) for k,v in packets.items()}
