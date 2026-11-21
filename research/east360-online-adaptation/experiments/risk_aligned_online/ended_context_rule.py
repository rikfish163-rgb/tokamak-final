"""Learn mature-signal loss weights from SEALED and ENDED EAST experience.

Inter-trained intra adaptation, not pure source-only intra. The rule is learned
after the first30 shots, frozen before later shots, and never consumes a current
outcome. Source classifier, decoder, input coordinates and alarm stay fixed.
Only a disposable additive FFN state changes within each shot. This bounded
pilot uses a sealed-peak surrogate; it is not proof of physical-law learning.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import simple_data
from .channel_gradient_audit import channel_losses
from .fast_update import normalized_step
from .ffn_intra_20261005.loss import context_features
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.intra import _batch_windows


class ContextRule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(92, 32), nn.SiLU(), nn.Linear(32, 23))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(3./4.))

    def forward(self, context):
        return 2. - 1.75*torch.sigmoid(-self.net(context))


class ContextFfnModel(PersistentFfnModel):
    def __init__(self, base):
        super().__init__(base)
        self.memory.requires_grad_(False)
        self.rule = ContextRule().to(device=self.center.device, dtype=self.center.dtype)

    def prepare_intra_forecast_loss(self, issued, truth, observed):
        return fixed_signal_loss(self, issued, truth, observed)


def fixed_signal_loss(model, issued, truth, observed):
    if issued.shape[0] != 1:
        raise ValueError('one same-shot mature forecast context required')
    weights = model.rule(context_features(issued.detach(), truth, observed,
                                          model.center, model.scale))[0]

    def loss(prediction, target, mask):
        # At initial weights=1, use the EXACT archived total loss/gradient,
        # while keeping a differentiable route from the rule through the step.
        parts = channel_losses(prediction, target, mask, model.center, model.scale)
        return model.forecast_loss(prediction, target, mask) + ((weights-1.)*parts).sum()

    loss.weights = weights
    return loss


def peak_index(shot, scores):
    if (shot.get('device') != 'EAST' or '_sealed_scores' not in shot
            or shot.get('horizon') != 125. or 'is_disrupt' not in shot):
        raise ValueError('sealed ended EAST shot with original125ms label contract required')
    scores = np.asarray(scores)
    valid = np.asarray(shot['valid'], bool).copy()
    if scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError('one finite sealed probability per endpoint required')
    if shot['is_disrupt']:
        lead = np.round((shot['down']-shot['times'])*1000., 9)
        valid &= lead > 30.
    eligible = np.flatnonzero(valid)
    if not len(eligible):
        raise ValueError('no eligible operational peak')
    return int(eligible[np.argmax(scores[eligible])])


def causal_peak_logit(model, shot, query, *, create_graph=False, step=.05, radius=1.,
                      stride=100, unroll_depth=4):
    if type(query) is not int or not 0 <= query < len(shot['times']):
        raise ValueError('existing query endpoint required')
    signals = {k: shot[k] for k in ('values', 'available')}
    fast = model.initial_fast(1).requires_grad_()
    origins = list(range(0, query-25, stride))
    burn = max(0, len(origins)-unroll_depth)
    accepted = differentiated = 0
    for j, origin in enumerate(origins):
        x, a = _batch_windows(simple_data, signals, [origin], model, model.center.device)
        truth = torch.as_tensor(signals['values'][origin+1:origin+26],
                               device=model.center.device, dtype=model.center.dtype)[None]
        observed = torch.as_tensor(signals['available'][origin+1:origin+26],
                                   device=model.center.device)[None] & torch.isfinite(truth)
        if observed.any():
            out = model(x, a, fast)
            loss_fn = model.prepare_intra_forecast_loss(out['forecast'].detach(), truth, observed)
            before = loss_fn(out['forecast'], truth, observed)
            differentiating = create_graph and j >= burn
            gradient, = torch.autograd.grad(before, fast, create_graph=differentiating)
            candidate = normalized_step(fast, torch.zeros_like(fast), gradient,
                                        step=step, radius=radius)
            after = loss_fn(model(x, a, candidate)['forecast'], truth, observed)
            tolerance = 8*torch.finfo(before.dtype).eps*max(
                abs(float(before.detach())), abs(float(after.detach())), 1e-12)
            if float((before-after).detach()) > tolerance:
                fast = candidate
                accepted += 1
                differentiated += int(differentiating)
        if j < burn or not create_graph:
            fast = fast.detach().requires_grad_()
    x, a = _batch_windows(simple_data, signals, [query], model, model.center.device)
    logit = model(x, a, fast)['logit'].reshape(())
    return logit, dict(mature_events=len(origins), burn_in_events=burn,
                       accepted_updates=accepted, differentiated_updates=differentiated,
                       fast_norm=float(fast.detach().double().norm()))


def train_pair(model, pair, optimizer, *, boundary):
    allowed = {id(p) for p in model.rule.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != allowed or len(actual) != len(allowed):
        raise ValueError('only contextual rule parameters may be optimized')
    if (len(pair) != 2 or sorted(int(s['is_disrupt']) for s, _ in pair) != [0, 1]
            or any(s.get('device') != 'EAST' or '_sealed_scores' not in s
                   or s.get('horizon') != 125. for s, _ in pair)):
        raise ValueError('one balanced pair of sealed ended EAST shots required')
    if not math.isfinite(boundary):
        raise ValueError('finite frozen alarm logit boundary required')
    frozen = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
    rule_before = [p.detach().clone() for p in model.rule.parameters()]
    model.zero_grad(set_to_none=True)
    records = []
    for shot, query in pair:
        logit, row = causal_peak_logit(model, shot, query, create_graph=True)
        # This is the risk AT THE CAUSALLY ADAPTED state. Neither a before-only
        # risk loss nor the signal loss can substitute for this outer objective.
        risk = F.softplus((1-2*int(shot['is_disrupt']))*(logit-boundary))
        if risk.requires_grad:
            (risk/2).backward()
        records.append(dict(**row, shot=shot['shot'], label=int(shot['is_disrupt']),
                            query=query, probability=float(logit.detach().sigmoid()),
                            adapted_risk_loss=float(risk.detach())))
    norm = torch.nn.utils.clip_grad_norm_(model.rule.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.rule.parameters()):
        raise FloatingPointError('nonfinite learned rule')
    if model.memory.any() or model.memory.grad is not None:
        raise AssertionError('rule training changed or differentiated persistent FFN')
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
    delta = sum(float((p.detach().double()-q.double()).square().sum())
                for p, q in zip(model.rule.parameters(), rule_before))**.5
    return dict(records=records, gradient_norm=float(norm), rule_delta_norm=delta,
                objective=float(np.mean([r['adapted_risk_loss'] for r in records])))
