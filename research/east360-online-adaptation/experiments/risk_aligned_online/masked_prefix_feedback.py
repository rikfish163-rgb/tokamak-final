"""Single-shot internal adaptation from withheld, ALREADY observed patches.

Inspired by masked-reconstruction test-time training, not a reproduction of
image MAE. A source-only auxiliary decoder provides the signal loss; it never
enters the risk path. The qualified classifier and its input scale stay fixed.
Only a disposable delta of its final Transformer FFN changes inside an EAST
shot. Fresh/stale arms share update times, masks, step size and parameter scope.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import simple_data
from .fast_update import normalized_step
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.intra import _batch_windows


def patch_mask(available, *, phase):
    """Hide every fourth 8-point patch. Four phases partition the 128 points."""
    if (available.ndim != 3 or available.shape[1:] != (128, 23)
            or available.dtype != torch.bool or type(phase) is not int or phase not in range(4)):
        raise ValueError('boolean Bx128x23 availability and integer phase 0..3 required')
    temporal = ((torch.arange(128, device=available.device)//8) % 4 == phase)
    return available & temporal[None, :, None]


class MaskedPrefixModel(PersistentFfnModel):
    def __init__(self, base):
        super().__init__(base)
        if base.cfg.patch != 8:
            raise ValueError('source patch length 8 required')
        self.decoder = nn.Linear(64, 8*23).to(self.center)
        self.requires_grad_(False)

    def reconstruction(self, values, available, fast, *, phase):
        available = available & torch.isfinite(values)
        hidden = patch_mask(available, phase=phase)
        visible = available & ~hidden
        masked = torch.where(visible, values.detach(), self.center)
        target = torch.asinh((torch.where(available, values.detach(), self.center)-self.center)/self.scale)
        features = []
        # The original risk forward zeroes invalid patch outputs after norm.
        # Capture BEFORE that operation: withheld queries attend to observed
        # keys, but neither their raw values nor their mask bits enter encoder.
        handle = self.base.norm.register_forward_hook(lambda module, inputs, output: features.append(output))
        try:
            self(masked, visible, fast)
        finally:
            handle.remove()
        if len(features) != 1:
            raise RuntimeError('exactly one original norm activation required')
        prediction = self.decoder(features[0][:, :16]).reshape(-1, 128, 23)
        return prediction, target.detach(), hidden

    def reconstruction_loss(self, values, available, fast, *, phase):
        prediction, target, hidden = self.reconstruction(values, available, fast, phase=phase)
        error = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~hidden, 0.)
        # Equal episode weighting; absent target coordinates contribute zero.
        return (error.sum((1, 2))/hidden.sum((1, 2)).clamp_min(1)).mean()


def predict_shot(model, shot, *, intra, lag=0, device='cpu', stride=100, step=.05,
                 radius=1., objective='level', data_api=simple_data):
    """Predict first; feedback at tick t affects only t+1 and later endpoints.

    Both arms start at t=152: t-25 has a complete 128-point window. The fresh
    arm reconstructs [t-127,t], the stale control [t-152,t-25]. No previous-shot
    statistics, natural outcomes, reconstruction targets or risk scores persist.
    """
    if set(shot) & {'y', 'is_disrupt', 'down', 'horizon', 'DownTime', 'IsDisrupt'}:
        raise ValueError('outcome fields are not allowed in single-shot prediction')
    if lag not in (0, 25) or stride != 100 or step != .05 or radius != 1. or objective != 'level':
        raise ValueError('locked comparison: lag 0/25, stride100, step.05, radius1, level')
    values, available = np.asarray(shot['values']), np.asarray(shot['available'])
    times = np.asarray(shot['times'])
    if (values.shape != (len(times), 23) or available.shape != values.shape
            or available.dtype != bool or not len(times) or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()):
        raise ValueError('nonempty ordered diagnostic record required')
    if model.memory.detach().count_nonzero():
        raise ValueError('pure intra experiment cannot start with persistent target memory')
    signals = dict(values=values, available=available)
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    fast = model.initial_fast(1).detach().requires_grad_()
    initial = fast.detach().clone()
    first = model.summarize_fast(fast)
    scores = np.empty(len(times), dtype=np.float32)
    ticks = list(range(152, len(times)-1, stride))
    attempts = []
    start = 0
    for block, end in enumerate([*ticks, len(times)-1]):
        # Each window reader receives only its own prefix. The batch is an
        # efficiency device: no normalization or attention crosses examples.
        for lo in range(start, end+1, 100):
            hi = min(lo+100, end+1)
            x, a = _batch_windows(data_api, signals, range(lo, hi), model, device)
            output = model(x, a, fast.expand(hi-lo, -1))['logit']
            scores[lo:hi] = output.detach().sigmoid().cpu().numpy()
        start = end+1
        if end not in ticks or not intra:
            continue
        feedback_end = end-lag
        x, a = _batch_windows(data_api, signals, [feedback_end], model, device)
        phase = block % 4
        hidden_count = int(patch_mask(a, phase=phase).sum())
        row = dict(update_index=end, feedback_end_index=feedback_end,
            feedback_end_time=float(times[feedback_end]), update_time=float(times[end]),
            predictions_sealed_through_index=end, effective_index=end+1,
            phase=phase, hidden_targets=hidden_count, accepted=False,
            fast_before=model.summarize_fast(fast), fast_after=model.summarize_fast(fast),
            loss_before=None, loss_after=None, reason='no_observed_hidden_targets')
        if hidden_count:
            loss = model.reconstruction_loss(x, a, fast, phase=phase)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite masked-prefix loss')
            gradient, = torch.autograd.grad(loss, fast)
            candidate = normalized_step(fast.detach(), initial, gradient.detach(), step=step, radius=radius)
            candidate = candidate.detach().requires_grad_()
            after = model.reconstruction_loss(x, a, candidate, phase=phase)
            if not torch.isfinite(after):
                raise FloatingPointError('nonfinite masked-prefix candidate loss')
            accepted = bool(after < loss.detach())
            row.update(loss_before=float(loss.detach()), loss_after=float(after.detach()),
                gradient_norm=float(gradient.detach().double().norm()), accepted=accepted,
                reason='same_observed_loss_decreased' if accepted else 'same_observed_loss_not_decreased')
            if accepted:
                fast = candidate
                row['fast_after'] = model.summarize_fast(fast)
            fast = fast.detach().requires_grad_()
        attempts.append(row)
    def change_norm(current, original):
        return float(torch.stack([(v.detach()-original[k]).double().square().sum()
                                  for k, v in current]).sum().sqrt())
    parameter_change = change_norm(model.named_parameters(), parameters)
    buffer_change = change_norm(model.named_buffers(), buffers)
    if parameter_change != 0 or buffer_change != 0:
        raise AssertionError('single-shot feedback changed registered model state')
    return scores, dict(intra=bool(intra), lag=lag, score_count=len(times),
        attempted_updates=len(attempts), accepted_updates=sum(r['accepted'] for r in attempts),
        attempts=attempts, fast_initial=first, fast_final=model.summarize_fast(fast),
        fast_discarded=True, auxiliary_decoder_updated=False, outcome_access=False,
        normalization='immutable source-only center/scale; no target record statistics',
        model_parameter_change_norm=parameter_change, model_buffer_change_norm=buffer_change)
