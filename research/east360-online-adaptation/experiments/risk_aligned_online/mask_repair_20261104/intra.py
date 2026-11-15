"""Observation-only, shot-local latent repair. No I/O, labels or risk loss.

The caller must seal predictions through the current endpoint before calling
step(), use only its arrived prefix, and discard/reset state at shot boundaries.
This module alone is not a prequential experiment or evidence of alarm gain.
"""

import math

import torch
from torch import nn

from . import core
from .training import native_embedding

MODES = ('real', 'hold', 'reverse', 'random', 'spatial', 'no_update')


def initial_state(repair):
    return next(repair.parameters()).new_zeros(16)


def _bound(raw, radius):
    rms2 = raw.double().square().mean(-1, keepdim=True)
    ceiling = radius * (1 - 4 * torch.finfo(raw.dtype).eps)
    factor = torch.where(rms2 > radius**2,
                         ceiling / rms2.clamp_min(radius**2).sqrt(), 1.0)
    return (raw.double() * factor).to(raw.dtype)


class LocalRepair(core.TokenRepair):
    """Frozen persistent weights, differentiable 16-dimensional local state."""

    def __init__(self, persistent, state):
        nn.Module.__init__(self)
        if not isinstance(persistent, core.TokenRepair):
            raise TypeError('persistent TokenRepair required')
        parameter = next(persistent.parameters())
        if (state.shape != (16,) or state.dtype != parameter.dtype
                or state.device != parameter.device or not torch.isfinite(state).all()
                or any(p.requires_grad for p in persistent.parameters())
                or any(m.training for m in persistent.modules())):
            raise ValueError('frozen persistent repair and finite matched 16-state required')
        self.net = persistent.net
        self.state = state
        self.eval()

    def forward(self, context):
        hidden = self.net[1](self.net[0](context))
        with torch.no_grad():
            original = self.net[3](self.net[2](hidden))
            bounded = _bound(original, 0.25)
        changed = self.net[3](self.net[2](hidden + self.state))
        # Do not branch on state == 0: zero initialization must retain its
        # derivative. The original term is the exact persistent intervention.
        extra = 0.025 * torch.tanh((changed - original) / 0.025)
        return bounded + extra


def support(available):
    if available.shape != (1, 128, 23) or available.dtype != torch.bool:
        raise ValueError('one arrived 128x23 boolean mask required')
    hidden = available.clone()
    hidden[:, :, 1] = False
    real_frames = available.any(-1)
    unchanged = (real_frames == hidden.any(-1)).reshape(1, 16, 8).all(-1)
    enough = available[:, :, 1].reshape(1, 16, 8).sum(-1) >= 4
    eligible = unchanged & enough
    train, held = eligible.clone(), eligible.clone()
    train[:, 12:] = False
    held[:, :12] = False
    return train, held


def embedding_after(base, repair, x, available):
    """Exact core embedding intervention without unused Transformer outputs.

    Source-view patch construction is already fidelity-tested in training.py;
    retain core's source/input/parameter validation and gating here. Tensor and
    temporary-state derivative parity are tested against the complete forward.
    This is computation reuse, not a different auxiliary or risk objective.
    """
    core._check(base, repair, x, available, True)
    embedding = native_embedding(base, x, available)
    if not torch.isfinite(embedding).all():
        raise FloatingPointError('nonfinite source patch embedding')
    grouped = available.reshape(len(x), 16, 8, 23)
    valid = grouped.any(-1)
    count = valid.sum(-1)
    fractions = grouped[..., (1, 2)].sum(-2).to(x.dtype) / count.clamp_min(1)[..., None]
    tokens = (count > 0) & (fractions < 1).any(-1)
    if not tokens.any():
        return embedding
    raw = repair(torch.cat((embedding, fractions), -1)[tokens])
    if raw.shape != (int(tokens.sum()), 64) or not torch.isfinite(raw).all():
        raise FloatingPointError('finite eligible token intervention required')
    delta = torch.zeros_like(embedding)
    delta[tokens] = _bound(raw, .25)
    return torch.where(tokens[..., None], embedding + delta, embedding)


def _control_values(x, a, mode, seed):
    values = x.clone()
    idx = torch.nonzero(a[0, :, 1], as_tuple=False).flatten()
    if len(idx) and mode in ('hold', 'reverse', 'random'):
        observed = x[0, idx, 1]
        if mode == 'hold':
            target = observed[0].expand_as(observed)
        elif mode == 'reverse':
            target = observed.flip(0)
        else:
            generator = torch.Generator(device=x.device).manual_seed(seed)
            target = observed[torch.randperm(len(idx), generator=generator, device=x.device)]
        values[0, idx, 1] = target
    return values


def step(base, persistent, state, x, available, *, mode='real', seed=131):
    """One normalized update from extra-hidden real CIII, never from outcomes.

    Held-out patch errors are returned only as diagnostics, not an acceptor.
    All modes execute three persistent and two student embedding calls, one backward.
    No-update computes the same gradient then discards its proposal.
    """
    if (mode not in MODES or type(seed) is not int or x.shape != (1, 128, 23)
            or available.shape != x.shape or available.dtype != torch.bool
            or available[:, :, 2].any()):
        raise ValueError('declared mode and real no-MNM single-prefix input required')
    LocalRepair(persistent, state)
    if state.double().norm() > 0.2500001:
        raise ValueError('shot-local state exceeds its declared L2 radius')
    train, held = support(available)
    info = dict(mode=mode, train_patches=int(train.sum()), held_patches=int(held.sum()))
    if train.sum() < 4 or held.sum() < 1:
        return state.detach().clone(), dict(info, status='INSUFFICIENT_OBSERVATIONS', gradient_norm=0.0)
    masked = available.clone()
    masked[:, :, 1] = False
    with torch.no_grad():
        real = embedding_after(base, persistent, x, available).detach()
        hidden = embedding_after(base, persistent, x, masked).detach()
        controlled = embedding_after(base, persistent,
                                     _control_values(x, available, mode, seed), available).detach()
        if mode == 'spatial':
            generator = torch.Generator(device=x.device).manual_seed(seed)
            permutation = torch.randperm(64, generator=generator, device=x.device)
            target = hidden + (real - hidden)[..., permutation]
        else:
            target = controlled
    current = state.detach().clone().requires_grad_(True)
    before = embedding_after(base, LocalRepair(persistent, current), x, masked)
    loss = (before[train] - target[train]).square().mean() + 0.01 * current.square().sum()
    gradient, = torch.autograd.grad(loss, current)
    norm = float(gradient.double().norm())
    if not math.isfinite(norm) or not torch.isfinite(loss):
        raise FloatingPointError('nonfinite observation-only latent gradient')
    candidate = current.detach().double()
    if norm >= 1e-8:
        candidate = candidate - 0.025 * gradient.detach().double() / norm
        radius = float(candidate.norm())
        if radius > 0.25:
            candidate *= 0.25 * (1 - 4 * torch.finfo(state.dtype).eps) / radius
    proposed = candidate.to(state.dtype)
    result = torch.zeros_like(state) if mode == 'no_update' else proposed
    with torch.no_grad():
        after = embedding_after(base, LocalRepair(persistent, result), x, masked)
    return result.detach(), dict(
        info, status='CONTROL_NO_UPDATE' if mode == 'no_update' else ('UPDATED' if norm >= 1e-8 else 'ZERO_GRADIENT'),
        gradient_norm=norm, state_norm=float(result.double().norm()),
        proposal_norm=float(proposed.double().norm()),
        train_before=float((before.detach()[train] - target[train]).square().mean()),
        train_after=float((after[train] - target[train]).square().mean()),
        held_real_before=float((before.detach()[held] - real[held]).square().mean()),
        held_real_after=float((after[held] - real[held]).square().mean()),
        backward_calls=1, persistent_embedding_calls=3, student_embedding_calls=2,
        full_classifier_forward_calls_in_step=0)
