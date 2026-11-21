"""Source-learned update directions in the ORIGINAL final Transformer FFN.

No new risk/forecast head, target scaler, or output correction. Delta = B q,
where B has orthonormal columns and only q is disposable at deployment.
Source outer labels train B; matured signal targets alone update q online.
Inspired by MT-net's learned update subspace and MT3's post-update task
objective, not a reproduction of either paper or a physical mechanism claim.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ffn_intra_20261005.model import FfnModel
from .fast_update import normalized_step


class FfnSubspaceModel(nn.Module):
    adaptation_site = 'original final FFN; source-learned orthonormal parameter directions'

    def __init__(self, base, *, rank=8, seed=131):
        super().__init__()
        self.ffn = FfnModel(base)
        if type(rank) is not int or not 1 <= rank <= self.ffn.delta_count:
            raise ValueError('positive integer rank no larger than FFN dimension required')
        self.temporary_dim = rank
        generator = torch.Generator(device='cpu').manual_seed(seed)
        raw = torch.randn(self.ffn.delta_count, rank, generator=generator)
        basis, _ = torch.linalg.qr(raw, mode='reduced')
        self.basis = nn.Parameter(basis.to(device=base.center.device, dtype=base.center.dtype))
        self.eval()

    @property
    def base(self): return self.ffn.base
    @property
    def cfg(self): return self.base.cfg
    @property
    def center(self): return self.base.center
    @property
    def scale(self): return self.base.scale

    def train(self, mode=True):
        super().train(mode); self.ffn.eval(); return self

    def initial_fast(self, batch):
        if type(batch) is not int or batch < 0:
            raise ValueError('nonnegative integer batch required')
        return self.basis.new_zeros(batch, self.temporary_dim)

    def forward(self, values, available, fast=None):
        fast = self.initial_fast(len(values)) if fast is None else fast
        if (not len(values) or fast.shape != (len(values), self.temporary_dim)
                or fast.dtype != self.basis.dtype or fast.device != self.basis.device
                or not torch.isfinite(fast).all()):
            raise ValueError('matching finite source dtype/device temporary state required')
        # The replay batches many endpoints of ONE shot at the SAME fast state.
        # Independent source episodes must be evaluated separately, not silently
        # assigned the first episode's adaptation state.
        if not torch.equal(fast.detach(), fast[:1].detach().expand_as(fast)):
            raise ValueError('all batched endpoints must share the same shot state')
        return self.ffn(values, available, self.basis @ fast[0])

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)

    @torch.no_grad()
    def project_basis(self):
        if not torch.isfinite(self.basis).all():
            raise FloatingPointError('nonfinite update basis')
        q, r = torch.linalg.qr(self.basis, mode='reduced')
        if bool((r.diag().abs() <= 1e-9).any()):
            raise FloatingPointError('rank-deficient update basis')
        # Fixed positive diagonal convention avoids arbitrary QR sign flips.
        self.basis.copy_(q * r.diag().sign()[None])


def one_step(model, e, *, create_graph=False, step=.05, radius=1., fast=None):
    """A source diagnostic of exactly ONE online, signal-only proposal.

    The later-window qx/qa and source label y are not read here. Truth stays in
    the parent's fixed source-asinh coordinate. Acceptance uses the same
    roundoff-aware signal improvement as the generic causal replay.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (step, radius)):
        raise ValueError('positive finite temporary budget required')
    if len(e['x']) != 1 or e['truth'].shape != (1, 25, 23):
        raise ValueError('one independent direct-H25 episode required')
    mask = e['mask'] & torch.isfinite(e['truth'])
    if not mask.any():
        raise ValueError('at least one observed signal target required')
    fast = model.initial_fast(1).requires_grad_() if fast is None else fast
    if (fast.shape != (1, model.temporary_dim) or not fast.requires_grad
            or fast.dtype != model.basis.dtype or fast.device != model.basis.device
            or not torch.isfinite(fast).all()):
        raise ValueError('matching gradient-enabled current temporary state required')
    prediction = model(e['x'], e['a'], fast)['forecast']
    loss = model.forecast_loss(prediction, e['truth'], mask)
    gradient, = torch.autograd.grad(loss, fast, create_graph=create_graph,
                                   retain_graph=create_graph)
    proposed = normalized_step(fast, torch.zeros_like(fast), gradient, step=step, radius=radius)
    after = model(e['x'], e['a'], proposed)['forecast']
    new_loss = model.forecast_loss(after, e['truth'], mask)
    if not all(torch.isfinite(t).all() for t in (loss, gradient, proposed, new_loss)):
        raise FloatingPointError('nonfinite subspace signal update')
    tolerance = 8 * torch.finfo(loss.dtype).eps * max(float(loss.detach()),
                                                    float(new_loss.detach()), 1e-12)
    accepted = float((loss - new_loss).detach()) > tolerance
    return (proposed if accepted else fast), dict(
        signal_before=float(loss.detach()), signal_after=float(new_loss.detach()),
        accepted=accepted, coefficient_norm=float(proposed.detach().double().norm()),
        gradient_norm=float(gradient.detach().double().norm()))


def risk_objective(after, before, *, protect):
    """Source-only cost: ordinary BCE plus the SAME episode's worsening.

    The coefficient is one: a harmful infinitesimal change has twice the cost
    of a beneficial one. This is an experiment, NOT a no-regression guarantee.
    The reference is detached; no teacher constraint is used inside the online
    signal update and no labels enter that update.
    """
    return after + F.relu(after - before.detach()) if protect else after


def meta_step(model, optimizer, episodes, *, mode):
    """Train B only; risk and signal controls share the exact sample/update budget."""
    if mode not in ('risk', 'signal', 'risk_protect') or not episodes:
        raise ValueError('risk/signal mode and source episodes required')
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if ids != [id(model.basis)]:
        raise ValueError('optimizer must contain only the update basis')
    model.zero_grad(set_to_none=True)
    rows = []
    for e in episodes:
        if e['y'].shape != (1,) or not torch.isfinite(e['y']).all() or not ((e['y'] == 0) | (e['y'] == 1)).all():
            raise ValueError('actual binary SOURCE endpoint label required')
        fast, row = one_step(model, e, create_graph=True)
        before = model(e['qx'], e['qa'], model.initial_fast(1).requires_grad_())['logit']
        after = model(e['qx'], e['qa'], fast)['logit']
        risk_before = F.binary_cross_entropy_with_logits(before.detach(), e['y'])
        risk_after = F.binary_cross_entropy_with_logits(after, e['y'])
        objective = (risk_objective(risk_after, risk_before, protect=mode == 'risk_protect')
                     if mode != 'signal' else model.forecast_loss(
                         model(e['x'], e['a'], fast)['forecast'], e['truth'], e['mask']))
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite source meta objective')
        (objective / len(episodes)).backward()
        rows.append(dict(row, risk_before=float(risk_before), risk_after=float(risk_after.detach())))
    if model.basis.grad is None:
        raise RuntimeError('missing source update-geometry gradient')
    norm = torch.nn.utils.clip_grad_norm_([model.basis], 1., error_if_nonfinite=True)
    optimizer.step()
    model.project_basis()
    return dict(basis_grad_norm=float(norm), episodes=len(rows),
                accepted=sum(row['accepted'] for row in rows),
                risk_before=sum(row['risk_before'] for row in rows) / len(rows),
                risk_after=sum(row['risk_after'] for row in rows) / len(rows),
                signal_after=sum(row['signal_after'] for row in rows) / len(rows))
