"""Causal latent TTT-Linear memory on frozen risk-head representations.

This is a small TTT-inspired prototype, not a paper reproduction and not
validated for tokamak performance or safety.  It adapts only an explicit
16x16 latent learner state with an unlabeled K-to-V objective; the base risk
logits and the feature encoder remain outside this module.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentMemoryConfig:
    input_dim: int = 64
    state_dim: int = 16
    chunk_size: int = 16
    eta_max: float = .1
    state_radius: float = 1.
    grad_clip: float = 1.
    eps: float = 1e-6

    def __post_init__(self):
        if min(self.input_dim, self.state_dim, self.chunk_size) < 1:
            raise ValueError('input_dim, state_dim, and chunk_size must be positive')
        if self.eta_max <= 0 or self.state_radius <= 0 or self.grad_clip <= 0:
            raise ValueError('eta_max, state_radius, and grad_clip must be positive')
        if self.eps <= 0:
            raise ValueError('eps must be positive')


@dataclass(frozen=True)
class LatentMemoryState:
    """Per-shot learner state, including a not-yet-complete chunk."""

    W: torch.Tensor
    pending_q: torch.Tensor
    pending_k: torch.Tensor
    pending_v: torch.Tensor
    pending_observed: torch.Tensor


class LatentMemory(nn.Module):
    """Add a causal, unlabeledly adapted latent delta to frozen base logits."""

    def __init__(self, cfg):
        super().__init__()
        if not isinstance(cfg, LatentMemoryConfig):
            raise TypeError('cfg must be a LatentMemoryConfig')
        self.cfg = cfg
        self.embedding = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.key = nn.Linear(cfg.input_dim, cfg.state_dim, bias=False)
        self.value = nn.Linear(cfg.input_dim, cfg.state_dim, bias=False)
        self.query = nn.Linear(cfg.input_dim, cfg.state_dim, bias=False)
        self.W0 = nn.Parameter(torch.eye(cfg.state_dim))
        self.eta = nn.Parameter(torch.tensor(-2.))
        self.readout = nn.Linear(cfg.state_dim, 1, bias=False)
        nn.init.normal_(self.readout.weight, mean=0., std=.01)

    def initial_state(self, batch):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        empty = self.W0.new_empty(batch, 0, self.cfg.state_dim)
        empty_observed = torch.zeros(batch, 0, dtype=torch.bool,
                                     device=self.W0.device)
        return LatentMemoryState(
            W=self.W0.unsqueeze(0).expand(batch, -1, -1).clone(),
            pending_q=empty.clone(),
            pending_k=empty.clone(),
            pending_v=empty.clone(),
            pending_observed=empty_observed,
        )

    def _validate_inputs(self, features, base_logits, observed):
        if features.ndim != 3 or features.shape[-1] != self.cfg.input_dim:
            raise ValueError('features must be [batch, time, input_dim]')
        if base_logits.ndim != 2 or base_logits.shape != features.shape[:2]:
            raise ValueError('base_logits must be [batch, time]')
        if features.device != self.W0.device or base_logits.device != self.W0.device:
            raise ValueError('features, base_logits, and model must share a device')
        if observed is not None:
            if observed.shape != base_logits.shape or observed.dtype != torch.bool:
                raise ValueError('observed must be a matching boolean time mask')
            if observed.device != features.device:
                raise ValueError('observed and features must share a device')

    def _validate_state(self, state, batch, device):
        if not isinstance(state, LatentMemoryState):
            raise TypeError('state must be a LatentMemoryState returned by this model')
        if state.W.shape != (batch, self.cfg.state_dim, self.cfg.state_dim):
            raise ValueError('state W has the wrong shape')
        pending = (state.pending_q, state.pending_k, state.pending_v)
        if any(tensor.ndim != 3 or tensor.shape[0] != batch
               or tensor.shape[2] != self.cfg.state_dim for tensor in pending):
            raise ValueError('pending state has the wrong shape')
        lengths = {tensor.shape[1] for tensor in pending}
        if len(lengths) != 1 or state.pending_observed.shape != (batch, next(iter(lengths))):
            raise ValueError('pending state lengths do not agree')
        if state.pending_observed.dtype != torch.bool:
            raise ValueError('pending_observed must be boolean')
        tensors = (state.W, *pending, state.pending_observed)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError('state and inputs must share a device')
        if any(tensor.dtype != self.W0.dtype for tensor in (state.W, *pending)):
            raise ValueError('state and model must share a floating dtype')
        if not all(torch.isfinite(tensor).all() for tensor in (state.W, *pending)):
            raise FloatingPointError('nonfinite latent memory state')
        return state

    def _project(self, features, base_logits, observed):
        finite = torch.isfinite(features).all(dim=-1) & torch.isfinite(base_logits)
        if observed is None:
            observed = finite
        else:
            observed = observed & finite
        clean = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
        embedded = torch.tanh(self.embedding(clean.to(dtype=self.W0.dtype)))
        return (self.query(embedded), self.key(embedded), self.value(embedded), observed)

    def _bounded_update(self, state, key, value, observed):
        """Analytic mean-squared K-to-V update with a bounded W-W0 offset."""
        predicted = torch.bmm(key, state.transpose(1, 2))
        mask = observed.unsqueeze(-1).to(key.dtype)
        error = (predicted - value) * mask
        count = observed.sum(dim=1, keepdim=True).to(key.dtype)
        denominator = count.clamp_min(1.) * self.cfg.state_dim
        gradient = 2. * torch.bmm(error.transpose(1, 2), key)
        gradient = gradient / denominator.unsqueeze(-1)

        norm = gradient.square().sum(dim=(1, 2), keepdim=True).add(self.cfg.eps).sqrt()
        gradient = gradient * (self.cfg.grad_clip / norm).clamp(max=1.)
        eta = self.cfg.eta_max * torch.sigmoid(self.eta)
        candidate = state - eta * gradient

        anchor = self.W0.unsqueeze(0).expand_as(candidate)
        offset = candidate - anchor
        offset_norm = offset.square().sum(dim=(1, 2), keepdim=True).add(self.cfg.eps).sqrt()
        return anchor + offset * (self.cfg.state_radius / offset_norm).clamp(max=1.)

    def _delta(self, query, state):
        offset = state - self.W0.unsqueeze(0).expand_as(state)
        latent = torch.bmm(query, offset.transpose(1, 2))
        return 2. * torch.tanh(self.readout(latent).squeeze(-1))

    def forward(self, features, base_logits, observed=None, state=None, adapt=True):
        """Return base-plus-delta logits and explicit state for the next chunk."""
        self._validate_inputs(features, base_logits, observed)
        if not isinstance(adapt, bool):
            raise TypeError('adapt must be a boolean')
        batch, length = base_logits.shape
        state = self.initial_state(batch) if state is None else self._validate_state(
            state, batch, features.device)
        if not adapt:
            return dict(logits=base_logits, delta=torch.zeros_like(base_logits), state=state)

        query, key, value, observed = self._project(features, base_logits, observed)
        prefix = state.pending_q.shape[1]
        all_query = torch.cat((state.pending_q, query), dim=1)
        all_key = torch.cat((state.pending_k, key), dim=1)
        all_value = torch.cat((state.pending_v, value), dim=1)
        all_observed = torch.cat((state.pending_observed, observed), dim=1)
        total = all_query.shape[1]
        current_start, current_end = prefix, prefix + length
        current_state = state.W
        delta_parts = []
        cursor = 0

        while cursor + self.cfg.chunk_size <= total:
            end = cursor + self.cfg.chunk_size
            chunk_delta = self._delta(all_query[:, cursor:end], current_state)
            left, right = max(cursor, current_start), min(end, current_end)
            if left < right:
                delta_parts.append(chunk_delta[:, left - cursor:right - cursor])
            current_state = self._bounded_update(
                current_state, all_key[:, cursor:end],
                all_value[:, cursor:end], all_observed[:, cursor:end])
            cursor = end

        if cursor < total:
            chunk_delta = self._delta(all_query[:, cursor:], current_state)
            left, right = max(cursor, current_start), min(total, current_end)
            if left < right:
                delta_parts.append(chunk_delta[:, left - cursor:right - cursor])

        delta = (torch.cat(delta_parts, dim=1) if delta_parts else
                 base_logits.new_zeros(batch, 0))
        pending = LatentMemoryState(
            W=current_state,
            pending_q=all_query[:, cursor:],
            pending_k=all_key[:, cursor:],
            pending_v=all_value[:, cursor:],
            pending_observed=all_observed[:, cursor:],
        )
        return dict(logits=base_logits + delta, delta=delta, state=pending)


__all__ = ['LatentMemoryConfig', 'LatentMemoryState', 'LatentMemory']
