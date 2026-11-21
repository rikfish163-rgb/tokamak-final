"""A bounded TTT-Linear risk state for causal 23-channel streams.

This small prototype is inspired by *Learning to (Learn at Test Time): RNNs
with Expressive Hidden States* (ICML 2025, PMLR 267), but it is not a paper
reproduction and does not establish tokamak performance, physical validity, or
a safety guarantee.  Its inner update uses only the unlabeled K-to-V squared
loss; risk labels belong to an external outer training loop.
"""

from dataclasses import dataclass

import torch
from torch import nn


CHANNELS = 23


@dataclass(frozen=True)
class TTTConfig:
    """Fixed small defaults; state is explicit so a shot can be reset."""

    dim: int = 32
    state_dim: int = 16
    chunk_size: int = 16
    eta_max: float = .1
    state_radius: float = 1.
    grad_clip: float = 1.
    eps: float = 1e-6

    def __post_init__(self):
        if min(self.dim, self.state_dim, self.chunk_size) < 1:
            raise ValueError('dim, state_dim, and chunk_size must be positive')
        if self.eta_max <= 0 or self.state_radius <= 0 or self.grad_clip <= 0:
            raise ValueError('eta_max, state_radius, and grad_clip must be positive')
        if self.eps <= 0:
            raise ValueError('eps must be positive')

    def to_dict(self):
        return dict(
            dim=self.dim,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            eta_max=self.eta_max,
            state_radius=self.state_radius,
            grad_clip=self.grad_clip,
            eps=self.eps,
        )


@dataclass(frozen=True)
class TTTState:
    """Explicit per-shot learner state, including a not-yet-adapted tail."""

    W: torch.Tensor
    pending_q: torch.Tensor
    pending_k: torch.Tensor
    pending_v: torch.Tensor
    pending_observed: torch.Tensor


class TTTRiskState(nn.Module):
    """Causal point-embedded TTT-Linear risk state with explicit streaming state."""

    def __init__(self, cfg, center, scale):
        super().__init__()
        if not isinstance(cfg, TTTConfig):
            raise TypeError('cfg must be a TTTConfig')
        self.cfg = cfg
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if center.shape != (CHANNELS,) or scale.shape != (CHANNELS,):
            raise ValueError('exactly 23 source channels are required')
        if (not torch.isfinite(center).all() or not torch.isfinite(scale).all()
                or not (scale > 0).all()):
            raise ValueError('source center/scale must be finite with positive scale')
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

        # Pointwise embedding is causal: each time point sees only its own
        # values and mask.  It avoids a hidden fixed 128-step window.
        self.embedding = nn.Linear(2 * CHANNELS, cfg.dim)
        self.key = nn.Linear(cfg.dim, cfg.state_dim, bias=False)
        self.value = nn.Linear(cfg.dim, cfg.state_dim, bias=False)
        self.query = nn.Linear(cfg.dim, cfg.state_dim, bias=False)

        # The learner is deliberately a small 16x16 matrix by default.
        self.W0 = nn.Parameter(torch.eye(cfg.state_dim))
        self.eta = nn.Parameter(torch.tensor(-2.))
        self.risk_readout = nn.Linear(cfg.state_dim, 1)

    def initial_state(self, batch):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        empty = self.W0.new_empty(batch, 0, self.cfg.state_dim)
        empty_observed = torch.zeros(batch, 0, dtype=torch.bool,
                                     device=self.W0.device)
        return TTTState(
            W=self.W0.unsqueeze(0).expand(batch, -1, -1).clone(),
            pending_q=empty.clone(),
            pending_k=empty.clone(),
            pending_v=empty.clone(),
            pending_observed=empty_observed,
        )

    def _validate_state(self, state, batch, device, dtype):
        if not isinstance(state, TTTState):
            raise TypeError('state must be a TTTState returned by this model')
        if state.W.shape != (batch, self.cfg.state_dim, self.cfg.state_dim):
            raise ValueError('state W has the wrong batch or state dimensions')
        pending = (state.pending_q, state.pending_k, state.pending_v)
        lengths = {tensor.shape[1] for tensor in pending}
        if any(tensor.ndim != 3 or tensor.shape[0] != batch
               or tensor.shape[2] != self.cfg.state_dim for tensor in pending):
            raise ValueError('pending state tensors have the wrong shape')
        if len(lengths) != 1 or state.pending_observed.shape != (batch, next(iter(lengths))):
            raise ValueError('pending state lengths do not agree')
        if state.pending_observed.dtype != torch.bool:
            raise ValueError('pending observed mask must be boolean')
        tensors = (state.W, *pending, state.pending_observed)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError('state and input must be on the same device')
        if any(tensor.dtype != dtype for tensor in (state.W, *pending)):
            raise ValueError('state and input must use the same floating dtype')
        if not all(torch.isfinite(tensor).all() for tensor in (state.W, *pending)):
            raise FloatingPointError('nonfinite TTT state')
        return state

    def _project(self, values, available):
        if values.ndim != 3 or values.shape[-1] != CHANNELS:
            raise ValueError('values must be [batch, time, 23]')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('available must be a matching boolean mask')
        values = values.to(dtype=self.center.dtype)
        valid = available & torch.isfinite(values)
        clean = torch.where(valid, values, self.center)
        source = torch.asinh((clean - self.center) / self.scale)
        embedded = torch.tanh(self.embedding(torch.cat(
            (source, valid.to(source.dtype)), dim=-1)))
        return (self.query(embedded), self.key(embedded), self.value(embedded),
                valid.any(dim=-1))

    def _bounded_update(self, state, key, value, observed):
        """Apply the analytic gradient of mean ||K W^T - V||^2 without labels."""
        prediction = torch.bmm(key, state.transpose(1, 2))
        mask = observed.unsqueeze(-1).to(key.dtype)
        error = (prediction - value) * mask
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

    def forward(self, values, available, state=None, adapt=True):
        """Return current-time logits and explicit state for the next stream chunk.

        A complete chunk is scored with its old W first.  Only then is its
        unlabeled K-to-V loss used to produce the W for the next chunk.  A tail
        shorter than ``chunk_size`` remains in ``state`` and is not adapted.
        Passing ``state=None`` creates a fresh W0 state for a new shot.
        """
        if not isinstance(adapt, bool):
            raise TypeError('adapt must be a boolean')
        query, key, value, observed = self._project(values, available)
        batch, length = query.shape[:2]
        state = (self.initial_state(batch) if state is None else
                 self._validate_state(state, batch, query.device, query.dtype))
        prefix = state.pending_q.shape[1]
        all_query = torch.cat((state.pending_q, query), dim=1)
        all_key = torch.cat((state.pending_k, key), dim=1)
        all_value = torch.cat((state.pending_v, value), dim=1)
        all_observed = torch.cat((state.pending_observed, observed), dim=1)
        total = all_query.shape[1]
        current_start, current_end = prefix, prefix + length
        current_state = state.W
        logits_parts = []
        cursor = 0

        while cursor + self.cfg.chunk_size <= total:
            end = cursor + self.cfg.chunk_size
            chunk_logits = self.risk_readout(torch.bmm(
                all_query[:, cursor:end], current_state.transpose(1, 2))).squeeze(-1)
            left, right = max(cursor, current_start), min(end, current_end)
            if left < right:
                logits_parts.append(chunk_logits[:, left - cursor:right - cursor])
            if adapt:
                current_state = self._bounded_update(
                    current_state, all_key[:, cursor:end],
                    all_value[:, cursor:end], all_observed[:, cursor:end])
            cursor = end

        if cursor < total:
            chunk_logits = self.risk_readout(torch.bmm(
                all_query[:, cursor:], current_state.transpose(1, 2))).squeeze(-1)
            left, right = max(cursor, current_start), min(total, current_end)
            if left < right:
                logits_parts.append(chunk_logits[:, left - cursor:right - cursor])

        logits = (torch.cat(logits_parts, dim=1) if logits_parts else
                  query.new_empty(batch, 0))
        pending_q = all_query[:, cursor:]
        pending_k = all_key[:, cursor:]
        pending_v = all_value[:, cursor:]
        pending_observed = all_observed[:, cursor:]
        return dict(logits=logits, state=TTTState(
            W=current_state,
            pending_q=pending_q,
            pending_k=pending_k,
            pending_v=pending_v,
            pending_observed=pending_observed,
        ))


TTTStateModel = TTTRiskState


__all__ = ['TTTConfig', 'TTTState', 'TTTRiskState', 'TTTStateModel']
