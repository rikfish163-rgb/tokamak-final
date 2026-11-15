"""Causal context-conditioned token adaptation of a frozen flat23 Simple23.

The original Transformer, risk head, forecast head, normalization and learned
fast origin are reused unchanged. This is not a final-score residual model.
"""

import math

import torch
from torch import nn

from ..simple23 import Simple23


class ContextModel(nn.Module):
    """Map past-window context to the source's 128-dimensional token state.

    ``fast`` is a per-shot delta, NOT an absolute Simple23 fast state. The
    caller owns its updates and lifetime; this module never stores it.
    """

    def __init__(self, base, context_dim=8, context_mode='endpoint'):
        super().__init__()
        if not isinstance(base, Simple23):
            raise ValueError("a qualified flat23 Simple23 base is required")
        if base.cfg.dim != 64 or base.cfg.fast_site != "tokens" or base.cfg.input_view != "source":
            raise ValueError("base must use source-normalized flat23 and the original 128-dimensional token site")
        state_dim = 2 * base.cfg.dim
        if context_mode not in ('endpoint','ordered'):
            raise ValueError('context_mode must be endpoint or ordered')
        tokens = base.cfg.window // base.cfg.patch
        if context_mode == 'ordered' and tokens % 4:
            raise ValueError('ordered context requires four equal causal token segments')
        if (not isinstance(context_dim, int) or isinstance(context_dim, bool)
                or not 1 <= context_dim <= state_dim):
            raise ValueError("context_dim must be an integer in 1..128")
        self.base = base
        self.context_dim = context_dim
        self.context_mode = context_mode
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.base.eval()

        # An orthonormal DCT basis is fixed, reproducible and label/data-free.
        # Build in float64 on CPU, then follow the source's dtype and device.
        summary_dim = state_dim if context_mode == 'endpoint' else 4 * base.cfg.dim
        coordinate = torch.arange(summary_dim, dtype=torch.float64)[:, None] + .5
        frequency = torch.arange(context_dim, dtype=torch.float64)[None]
        projection = torch.cos(math.pi * coordinate * frequency / summary_dim) * math.sqrt(2 / summary_dim)
        projection[:, 0] = 1 / math.sqrt(summary_dim)
        self.register_buffer("context_projection", projection.to(base.center))

        # Linear's discarded random initialization must not perturb caller RNG.
        with torch.random.fork_rng(devices=[]):
            self.context_map = nn.Linear(context_dim, state_dim, bias=False)
        nn.init.zeros_(self.context_map.weight)
        self.context_map.to(base.center)

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def center(self):
        return self.base.center

    @property
    def scale(self):
        return self.base.scale

    def train(self, mode=True):
        super().train(mode)
        # Frozen inference semantics remain the same during persistent fitting.
        self.base.eval()
        return self

    def initial_fast(self, batch):
        """Fresh independent per-sample zero deltas, on the source device."""
        return self.center.new_zeros(batch, 2 * self.cfg.dim)

    @torch.no_grad()
    def _context(self, values, available):
        # Exactly the source-view input tokenization, before affine/position/CLS.
        # Detach only the context producer, not the map or adapted source path.
        observed = available & torch.isfinite(values)
        clean = torch.where(observed, values, self.center)
        level = torch.asinh((clean - self.center) / self.scale)
        inputs = torch.cat((level, observed.to(level.dtype)), -1)
        batch, tokens = len(values), self.cfg.window // self.cfg.patch
        embedded = self.base.embedding(inputs.reshape(batch, tokens, -1))
        valid = observed.reshape(batch, tokens, -1).any(-1)
        embedded = embedded.masked_fill(~valid[..., None], 0.)
        if self.context_mode == 'ordered':
            # Four ordered observed segments retain within-window timing that
            # a mean + endpoint delta erases. Empty segments have zero summary;
            # no future padding or unknown physical units are inferred.
            grouped = embedded.reshape(batch,4,tokens//4,self.cfg.dim)
            count = valid.reshape(batch,4,tokens//4).sum(-1).clamp_min(1)
            summary = (grouped.sum(2) / count[...,None]).flatten(1)
            return summary @ self.context_projection
        mean = embedded.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        first = valid.long().argmax(1)
        last = tokens - 1 - valid.flip(1).long().argmax(1)
        rows = torch.arange(batch, device=values.device)
        innovation = embedded[rows, last] - embedded[rows, first]
        return torch.cat((mean, innovation), -1) @ self.context_projection

    def forward(self, values, available, fast=None):
        if values.ndim != 3 or values.shape[1:] != (self.cfg.window, 23):
            raise ValueError("values must be [batch, past window, 23]")
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError("matching boolean availability required")
        batch = len(values)
        fast = self.initial_fast(batch) if fast is None else fast
        if not isinstance(fast, torch.Tensor) or fast.shape != (batch, 2 * self.cfg.dim):
            raise ValueError("one 128-dimensional temporary delta per shot required")
        persistent = self.context_map(self._context(values, available))
        absolute_fast = self.base.initial_fast(batch) + persistent + fast
        # Do not detach/no_grad this call: frozen weights still transmit both
        # heads' derivatives, including double backward, to the internal state.
        return self.base(values, available, absolute_fast)

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)

    def persistent_parameters(self):
        """Only the context-to-token map; no source/head/LayerNorm parameters."""
        return self.context_map.parameters()
