"""Amortized, context-dependent low-rank updates for frozen Q/K attention."""

import math
from numbers import Real

import torch
from torch import nn

from ..context_vector_20261030.teaching import context_controls


class ConditionalWeights(nn.Module):
    """Generate a bounded Q/K matrix pair without optimizing per-example state."""

    def __init__(self, radius=0.1):
        super().__init__()
        if (isinstance(radius, bool) or not isinstance(radius, Real)
                or not math.isfinite(radius) or not 0 <= radius <= 0.1):
            raise ValueError('radius must be finite and in [0, 0.1]')
        self.radius = float(radius)
        self.encoder = nn.Sequential(nn.LayerNorm(358), nn.Linear(358, 32), nn.SiLU())
        self.a_head = nn.Linear(32, 256)
        self.b_head = nn.Linear(32, 256)
        nn.init.normal_(self.a_head.weight, std=0.02)
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

    def forward(self, context, support):
        if (not isinstance(context, torch.Tensor) or context.layout != torch.strided
                or context.ndim != 2 or context.shape[1] != 358
                or not torch.is_floating_point(context)):
            raise ValueError('context must be a floating B-by-358 tensor')
        if (not isinstance(support, torch.Tensor) or support.layout != torch.strided
                or support.dtype != torch.bool or support.shape != (len(context),)
                or support.device != context.device):
            raise ValueError('support must be a same-device boolean B-vector')
        norm = self.encoder[0]
        if context.device != norm.weight.device:
            raise ValueError('context and generator must be on the same device')
        if not torch.isfinite(context).all():
            raise ValueError('context must be finite, including unsupported rows')

        context = context.detach().to(norm.weight.dtype)
        if not torch.isfinite(context).all():
            raise ValueError('context is nonfinite after conversion to generator dtype')
        normalized = norm(context)
        if not torch.isfinite(normalized).all():
            raise FloatingPointError('nonfinite LayerNorm output')
        hidden = self.encoder[2](self.encoder[1](normalized))
        if not torch.isfinite(hidden).all():
            raise FloatingPointError('nonfinite context encoder output')
        a = self.a_head(hidden).reshape(len(context), 2, 2, 64)
        b = self.b_head(hidden).reshape(len(context), 2, 2, 64)
        raw = (b.transpose(-1, -2) @ a).flatten(1)
        if not torch.isfinite(raw).all():
            raise FloatingPointError('nonfinite generated Q/K updates')

        if self.radius == 0:
            projected = raw * 0
        else:
            # Compute the norm after scaling by the largest component so that
            # finite, very large updates do not overflow while being projected.
            maximum = raw.abs().amax(dim=1, keepdim=True)
            safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
            direction = raw / safe_maximum
            direction_norm = direction.norm(dim=1, keepdim=True)
            safe_norm = torch.where(direction_norm > 0, direction_norm, torch.ones_like(direction_norm))
            bound = self.radius / safe_norm
            projected = torch.where(maximum > bound, direction * bound, raw)
        result = torch.where(support[:, None], projected, torch.zeros_like(projected))
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite projected Q/K updates')
        return result


def controlled_contexts(context, train_mean):
    """Use the frozen teaching protocol's real and matched-context controls."""
    return context_controls(context, train_mean)
