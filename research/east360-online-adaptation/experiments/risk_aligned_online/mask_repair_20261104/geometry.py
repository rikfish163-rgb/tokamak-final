"""Prefix-only context and SPD geometry for a 16-dimensional gradient.

This module does not read observations, labels or model state. The caller is
responsible for supplying an embedding and gradient from the current prefix.
"""

import torch
from torch import nn


_FLOATS = (torch.float32, torch.float64)


def context_from_embedding(e, a):
    """Summarize valid patch embeddings and CIII/MNM frame availability.

    The first/last four are the first/last *valid* tokens in temporal order;
    fewer than four use all available tokens. An entirely padded row is invalid.
    """
    if (not isinstance(e, torch.Tensor) or e.ndim != 3 or e.shape[1:] != (16, 64)
            or not len(e) or e.dtype not in _FLOATS or not isinstance(a, torch.Tensor)
            or a.shape != (len(e), 128, 23) or a.dtype != torch.bool or a.device != e.device):
        raise ValueError("matching nonempty Bx16x64 embeddings and boolean Bx128x23 availability required")
    if e.device.type == "meta":
        raise ValueError("materialized input tensors are required")
    if not torch.isfinite(e).all():
        raise ValueError("all embeddings must be finite")
    frames = a.any(dim=-1)
    valid = frames.reshape(len(e), 16, 8).any(dim=-1)
    count = valid.sum(dim=-1)
    if not (count > 0).all():
        raise ValueError("each prefix must contain at least one observed token; all-padding is invalid")
    ranks = valid.long().cumsum(dim=-1)
    first = valid & (ranks <= 4)
    last = valid & (ranks > count[:, None] - 4)

    def mean_tokens(selection):
        return (e * selection[..., None].to(e.dtype)).sum(dim=1) / selection.sum(dim=1)[:, None]

    average = mean_tokens(valid)
    contrast = mean_tokens(last) - mean_tokens(first)
    fractions = a[:, :, (1, 2)].sum(dim=1).to(e.dtype) / frames.sum(dim=1)[:, None]
    result = torch.cat((average, contrast, fractions), dim=-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite prefix context")
    return result


class ContextGeometry(nn.Module):
    """Bounded symmetric log-geometry; no additive update or risk head."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(130), nn.Linear(130, 32), nn.SiLU(), nn.Linear(32, 136))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or context.shape[1:] != (130,) or not len(context) or context.dtype not in _FLOATS):
            raise ValueError("nonempty Bx130 float32/float64 context required")
        if context.device.type == "meta":
            raise ValueError("materialized context and geometry parameters are required")
        if any(p.device != context.device or p.dtype != context.dtype for p in self.parameters()):
            raise ValueError("geometry parameters and context must share device and dtype")
        if not torch.isfinite(context).all() or any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("finite context and geometry parameters required")
        entries = .125 * torch.tanh(self.net(context))
        if not torch.isfinite(entries).all():
            raise FloatingPointError("nonfinite symmetric geometry entries")
        i, j = torch.triu_indices(16, 16, device=context.device)
        upper = context.new_zeros((len(context), 16, 16))
        upper[:, i, j] = entries
        symmetric = upper + upper.transpose(-1, -2) - torch.diag_embed(upper.diagonal(dim1=-2, dim2=-1))
        matrix = torch.linalg.matrix_exp(symmetric)
        # Restore exact symmetry against floating-point matrix-exp roundoff.
        matrix = .5 * (matrix + matrix.transpose(-1, -2))
        if not torch.isfinite(matrix).all():
            raise FloatingPointError("nonfinite matrix exponential")
        return matrix

    def propose(self, state, g, context):
        """Return (new state, M), without mutating inputs or detaching gradients.

        Apply M@g, then the existing float64 0.025 normalized descent step,
        1e-8 minimum direction norm and 0.25 per-sample state-radius projection.
        A caller must provide an already radius-bounded state.
        """
        if (not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1:] != (16,)
                or not len(state) or state.dtype not in _FLOATS or not isinstance(g, torch.Tensor)
                or g.shape != state.shape or g.dtype != state.dtype or g.device != state.device
                or not isinstance(context, torch.Tensor) or context.shape != (len(state), 130)
                or context.dtype != state.dtype or context.device != state.device):
            raise ValueError("matching nonempty Bx16 state/gradient and Bx130 context required")
        if state.device.type == "meta":
            raise ValueError("materialized state, gradient and context are required")
        if not torch.isfinite(state).all() or not torch.isfinite(g).all():
            raise ValueError("state and reconstruction gradient must be finite")
        if (torch.linalg.vector_norm(state.double(), dim=-1) > .25).any():
            raise ValueError("input state must be within the L2 radius .25")
        matrix = self(context)
        direction = torch.bmm(matrix.double(), g.double().unsqueeze(-1)).squeeze(-1)
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(direction).all() or not torch.isfinite(norm).all():
            raise FloatingPointError("nonfinite preconditioned gradient or norm")
        active = norm >= 1e-8
        candidate = state.double() - .025 * direction / norm.clamp_min(1e-8)
        radius = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
        ceiling = .25 * (1 - 4 * torch.finfo(state.dtype).eps)
        factor = torch.where(radius > .25, ceiling / radius.clamp_min(.25), 1.)
        proposed = (candidate * factor).to(state.dtype)
        # A float64 point just inside the ball can round outside on casting.
        rounded_radius = torch.linalg.vector_norm(proposed.double(), dim=-1, keepdim=True)
        rounded_factor = torch.where(rounded_radius > .25, ceiling / rounded_radius.clamp_min(.25), 1.)
        proposed = (proposed.double() * rounded_factor).to(state.dtype)
        if (torch.linalg.vector_norm(proposed.double(), dim=-1) > .25).any():
            raise FloatingPointError("rounded state exceeds the declared L2 radius")
        return torch.where(active, proposed, state), matrix
