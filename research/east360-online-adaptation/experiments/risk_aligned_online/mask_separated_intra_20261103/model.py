"""Explicit source-equivalent choice for an otherwise unobserved mask pattern.

This is a static internal representation hypothesis, not a physical calibration
or online gain. Masks remain truthful; original tensors are never reassigned.
"""

import torch
from torch.nn import functional as F

from ..context_memory_20261028.model import ContextAdapter


def mask_embedding_shift(x, weight):
    if (
        x.shape[-1:] != (368,)
        or weight.shape != (64, 368)
        or not x.is_floating_point()
        or not weight.is_floating_point()
        or x.device != weight.device
        or x.dtype != weight.dtype
    ):
        raise ValueError(
            "source patch8x46 and matching64x368 embedding weight required"
        )
    if not torch.isfinite(x).all() or not torch.isfinite(weight).all():
        raise FloatingPointError("finite source embedding operands required")
    patch = x.reshape(*x.shape[:-1], 8, 46)
    dc = patch[..., 23] - patch[..., 24]
    dm = patch[..., 23] - patch[..., 25]
    positions = torch.arange(8, device=x.device) * 46
    return F.linear(dc, weight[:, positions + 24]) + F.linear(
        dm, weight[:, positions + 25]
    )


class MaskSeparatedContext(ContextAdapter):
    def __init__(self, base, *, policy="reference_bt"):
        if policy not in ("original", "reference_bt"):
            raise ValueError("known original or reference_bt representation required")
        if base.training or any(p.requires_grad for p in base.parameters()):
            raise ValueError("qualified base must already be frozen and eval-mode")
        if (base.cfg.patch, base.cfg.dim, base.cfg.input_view) != (8, 64, "source"):
            raise ValueError("source patch8/dim64/source-view base required")
        super().__init__(base, readout="full", feedback="progressive")
        self.policy = policy
        self.eval().requires_grad_(False)

    def tokens(self, values, available):
        if self.policy == "original":
            return super().tokens(values, available)

        def separate(_module, args, output):
            return output + mask_embedding_shift(args[0], self.base.embedding.weight)

        hook = self.base.embedding.register_forward_hook(separate)
        try:
            return super().tokens(values, available)
        finally:
            hook.remove()
