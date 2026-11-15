"""Isolated, zero-initialized adapters for an already-qualified Simple23.

This module neither loads data/checkpoints nor chooses a loss or optimizer.
The caller is responsible for qualifying the source and configuring numerical
backends consistently for the paired source/adapter comparison.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils import parametrize

from ..simple23 import Simple23


class _QVLoRA(nn.Module):
    """Rank-2 Q/V offsets on a frozen packed Q,K,V projection, alpha/r=1."""

    def __init__(self, weight):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(20261104)

        def initial_a():
            value = torch.randn(2, 64, generator=generator, dtype=weight.dtype, device="cpu")
            return nn.Parameter((value * 0.02).to(device=weight.device))

        self.lora_A_q = initial_a()
        self.lora_B_q = nn.Parameter(weight.new_zeros(64, 2))
        self.lora_A_v = initial_a()
        self.lora_B_v = nn.Parameter(weight.new_zeros(64, 2))

    def forward(self, weight):
        q = self.lora_B_q @ self.lora_A_q
        v = self.lora_B_v @ self.lora_A_v
        return weight + torch.cat((q, torch.zeros_like(q), v), dim=0)


def _last_layer(base):
    if not isinstance(base, Simple23):
        raise TypeError("an already-qualified Simple23 source is required")
    if base.cfg.dim != 64 or len(base.encoder.layers) != 2:
        raise ValueError("the qualified two-layer, 64-dimensional source is required")
    layer = base.encoder.layers[-1]
    attention = layer.self_attn
    if (parametrize.is_parametrized(attention, "in_proj_weight")
            or attention.in_proj_weight is None
            or attention.in_proj_weight.shape != (192, 64)):
        raise ValueError("an unparametrized packed 192-by-64 QKV projection is required")
    if any(parameter.requires_grad for parameter in base.parameters()) or any(
            module.training for module in base.modules()):
        raise ValueError("source must already be frozen and in eval mode")
    for norm in (layer.norm1, layer.norm2):
        if (not isinstance(norm, nn.LayerNorm) or norm.weight is None or norm.bias is None
                or norm.weight.shape != (64,) or norm.bias.shape != (64,)):
            raise ValueError("two affine 64-dimensional final-layer norms are required")
    return layer


def make_adapter(base, mode):
    """Return an independent eval clone with only the requested parameters open.

    ``mode`` is exactly ``norm`` or ``qv_lora``. The original model, flags,
    gradients and buffers are not modified; all clone storage is independent.
    """
    if mode not in ("norm", "qv_lora"):
        raise ValueError("mode must be 'norm' or 'qv_lora'")
    _last_layer(base)
    clone = deepcopy(base)
    clone.eval()
    for parameter in clone.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    layer = clone.encoder.layers[-1]
    if mode == "norm":
        for norm in (layer.norm1, layer.norm2):
            norm.weight.requires_grad_(True)
            norm.bias.requires_grad_(True)
    else:
        weight = layer.self_attn.in_proj_weight
        parametrize.register_parametrization(layer.self_attn, "in_proj_weight", _QVLoRA(weight))
        # Newly registered modules default to training mode even if their
        # parent was already eval; keep the whole returned clone in eval.
        clone.eval()
    return clone


def named_trainables(model):
    """Return names and tensors currently marked trainable, in model order."""
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def trainable_parameters(model):
    """Count trainable scalar parameters, not tensors."""
    return sum(parameter.numel() for _, parameter in named_trainables(model))
