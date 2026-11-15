"""Existing last Transformer FFN, with one disposable additive delta per shot.

No new risk head, no persistent target parameters. All registered tensors remain
the fixed source/decoder. Functional calls retain gradients through an external
delta, allowing source-only meta-training of the signal-loss weights.
"""
import torch
from torch import nn
from torch.func import functional_call

from ..simple23 import Simple23


class FfnModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        if not isinstance(base, Simple23) or (
                base.cfg.fast_site, base.cfg.input_view, base.cfg.dim,
                base.cfg.window, base.cfg.horizon) != ('tokens', 'source', 64, 128, 25):
            raise ValueError('qualified source-view token model W128/H25/d64 required')
        self.base = base.eval().requires_grad_(False)
        last = base.cfg.layers - 1
        self.parameter_names = tuple(f'encoder.layers.{last}.{layer}.{part}'
            for layer in ('linear1', 'linear2') for part in ('weight', 'bias'))
        parameters = dict(base.named_parameters())
        self.delta_count = sum(parameters[n].numel() for n in self.parameter_names)
        self.eval()

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def center(self):
        return self.base.center

    @property
    def scale(self):
        return self.base.scale

    def initial_delta(self):
        return self.center.new_zeros(self.delta_count, requires_grad=True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forecast_overrides(self):
        # Empty on every archived model. The ended-EAST decoder experiment
        # overrides only the existing auxiliary head, not classifier weights.
        return {}

    def forward(self, values, available, delta):
        if not torch.is_grad_enabled():
            raise RuntimeError('use enabled autograd and detach outputs; fixed attention dispatch required')
        if (delta.shape != (self.delta_count,) or delta.dtype != self.center.dtype
                or delta.device != self.center.device or not torch.isfinite(delta).all()):
            raise ValueError('one finite flat temporary delta per shot required')
        parameters = dict(self.base.named_parameters())
        replacements, offset = self.forecast_overrides(), 0
        for name in self.parameter_names:
            original = parameters[name]
            replacements[name] = original + delta[offset:offset+original.numel()].view_as(original)
            offset += original.numel()
        # A fixed leaf keeps EARLY attention on the source replay's gradient
        # path too. Without it, zero FFN delta differs under fused inference on
        # CUDA. This leaf is never updated; only last-FFN delta is differentiated.
        fast = self.base.initial_fast(len(values)).detach().requires_grad_(True)
        return functional_call(self.base, replacements, (values, available, fast), strict=False)
