"""Joint diagnostic decoder on an isolated, Q/V-LoRA-adaptable source.

This module creates no targets, reads no data or labels, and chooses no loss,
optimizer, or adaptation schedule. Qualification of the supplied source is
the caller's responsibility; its expected architecture is checked here.
"""

import torch
from torch import nn

from ..cold20_compare_20261104.adapters import make_adapter
from ..direct_fast_memory_20261104.memory import dynamic_key
from ..direct_fast_memory_20261104.model import levels
from ..simple23 import Simple23


class JointLoRA(nn.Module):
    """Frozen source clone, 512 Q/V LoRA parameters, and zero residual head."""

    def __init__(self, base, initial_forecaster):
        super().__init__()
        if not isinstance(base, Simple23):
            raise TypeError("base must be the qualified Simple23 source")
        cfg = base.cfg
        if (
            cfg.dim,
            cfg.layers,
            cfg.heads,
            cfg.window,
            cfg.patch,
            cfg.horizon,
            cfg.input_view,
            cfg.fast_site,
            cfg.fast_rule,
        ) != (64, 2, 4, 128, 8, 25, "source", "tokens", "normalized"):
            raise ValueError("qualified source configuration is required")
        if (
            len(base.encoder.layers) != 2
            or not isinstance(base.norm, nn.LayerNorm)
            or base.norm.normalized_shape != (64,)
            or base.position.shape != (1, 17, 64)
            or base.cls.shape != (1, 1, 64)
        ):
            raise ValueError("qualified source encoder and normalization are required")
        for layer in base.encoder.layers:
            attention = layer.self_attn
            if (
                not isinstance(attention, nn.MultiheadAttention)
                or attention.embed_dim != 64
                or attention.num_heads != 4
            ):
                raise ValueError("qualified source attention geometry is required")
        for layer, shape in (
            (base.embedding, (64, 368)),
            (base.risk_head, (1, 64)),
            (base.forecast_head, (575, 1088)),
        ):
            if (
                not isinstance(layer, nn.Linear)
                or layer.weight.shape != shape
                or layer.bias is None
                or layer.bias.shape != (shape[0],)
            ):
                raise ValueError("qualified source embedding and heads are required")
        if (
            base.center.shape != (23,)
            or base.scale.shape != (23,)
            or base.center.dtype != torch.float32
            or base.scale.dtype != torch.float32
            or not torch.isfinite(base.center).all()
            or not torch.isfinite(base.scale).all()
            or not (base.scale > 0).all()
        ):
            raise ValueError("finite float32 source normalization is required")
        if any(
            p.device != base.center.device or p.dtype != torch.float32
            for p in base.parameters()
        ):
            raise ValueError("source parameters must share the float32 source device")
        if base.scale.device != base.center.device:
            raise ValueError("source normalization must share a device")
        if not isinstance(initial_forecaster, torch.Tensor):
            raise TypeError("initial forecaster must be a tensor")
        if (
            initial_forecaster.shape != (168, 70)
            or initial_forecaster.dtype != torch.float32
            or not torch.isfinite(initial_forecaster).all()
        ):
            raise ValueError("finite float32 168-by-70 initial forecaster required")

        # This existing helper verifies frozen/eval status, deep-copies source
        # parameters and buffers, and opens only the last layer's Q/V LoRA.
        self.clone = make_adapter(base, "qv_lora")
        self.register_buffer(
            "W0", initial_forecaster.detach().to(base.center.device).clone()
        )
        self.residual_decoder = nn.Linear(
            192, 168, bias=True, device=base.center.device, dtype=torch.float32
        )
        nn.init.zeros_(self.residual_decoder.weight)
        nn.init.zeros_(self.residual_decoder.bias)
        neural = self.neural_parameters()
        if sum(p.numel() for p in neural) != 512 or {id(p) for p in neural} != {
            id(p) for p in self.clone.parameters() if p.requires_grad
        }:
            raise AssertionError(
                "only the 512 Q/V LoRA source parameters may be trainable"
            )
        self.eval()

    def train(self, mode=True):
        """Keep the qualified clone in eval even during decoder/LoRA training."""
        super().train(mode)
        self.clone.eval()
        return self

    def neural_parameters(self):
        """Return the stable Q/V LoRA scope, even if a caller freezes it."""
        lora = self.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[
            0
        ]
        return tuple(lora.parameters())

    def decoder_parameters(self):
        """Return only the residual diagnostic decoder's weight and bias."""
        return tuple(self.residual_decoder.parameters())

    def forward(self, x, a):
        # CLS is the last token in Simple23, not the first. Keep the graph in
        # this local list only, and remove the hook even if source raises.
        captured = []
        hook = self.clone.norm.register_forward_hook(
            lambda _m, _args, output: captured.append(output)
        )
        try:
            source = self.clone(x, a)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (len(x), 17, 64):
            raise RuntimeError("exactly one 17-by-64 source norm output is required")
        tokens = captured[0]
        cls, last, previous = tokens[:, -1], tokens[:, -2], tokens[:, -3]
        latent = torch.cat((cls, last, last - previous), dim=1)
        key = dynamic_key(levels(self.clone, x, a), a)
        diagnostic = key @ self.W0.T + self.residual_decoder(latent)
        return dict(
            logit=source["logit"], diagnostic=diagnostic, key=key, latent=latent
        )
