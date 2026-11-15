"""Full input-patch-embedding deltas outside a frozen flat23 source model.

This is ordinary feature-encoder adaptation, not a physical calibration or a
score correction. Calls sharing the same base must be serialized: PyTorch's
functional_call temporarily substitutes registered tensors during forward.
"""

import torch
from torch import nn
from torch.func import functional_call

from ..simple23 import Simple23


class EmbeddingModel(nn.Module):
    """Reuse both source heads; adapt only its effective input embedding.

    The caller qualifies the source checkpoint. ``fast`` is a temporary
    per-shot delta at the original token site, never stored by this wrapper.
    Persistent deltas start at zero and are owned outside ``base``.
    """

    def __init__(self, base):
        super().__init__()
        if not isinstance(base, Simple23):
            raise ValueError("a qualified flat23 Simple23 base is required")
        cfg = base.cfg
        if (cfg.dim != 64 or cfg.window != 128 or cfg.patch != 8
                or cfg.horizon != 25 or cfg.fast_site != "tokens"
                or cfg.input_view != "source"):
            raise ValueError("base requires dim64/window128/patch8/horizon25, source flat23 and token fast128")
        embedding = base.embedding
        if (not isinstance(embedding, nn.Linear)
                or embedding.weight.shape != (64, 368)
                or embedding.bias is None or embedding.bias.shape != (64,)
                or "embedding.weight" not in dict(base.named_parameters())
                or "embedding.bias" not in dict(base.named_parameters())
                or base.fast_origin.shape != (128,)):
            raise ValueError("the original biased 64x368 patch embedding and fast128 origin are required")
        if (base.center.shape != (23,) or base.scale.shape != (23,)
                or not torch.isfinite(base.center).all()
                or not torch.isfinite(base.scale).all() or not (base.scale > 0).all()):
            raise ValueError("finite source center and positive scale for exactly 23 features required")

        self.base = base
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.base.eval()
        # No random initialization, copied model, low-rank factors or extra head.
        self.delta_weight = nn.Parameter(torch.zeros_like(embedding.weight))
        self.delta_bias = nn.Parameter(torch.zeros_like(embedding.bias))

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
        self.base.eval()
        return self

    def initial_fast(self, batch):
        """Independent zero temporary deltas, not the learned source origin."""
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 0:
            raise ValueError("batch must be a nonnegative integer")
        return self.delta_weight.new_zeros(batch, 128)

    def forward(self, values, available, fast=None):
        if (not isinstance(values, torch.Tensor) or values.ndim != 3
                or values.shape[1:] != (128, 23) or len(values) == 0
                or values.dtype != self.delta_weight.dtype
                or values.device != self.delta_weight.device):
            raise ValueError("values must be nonempty [batch,128,23] with the source dtype/device")
        if (not isinstance(available, torch.Tensor) or available.shape != values.shape
                or available.dtype != torch.bool or available.device != values.device):
            raise ValueError("matching boolean availability on the input device required")
        batch = len(values)
        fast = self.initial_fast(batch) if fast is None else fast
        if (not isinstance(fast, torch.Tensor) or fast.shape != (batch, 128)
                or fast.dtype != values.dtype or fast.device != values.device
                or not torch.isfinite(fast).all()):
            raise ValueError("one finite source-dtype/device fast128 temporary delta per sample required")

        replacements = {
            "embedding.weight": self.base.embedding.weight + self.delta_weight,
            "embedding.bias": self.base.embedding.bias + self.delta_bias,
        }
        absolute_fast = self.base.initial_fast(batch) + fast
        # Always take the differentiable path, including at exactly zero deltas.
        # Frozen Transformer/head weights still transmit first/second derivatives.
        return functional_call(self.base, replacements,
                               args=(values, available, absolute_fast), strict=False)

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)

    def persistent_parameters(self):
        """Exactly 23,616 coordinates outside the frozen source model."""
        return iter((self.delta_weight, self.delta_bias))
