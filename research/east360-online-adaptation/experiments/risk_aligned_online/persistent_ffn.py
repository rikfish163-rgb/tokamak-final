"""Fast/slow copies of the ORIGINAL final Transformer feed-forward weights.

The immutable J-TEXT parent is never overwritten. A registered additive memory
is trained only by the post-shot runner; an external same-shot additive delta
is reset/discarded by the signal-only runner. Both change the same internal
linear1/linear2 weights, not the risk head or final risk score.
"""
import hashlib

import torch
from torch import nn

from .ffn_intra_20261005.model import FfnModel


class PersistentFfnModel(FfnModel):
    adaptation_site = 'original final Transformer FFN linear1/linear2'

    def __init__(self, base):
        super().__init__(base)
        self.temporary_dim = self.delta_count
        self.memory = nn.Parameter(self.center.new_zeros(self.delta_count))

    def initial_fast(self, batch):
        if type(batch) is not int or batch < 0:
            raise ValueError('nonnegative integer batch required')
        return self.memory.new_zeros(batch, self.temporary_dim)

    def persistent_parameters(self):
        return iter((self.memory,))

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)

    def summarize_fast(self, tensor):
        """Bounded receipts for 16,576 values; no lossy state reconstruction.

        Count/norm/hash identify the actual state used. This is not a saved
        replayable tensor; final persistent memory is saved separately by the
        experiment entry point. Compact receipts avoid multi-GB JSON vectors.
        """
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        if flat.numel() != self.temporary_dim or not torch.isfinite(flat).all():
            raise ValueError('finite declared FFN state required for receipt')
        return dict(count=flat.numel(), dtype=str(flat.dtype),
                    l2=float(flat.double().norm()),
                    sha256=hashlib.sha256(flat.numpy().tobytes()).hexdigest())

    def forward(self, values, available, fast=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('inference_mode cannot preserve the gradient-path FFN dispatch')
        if values.ndim != 3 or not len(values):
            raise ValueError('nonempty batched source windows required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape != (len(values), self.temporary_dim)
                or fast.device != self.memory.device or fast.dtype != self.memory.dtype
                or not torch.isfinite(fast).all()):
            raise ValueError('finite source dtype/device batched FFN state required')
        if not torch.equal(fast, fast[:1].expand_as(fast)):
            raise ValueError('one same shot FFN state is required across the endpoint batch')
        enabled = torch.is_grad_enabled()
        # The archived functional FFN uses enabled autograd for a fixed CUDA
        # attention dispatch. no_grad evaluation computes the SAME path then
        # explicitly detaches all outputs; no gradient/update escapes the call.
        with torch.enable_grad():
            out = super().forward(values, available, self.memory + fast[0])
        return out if enabled else {key: value.detach() for key, value in out.items()}
