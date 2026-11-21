"""Source-state-conditioned weights INSIDE the original final FFN.

Unlike routed_ffn's hidden-feature offsets, these matrices change how the
current hidden activations are transformed. The original source parameters,
router keys, risk head and direct forecast head are frozen. Temporary matrices
are external same-shot state; only sealed, revealed outcomes train memory.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .routed_ffn import RoutedFfnModel


class RoutedWeightFfnModel(RoutedFfnModel):
    adaptation_site = 'source-keyed conditional linear2 weights inside last Transformer FFN'

    def __init__(self, base, router, *, routing='local'):
        super().__init__(base,router,routing=routing)
        self.temporary_dim=8*64*128
        self.memory=nn.Parameter(self.center.new_zeros(8,64,128))

    def route(self, hidden):
        # Unit L2 gates equalize the linear combination's norm bound between
        # two-key and uniform controls. These are activation coefficients,
        # NOT risk probabilities and NOT eight physical disruption classes.
        return F.normalize(super().route(hidden),dim=-1,eps=1e-12)

    def increment(self, hidden, working):
        if working.shape!=(8,64,128):
            raise ValueError('eight internal 64 by 128 matrices required')
        return torch.einsum('...k,koi,...i->...o',self.route(hidden),working,hidden)

    def forward(self, values, available, fast=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('inference_mode changes the declared source path')
        if values.ndim!=3 or not len(values) or values.shape[1:]!=(128,23):
            raise ValueError('nonempty W128 flat23 batch required')
        fast=self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape!=(len(values),self.temporary_dim) or fast.device!=self.center.device
                or fast.dtype!=self.center.dtype or not torch.isfinite(fast).all()):
            raise ValueError('finite source-device temporary weights required')
        if not torch.equal(fast,fast[:1].expand_as(fast)):
            raise ValueError('one same shot temporary state is required')
        enabled=torch.is_grad_enabled()
        with torch.enable_grad():
            working=self.memory+fast[0].reshape_as(self.memory)
            def adapt(module,inputs,output):
                return output+self.increment(inputs[0],working)
            hook=self.base.encoder.layers[-1].linear2.register_forward_hook(adapt)
            try:
                source_fast=self.base.initial_fast(len(values)).detach().requires_grad_(True)
                result=self.base(values,available,source_fast)
            finally:
                hook.remove()
        return result if enabled else {k:v.detach() for k,v in result.items()}
