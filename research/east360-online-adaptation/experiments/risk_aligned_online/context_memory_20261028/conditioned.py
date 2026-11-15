"""Parameter-matched source probe of content-conditioned INTERNAL adaptation.

The source classifier is immutable. Only a diagnostic-to-token readout learns.
This experiment is not accepted by any EAST checkpoint loader.
"""
import torch

from .model import ContextAdapter


class ConditionedReadout(ContextAdapter):
    def __init__(self, base, *, kind):
        if kind not in ('additive', 'state_conditioned'):
            raise ValueError('explicit readout comparison arm required')
        super().__init__(base)
        self.kind = kind
        self.requires_grad_(False)
        self.memory.output.requires_grad_(True)

    def internal_shift(self, last, delta):
        effect = .1*torch.tanh(self.memory.output(delta))
        if self.kind == 'state_conditioned':
            # Current token only: no batch/target statistics or future samples.
            # Same L2 upper bound .1*sqrt(dim) as the additive arm. The actual
            # magnitudes can still differ; this is not a norm-matched ablation.
            content = last/torch.sqrt(last.square().mean(-1, keepdim=True)+1e-8)
            effect = effect*content
        return effect
