"""Shot-local residual context relative to previously issued expectations."""
import copy

import torch

from ..context_memory_20261028.model import ContextAdapter
from ..context_memory_20261028.scaled import ScaledContextMemory
from .forecast import observed_features


class ExpectedMemory(ScaledContextMemory):
    def __init__(self, dim, weights, predictor, *, use_expected):
        if not isinstance(use_expected, bool):
            raise ValueError('explicit boolean expectation arm required')
        super().__init__(dim, weights)
        self.predictor = copy.deepcopy(predictor).eval().requires_grad_(False)
        self.use_expected = use_expected

    @torch.no_grad()
    def expected_forecast(self, clean, complete):
        if not self.use_expected:
            return None
        return self.predictor(observed_features(clean, complete))


class ExpectedAdapter(ContextAdapter):
    def __init__(self, base, weights, predictor, *, use_expected):
        super().__init__(base)
        self.memory = ExpectedMemory(base.cfg.dim, weights, predictor, use_expected=use_expected)
        self.slow.requires_grad_(False)
        self.memory.requires_grad_(True)
        self.memory.predictor.requires_grad_(False)
