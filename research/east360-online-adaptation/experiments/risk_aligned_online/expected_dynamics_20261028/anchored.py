"""Training-time zero-change constraint, not a risk-output correction."""
import torch

from .forecast import DirectForecast

ARMS = {
    'direct_native': (False, False),
    'anchor_native': (True, False),
    'direct_no_mnm': (False, True),
    'anchor_no_mnm': (True, True),
}


class AnchoredForecast(DirectForecast):
    def __init__(self, *, anchored, omit_mnm):
        if not isinstance(anchored, bool) or not isinstance(omit_mnm, bool):
            raise ValueError('explicit boolean structural factors required')
        super().__init__()
        self.anchored = anchored
        self.omit_mnm = omit_mnm

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 92:
            raise ValueError('observed 92-dimensional prefix features required')
        if self.omit_mnm:
            features = features.clone()
            features[:, [2, 25, 48, 71]] = 0
        full = super().forward(features)
        if not self.anchored:
            return full
        zero_change = torch.cat((features[:, :46], torch.zeros_like(features[:, 46:69]), features[:, 69:]), -1)
        return full - super().forward(zero_change)
