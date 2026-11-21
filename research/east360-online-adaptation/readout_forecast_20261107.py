"""Forecast from one token in Simple23's flattened 17-token representation."""

from torch import nn


class TokenForecast(nn.Module):
    def __init__(self, dim, horizon, site):
        super().__init__()
        if site not in ('cls', 'last_data'):
            raise ValueError("site must be 'cls' or 'last_data'")
        self.site = site
        self.projection = nn.Linear(dim, horizon * 23)

    def forward(self, x):
        dim = self.projection.in_features
        if x.ndim != 2 or x.shape[1] != 17 * dim:
            raise ValueError('expected flattened [batch, 17 * dim] tokens')
        tokens = x.reshape(x.shape[0], 17, dim)
        return self.projection(tokens[:, -1 if self.site == 'cls' else -2])
