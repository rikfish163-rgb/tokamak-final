"""Source-only direct diagnostic expectation, separate from the risk model."""
import numpy as np
import torch
from torch import nn


def observed_features(x, available):
    if (x.ndim != 2 or x.shape[1] != 23 or available.shape != x.shape
            or available.dtype != torch.bool or not torch.isfinite(x[available]).all()):
        raise ValueError('finite observed 23-channel patches required')
    clean = torch.where(available, x, 0.)
    pair = torch.zeros_like(available)
    change = torch.zeros_like(clean)
    if len(x) > 1:
        pair[1:] = available[1:] & available[:-1]
        change[1:] = torch.where(pair[1:], clean[1:] - clean[:-1], 0.)
    return torch.cat((clean, available.to(x), change, pair.to(x)), -1)


def direct_examples(x, available):
    features = observed_features(x, available)
    n = len(x)
    if n <= 4:
        raise ValueError('no complete four-horizon origin; never invent future targets')
    clean = torch.where(available, x, 0.)
    mask = torch.stack([available[h:n-4+h] & available[:-4] for h in range(1, 5)], 1)
    changes = torch.stack([clean[h:n-4+h] - clean[:-4] for h in range(1, 5)], 1)
    return features[:-4], torch.where(mask, changes, 0.), mask


class DirectForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(92, 64), nn.SiLU(), nn.Linear(64, 92))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 92:
            raise ValueError('observed prefix features, not risk/phase/outcome input, required')
        return self.net(features).reshape(-1, 4, 23)


def forecast_screen(rows):
    if (len(rows) != 160 or sum(bool(r['disruptive']) for r in rows) != 80
            or any(not np.isfinite(r[k]) or r[k] < 0 for r in rows for k in ('prediction', 'persistence'))):
        raise ValueError('complete finite source VAL160 with 80/80 classes required')
    mean = lambda part, key: float(np.mean([r[key] for r in part]))
    safe = [r for r in rows if not r['disruptive']]
    disruptive = [r for r in rows if r['disruptive']]
    checks = dict(average_improves=mean(rows, 'prediction') < mean(rows, 'persistence'),
        safe_not_worse=mean(safe, 'prediction') <= mean(safe, 'persistence'),
        disruptive_not_worse=mean(disruptive, 'prediction') <= mean(disruptive, 'persistence'),
        majority=sum(r['prediction'] < r['persistence'] for r in rows) >= 81,
        safe_majority=sum(r['prediction'] < r['persistence'] for r in safe) >= 41)
    return dict(eligible=all(checks.values()), checks=checks,
                scope='FORECAST_COMPONENT_SCREEN_NOT_RISK_OR_EAST_ACCEPTANCE')
