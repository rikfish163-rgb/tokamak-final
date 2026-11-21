"""Direct, per-channel dynamic response fitted only to matured observations.

This is a small ridge-regression baseline, not a new physical law. The state
resets on every trace. It never consumes a predicted value as an observation.
"""
import torch
from torch import nn

from ..lifecycle_stream_20261025.model import LifecycleAdapter


def window_sum(values, width=16):
    total = values.cumsum(0)
    if len(values) > width:
        total = total - torch.cat((torch.zeros_like(total[:width]), total[:-width]), 0)
    return total


class SlopeMemory(nn.Module):
    def __init__(self, dim, prior, ridge, weights=None, *, static_features=False):
        super().__init__()
        if (prior.shape != (4, 23) or ridge.shape != prior.shape
                or not torch.isfinite(prior).all() or not torch.isfinite(ridge).all()
                or (ridge <= 0).any()):
            raise ValueError('finite source coefficients and positive source ridge scales required')
        self.register_buffer('prior', prior.detach().clone())
        self.register_buffer('ridge', ridge.detach().clone())
        if weights is None:
            weights = torch.ones_like(prior)
            weights[:, 2] = 0
        if (weights.shape != prior.shape or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('fixed finite forecast weights required')
        self.register_buffer('forecast_weights', weights.detach().clone())
        self.output = nn.Linear(92, dim, bias=False, dtype=prior.dtype)
        nn.init.zeros_(self.output.weight)
        self.static_features = static_features

    def trace(self, means, complete, *, mode, reference_norm=None, return_before=False):
        if (mode not in ('off', 'memory', 'reverse', 'hold') or means.ndim != 2
                or means.shape[1] != 23 or complete.shape != means.shape
                or complete.dtype != torch.bool or not torch.isfinite(means[complete]).all()):
            raise ValueError('finite observed 23-channel patches and declared feedback mode required')
        # MNM is absent under the existing EAST input contract. Keep the risk
        # backbone unchanged; only this auxiliary response uses common22.
        complete = complete.clone(); complete[:, 2] = False
        x = torch.where(complete, means, 0.)
        n = len(x)
        pair = torch.zeros_like(complete)
        d = torch.zeros_like(x)
        if n > 1:
            pair[1:] = complete[1:] & complete[:-1]
            d[1:] = torch.where(pair[1:], x[1:]-x[:-1], 0.)
        origin_d = x.new_zeros(n, 4, 23)
        target = x.new_zeros(n, 4, 23)
        mask = complete.new_zeros(n, 4, 23)
        for h in range(1, 5):
            if n > h:
                origin_d[h:, h-1] = d[:-h]
                target[h:, h-1] = x[h:]-x[:-h]
                mask[h:, h-1] = complete[h:] & pair[:-h]
        if mode == 'reverse' and n:
            count = torch.arange(n, device=x.device).clamp_max(4)
            h = torch.arange(4, device=x.device)[None]
            perm = (count[:, None]-h-1).clamp_min(0)
            perm = perm[..., None].expand(n, 4, 23)
            target = target.gather(1, perm)
            mask = mask & mask.gather(1, perm) & (h < count[:, None])[..., None]
        elif mode == 'hold':
            target = torch.zeros_like(target)
        p = self.prior.to(x)
        r = self.ridge.to(x)
        if mode == 'off':
            coefficients = p.expand(n, -1, -1)
        else:
            # Sufficient statistics enter at arrival index, never origin index.
            xy = torch.where(mask, origin_d*target, 0.)
            xx = torch.where(mask, origin_d.square(), 0.)
            # Double cumulative sums reduce cancellation in long shots.
            coefficients = ((window_sum(xy.double())+r.double()*p.double()) /
                            (window_sum(xx.double())+r.double())).to(x)
        forecast = coefficients*d[:, None]
        baseline = p*d[:, None]
        delta = (forecast-baseline).flatten(1)
        before_coefficients = torch.cat((p[None], coefficients[:-1]), 0) if n else coefficients
        before = ((before_coefficients-p)*d[:, None]).flatten(1)
        if self.static_features and mode != 'off':
            delta = baseline.flatten(1)
            before = delta
        if n > 4:
            y = torch.stack([x[h:n-4+h]-x[:-4] for h in range(1, 5)], 1)
            valid = torch.stack([complete[h:n-4+h]&complete[:-4] for h in range(1, 5)], 1)
            w = valid*self.forecast_weights.to(x)
            loss = ((forecast[:-4]-y).square()*w).sum()/w.sum().clamp_min(1)
        else:
            loss = x.sum()*0
        result = dict(delta=delta, forecast=forecast, baseline=baseline,
            coefficients=coefficients, forecast_loss=loss,
            norm=(coefficients-p).flatten(1).norm(dim=1),
            zero_direction=complete.new_zeros(n),
            updates=int(mask.any(-1).any(-1).sum()) if mode != 'off' else 0,
            horizon_targets=mask.any(-1).sum(0).tolist(), target_scalars=int(mask.sum()))
        if return_before:
            result['before_delta'] = before
        return result


class SlopeAdapter(LifecycleAdapter):
    def __init__(self, base, prior, ridge, weights, *, static_features=False):
        super().__init__(base, readout='full', feedback='progressive')
        self.slow.requires_grad_(False)
        self.memory = SlopeMemory(base.cfg.dim, prior, ridge, weights, static_features=static_features)
