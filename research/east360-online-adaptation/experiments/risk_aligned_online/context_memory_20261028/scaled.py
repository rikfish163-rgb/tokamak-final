"""Source TRAIN increment metric shared by teaching and causal context writes."""
from pathlib import Path

import torch

from ..simple_train import sha
from .checkpoint import read
from .model import ContextMemory


def weights_from_moments(moments):
    if (moments.shape != (4, 23) or not torch.isfinite(moments).all() or (moments < 0).any()):
        raise ValueError('finite nonnegative source second moments, four horizons and 23 channels required')
    middle = moments.median(1, keepdim=True).values
    if (middle <= 0).any():
        raise ValueError('nondegenerate source increment scale required at every horizon')
    raw = (middle/torch.maximum(moments, middle)).clamp(.01, 1.)
    return (raw/raw.mean(1, keepdim=True)).float()


def validate_weights(weights):
    if (weights.shape != (4, 23) or not torch.isfinite(weights).all() or (weights <= 0).any()
            or not torch.allclose(weights.mean(1), weights.new_ones(4), atol=1e-6, rtol=1e-6)
            or weights.max() > 23/12+1e-6
            or (weights.max(1).values/weights.min(1).values).max() > 100+1e-4):
        raise ValueError('positive bounded TRAIN-derived unit-mean horizon weights required')


def estimate(cache, train_ids):
    cache = Path(cache)
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    if len(train_ids) != 1353 or len(set(train_ids)) != 1353:
        raise ValueError('complete source TRAIN1353 required')
    total = torch.zeros(4, 23, dtype=torch.float64)
    shots = torch.zeros(4, 23, dtype=torch.long)
    ineligible = []
    for sid in train_ids:
        path = cache/'cache'/f'{sid}.pt'
        if inventory[sid]['role'] != 'train' or sha(path) != inventory[sid]['cache_sha256']:
            raise ValueError('canonical training cache required for fixed scale estimation')
        e = torch.load(path, map_location='cpu', weights_only=True)
        x, a = e['means'].double(), e['complete']
        if not torch.isfinite(x[a]).all():
            raise ValueError('finite observed source values required')
        if len(x) <= 4:
            ineligible.append(dict(shot=sid, patches=len(x), reason='no_complete_four_horizon_origin'))
            continue
        x = torch.where(a, x, 0.)
        y = torch.stack([x[h:len(x)-4+h]-x[:-4] for h in range(1,5)],1)
        mask = torch.stack([a[h:len(x)-4+h]&a[:-4] for h in range(1,5)],1)
        count = mask.sum(0)
        total += torch.where(mask,y.square(),0.).sum(0)/count.clamp_min(1)
        shots += count > 0
    if (shots == 0).any():
        raise ValueError('training observations missing for a channel/horizon; no scale imputation')
    moments = total/shots
    weights = weights_from_moments(moments)
    validate_weights(weights)
    return weights, dict(method='equal_shot_increment_second_moment_median_floor_ratio100_v1',
        scope='JTEXT_TRAIN_ONLY_NOT_PHYSICAL_UNIT_CALIBRATION', train_ids=train_ids,
        cache_inventory_sha256=sha(cache/'inventory.json'), ineligible=ineligible,
        observed_shots=shots.tolist(), moments=moments.tolist(), weights=weights.tolist())


class ScaledContextMemory(ContextMemory):
    def __init__(self, dim, weights):
        validate_weights(weights)
        super().__init__(dim)
        self.register_buffer('forecast_weights', weights.detach().clone())

    def write(self, context, basis, target, mask):
        if (context.shape != (8,) or target.ndim != 2 or target.shape[1] != 23
                or not 1 <= len(target) <= 4 or basis.shape != (*target.shape, 8)
                or mask.shape != target.shape or mask.dtype != torch.bool):
            raise ValueError('arrived horizons in original order and matching observed basis required')
        # Base trace passes h=1..count, even for the reverse-target control.
        # Weights must stay with the issued forecast horizon, not the permuted target.
        w = torch.where(mask, self.forecast_weights[:len(target)], 0.)
        error = torch.where(mask, basis@context-target, 0.)
        safe_basis = torch.where(mask[..., None],basis,0.)
        gradient = (safe_basis*(w*error)[...,None]).sum((0,1))
        candidate = context-self.eta*gradient/(1+(safe_basis.square()*w[...,None]).sum())
        return candidate/candidate.norm().clamp_min(1.)

    def trace(self, means, complete, **kwargs):
        result = super().trace(means, complete, **kwargs)
        result['unweighted_forecast_loss'] = result['forecast_loss']
        if len(means) > 4:
            x = torch.where(complete,means,0.)
            target = torch.stack([x[h:len(x)-4+h]-x[:-4] for h in range(1,5)],1)
            valid = torch.stack([complete[h:len(x)-4+h]&complete[:-4] for h in range(1,5)],1)
            w = valid*self.forecast_weights
            numerator = ((result['forecast'][:-4]-target).square()*w).sum()
            # No observed targets means zero contribution, matching the original
            # trainer. No artificial unit denominator when a small weight is valid.
            denominator = w.sum()
            result['forecast_loss'] = numerator/torch.where(denominator>0,denominator,denominator.new_ones(()))
        return result
