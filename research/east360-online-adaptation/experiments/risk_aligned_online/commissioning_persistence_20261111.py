"""Causal sustained innovations with a disjoint operating-context reference."""

import numpy as np

from .commissioning_weak_shape_data_20261111 import GROUPS


WINDOW, STRIDE = 25, 32
FIRST_EFFECT = 127 + 2*STRIDE + WINDOW + 1
HISTORY, NEIGHBORS, PRIOR_STRENGTH = 24, 4, 4.
VIEWS = ('sustained', 'current_equal_norm', 'spaced', 'opposite', 'lagged')
_GROUP = np.asarray(GROUPS)


def levels(forecast, target, observed):
    """Masked mean of signed clipped residuals after a complete 25-point window."""
    n = observed.sum(1)
    supported = n >= (WINDOW+1)//2
    family = np.column_stack([supported[:, _GROUP==i].any(1) for i in np.unique(_GROUP)])
    valid = supported & (family.sum(1) >= 2)[:, None]
    clipped = np.where(observed, np.clip(target-forecast, -1., 1.), 0.)
    mean = clipped.sum(1) / np.maximum(n, 1)
    return np.where(valid, mean, 0.).astype(np.float32), valid


def location_scale(level, valid):
    """Global quiet prior, fitted exclusively on observed safe source data."""
    loc, scale = [], []
    for i in range(level.shape[1]):
        x = level[valid[:, i], i]
        if not len(x):
            raise ValueError('source prior lacks observed support for a diagnostic')
        median = np.median(x)
        loc.append(median)
        scale.append(max(.15, 1.4826*np.median(np.abs(x-median))))
    return np.asarray(loc, np.float32), np.asarray(scale, np.float32)


def local_reference(level, valid, context, usable, j, prior):
    """Robust regularized moments; j-2, j-1, j are never reference candidates."""
    first, stop = max(0, j-2-HISTORY), j-2
    earlier = np.arange(first, stop)
    earlier = earlier[usable[earlier] & valid[earlier].any(1)][::-1]
    distances = np.square(context[earlier] - context[j]).sum(1)
    selected = earlier[np.argsort(distances, kind='stable')[:NEIGHBORS]]
    a, s = prior
    mask = valid[selected]
    prior_units = np.where(mask, np.clip((level[selected]-a)/s, -1.5, 1.5), 0.)
    den = PRIOR_STRENGTH + mask.sum(0)
    mean = prior_units.sum(0)/den
    var = (PRIOR_STRENGTH + (prior_units**2).sum(0))/den - mean**2
    return a+s*mean, s*np.sqrt(var)


def _common_direction(a, b):
    return np.where(a*b > 0., np.sign(a)*np.minimum(abs(a), abs(b)), 0.)


def replay(level, valid, origins, context, usable, length, prior):
    """Consecutive-vs-isolated evidence; output first changes after real arrival.

    The controls retain current measurements, masks, center, scale and the
    inherited forecaster. Only the second factor of directional agreement
    differs in the spaced view; the current-only view has exactly the
    sustained view's norm but preserves the full current direction.
    """
    if (level.shape != valid.shape or level.shape != (len(origins), len(_GROUP))
            or context.shape != (len(origins), 3) or usable.shape != origins.shape
            or np.any(np.diff(origins) != STRIDE)
            or np.any((origins < 0) | (origins >= length))):
        raise ValueError('aligned inputs on the fixed 32-point issuance grid required')
    arms = {k: np.zeros((length, level.shape[1]), np.float32) for k in VIEWS[:3]}
    receipt = dict(visible_comparisons=0, active_sustained=0, active_spaced=0)
    for j in range(2, len(origins)):
        start = int(origins[j]+WINDOW+1)
        if start >= length:
            break
        if not usable[j-2:j+1].all():
            continue
        available = valid[j-2:j+1].all(0)
        if np.unique(_GROUP[available]).size < 2:
            continue
        a, s = local_reference(level, valid, context, usable, j, prior)
        z = np.where(available, (level[j-2:j+1]-a)/s, 0.)
        evidence = np.sign(z)*np.clip((abs(z)-.50)/1.50, 0., 1.)
        earlier, last, current = evidence
        d2 = ((context[j-2:j] - context[j])**2).sum(1).max()
        gate = np.exp(-.5*d2)
        adjacent = gate*_common_direction(current, last)
        spaced = gate*_common_direction(current, earlier)
        power = np.linalg.norm(current)
        same_budget = current * (np.linalg.norm(adjacent)/power if power else 0.)
        stop = min(length, start+STRIDE)
        for k, value in zip(arms, (adjacent, same_budget, spaced), strict=True):
            arms[k][start:stop] = value
        receipt['visible_comparisons'] += 1
        receipt['active_sustained'] += bool(adjacent.any())
        receipt['active_spaced'] += bool(spaced.any())
    arms['opposite'] = -arms['sustained']
    delayed = np.zeros_like(arms['sustained'])
    delayed[STRIDE:] = arms['sustained'][:-STRIDE]
    arms['lagged'] = delayed
    return arms, receipt
