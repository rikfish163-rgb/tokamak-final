"""Per-shot, outcome-free numerical feedback in fixed scaled coordinates.

The caller is responsible for releasing each origin only after q+25 arrives,
and for calling ``innovations`` separately for each shot.
"""

import numpy as np

from .commissioning_transition_data_20261110 import _indices, _series_and_mask, _window_mean


GROUPS = (0, 0, 0, 1, 1, 3, 2, 2, 2, 2)
HORIZON = 25


def _observed_center(mask, origins, start, length, counts):
    offsets = np.arange(start, start+length)
    weighted = (mask[origins[:,None]+offsets] * offsets[None,:,None]).sum(1)
    return np.divide(weighted,counts,out=np.zeros(counts.shape,np.float64),where=counts>0)


def raw_feedback(z, mask, origins):
    """Fit each channel's residual future slope after a linear past trend.

    An origin requires q >= 31 and the full future horizon. A channel is
    observed with at least eight finite, masked samples in both history bins
    and three in every future bin. Unobserved output is zero.
    """
    z, mask = _series_and_mask(z, mask)
    if z.shape[1] != len(GROUPS):
        raise ValueError('z must have ten channels in CUE order')
    mask = mask & np.isfinite(z)
    origins = _indices(origins, 31, len(z) - HORIZON)
    n = len(origins)
    if not n:
        return np.empty((0, 10), np.float32), np.empty((0, 10), bool)

    previous, previous_count = _window_mean(z, mask, origins, -31, 16)
    recent, recent_count = _window_mean(z, mask, origins, -15, 16)
    future = np.empty((n, 5, 10), np.float64)
    counts = np.empty((n, 5, 10), np.int64)
    centers = np.empty_like(future)
    for k in range(5):
        future[:, k], counts[:, k] = _window_mean(z, mask, origins, 1 + 5*k, 5)
        centers[:, k] = _observed_center(mask,origins,1+5*k,5,counts[:,k])
    observed = (previous_count >= 8) & (recent_count >= 8) & (counts >= 3).all(axis=1)
    duration = (_observed_center(mask,origins,-15,16,recent_count)
                - _observed_center(mask,origins,-31,16,previous_count))
    past_rate = np.divide(recent-previous,duration,out=np.zeros_like(recent),where=observed)
    elapsed = centers[:,1:] - centers[:,:1]
    residual = future[:,1:] - future[:,:1] - elapsed*past_rate[:,None]
    factors = elapsed / 20.
    numerator = (residual*factors).sum(1)
    denom = np.square(factors).sum(1)
    surplus = np.divide(numerator,denom,out=np.zeros_like(numerator),where=observed).astype(np.float32)
    return surplus, observed


def physical_keys(context, mask):
    """Return causal keys from fixed-scaled asinh Ip and Bt at each sample."""
    context, mask = _series_and_mask(context, mask)
    if context.shape[1] != 2:
        raise ValueError('context must have shape (T, 2)')
    mask = mask & np.isfinite(context)
    n = len(context)
    if not n:
        return np.empty((0, 4), np.float32)
    endpoints = np.arange(n)
    recent, recent_count = _window_mean(context, mask, endpoints, -15, 16)
    previous, previous_count = _window_mean(context, mask, endpoints, -31, 16)
    recent = np.where(recent_count >= 8, recent, 0)
    change = np.where((recent_count[:, 0] >= 8) & (previous_count[:, 0] >= 8),
                      recent[:, 0] - previous[:, 0], 0)
    return (0.5 * np.column_stack((np.ones(n), np.tanh(recent[:, 0]),
                                   np.tanh(recent[:, 1]), np.tanh(4 * change)))).astype(np.float32)


def _features(e, observed, previous_e, previous_observed, adjacent):
    persistent = np.where(observed & previous_observed & adjacent & (e * previous_e > 0),
                          np.sign(e) * np.minimum(np.abs(e), np.abs(previous_e)), 0)
    group_e = np.zeros(4, np.float64)
    group_persistent = np.zeros(4, np.float64)
    groups = np.asarray(GROUPS)
    for group in range(4):
        valid = observed & (groups == group)
        if valid.any():
            group_e[group] = np.sqrt(np.mean(e[valid] ** 2))
            group_persistent[group] = np.sqrt(np.mean(persistent[valid] ** 2))
    return np.tanh(np.concatenate((e, persistent, group_e, group_persistent)))


def innovations(surplus, observed, scale):
    """Build matched per-shot no-background, intra and lag feedback features.

    ``scale`` is the fixed positive ten-channel TRAIN scale. A block is
    eligible if at least two CUE groups are observed; non-finite surplus is
    unobserved. Lag uses the preceding eligible intra vector, with the first
    eligible vector copied from its own intra vector. Audit background_updates
    counts channel updates passing the robust gate, not blocks.
    """
    surplus, observed = _series_and_mask(surplus, observed)
    if surplus.shape[1] != len(GROUPS):
        raise ValueError('surplus must have ten channels in CUE order')
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (10,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('scale must contain ten finite positive TRAIN scales')
    observed = observed & np.isfinite(surplus)
    with np.errstate(over='ignore', invalid='ignore'):
        r = np.clip(np.where(observed, surplus, 0).astype(np.float64) / scale, -8, 8)
    groups = np.asarray(GROUPS)
    eligible = np.stack([observed[:, groups == g].any(axis=1) for g in range(4)], axis=1).sum(axis=1) >= 2
    n = len(surplus)
    arms = {name: np.zeros((n, 28), np.float32) for name in ('no_background', 'intra', 'lag')}
    background = np.zeros(10, np.float64)
    previous = {name: np.zeros(10, np.float64) for name in ('no_background', 'intra')}
    previous_observed = np.zeros(10, bool)
    last_row = -2
    last_intra = None
    updates = 0
    for i in np.flatnonzero(eligible):
        valid = observed[i]
        for name in ('no_background', 'intra'):
            e = np.where(valid, r[i] - (background if name == 'intra' else 0), 0)
            arms[name][i] = _features(e, valid, previous[name], previous_observed, i == last_row + 1)
            previous[name] = e
        arms['lag'][i] = arms['intra'][i] if last_intra is None else last_intra
        last_intra = arms['intra'][i].copy()
        e = previous['intra']
        update = valid & (np.abs(e) <= 1.5)
        background[update] = np.clip(background[update] + 0.08 * np.clip(e[update], -0.5, 0.5), -1, 1)
        updates += int(update.sum())
        previous_observed = valid
        last_row = i
    return arms, eligible, {'background_updates': updates, 'eligible_blocks': int(eligible.sum())}
