"""Causal weak-shape inputs and matched, observed future-shape targets.

The ten channels follow the CUE order in commissioning_event_roughness.
This module constructs targets only; their release time is the caller's duty.
"""

import numpy as np

from .commissioning_transition_data_20261110 import _indices, _series_and_mask, _window_mean


# Magnetic, position, radiation, loop, in CUE channel order.
GROUPS = (0, 0, 0, 1, 1, 3, 2, 2, 2, 2)
_FLOOR = 0.05


def _signal(values, mask):
    values, mask = _series_and_mask(values, mask)
    if values.shape[1] != len(GROUPS):
        raise ValueError('z must have the ten channels in CUE order')
    return values, mask & np.isfinite(values)


def inputs(z, mask, context, context_mask, ends):
    """Return causal design vectors and seven matching coordinates.

    Design columns are recent C channel means; 8*C centered, change-RMS
    scaled bin means in time-major order; C centered-bin RMS amplitudes; 8*C
    binary bin masks in time-major order; recent Ip, Bt, signed Ip change;
    and availability of each of those three physical-context features.
    Bins cover [q-127,q], oldest first; eight observations make a bin valid.

    Matching columns are recent Ip, recent Bt, Ip change, then RMS channel
    change for the magnetic, position, radiation and loop families. A channel
    change compares [q-15,q] against [q-47,q-16], requiring 8 and 16 observed
    samples respectively. Missing quantities are zero, never record length.
    """
    z, mask = _signal(z, mask)
    context, context_mask = _series_and_mask(context, context_mask)
    if context.shape != (z.shape[0], 2):
        raise ValueError('context must have shape (N, 2)')
    context_mask = context_mask & np.isfinite(context)
    ends = _indices(ends, 0, len(z))
    n, c = len(ends), z.shape[1]
    if not n:
        return np.empty((0, 18*c + 6), np.float32), np.empty((0, 7), np.float32)

    means = np.empty((n, 8, c), dtype=np.float64)
    counts = np.empty((n, 8, c), dtype=np.int64)
    for b in range(8):
        means[:, b], counts[:, b] = _window_mean(z, mask, ends, -127 + 16*b, 16)
    valid = counts >= 8
    num = valid.sum(axis=1)
    center = np.divide(np.where(valid, means, 0).sum(axis=1), num,
                       out=np.zeros((n, c)), where=num > 0)
    centered = np.where(valid, means - center[:, None, :], 0)
    amplitude = np.sqrt(np.divide((centered**2).sum(axis=1), num,
                                  out=np.zeros((n, c)), where=num > 0))
    adjacent = valid[:, 1:] & valid[:, :-1]
    diff = np.where(adjacent, means[:, 1:] - means[:, :-1], 0)
    adjacent_count = adjacent.sum(axis=1)
    change_rms = np.sqrt(np.divide((diff**2).sum(axis=1), adjacent_count,
                                   out=np.zeros((n, c)), where=adjacent_count > 0))
    shape = centered / np.maximum(change_rms[:, None, :], _FLOOR)
    recent = np.where(valid[:, -1], means[:, -1], 0)

    physical, physical_counts = _window_mean(context, context_mask, ends, -15, 16)
    prior, prior_counts = _window_mean(context, context_mask, ends, -31, 16)
    physical_valid = physical_counts >= 8
    ip_valid = physical_valid[:, 0] & (prior_counts[:, 0] >= 8)
    ip_change = np.where(ip_valid, physical[:, 0] - prior[:, 0], 0)
    physical = np.where(physical_valid, physical, 0)
    physical_features = np.column_stack((physical, ip_change))
    physical_available = np.column_stack((physical_valid, ip_valid))

    early, early_counts = _window_mean(z, mask, ends, -47, 32)
    family_valid = valid[:, -1] & (early_counts >= 16)
    family_change = np.where(family_valid, means[:, -1] - early, 0)
    families = np.zeros((n, 4), dtype=np.float64)
    groups = np.asarray(GROUPS)
    for group in range(4):
        select = groups == group
        observed = family_valid[:, select].sum(axis=1)
        families[:, group] = np.sqrt(np.divide(
            (family_change[:, select]**2).sum(axis=1), observed,
            out=np.zeros(n), where=observed > 0))

    design = np.concatenate((recent, shape.reshape(n, 8*c), amplitude,
                             valid.reshape(n, 8*c), physical_features,
                             physical_available), axis=1)
    pair = np.column_stack((physical_features, families))
    return design.astype(np.float32), pair.astype(np.float32)


def feedback(z, mask, origins):
    """Return unit real/shifted future shapes and joint eligibility.

    Origins q must have a complete q+1..q+25 horizon. Five consecutive
    5-sample bin means are centered and divided by their channel RMS (floor
    0.05). A channel is observed only with at least 3 samples in every bin.
    At least two observed families and a nonzero global norm are required.
    The null rolls the same centered shape forward by two bins; the caller
    must not release either target before q+26.
    """
    z, mask = _signal(z, mask)
    origins = _indices(origins, 0, len(z) - 25)
    n, c = len(origins), z.shape[1]
    if not n:
        return (np.empty((0, 5*c), np.float32), np.empty((0, 5*c), np.float32),
                np.empty(0, dtype=bool))
    means = np.empty((n, 5, c), dtype=np.float64)
    valid_bins = np.empty((n, 5, c), dtype=bool)
    for b in range(5):
        means[:, b], count = _window_mean(z, mask, origins, 1 + 5*b, 5)
        valid_bins[:, b] = count >= 3
    channel_valid = valid_bins.all(axis=1)
    centered = np.where(channel_valid[:, None, :], means - means.mean(axis=1, keepdims=True), 0)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    shape = centered / np.maximum(rms[:, None, :], _FLOOR)
    norm = np.linalg.norm(shape.reshape(n, 5*c), axis=1)
    groups = np.asarray(GROUPS)
    observed_families = np.stack([channel_valid[:, groups == g].any(axis=1)
                                  for g in range(4)], axis=1).sum(axis=1)
    eligible = (observed_families >= 2) & (norm > 0)
    unit = np.divide(shape, norm[:, None, None], out=np.zeros_like(shape),
                     where=eligible[:, None, None])
    shifted = np.roll(unit, 2, axis=1)
    return unit.reshape(n, 5*c).astype(np.float32), shifted.reshape(n, 5*c).astype(np.float32), eligible
