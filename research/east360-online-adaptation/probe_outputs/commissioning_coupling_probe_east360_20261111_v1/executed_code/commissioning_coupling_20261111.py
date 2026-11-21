"""Separate per-family activity from its observed inter-family time ordering."""

from itertools import combinations

import numpy as np

from .commissioning_transition_data_20261110 import _indices, _window_mean
from .commissioning_weak_shape_data_20261111 import GROUPS, _signal


FAMILIES = ('magnetic', 'position', 'radiation', 'loop')
BIN, BINS = 8, 8
FAMILY_PAIRS = tuple(combinations(range(len(FAMILIES)), 2))


def _mean(values, mask, axis):
    n = mask.sum(axis)
    total = np.where(mask, values, 0.).sum(axis)
    return np.divide(total, n, out=np.zeros_like(total), where=n > 0)


def activity(z, mask, ends):
    """Eight adjacent-bin changes from 72 arrived samples, log-RMS per family."""
    z, mask = _signal(z, mask)
    ends = _indices(ends, (BINS + 1) * BIN - 1, len(z))
    means, counts = (np.stack(a, axis=1) for a in zip(*(
        _window_mean(z, mask, ends, 1 - (BINS + 1 - j) * BIN, BIN)
        for j in range(BINS + 1)), strict=True))
    observed = (counts[:, 1:] >= BIN // 2) & (counts[:, :-1] >= BIN // 2)
    square = np.square(means[:, 1:] - means[:, :-1])
    group = np.asarray(GROUPS)
    values = np.stack([np.log1p(np.sqrt(_mean(square[..., group == g],
                                            observed[..., group == g], 2)))
                       for g in range(len(FAMILIES))], axis=2)
    valid = np.stack([observed[..., group == g].any(2) for g in range(len(FAMILIES))], axis=2)
    return values, valid


def alignment(values, observed):
    """Even and odd parts of centered cross-products at 0, 8 and 16 samples.

    The forward/backward estimates at each lag share the intersection of the
    four availability masks. Reversing both input time axes therefore negates
    only the odd part; the signs are not assumed to be physical causation.
    """
    centered = values - _mean(values, observed, 1)[:, None, :]
    norm = np.sqrt(_mean(centered ** 2, observed, 1)).clip(.05)
    standardized = np.where(observed, centered / norm[:, None, :], 0.)
    symmetric, directed = [], []
    for i, j in FAMILY_PAIRS:
        a, b = standardized[:, :, i], standardized[:, :, j]
        ma, mb = observed[:, :, i], observed[:, :, j]
        same = ma & mb
        symmetric.extend((_mean(a*b, same, 1), same.mean(1)))
        for lag in (1, 2):
            keep = ma[:, :-lag] & mb[:, :-lag] & ma[:, lag:] & mb[:, lag:]
            forward, backward = a[:, :-lag] * b[:, lag:], b[:, :-lag] * a[:, lag:]
            keep &= (keep.sum(1) >= 3)[:, None]
            sym = _mean(forward + backward, keep, 1) * .5
            order = _mean(forward - backward, keep, 1) * .5
            symmetric.extend((sym, np.abs(order), keep.mean(1)))
            directed.append(order)
    return np.column_stack(symmetric), np.column_stack(directed)


def observed_features(z, mask, context, context_mask, ends):
    """Return marginal, unsigned coupling, signed coupling and matched contexts."""
    values, valid = activity(z, mask, ends)
    phys, count = (np.stack(a, axis=1) for a in zip(*(
        _window_mean(context, context_mask, ends, offset, 16)
        for offset in (-63, -47, -31, -15)), strict=True))
    first, last = _mean(values[:, :4], valid[:, :4], 1), _mean(values[:, 4:], valid[:, 4:], 1)
    ip_change = phys[:, 1::2, 0] - phys[:, ::2, 0]
    operating = np.concatenate((phys[:, 1::2], ip_change[..., None]), axis=2)
    matching = np.concatenate((operating, np.stack((first, last), axis=1)), axis=2)
    persistence = _mean(np.minimum(values[:, 1:], values[:, :-1]),
                        valid[:, 1:] & valid[:, :-1], 1)
    unilateral = np.column_stack((operating.reshape(-1, 6), first, last,
                                  _mean(values[:, -2:], valid[:, -2:], 1),
                                  values.max(1), persistence, valid.mean(1)))
    even, odd = alignment(values, valid)
    return dict(context=matching, marginal=unilateral, even=even, odd=odd,
                eligible=(count >= 8).all((1, 2)) & ((valid.sum(1) >= 6).sum(1) >= 3))


def outcome_endpoints(times, grid, eligible, down):
    """Post-outcome diagnostic anchors, not a change to source training labels."""
    lead = np.round((down - times[grid]) * 1000., 9)
    selected = []
    for low, high, target in ((30., 60., 45.), (60., 95., 75.), (95., 125., 105.)):
        ids = np.flatnonzero(eligible & (lead > low) & (lead <= high))
        if len(ids):
            selected.append(grid[ids[np.argmin(np.abs(lead[ids] - target))]])
    return np.asarray(selected, dtype=np.int64)
