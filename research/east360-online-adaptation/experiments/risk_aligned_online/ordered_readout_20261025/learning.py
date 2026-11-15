"""Small, independently tested readout diagnostics; not a deployed policy.

The earlier ten-shot proposal was superseded by complete-stream experiments.
These mathematical utilities are retained without importing the old EAST fit.
"""
from itertools import combinations

import numpy as np


def decompose(x, mode):
    if mode not in ('full', 'shape', 'level') or x.shape[-1] != 92:
        raise ValueError('declared readout and four-by-23 coordinates required')
    level = x.reshape(*x.shape[:-1], 4, 23).mean(-2, keepdim=True)
    level = level.expand(*x.shape[:-1], 4, 23).reshape_as(x)
    return x if mode == 'full' else level if mode == 'level' else x-level


def minnorm(gradients):
    """Exact small convex-hull problem by enumerating all nonempty faces."""
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2 or not 1 <= len(g) <= 8 or not np.isfinite(g).all():
        raise ValueError('one to eight finite gradient rows required')
    q = g @ g.T
    best = None
    for n in range(1, len(g)+1):
        for face in combinations(range(len(g)), n):
            ix = np.array(face)
            system = np.block([[q[np.ix_(ix, ix)], np.ones((n, 1))],
                               [np.ones((1, n)), np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(n), 1.]
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0][:n]
            if solution.min() < -1e-10 or abs(solution.sum()-1) > 1e-8:
                continue
            a = np.zeros(len(g)); a[ix] = np.maximum(solution, 0)
            a /= a.sum()
            value = float(a @ q @ a)
            if best is None or value < best[0]:
                best = value, a
    if best is None:
        raise ArithmeticError('convex hull has no numerically feasible face')
    value, a = best
    directional = q @ a
    return a, dict(norm_squared=max(0., value),
                   minimum_directional_derivative=float(directional.min()),
                   kkt_gap=float(max(abs(directional[a > 1e-9]-value).max(initial=0),
                                     max(0., value-directional.min()))))


def accepted(old, new, mode):
    old, new = np.asarray(old, float), np.asarray(new, float)
    if (mode not in ('mean', 'guard') or old.ndim != 1 or old.shape != new.shape
            or not len(old) or not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError('finite matching group losses and declared rule required')
    return bool(new.mean() < old.mean() and (mode == 'mean' or np.all(new <= old)))


def group_values(records):
    if not records or len({r['shot'] for r in records}) != len(records):
        raise ValueError('distinct records required')
    dangerous = sorted((r for r in records if r['flag']), key=lambda r: r['shot'])
    safe = [r['loss'] for r in records if not r['flag']]
    if not dangerous or not safe:
        raise ValueError('both outcome classes required')
    values = np.asarray([r['loss'] for r in dangerous]+[np.mean(safe)], float)
    if not np.isfinite(values).all():
        raise ValueError('finite losses required')
    return [r['shot'] for r in dangerous]+['safe_mean'], values
