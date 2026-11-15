"""Fold-local fitting of causal input to finite, actually measured gradient actions."""

import numpy as np


SELECTION_FLOOR = 0.02
MIN_ROWS = 12  # Six matched pairs, shared arm availability on both sides.
RIDGE = 4.0


def utility(native, future, offset, positive, protect, tau):
    """Truth and completed curves are supplied only to the offline value fit."""
    n = np.asarray(native, dtype=np.float64)
    z = np.asarray(future, dtype=np.float64)
    a = np.asarray(offset, dtype=np.float64)
    pos = positive[:, None, None, :]
    prot = protect[:, None, None, :]
    npd = np.where(positive, n[:, 0], -np.inf).max(1)[:, None, None]
    nps = n[:, 1].max(1)[:, None, None]
    actual_safe = z[:, 1].max(-1)
    safe_hit = (nps < tau) & (actual_safe >= tau)
    lost = protect.any(1)[:, None, None] & (
        np.where(prot, z[:, 0], -np.inf).max(-1) < tau
    )
    cost = 0.75 * np.stack((lost, safe_hit), 1)
    rewards = []
    for v in (z, z - a[..., None]):
        d = np.where(pos, v[:, 0], -np.inf).max(-1) - npd
        s = nps - v[:, 1].max(-1)
        rewards.append(np.stack((d, s), 1).clip(-0.25, 0.25) - cost)
    return tuple(rewards)


def standardize(x, mean, scale):
    values = ((x - mean) / scale).clip(-4.0, 4.0)
    return np.column_stack((np.ones(len(x)), values))


def fit_values(x, reward, available):
    x, reward, available = (
        np.asarray(x, dtype=np.float64),
        np.asarray(reward),
        np.asarray(available),
    )
    if (
        x.ndim != 2
        or not len(x)
        or reward.ndim != 2
        or len(reward) != len(x)
        or available.shape != reward.shape
        or available.dtype != bool
        or not np.isfinite(x).all()
        or not np.isfinite(reward[available]).all()
    ):
        raise ValueError(
            "aligned causal predictors, available actions and finite rewards required"
        )
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.05)
    zz = standardize(x, mean, scale)
    reg = np.diag(np.r_[0.0, np.full(x.shape[1], RIDGE)])
    weights = np.zeros((zz.shape[1], reward.shape[1]), dtype=np.float64)
    active = available.sum(0) >= MIN_ROWS
    for k in np.flatnonzero(active):
        ok = available[:, k]
        xx = zz[ok]
        weights[:, k] = np.linalg.solve(xx.T @ xx + reg, xx.T @ reward[ok, k])
    return dict(mean=mean, scale=scale, weights=weights, trained=active)


def choose_scores(value, available):
    score = np.where(available, value, -np.inf)
    best = score.argmax(1)
    return np.where(score[np.arange(len(best)), best] > SELECTION_FLOOR, best, -1)


def choose(fit, features, available):
    z = standardize(np.asarray(features, dtype=np.float64), fit["mean"], fit["scale"])
    return choose_scores(z @ fit["weights"], available & fit["trained"])


def grouped_folds(ids):
    """Hold both sides out in the same fold, by fixed sorted danger ID blocks."""
    ids = np.asarray(ids)
    if (
        ids.ndim != 2
        or ids.shape[1] != 2
        or len(ids) < 5
        or np.unique(ids).size != ids.size
    ):
        raise ValueError("five folds require unique, disjoint source shot pairs")
    order = np.argsort(ids[:, 0], kind="stable")
    return [(np.setdiff1d(order, hold), hold) for hold in np.array_split(order, 5)]
