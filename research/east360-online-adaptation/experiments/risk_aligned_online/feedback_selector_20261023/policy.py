"""Prefix-trained selection of an existing unlabeled encoder update.

The regressor predicts finite *risk-loss differences*, never risk scores.
Deployment has no outcome input and leaves the source risk head unchanged.
"""
from collections import Counter

import numpy as np
import torch

ACTIONS = ('hold', 'recent', 'lag_bias')
N_FEATURES = 8 * 23 + 1


def make_context(prediction, truth, mask, level, available, logit):
    if (prediction.shape != (4, 25, 23) or truth.shape != prediction.shape
            or mask.shape != prediction.shape or mask.dtype != torch.bool
            or level.shape != (128, 23) or available.shape != level.shape
            or available.dtype != torch.bool):
        raise ValueError('chronological four forecasts and arrived W128 context required')
    if (not torch.isfinite(prediction[mask]).all() or not torch.isfinite(truth[mask]).all()
            or not torch.isfinite(level[available]).all() or not np.isfinite(logit)):
        raise FloatingPointError('nonfinite observed context')

    def mean(value, observed):
        axes = tuple(range(value.ndim - 1))
        # Zero denotes absent context, accompanied by an observation fraction;
        # it is not imputed into a forecast target or risk input.
        return torch.where(observed, value.detach(), 0.).sum(axes) / observed.sum(axes).clamp_min(1)

    residual = torch.where(mask, prediction.detach() - truth.detach(), 0.)
    pieces = [mean(residual[:2], mask[:2]), mean(residual[2:], mask[2:]),
              mean(residual[2:].square(), mask[2:]).sqrt(), mean(level, available),
              mean(level[64:], available[64:]) - mean(level[:64], available[:64]),
              available.float().mean(0), mask[:2].float().mean((0, 1)),
              mask[2:].float().mean((0, 1))]
    return np.concatenate([p.cpu().numpy().astype(np.float64) for p in pieces] +
                          [np.array([logit], dtype=np.float64)])


def fit_policy(rows, *, allowed_ids):
    allowed = set(allowed_ids)
    if (len(allowed) != len(allowed_ids) or not rows
            or {r['shot'] for r in rows} != allowed):
        raise ValueError('all and only permitted ended shots must supply training rows')
    labels = {}
    trajectories = {}
    counts = Counter()
    for r in rows:
        if type(r['is_disrupt']) is not bool:
            raise ValueError('explicit boolean ended-shot outcome required')
        sid = r['shot']; label = r['is_disrupt']; trajectory = r['trajectory']
        if sid in labels and labels[sid] != label:
            raise ValueError('inconsistent shot outcome')
        labels[sid] = label
        trajectories.setdefault(sid, set()).add(trajectory)
        counts[(sid, trajectory)] += 1
    if set(labels.values()) != {False, True}:
        raise ValueError('both outcome classes required; no one-class substitution')
    if any(t != {'static', 'recent', 'lag_bias'} for t in trajectories.values()):
        raise ValueError('each shot must supply all three declared trajectories')
    class_count = Counter(labels.values())
    weights = np.array([.5 / class_count[r['is_disrupt']] / len(trajectories[r['shot']]) /
                        counts[(r['shot'], r['trajectory'])] for r in rows])
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-12)
    x = np.asarray([r['context'] for r in rows], dtype=np.float64)
    y = np.asarray([r['utility'] for r in rows], dtype=np.float64)
    if x.shape != (len(rows), N_FEATURES) or y.shape != (len(rows), 2):
        raise ValueError('185-dimensional context and two candidate utilities required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError('nonfinite prefix training data')
    mean = weights @ x
    scale = np.maximum(np.sqrt(weights @ ((x - mean) ** 2)), .0001)
    design = np.column_stack(((x - mean) / scale, np.ones(len(rows))))
    penalty = np.eye(N_FEATURES + 1) * .01
    penalty[-1, -1] = 0.  # Intercept is unpenalized; sum of weights is one.
    coefficient = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                  design.T @ (weights[:, None] * y))
    if not np.isfinite(coefficient).all():
        raise FloatingPointError('nonfinite ridge solution')
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coefficient.tolist(),
                allowed_ids=list(allowed_ids), ridge=.01, intercept_penalized=False,
                class_counts={str(k): v for k, v in class_count.items()},
                trajectories=sorted(next(iter(trajectories.values()))),
                n_rows=len(rows), weights=weights.tolist(),
                weighted_fit_mse=(weights @ ((design @ coefficient - y) ** 2)).tolist())


def choose_action(policy, context, *, mean_context=False):
    x = np.asarray(context, dtype=np.float64)
    if x.shape != (N_FEATURES,) or not np.isfinite(x).all():
        raise ValueError('finite signal-only context required')
    if mean_context:
        x = np.asarray(policy['mean'])
    scaled = (x - policy['mean']) / policy['scale']
    utilities = np.r_[0., np.r_[scaled, 1.] @ np.asarray(policy['coefficients'])]
    if not np.isfinite(utilities).all():
        raise FloatingPointError('nonfinite predicted utility')
    return ACTIONS[int(utilities.argmin())], utilities.tolist()
