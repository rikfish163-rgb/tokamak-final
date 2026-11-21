"""Masked, causal dual-dynamics features with a per-shot linear adaptation.

``blocks`` in :func:`replay` is the ``(q, x, y, m)`` tuple from
:func:`prepare`.  Feature columns are grouped as C margins, C signed
persistence values, then C observed fractions.  Huber losses use a 0.5
transition without the normalization used by some SmoothL1 definitions.
"""

import numpy as np


_HORIZON = 25


def _window_sums(values, valid, start, end):
    sums = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(np.where(valid, values, 0.0), axis=0)))
    counts = np.vstack((np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)))
    return sums[end] - sums[start], counts[end] - counts[start]


def _mean_with_minimum(sums, counts, minimum):
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts >= minimum)


def prepare(z, mask, context, context_mask):
    """Build causal design vectors and the fully arrived future target blocks."""
    z = np.asarray(z, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    context = np.asarray(context, dtype=float)
    context_mask = np.asarray(context_mask, dtype=bool)
    if z.ndim != 2 or mask.shape != z.shape:
        raise ValueError('z and mask must have the same two-dimensional shape')
    if context.shape != (z.shape[0], 2) or context_mask.shape != context.shape:
        raise ValueError('context and context_mask must have shape (N, 2)')

    length, channels = z.shape
    q = np.arange(127, length - _HORIZON, 32, dtype=np.int64)
    if not q.size:
        return (
            q,
            np.empty((0, 3 + 2 * channels), dtype=float),
            np.empty((0, _HORIZON, channels), dtype=float),
            np.empty((0, _HORIZON, channels), dtype=bool),
        )

    recent_sum, recent_count = _window_sums(z, mask, q - 15, q + 1)
    previous_sum, previous_count = _window_sums(z, mask, q - 31, q - 15)
    earlier_sum, earlier_count = _window_sums(z, mask, q - 63, q - 31)
    context_sum, context_count = _window_sums(context, context_mask, q - 15, q + 1)
    recent = _mean_with_minimum(recent_sum, recent_count, 8)
    previous = _mean_with_minimum(previous_sum, previous_count, 8)
    earlier = _mean_with_minimum(earlier_sum, earlier_count, 16)
    context_mean = _mean_with_minimum(context_sum, context_count, 8)
    near_change = np.where((recent_count >= 8) & (previous_count >= 8), recent - previous, 0.0)
    prior_change = np.where((previous_count >= 8) & (earlier_count >= 16), previous - earlier, 0.0)
    x = np.concatenate((np.ones((q.size, 1)), np.clip(context_mean, -4, 4),
                        np.clip(near_change, -4, 4), np.clip(prior_change, -4, 4)), axis=1)
    x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))

    future_indices = q[:, None] + np.arange(1, _HORIZON + 1)
    m = mask[future_indices] & (recent_count >= 8)[:, None, :]
    y = np.where(m, z[future_indices] - recent[:, None, :], 0.0)
    return q, x, y, m


def fit(x, y, m, weights):
    """Fit independent masked ridge regressions, with zero as the prior."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.ndim != 3 or y.shape != m.shape or y.shape[:2] != (len(x), _HORIZON):
        raise ValueError('expected x[B,D] and matching y/m[B,25,C]')
    if weights.shape != (len(x),) or np.any(weights < 0):
        raise ValueError('weights must be a nonnegative vector of length B')

    dimensions = x.shape[1]
    channels = y.shape[2]
    weighted_mask = m * weights[:, None, None]
    total_weight = weighted_mask.sum(axis=0)
    gram = np.zeros((_HORIZON, channels, dimensions, dimensions))
    # Bound the temporary weighted-design array when a source stream is long.
    for start in range(0, len(x), 512):
        stop = start + 512
        gram += np.einsum('bi,bj,bhc->hcij', x[start:stop], x[start:stop],
                          weighted_mask[start:stop], optimize=True)
    rhs = np.einsum('bi,bhc->hci', x, weighted_mask * np.where(m, y, 0.0), optimize=True)
    gram += 0.05 * total_weight[:, :, None, None] * np.eye(dimensions)
    solution = np.zeros((_HORIZON, channels, dimensions))
    active = total_weight > 0
    if np.any(active):
        solution[active] = np.linalg.solve(gram[active], rhs[active, :, None])[..., 0]
    return np.moveaxis(solution, -1, 0)


def _huber(residual):
    magnitude = np.abs(residual)
    return np.where(magnitude <= 0.5, 0.5 * residual**2, 0.5 * (magnitude - 0.25))


def replay(length, blocks, benign, disruptive):
    """Replay one shot, releasing each complete block only after its last sample.

    Both adaptive arms update against the benign waveform; the alternative
    waveform shares that arm's same offset.  The gate always uses the two
    unadapted waveforms and the actual observations.  Audit update attempts
    count mature blocks with any observations; parameter changes count those
    attempts that actually altered at least one coefficient.
    """
    q, x, y, m = blocks
    q = np.asarray(q, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=bool)
    benign = np.asarray(benign, dtype=float)
    disruptive = np.asarray(disruptive, dtype=float)
    channels = benign.shape[2]
    features = {arm: np.zeros((length, 3 * channels), dtype=np.float32)
                for arm in ('fixed', 'real', 'reverse')}
    offsets = {arm: np.zeros_like(benign) for arm in ('real', 'reverse')}
    previous = {arm: np.zeros(channels) for arm in features}
    audit = {
        'mature_blocks': 0,
        'update_attempts': {'real': 0, 'reverse': 0},
        'parameter_changes': {'real': 0, 'reverse': 0},
        'benign_huber_sum': {arm: 0.0 for arm in features},
        'benign_huber_count': {arm: 0 for arm in features},
        'hold_huber_sum': 0.0,
        'hold_huber_count': 0,
    }

    for index, origin in enumerate(q):
        if origin + _HORIZON >= length:
            break
        design = x[index]
        observed = m[index]
        target = np.where(observed, y[index], 0.0)
        counts = observed.sum(axis=0)
        total_count = int(counts.sum())
        static_benign = np.einsum('d,dhc->hc', design, benign)
        static_disruptive = np.einsum('d,dhc->hc', design, disruptive)
        static_b_loss = _huber(static_benign - target)
        static_d_loss = _huber(static_disruptive - target)
        gate = (np.clip(0.5 + ((static_d_loss - static_b_loss)[observed].mean()) / 0.1, 0, 1)
                if total_count else 0.0)
        audit['mature_blocks'] += 1
        audit['hold_huber_sum'] += float(_huber(target)[observed].sum())
        audit['hold_huber_count'] += total_count
        release = int(origin + _HORIZON + 1)
        next_release = int(q[index + 1] + _HORIZON + 1) if index + 1 < len(q) else length
        adaptive_benign = {}

        for arm in features:
            adjustment = 0.0 if arm == 'fixed' else np.einsum('d,dhc->hc', design, offsets[arm])
            benign_prediction = static_benign + adjustment
            disruptive_prediction = static_disruptive + adjustment
            b_loss = _huber(benign_prediction - target)
            d_loss = _huber(disruptive_prediction - target)
            margin = np.divide(np.where(observed, b_loss - d_loss, 0.0).sum(axis=0), counts,
                               out=np.zeros(channels), where=counts > 0)
            persistence = np.where(margin * previous[arm] > 0,
                                   np.sign(margin) * np.minimum(np.abs(margin), np.abs(previous[arm])), 0.0)
            features[arm][release:min(length, next_release)] = np.concatenate((margin, persistence, counts / _HORIZON))
            previous[arm] = margin
            audit['benign_huber_sum'][arm] += float(b_loss[observed].sum())
            audit['benign_huber_count'][arm] += total_count
            if arm != 'fixed':
                adaptive_benign[arm] = benign_prediction

        if not total_count:
            continue
        reverse_target = target.copy()
        for channel in range(channels):
            positions = np.flatnonzero(observed[:, channel])
            reverse_target[positions, channel] = target[positions[::-1], channel]
        rate = 0.20 * gate * design / (np.dot(design, design) + 0.25)
        for arm, update_target in (('real', target), ('reverse', reverse_target)):
            audit['update_attempts'][arm] += 1
            gradient = np.clip(adaptive_benign[arm] - update_target, -0.5, 0.5)
            candidate = np.where(observed[None, :, :],
                                 0.98 * offsets[arm] - rate[:, None, None] * gradient[None, :, :],
                                 offsets[arm])
            norms = np.linalg.norm(candidate, axis=0)
            candidate /= np.maximum(1.0, norms / 0.5)[None, :, :]
            if np.any(candidate != offsets[arm]):
                audit['parameter_changes'][arm] += 1
            offsets[arm] = candidate

    return features, audit
