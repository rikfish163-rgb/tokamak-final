"""Masked innovation steps and their within-shot delayed displacement."""

import numpy as np

from .commissioning_transition_data_20261110 import _indices
from .commissioning_transition_net_20261110 import HORIZON
from .commissioning_weak_shape_data_20261111 import _signal


RETAIN = 0.85
STEP = 0.225
STABILIZER = 0.02


def observed_mask(z, mask, origins):
    """Mark all five shape coordinates for each fully observed channel.

    Each future bin covers five samples, starting at q+1. A channel is
    observed only if every bin has at least three finite available samples.
    """
    z, mask = _signal(z, mask)
    origins = _indices(origins, 0, len(z) - HORIZON)
    channel_valid = np.ones((len(origins), z.shape[1]), dtype=bool)
    for b in range(5):
        indices = origins[:, None] + 1 + 5*b + np.arange(5)
        channel_valid &= mask[indices].sum(axis=1) >= 3
    return np.broadcast_to(channel_valid[:, None, :],
                           (len(origins), 5, z.shape[1])).reshape(len(origins), 50).copy()


def anchor_steps(pre, key, target, observed, decoder_weight, decoder_bias):
    """Return normalized negative loss gradients at zero fast state."""
    pre, key = np.asarray(pre), np.asarray(key)
    target, observed = np.asarray(target), np.asarray(observed)
    decoder_weight, decoder_bias = np.asarray(decoder_weight), np.asarray(decoder_bias)
    if pre.ndim != 2 or key.ndim != 2 or pre.shape[0] != key.shape[0]:
        raise ValueError('pre and key must have matching batch dimensions')
    if (target.ndim != 2 or target.shape[0] != len(pre)
            or observed.shape != target.shape or observed.dtype != np.bool_):
        raise ValueError('target and boolean observed mask must have matching batch shapes')
    if (decoder_weight.shape != (target.shape[1], pre.shape[1])
            or decoder_bias.shape != (target.shape[1],)):
        raise ValueError('decoder shapes must match the hidden and target dimensions')

    hidden = np.tanh(pre)
    residual = np.where(observed, hidden @ decoder_weight.T + decoder_bias - target, 0)
    direction = (residual @ decoder_weight) * (1 - hidden**2)
    gradient = direction[:, :, None] * key[:, None, :]
    norm = np.linalg.norm(gradient, axis=(1, 2), keepdims=True)
    return (-STEP * gradient / (norm + STABILIZER)).astype(np.float32)


def released_displacement(key, origins, steps, eligible):
    """Apply only matured updates, with a fresh fast state for each call.

    An update at q first affects q+HORIZON+1. Ineligible updates do not
    decay the state, and updates outside the record are not counted.
    """
    key, origins = np.asarray(key), np.asarray(origins)
    steps, eligible = np.asarray(steps), np.asarray(eligible)
    if key.ndim != 2 or origins.ndim != 1:
        raise ValueError('key must be two-dimensional and origins one-dimensional')
    if origins.size and not np.issubdtype(origins.dtype, np.integer):
        raise TypeError('origins must be integers')
    if np.any(origins < 0):
        raise ValueError('origins must be nonnegative')
    if (steps.ndim != 3 or steps.shape[0] != len(origins)
            or steps.shape[2] != key.shape[1]):
        raise ValueError('steps must have shape (origins, hidden, key width)')
    if eligible.shape != origins.shape or (eligible.size and eligible.dtype != np.bool_):
        raise ValueError('eligible must be a boolean value for each origin')

    delta = np.zeros((len(key), steps.shape[1]), dtype=np.float32)
    if not len(origins):
        return delta, 0
    visible = np.flatnonzero(eligible & (origins < len(key) - HORIZON - 1))
    visible = visible[np.argsort(origins[visible], kind='stable')]
    state = np.zeros(steps.shape[1:], dtype=np.result_type(steps.dtype, np.float32))
    for i, j in enumerate(visible):
        state = RETAIN * state + steps[j]
        start = int(origins[j]) + HORIZON + 1
        stop = (int(origins[visible[i + 1]]) + HORIZON + 1
                if i + 1 < len(visible) else len(key))
        delta[start:stop] = key[start:stop] @ state.T
    return delta, int(len(visible))
