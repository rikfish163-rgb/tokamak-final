"""Tensor primitives for causal, bounded direct diagnostic fast memory.

Inputs named ``level`` and ``future`` are already in source-normalized asinh
coordinates. This module neither reads signals nor converts physical units.
The caller owns shot-local state, target arrival, and predict-before-reveal.
"""

import math
from numbers import Real

import torch


_CHANNELS = tuple(index for index in range(23) if index not in (2, 10))


def _floating(value, shape, name, device=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.shape != shape):
        raise ValueError(f'{name} must be a floating tensor of shape {shape}')
    if device is not None and value.device != device:
        raise ValueError(f'{name} must be on the same device as the other inputs')


def _mask(value, shape, name, device):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
            or value.shape != shape):
        raise ValueError(f'{name} must be a boolean tensor of shape {shape}')
    if value.device != device:
        raise ValueError(f'{name} must be on the same device as the other inputs')


def _finite(value, name):
    if value.device.type == 'meta' or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def _window(level, available):
    if not isinstance(level, torch.Tensor) or level.ndim != 3:
        raise ValueError('level must have shape (B,128,23)')
    shape = (level.shape[0], 128, 23)
    _floating(level, shape, 'level')
    _mask(available, shape, 'available', level.device)
    if level.device.type == 'meta':
        raise ValueError('level must contain materialized values')
    if bool((available & ~torch.isfinite(level)).any()):
        raise ValueError('observed level values must be finite')
    return torch.where(available, level, 0.)


def _compatible(value, reference, name):
    if value.dtype != reference.dtype:
        raise ValueError(f'{name} must share the floating dtype')


def _origin(origin, weight):
    if origin is None:
        return torch.zeros_like(weight)
    if not isinstance(origin, torch.Tensor) or origin.shape not in (
            weight.shape, weight.shape[1:]):
        raise ValueError('origin must have shape (B,168,70) or (168,70)')
    _floating(origin, origin.shape, 'origin', weight.device)
    _compatible(origin, weight, 'origin')
    _finite(origin, 'origin')
    return origin


def dynamic_key(level, available):
    """Return (B,70): masked last, 8-vs-8 mean change, fractions, bias.

    A mean change is present only when both eight-step halves have observed
    values. The first 69 components are divided by sqrt(69); the bias is 1.
    An unavailable final sample is zero, not an older observed value.
    """
    clean = _window(level, available)
    counts = available[:, -16:].reshape(-1, 2, 8, 23).sum(dim=2)
    blocks = clean[:, -16:].reshape(-1, 2, 8, 23)
    # Divide first so summing finite inputs is less likely to overflow.
    means = (blocks / counts.clamp_min(1).unsqueeze(2)).sum(dim=2)
    delta = torch.where((counts > 0).all(dim=1), means[:, 1] - means[:, 0], 0.)
    fraction = counts.sum(dim=1).to(level.dtype) / 16
    features = torch.cat((clean[:, -1], delta, fraction), dim=1) / math.sqrt(69)
    result = torch.cat((features, level.new_ones((level.shape[0], 1))), dim=1)
    _finite(result, 'dynamic key')
    return result


def direct_target(level, available, future, future_available):
    """Return time-major (B,168) future-minus-last targets and validity.

    Both the actual final history sample and the corresponding future sample
    must be observed. RAW23 indices 2 and 10 are excluded; invalid targets
    are exactly zero. The caller may supply future samples only after arrival.
    """
    clean = _window(level, available)
    shape = (level.shape[0], 8, 23)
    _floating(future, shape, 'future', level.device)
    _mask(future_available, shape, 'future_available', level.device)
    _compatible(future, level, 'future')
    if bool((future_available & ~torch.isfinite(future)).any()):
        raise ValueError('observed future values must be finite')
    clean_future = torch.where(future_available, future, 0.)
    valid = future_available[:, :, _CHANNELS] & available[:, -1:, _CHANNELS]
    delta = clean_future[:, :, _CHANNELS] - clean[:, -1:, _CHANNELS]
    target = torch.where(valid, delta, 0.).reshape(level.shape[0], 168)
    _finite(target, 'direct target')
    return target, valid.reshape(level.shape[0], 168)


def _state(weight, key):
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        raise ValueError('weight must have shape (B,168,70)')
    batch = weight.shape[0]
    _floating(weight, (batch, 168, 70), 'weight')
    _floating(key, (batch, 70), 'key', weight.device)
    _compatible(key, weight, 'key')
    _finite(weight, 'weight')
    _finite(key, 'key')
    return batch


def _positive_scalar(value, name, *, upper=None):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0
            or (upper is not None and value > upper)):
        raise ValueError(f'{name} must be a finite positive scalar'
                         + (f' at most {upper}' if upper is not None else ''))


def fast_step(weight, key, target, mask, *, lr=.5, radius=2., origin=None):
    """Take one functional normalized LMS step and return (weight, row_loss).

    row_loss is the pre-update mean of half-squared observed errors (zero for
    an unsupported row). The step instead uses each output's own gradient,
    without division by the number of outputs or observations. Active rows
    are projected into the Frobenius ball centered on ``origin``; an inactive
    row is exactly unchanged even if the caller supplied an out-of-ball state.
    The default origin is zero. A shared (168,70) or batched origin is accepted.
    """
    batch = _state(weight, key)
    _floating(target, (batch, 168), 'target', weight.device)
    _mask(mask, (batch, 168), 'mask', weight.device)
    _compatible(target, weight, 'target')
    _finite(target, 'target')
    _positive_scalar(lr, 'lr', upper=1)
    _positive_scalar(radius, 'radius')
    origin = _origin(origin, weight)

    prediction = torch.bmm(weight, key.unsqueeze(-1)).squeeze(-1)
    _finite(prediction, 'prediction')
    error = torch.where(mask, prediction - target, 0.)
    _finite(error, 'observed error')
    loss = .5 * error.square().sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    _finite(loss, 'row loss')
    denominator = 1 + key.square().sum(dim=1, keepdim=True)
    _finite(denominator, 'LMS denominator')
    proposed = weight - (lr * error.unsqueeze(-1) * key.unsqueeze(1)
                         / denominator.unsqueeze(-1))
    _finite(proposed, 'unprojected state')
    difference = proposed - origin
    _finite(difference, 'origin displacement')
    # Higher-precision norms prevent ordinary float32 sum-of-squares overflow.
    norm = torch.linalg.vector_norm(difference.to(torch.float64).flatten(1),
                                    dim=1).reshape(batch, 1, 1)
    _finite(norm, 'displacement norm')
    scale = (radius / norm.clamp_min(radius)).to(weight.dtype)
    projected = origin + difference * scale
    bounded = torch.where(norm > radius, projected, proposed)
    result = torch.where(mask.any(dim=1).reshape(batch, 1, 1), bounded, weight)
    _finite(result, 'updated state')
    return result, loss


def innovation(weight, origin, key):
    """Return (W - W0) @ key, using only the current key and saved weights."""
    _state(weight, key)
    if origin is None:
        raise ValueError('innovation requires an explicit origin')
    origin = _origin(origin, weight)
    result = torch.bmm(weight - origin, key.unsqueeze(-1)).squeeze(-1)
    _finite(result, 'innovation')
    return result
