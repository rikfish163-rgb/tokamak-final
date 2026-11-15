"""Projected, no-forgetting recursive ridge for direct diagnostic memory.

The caller owns the prior, shot reset, observation timing, and outcome boundary.
The covariance is inverse feature precision, not a calibrated uncertainty.
"""

from numbers import Integral

import torch

from .memory import _compatible, _finite, _floating, _mask, _origin, _positive_scalar


def initial_covariance(batch, reference):
    """Create independent identity priors, one per batch row and output."""
    if isinstance(batch, bool) or not isinstance(batch, Integral) or batch < 0:
        raise ValueError('batch must be a nonnegative integer')
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise ValueError('reference must be a finite floating tensor')
    _finite(reference, 'reference')
    eye = torch.eye(70, dtype=reference.dtype, device=reference.device)
    return eye.reshape(1, 1, 70, 70).expand(int(batch), 168, 70, 70).clone()


def ridge_step(weight, covariance, key, target, mask, *, origin, radius=2.):
    """Return new weight, inverse precision, and pre-update row loss.

    Each output has its own inverse precision and is updated only when its
    target is observed. Origin is mandatory: it may be shared (168,70) or
    batched (B,168,70). Active rows are projected into its Frobenius ball;
    inactive rows and all unobserved output covariances stay exactly intact.
    The row loss is the mean half-squared observed error, or zero if absent.

    With an identity prior, no forgetting, and no active projection, this is
    classical recursive ridge. Once projection acts, covariance still tracks
    inverse feature precision, but the weights need not be the ridge solution.
    """
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        raise ValueError('weight must have shape (B,168,70)')
    batch = weight.shape[0]
    _floating(weight, (batch, 168, 70), 'weight')
    _floating(covariance, (batch, 168, 70, 70), 'covariance', weight.device)
    _floating(key, (batch, 70), 'key', weight.device)
    _floating(target, (batch, 168), 'target', weight.device)
    _mask(mask, (batch, 168), 'mask', weight.device)
    for name, value in (('covariance', covariance), ('key', key), ('target', target)):
        _compatible(value, weight, name)
    if origin is None:
        raise ValueError('origin is required')
    origin = _origin(origin, weight)
    for name, value in (('weight', weight), ('covariance', covariance),
                        ('key', key), ('target', target)):
        _finite(value, name)
    _positive_scalar(radius, 'radius')

    pk = torch.matmul(covariance, key[:, None, :, None]).squeeze(-1)
    _finite(pk, 'covariance-key product')
    denominator = 1 + (pk * key[:, None, :]).sum(dim=-1)
    _finite(denominator, 'RLS denominator')
    if bool((denominator <= 0).any()):
        raise ValueError('RLS denominator must be positive')
    gain = torch.where(mask[:, :, None], pk / denominator[:, :, None], 0.)
    prediction = torch.bmm(weight, key.unsqueeze(-1)).squeeze(-1)
    _finite(prediction, 'prediction')
    error = torch.where(mask, target - prediction, 0.)
    _finite(error, 'observed error')
    loss = .5 * error.square().sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    _finite(loss, 'row loss')

    proposed = weight + error[:, :, None] * gain
    _finite(proposed, 'unprojected weight')
    proposed_covariance = covariance - gain.unsqueeze(-1) * pk.unsqueeze(-2)
    # Scaling first avoids overflowing the sum of two otherwise finite entries.
    symmetric = .5 * proposed_covariance + .5 * proposed_covariance.transpose(-1, -2)
    updated_covariance = torch.where(mask[:, :, None, None], symmetric, covariance)
    _finite(updated_covariance, 'updated covariance')

    displacement = proposed - origin
    _finite(displacement, 'origin displacement')
    norm = torch.linalg.vector_norm(displacement.to(torch.float64).flatten(1),
                                    dim=1).reshape(batch, 1, 1)
    _finite(norm, 'displacement norm')
    scale = (radius / norm.clamp_min(radius)).to(weight.dtype)
    projected = origin + displacement * scale
    bounded = torch.where(norm > radius, projected, proposed)
    updated_weight = torch.where(mask.any(dim=1).reshape(batch, 1, 1), bounded, weight)
    _finite(updated_weight, 'updated weight')
    return updated_weight, updated_covariance, loss
