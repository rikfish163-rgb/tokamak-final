"""Matured-target decisions between already-issued diagnostic forecasts.

This module does not update state, issue forecasts, or decide when targets
arrive. The caller owns snapshot provenance and the predict-before-reveal
boundary; no outcome labels are used here.
"""

import torch


def choose_commit(candidate_prediction, active_prediction, target, mask, *, policy='guard'):
    """Return acceptance, candidate/active float64 MSE, and integer support.

    All floating inputs have shape (B,168), the same dtype, and the same
    device as the matching boolean mask. Only observed positions must be
    finite. Guard accepts a supported row only for strictly lower candidate
    MSE; delay accepts every supported row. Unsupported rows return zero
    errors for logging and must not be treated as measured performance.
    """
    if not isinstance(policy, str) or policy not in ('guard', 'delay'):
        raise ValueError('policy must be guard or delay')
    if (not isinstance(candidate_prediction, torch.Tensor)
            or candidate_prediction.ndim != 2 or candidate_prediction.shape[1] != 168
            or not candidate_prediction.is_floating_point()):
        raise ValueError('candidate_prediction must be floating with shape (B,168)')
    shape = candidate_prediction.shape
    for name, value in (('active_prediction', active_prediction), ('target', target)):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.shape != shape):
            raise ValueError(f'{name} must be floating with shape (B,168)')
        if value.device != candidate_prediction.device:
            raise ValueError(f'{name} must be on the same device as the candidate')
        if value.dtype != candidate_prediction.dtype:
            raise ValueError(f'{name} must have the same floating dtype as the candidate')
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != shape:
        raise ValueError('mask must be boolean with shape (B,168)')
    if mask.device != candidate_prediction.device:
        raise ValueError('mask must be on the same device as the predictions')
    if candidate_prediction.device.type == 'meta':
        raise ValueError('inputs must contain materialized values')

    for name, value in (('candidate_prediction', candidate_prediction),
                        ('active_prediction', active_prediction), ('target', target)):
        if bool((mask & ~torch.isfinite(value)).any()):
            raise ValueError(f'observed {name} must be finite')
    candidate = torch.where(mask, candidate_prediction, 0.).to(torch.float64)
    active = torch.where(mask, active_prediction, 0.).to(torch.float64)
    actual = torch.where(mask, target, 0.).to(torch.float64)
    support = mask.sum(dim=1)
    divisor = support.clamp_min(1).to(torch.float64).unsqueeze(1)
    # Scale before squaring so a representable mean does not overflow merely
    # because one unnormalized squared error exceeds the float64 range.
    scale = divisor.sqrt()
    candidate_mse = ((candidate - actual) / scale).square().sum(dim=1)
    active_mse = ((active - actual) / scale).square().sum(dim=1)
    if not bool(torch.isfinite(candidate_mse).all() and torch.isfinite(active_mse).all()):
        raise ValueError('observed mean squared errors must be representable as float64')
    accepted = support > 0
    if policy == 'guard':
        accepted = accepted & (candidate_mse < active_mse)
    return accepted, candidate_mse, active_mse, support
