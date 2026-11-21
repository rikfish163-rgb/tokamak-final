"""Robust, damped measurement assimilation in named internal coordinates.

The two trajectory coefficients have different curvatures in the signal
loss. A 2x2 Gauss-Newton solve per diagnostic accounts for that curvature,
instead of shrinking all dimensions by the average 150-element loss. Huber
weights limit spike influence; ridge .01 makes partially observed systems
well-posed. This uses only arrived signals, never a risk/outcome label.
"""
import torch

from .diagnostic_adapter import bounded_step


def assimilate(current, prediction, target, mask, basis):
    if prediction.shape != target.shape or mask.shape != target.shape or prediction.shape[-1] != 6:
        raise ValueError('six matching forecast trajectories and masks required')
    if current.shape != (len(target), 12) or basis.shape != (target.shape[1], 2):
        raise ValueError('12 named coordinates and two-column trajectory basis required')
    residual = (target.detach()-prediction).masked_fill(~mask, 0.)
    weights = mask.to(target.dtype)/residual.detach().abs().clamp_min(1.)
    curvature = torch.einsum('btc,tk,tl->bckl', weights, basis, basis)
    curvature = curvature+.01*torch.eye(2, device=target.device, dtype=target.dtype)
    rhs = torch.einsum('btc,btc,tk->bck', weights, residual, basis)
    correction = torch.linalg.solve(curvature, rhs.unsqueeze(-1)).squeeze(-1).flatten(1)
    # bounded_step subtracts a gradient; the solved correction is an additive
    # observation innovation. Bounds are identical to the gradient control.
    return bounded_step(current, -correction)
