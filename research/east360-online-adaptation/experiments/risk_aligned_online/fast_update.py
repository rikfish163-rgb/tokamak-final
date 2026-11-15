"""Shared differentiable, bounded temporary update for training and replay."""
import torch


def normalized_step(current, initial, gradient, *, step=.05, radius=1.):
    if current.shape!=initial.shape or current.shape!=gradient.shape or current.ndim!=2:
        raise ValueError('matching batch by parameter tensors required')
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('nonfinite temporary gradient')
    # Smooth denominator gives a finite second derivative even at zero.
    # This epsilon is in gradient coordinates, not a diagnostic unit scale.
    norm=(gradient.square().sum(-1,keepdim=True)+1e-12).sqrt()
    proposed=current-step*gradient/norm
    offset=proposed-initial
    length=(offset.square().sum(-1,keepdim=True)+1e-12).sqrt()
    return initial+offset*(radius/length).clamp(max=1.)
