"""Ordered arrived-signal context and a differentiable, bounded Q/K step."""

import math
from numbers import Real

import torch
from torch import nn


def _floating(value, shape, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.shape != shape or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite floating tensor of shape {shape}')


@torch.no_grad()
def ordered_context(h_cls, h_last, residual, available):
    """Return detached (B,358) context and a boolean (B,) feedback gate.

    ``residual`` is the caller's arrived asinh signal minus its origin value;
    this component does not call a forecast head. Layout: h_cls, h_last, then
    five chronological blocks, each containing 23 masked means followed by
    23 valid fractions. A channel with no observations has mean/fraction zero.
    """
    if not isinstance(h_cls, torch.Tensor) or h_cls.ndim != 2:
        raise ValueError('h_cls must have shape (B,64)')
    size = len(h_cls)
    _floating(h_cls, (size, 64), 'h_cls')
    _floating(h_last, (size, 64), 'h_last')
    if (not isinstance(residual, torch.Tensor) or not residual.is_floating_point()
            or residual.shape != (size, 25, 23)
            or not isinstance(available, torch.Tensor) or available.dtype != torch.bool
            or available.shape != residual.shape):
        raise ValueError('floating residual and boolean availability must have shape (B,25,23)')
    if (any(v.device != h_cls.device or v.dtype != h_cls.dtype for v in (h_last, residual))
            or available.device != h_cls.device):
        raise ValueError('context tensors must share floating dtype and device')
    if bool((available & ~torch.isfinite(residual)).any()):
        raise ValueError('available residuals must be finite')
    safe = torch.where(available, residual, 0.).reshape(size, 5, 5, 23)
    counts = available.reshape(size, 5, 5, 23).sum(dim=2).to(residual.dtype)
    # Divide before summing so large finite residuals need not overflow a sum.
    means = (safe / counts.clamp_min(1)[:, :, None, :]).sum(dim=2)
    if not torch.isfinite(means).all():
        raise ValueError('masked residual means must be finite')
    feedback = torch.cat((means, counts / 5), dim=-1).flatten(1)
    context = torch.cat((h_cls, h_last, feedback), dim=1)
    return context.detach(), available.flatten(1).any(dim=1)


class ContextTarget(nn.Module):
    """Map detached context to a bounded 64-vector; unsupported rows are zero."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(358), nn.Linear(358, 32), nn.SiLU(),
                                     nn.Linear(32, 64), nn.Tanh())
        nn.init.zeros_(self.network[3].weight)
        nn.init.zeros_(self.network[3].bias)

    def forward(self, context, support):
        if not isinstance(context, torch.Tensor) or context.ndim != 2:
            raise ValueError('context must have shape (B,358)')
        _floating(context, (len(context), 358), 'context')
        if (not isinstance(support, torch.Tensor) or support.dtype != torch.bool
                or support.shape != (len(context),) or support.device != context.device):
            raise ValueError('support must be a same-device boolean tensor of shape (B,)')
        delta = self.network(context.detach())
        return torch.where(support[:, None], delta, torch.zeros_like(delta))


def representation_step(h_with_graph, reference_h, delta, state, eta=.1, radius=.1,
                        create_graph=False):
    """Return a fresh (B,8192) step; never mutate or accumulate into ``state``.

    Differentiate 0.5 * sum_b sum_64 (h - stopgrad(reference_h) - delta)^2.
    Set create_graph=True when training ContextTarget through an outer loss:
    delta stays connected, including at its exact-zero initialization.
    Projection only clips rows exceeding radius; small gradients keep magnitude.
    The caller's representation forward must be independent across batch rows.
    """
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError('state must have shape (B,8192)')
    size = len(state)
    _floating(state, (size, 8192), 'state')
    for name, value in (('h_with_graph', h_with_graph), ('reference_h', reference_h), ('delta', delta)):
        _floating(value, (size, 64), name)
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError('representations and state must share floating dtype and device')
    if not state.requires_grad or not h_with_graph.requires_grad:
        raise ValueError('state and h_with_graph must retain their autograd graph')
    for name, value in (('eta', eta), ('radius', radius)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a finite nonnegative scalar')
    if not isinstance(create_graph, bool):
        raise ValueError('create_graph must be boolean')
    loss = .5 * (h_with_graph - reference_h.detach() - delta).square().sum()
    if not torch.isfinite(loss):
        raise ValueError('representation loss must be finite')
    gradient, = torch.autograd.grad(loss, state, create_graph=create_graph)
    step = -eta * gradient
    if not torch.isfinite(step).all():
        raise ValueError('representation step must be finite')
    if radius == 0:
        return step * 0
    norm = torch.linalg.vector_norm(step.to(torch.float64), dim=1, keepdim=True)
    # Clamp the denominator, not the gradient: at zero the scale is one and
    # the derivative through -eta*g remains available to the outer objective.
    return step * (radius / norm.clamp_min(radius)).to(step.dtype)
