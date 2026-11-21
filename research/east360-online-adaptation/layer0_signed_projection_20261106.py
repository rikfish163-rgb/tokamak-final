"""Finite-checked signed projection of unsupported layer-0 actions.

This module neither decides which actions lack support nor accesses outcomes.
The caller supplies actions, and every retained correction is checked against
the zero-action logit at the same site.
"""

from dataclasses import dataclass

import torch
from torch import nn

import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy


EPS = 1e-5
ACTION_WIDTH = 128
BATCH = 128


@dataclass(frozen=True)
class SignedProjection:
    values: torch.Tensor
    native: torch.Tensor
    initial: torch.Tensor
    final: torch.Tensor
    restricted: torch.Tensor
    at_zero: torch.Tensor


def _actions(value):
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != 2
        or value.shape[1] != ACTION_WIDTH
    ):
        raise ValueError("actions must be a floating-point N x 128 tensor")
    if not torch.isfinite(value).all():
        raise ValueError("actions must be finite")


def _checked_tensor(value, shape, like, name, *, finite):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != like.dtype
        or value.device != like.device
    ):
        raise ValueError(f"{name} must have the expected shape, dtype, and device")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach()


@torch.no_grad()
def project_unsupported(actions, evaluate, gradient):
    """Remove unsupported positive effects using at most two checked steps.

    ``evaluate(indices, values)`` returns one actual logit per row, and
    ``gradient(indices, values)`` returns its derivative at those values. Row
    indices always refer to the original action tensor. Outputs are detached;
    neither the input nor any caller-owned gradient is modified.
    """
    _actions(actions)
    source = actions.detach().clone()
    count = len(source)
    if not count:
        empty = source.new_empty((0,))
        flags = torch.empty(0, dtype=torch.bool, device=source.device)
        return SignedProjection(source, empty, empty.clone(), empty.clone(), flags, flags.clone())

    indices = torch.arange(count, device=source.device)
    native = _checked_tensor(
        evaluate(indices, torch.zeros_like(source)), (count,), source,
        "native logits", finite=True,
    ).clone()
    initial = _checked_tensor(
        evaluate(indices, source.clone()), (count,), source,
        "initial logits", finite=True,
    ).clone()
    # Comparing the represented logits in float64 avoids rounding native+EPS
    # back to native in lower-precision input dtypes.
    restricted = (initial.to(torch.float64) - native.to(torch.float64)) > EPS
    values = source.clone()
    final = initial.clone()
    values[restricted] = 0
    final[restricted] = native[restricted]

    pending = indices[restricted]
    current = source[restricted]
    current_logits = initial[restricted]
    original_norm = source.to(torch.float64).norm(dim=1)

    for _ in range(2):
        if not len(pending):
            break
        with torch.enable_grad():
            grad = _checked_tensor(
                gradient(pending, current.clone()), current.shape, source,
                "gradient", finite=False,
            )

        g64 = grad.to(torch.float64)
        g2 = g64.square().sum(dim=1)
        effect = current_logits.to(torch.float64) - native[pending].to(torch.float64)
        radii = original_norm[pending]
        valid = (
            torch.isfinite(g64).all(dim=1)
            & torch.isfinite(g2) & (g2 > 0)
            & torch.isfinite(effect)
            & torch.isfinite(radii)
        )
        pending, current, g64, g2, effect, radii = (
            tensor[valid] for tensor in (pending, current, g64, g2, effect, radii)
        )
        if not len(pending):
            break

        # The unbounded step satisfies g.T @ (candidate-current) = -effect.
        proposal64 = current.to(torch.float64) - (effect / g2)[:, None] * g64
        norm64 = proposal64.norm(dim=1)
        valid = torch.isfinite(proposal64).all(dim=1) & torch.isfinite(norm64)
        pending, proposal64, norm64, radii = (
            tensor[valid] for tensor in (pending, proposal64, norm64, radii)
        )
        if not len(pending):
            break
        scale = torch.ones_like(norm64)
        over = norm64 > radii
        scale[over] = radii[over] / norm64[over]
        proposal = (proposal64 * scale[:, None]).to(source.dtype)

        # Casting a boundary point can round it outside the original ball.
        # Move only those rows slightly inward, then verify the represented norm.
        norm = proposal.to(torch.float64).norm(dim=1)
        over = torch.isfinite(norm) & (norm > radii)
        if over.any():
            margin = 1 - 4 * torch.finfo(source.dtype).eps
            scale = (radii[over] / norm[over] * margin).to(source.dtype)
            proposal[over] = proposal[over] * scale[:, None]
            norm = proposal.to(torch.float64).norm(dim=1)
        valid = torch.isfinite(proposal).all(dim=1) & torch.isfinite(norm) & (norm <= radii)
        pending, proposal = pending[valid], proposal[valid]
        if not len(pending):
            break

        observed = _checked_tensor(
            evaluate(pending, proposal.clone()), (len(pending),), source,
            "proposal logits", finite=False,
        )
        finite = torch.isfinite(observed)
        safe = finite & ((observed.to(torch.float64) - native[pending].to(torch.float64)) <= EPS)
        values[pending[safe]] = proposal[safe]
        final[pending[safe]] = observed[safe]

        # Only a finite, actually evaluated violation can seed the next step.
        remaining = finite & ~safe
        pending, current, current_logits = (
            pending[remaining], proposal[remaining], observed[remaining]
        )

    at_zero = restricted & (values == 0).all(dim=1)
    return SignedProjection(values, native, initial, final, restricted, at_zero)


def project_layer0(base, ctx, actions):
    """Project anchor actions through the frozen source's real layer-0 route.

    Row k uses only the native window at FIRST + STRIDE*k. The supplied rows
    may be a prefix of the available anchors. All source calls are bounded to
    128 rows, and differentiation is limited to a temporary 128-D action.
    """
    _actions(actions)
    if not isinstance(base, nn.Module):
        raise ValueError("a frozen evaluation-mode source module is required")
    if any(module.training for module in base.modules()) or any(
        parameter.requires_grad for parameter in base.parameters()
    ):
        raise ValueError("source must already be frozen and in eval mode")
    if not isinstance(ctx, (tuple, list)) or len(ctx) < 2:
        raise ValueError("native context must contain windows and availability")
    windows, available = ctx[:2]
    if (
        not isinstance(windows, torch.Tensor)
        or not windows.is_floating_point()
        or windows.ndim != 3
        or windows.shape[1:] != (128, 23)
        or not isinstance(available, torch.Tensor)
        or available.shape != windows.shape
        or available.dtype != torch.bool
        or windows.dtype != actions.dtype
        or windows.device != actions.device
        or available.device != actions.device
    ):
        raise ValueError("matching native windows, availability, action dtype and device required")
    if len(actions) and policy.FIRST + policy.STRIDE * (len(actions) - 1) >= len(windows):
        raise ValueError("one native anchor window per action is required")
    for tensor in list(base.parameters()) + list(base.buffers()):
        if tensor.device != actions.device or (tensor.is_floating_point() and tensor.dtype != actions.dtype):
            raise ValueError("source and actions must have matching dtype and device")

    windows, available = windows.detach(), available.detach()

    def forward(indices, value):
        anchors = policy.FIRST + policy.STRIDE * indices
        x = windows[anchors].contiguous()
        a = available[anchors].contiguous()
        if not torch.isfinite(x).all():
            raise ValueError("selected native windows must be finite")
        routes = latent.routes_all(a)
        delta = torch.cat((value, torch.zeros_like(value)), dim=1)
        return policy.routed_pair(base, x, a, routes, delta)

    def evaluate(indices, value):
        return torch.cat([
            forward(indices[lo:lo + BATCH], value[lo:lo + BATCH])
            for lo in range(0, len(indices), BATCH)
        ])

    def gradient(indices, value):
        parts = []
        with torch.inference_mode(False), torch.enable_grad():
            for lo in range(0, len(indices), BATCH):
                temporary = value[lo:lo + BATCH].detach().clone().requires_grad_(True)
                z = forward(indices[lo:lo + BATCH], temporary)
                z = _checked_tensor_for_gradient(z, len(temporary), temporary)
                (g,) = torch.autograd.grad(z.sum(), temporary)
                g = g.detach()
                g[~torch.isfinite(z.detach())] = float("nan")
                parts.append(g)
        return torch.cat(parts)

    return project_unsupported(actions, evaluate, gradient)


def _checked_tensor_for_gradient(value, count, like):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (count,)
        or value.dtype != like.dtype
        or value.device != like.device
    ):
        raise ValueError("gradient logits must have the expected shape, dtype, and device")
    return value
