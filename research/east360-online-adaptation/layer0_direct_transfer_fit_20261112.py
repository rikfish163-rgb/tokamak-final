"""Fit a single global layer-0 action on revealed teaching windows."""

import math

import numpy as np
import torch
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import layer0_local_memory_20261105 as memory
import trajectory_policy_20261105 as policy


def fit_direct(base, history, initial):
    """Return a detached candidate and fit diagnostics.

    Each history item is (case, absolute int64 endpoints, revealed outcome).
    Cases without endpoints contribute nothing. The caller owns eligibility,
    replay and acceptance; case keys are unused by the global action.
    """
    if any(m.training for m in base.modules()) or any(p.requires_grad for p in base.parameters()):
        raise ValueError("source must already be frozen and in eval mode")
    if (
        not isinstance(initial, torch.Tensor)
        or not initial.is_floating_point()
        or initial.shape != (memory.ACTION_WIDTH,)
        or not torch.isfinite(initial).all()
        or initial.detach().norm() > memory.EPISODE_RADIUS
    ):
        raise ValueError("initial must be a finite bounded floating 128-vector")

    windows, masks, actions, signs, counts = [], [], [], [], []
    for (ctx, fast, _keys), indices, disruptive in history:
        if not isinstance(disruptive, (bool, np.bool_)):
            raise ValueError("binary revealed outcome required")
        if (
            not isinstance(indices, np.ndarray)
            or indices.dtype != np.int64
            or indices.ndim != 1
            or (np.diff(indices) <= 0).any()
        ):
            raise ValueError("unique sorted int64 window indices required")
        if not len(indices):
            continue
        x, available = ctx[:2]
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or x.shape[1:] != (128, 23)
            or x.dtype != initial.dtype
            or x.device != initial.device
            or not isinstance(available, torch.Tensor)
            or available.dtype != torch.bool
            or available.shape != x.shape
            or available.device != x.device
        ):
            raise ValueError("native windows and boolean availability matching the initial vector required")
        if indices[0] < policy.FIRST or indices[-1] >= len(x):
            raise ValueError("window index outside eligible native anchors")
        anchors = len(range(policy.FIRST, len(x), policy.STRIDE))
        if (
            not isinstance(fast, torch.Tensor)
            or fast.shape != (anchors, memory.ACTION_WIDTH)
            or fast.dtype != x.dtype
            or fast.device != x.device
        ):
            raise ValueError("one matching fast action per native anchor required")
        q = torch.as_tensor(indices, device=x.device)
        k = (q - policy.FIRST) // policy.STRIDE
        windows.append(x.detach()[q])
        masks.append(available[q])
        actions.append(fast.detach()[k])
        counts.append(len(indices))
        signs.append(x.new_full((len(indices),), -1 if disruptive else 1))

    delta = initial.detach().clone()
    stats = dict(
        steps=0, initial_loss=None, final_loss=None, norm=float(delta.norm()),
        teaching_shots=len(counts), teaching_points=sum(counts),
    )
    if not counts:
        return delta, stats

    x, available, fast, sign = map(torch.cat, (windows, masks, actions, signs))
    route = latent.routes_all(available)
    zero = torch.zeros_like(fast)
    threshold_logit = math.log(memory.THRESHOLD / (1 - memory.THRESHOLD))
    delta.requires_grad_()
    optimizer = torch.optim.Adam([delta], lr=memory.LR)

    def objective():
        slow = delta.expand_as(fast)
        inter = policy.routed_pair(base, x, available, route, memory.total_action(zero, slow))
        both = policy.routed_pair(base, x, available, route, memory.total_action(fast, slow))
        point_loss = F.softplus(sign * (torch.stack((inter, both)) - threshold_logit)).mean(dim=0)
        shot_loss = torch.stack([points.mean() for points in point_loss.split(counts)])
        loss = shot_loss.mean() + 0.10 * delta.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite direct-transfer objective")
        return loss

    with torch.no_grad():
        stats["initial_loss"] = float(objective())
    with torch.enable_grad():
        for _ in range(memory.STEPS):
            (gradient,) = torch.autograd.grad(objective(), delta)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite direct-transfer gradient")
            delta.grad = gradient
            optimizer.step()
            with torch.no_grad():
                norm = delta.norm()
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite direct-transfer action")
                if norm > memory.EPISODE_RADIUS:
                    margin = 1 - 4 * torch.finfo(delta.dtype).eps
                    delta.mul_(memory.EPISODE_RADIUS * margin / norm)
    with torch.no_grad():
        stats.update(steps=memory.STEPS, final_loss=float(objective()), norm=float(delta.norm()))
    return delta.detach(), stats
