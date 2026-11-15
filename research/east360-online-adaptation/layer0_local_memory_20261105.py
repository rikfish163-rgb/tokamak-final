"""Bounded layer-0 episode memory; selection and replay remain the caller's job."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy


KEY_WIDTH = 192
ACTION_WIDTH = 128
EPISODE_RADIUS = 0.60
TOTAL_RADIUS = 2.0
STEPS = 20
LR = 0.04
# The frozen operating threshold used by the existing trajectory replay.
THRESHOLD = 0.7258946895599365


def _matrix(value, width, name, *, nonempty=False, finite=True):
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != 2
        or value.shape[1] != width
        or (nonempty and not len(value))
    ):
        raise ValueError(f"{name} must be a floating-point matrix with width {width}")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _same_type(left, right):
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("matching tensor device and dtype required")


def _bandwidth(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite positive bandwidth required") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("finite positive bandwidth required")
    return result


def keys_for(rule, ctx):
    """Native anchor state and history in the rule's clock-free coordinates."""
    state, history = ctx[4], ctx[5]
    _matrix(state, 66, "state", finite=False)
    _matrix(state[:, 2:], 64, "clock-free state")
    _matrix(history, 128, "history")
    _same_type(state, history)
    if len(state) != len(history):
        raise ValueError("one history per native anchor required")
    mean, scale = rule.mean, rule.scale
    if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("tensor normalization required")
    if mean.shape != (194,) or scale.shape != (194,):
        raise ValueError("194-dimensional rule normalization required")
    _same_type(state, mean)
    _same_type(state, scale)
    # The first two entries are deliberately ignored, including their values.
    mean, scale = mean[2:], scale[2:]
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("finite positive clock-free normalization required")
    result = torch.asinh((torch.cat((state[:, 2:], history), dim=1) - mean) / scale)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite memory keys")
    return result


def fit_bandwidth(groups):
    """Pooled 75th percentile of each key's nearest key in a different shot."""
    if len(groups) < 2:
        raise ValueError("at least two shots required")
    for group in groups:
        _matrix(group, KEY_WIDTH, "shot keys", nonempty=True)
        _same_type(groups[0], group)
    nearest = []
    with torch.no_grad():
        for i, group in enumerate(groups):
            source = group.detach().to(dtype=torch.float64)
            best = source.new_full((len(source),), float("inf"))
            for j, other in enumerate(groups):
                if i == j:
                    continue
                distances = torch.cdist(
                    source, other.detach().to(dtype=torch.float64),
                    compute_mode="donot_use_mm_for_euclid_dist",
                ) / math.sqrt(KEY_WIDTH)
                best = torch.minimum(best, distances.min(dim=1).values)
            nearest.append(best)
        distance = torch.cat(nearest)
        if not torch.isfinite(distance).all():
            raise ValueError("nonfinite cross-shot distances")
        result = float(torch.quantile(distance, 0.75).item())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("nearest-other-shot bandwidth must be strictly positive")
    return result


@dataclass
class Episode:
    keys: torch.Tensor
    delta: torch.Tensor

    def __post_init__(self):
        _matrix(self.keys, KEY_WIDTH, "episode keys", nonempty=True)
        if (
            not isinstance(self.delta, torch.Tensor)
            or not self.delta.is_floating_point()
            or self.delta.shape != (ACTION_WIDTH,)
        ):
            raise ValueError("episode delta must be a floating 128-vector")
        _same_type(self.keys, self.delta)
        if not torch.isfinite(self.delta).all():
            raise ValueError("episode delta must be finite")
        if self.delta.detach().norm() > EPISODE_RADIUS:
            raise ValueError("episode delta exceeds its norm bound")


def _memory_actions(keys, episodes, bandwidth, *, fixed_prefix=0):
    _matrix(keys, KEY_WIDTH, "query keys")
    h = _bandwidth(bandwidth)
    if not episodes:
        return keys.new_zeros((len(keys), ACTION_WIDTH))
    key_parts, deltas, counts = [], [], []
    for i, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            raise ValueError("Episode objects required")
        _matrix(episode.keys, KEY_WIDTH, "episode keys", nonempty=True, finite=False)
        if (
            not isinstance(episode.delta, torch.Tensor)
            or not episode.delta.is_floating_point()
            or episode.delta.shape != (ACTION_WIDTH,)
        ):
            raise ValueError("episode delta must be a floating 128-vector")
        _same_type(keys, episode.keys)
        _same_type(keys, episode.delta)
        key_parts.append(episode.keys.detach() if i < fixed_prefix else episode.keys)
        deltas.append(episode.delta.detach() if i < fixed_prefix else episode.delta)
        counts.append(len(episode.keys))
    packed, delta = torch.cat(key_parts), torch.stack(deltas)
    # Revalidate mutable episodes in one device check, not once per episode.
    if not (torch.isfinite(packed).all() & torch.isfinite(delta).all() &
            (delta.detach().norm(dim=1) <= EPISODE_RADIUS).all()):
        raise ValueError("finite bounded episode keys and deltas required")
    group = torch.repeat_interleave(
        torch.arange(len(episodes), device=keys.device),
        torch.as_tensor(counts, device=keys.device),
        output_size=len(packed),
    )
    weight = keys.new_zeros((len(keys), len(episodes)))
    # Chunk prototypes to bound temporary distance storage on long replays.
    chunk = max(1, 2_000_000 // max(1, len(keys) * KEY_WIDTH))
    query64 = keys.to(torch.float64)
    limit = 2 * math.sqrt(KEY_WIDTH)
    for start in range(0, len(packed), chunk):
        stop = min(start + chunk, len(packed))
        relative = ((query64[:, None, :] - packed[start:stop].to(torch.float64)[None, :, :]) / h)
        # Any coordinate this far away is already outside compact support.
        relative = relative.clamp(-limit, limit)
        kernel = (1.0 - relative.square().mean(dim=2) / 4.0).clamp_min(0).square()
        index = group[start:stop].expand(len(keys), -1)
        weight = weight.scatter_reduce(1, index, kernel.to(keys.dtype), reduce="amax", include_self=True)
    result = (weight @ delta) / weight.sum(dim=1, keepdim=True).clamp_min(1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite memory action")
    return result


def memory_actions(keys, episodes, bandwidth):
    """Combine compact-support episode kernels without amplifying sparse memory."""
    return _memory_actions(keys, episodes, bandwidth)


def total_action(fast, slow):
    """Add as much slow action as fits, retaining the fast action exactly."""
    _matrix(fast, ACTION_WIDTH, "fast action")
    _matrix(slow, ACTION_WIDTH, "slow action")
    _same_type(fast, slow)
    if fast.shape != slow.shape:
        raise ValueError("matching action shapes required")
    # Float64 prevents cancellation in the constrained quadratic root.
    f, s = fast.to(torch.float64), slow.to(torch.float64)
    f2 = f.square().sum(dim=1, keepdim=True)
    if (f2 > TOTAL_RADIUS**2).any():
        raise ValueError("fast action exceeds its norm bound")
    a = s.square().sum(dim=1, keepdim=True)
    b = (f * s).sum(dim=1, keepdim=True)
    remaining = (TOTAL_RADIUS**2 - f2).clamp_min(0)
    clipped = (f + s).square().sum(dim=1, keepdim=True) > TOTAL_RADIUS**2
    discriminant = b.square() + a * remaining
    if not torch.isfinite(discriminant).all():
        raise ValueError("nonfinite action composition")
    # At the tangent point the largest admissible lambda is exactly zero.
    root = torch.sqrt(torch.where(discriminant > 0, discriminant, torch.ones_like(discriminant)))
    denominator = root + b
    positive = remaining / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    negative = (-b + root) / torch.where(a > 0, a, torch.ones_like(a))
    limit = torch.where(b >= 0, positive, negative).clamp(0, 1)
    amount = torch.where(clipped, limit, torch.ones_like(limit)).to(dtype=fast.dtype)
    combined = fast + amount * slow
    if not torch.isfinite(combined).all():
        raise ValueError("nonfinite combined action")
    return torch.cat((combined, torch.zeros_like(combined)), dim=1)


def fit_episode(base, ctx, fast, keys, episodes, bandwidth, indices, disruptive):
    """Fit one ended-shot correction on caller-supplied, already eligible windows."""
    if any(module.training for module in base.modules()) or any(p.requires_grad for p in base.parameters()):
        raise ValueError("source must already be frozen and in eval mode")
    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError("binary revealed outcome required")
    if (
        not isinstance(indices, np.ndarray)
        or indices.dtype != np.int64
        or indices.ndim != 1
        or not len(indices)
        or (np.diff(indices) <= 0).any()
    ):
        raise ValueError("unique sorted nonempty int64 window indices required")
    x, available = ctx[0], ctx[1]
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or x.shape[1:] != (128, 23)
        or not isinstance(available, torch.Tensor)
        or available.dtype != torch.bool
        or available.shape != x.shape
        or available.device != x.device
    ):
        raise ValueError("native window views and availability required")
    _matrix(keys, KEY_WIDTH, "anchor keys")
    _matrix(fast, ACTION_WIDTH, "anchor fast actions")
    _same_type(keys, fast)
    _same_type(keys, x)
    if len(keys) != len(fast) or len(keys) != len(ctx[4]):
        raise ValueError("one key and fast action per native anchor required")
    if indices[0] < policy.FIRST or indices[-1] >= len(x):
        raise ValueError("window index outside eligible native anchors")
    q = torch.as_tensor(indices, device=x.device)
    k = (q - policy.FIRST) // policy.STRIDE
    if int(k[-1]) >= len(keys):
        raise ValueError("window has no native anchor")
    # All nonlearned tensors are detached; source and prior memory stay untouched.
    selected_keys = keys.detach()[k]
    prototype = keys.detach()[torch.unique(k, sorted=True)].clone()
    selected_fast = fast.detach()[k]
    selected_x, selected_a = x.detach()[q].contiguous(), available.detach()[q].contiguous()
    previous = list(episodes)
    h = _bandwidth(bandwidth)
    route = latent.routes_all(selected_a)
    delta = torch.nn.Parameter(keys.new_zeros(ACTION_WIDTH))
    optimizer = torch.optim.Adam([delta], lr=LR)
    threshold_logit = math.log(THRESHOLD / (1.0 - THRESHOLD))
    sign = -1.0 if disruptive else 1.0

    def objective():
        slow = _memory_actions(
            selected_keys, [*previous, Episode(prototype, delta)], h,
            fixed_prefix=len(previous),
        )
        persistent = total_action(torch.zeros_like(selected_fast), slow)
        combined = total_action(selected_fast, slow)
        logits = torch.stack((
            policy.routed_pair(base, selected_x, selected_a, route, persistent),
            policy.routed_pair(base, selected_x, selected_a, route, combined),
        ))
        loss = F.softplus(sign * (logits - threshold_logit)).mean() + 0.10 * delta.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite episode objective")
        return loss

    with torch.enable_grad():
        initial_loss = float(objective().detach())
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            (gradient,) = torch.autograd.grad(loss, delta)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite episode gradient")
            delta.grad = gradient
            optimizer.step()
            with torch.no_grad():
                norm = delta.norm()
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite fitted episode")
                if norm > EPISODE_RADIUS:
                    # A tiny inward margin keeps rounding inside the closed ball.
                    margin = 1.0 - 4.0 * torch.finfo(delta.dtype).eps
                    delta.mul_(EPISODE_RADIUS * margin / norm)
        final_loss = float(objective().detach())
    episode = Episode(prototype, delta.detach().clone())
    return episode, dict(
        steps=STEPS,
        initial_loss=initial_loss,
        final_loss=final_loss,
        norm=float(episode.delta.norm()),
        n_keys=len(prototype),
    )
