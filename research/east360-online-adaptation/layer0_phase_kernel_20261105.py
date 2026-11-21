"""Phase-competing layer-0 memory for already revealed teaching episodes.

Only the caller may select historical early anchors.  Online action lookup has
no clock or outcome input, and never treats the early reference as a safe label.
"""

from dataclasses import dataclass
import math
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
import trajectory_policy_20261105 as policy


# The caller may temporarily replace memory.fit_episode while invoking us.
_ORIGINAL_FIT_EPISODE = memory.fit_episode


@dataclass
class PhaseEpisode(memory.Episode):
    """A native episode plus a nonempty, detached same-shot early key bank."""

    early_keys: torch.Tensor

    def __post_init__(self):
        super().__post_init__()
        memory._matrix(self.early_keys, memory.KEY_WIDTH, "early keys", nonempty=True)
        memory._same_type(self.keys, self.early_keys)
        self.early_keys = self.early_keys.detach()


def early_anchor_indices(times, down, n_pos, mode):
    """Return eligible historical sample indices, never native-anchor ordinals.

    For an already revealed disruptive shot, anchors must be at least 500 ms
    before its disruption time.  Matched mode returns exactly ``n_pos`` evenly
    spaced anchors, or an empty int64 array when that many are unavailable.
    All mode returns every eligible anchor; an empty result is unavailable.
    """
    if mode not in ("all", "matched"):
        raise ValueError("early reference mode must be 'all' or 'matched'")
    if isinstance(n_pos, (bool, np.bool_)) or not isinstance(n_pos, (int, np.integer)) or n_pos < 0:
        raise ValueError("nonnegative integer positive-key count required")
    try:
        time = np.asarray(times, dtype=np.float64)
        end = float(down)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite one-dimensional times and disruption time required") from exc
    if time.ndim != 1 or not np.isfinite(time).all() or not math.isfinite(end):
        raise ValueError("finite one-dimensional times and disruption time required")
    anchors = np.arange(policy.FIRST, len(time), policy.STRIDE, dtype=np.int64)
    eligible = anchors[np.round((end - time[anchors]) * 1000, 9) >= 500]
    if mode == "all":
        return eligible
    if len(eligible) < n_pos:
        return np.empty(0, dtype=np.int64)
    return eligible[np.linspace(0, len(eligible) - 1, n_pos, dtype=np.int64)]


def _nearest_and_kernel(query, packed, groups, count, bandwidth, *, with_kernel):
    """Nearest half-RMS distances and, for positives, native kernel maxima."""
    nearest = query.new_full((len(query), count), float("inf"), dtype=torch.float64)
    weight = query.new_zeros((len(query), count)) if with_kernel else None
    chunk = max(1, 2_000_000 // max(1, len(query) * memory.KEY_WIDTH))
    query64 = query.to(torch.float64)
    limit = 2 * math.sqrt(memory.KEY_WIDTH)
    for start in range(0, len(packed), chunk):
        stop = min(start + chunk, len(packed))
        prototypes = packed[start:stop].to(torch.float64)
        index = groups[start:stop].expand(len(query), -1)
        # Halving first avoids overflow for opposite-sign finite float64 keys.
        # Pairwise scaling avoids squaring away very small nonzero distances.
        # The common factor 1/2 is immaterial to the strict comparison.
        with torch.no_grad():
            difference = query64.detach()[:, None, :] * 0.5 - prototypes.detach()[None, :, :] * 0.5
            scale = difference.abs().amax(dim=2)
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            distance = scale * (difference / safe_scale[:, :, None]).square().mean(dim=2).sqrt()
            nearest = nearest.scatter_reduce(1, index, distance, reduce="amin", include_self=True)
        if with_kernel:
            # Identical arithmetic and dtype conversion to the native memory:
            # max over K(d) is the kernel at the nearest positive key.
            relative = (query64[:, None, :] - prototypes[None, :, :]) / bandwidth
            relative = relative.clamp(-limit, limit)
            kernel = (1.0 - relative.square().mean(dim=2) / 4.0).clamp_min(0).square()
            weight = weight.scatter_reduce(
                1, index, kernel.to(query.dtype), reduce="amax", include_self=True
            )
    return nearest, weight


def phase_actions(keys, episodes, bandwidth, *, fixed_prefix=0):
    """Return compact-kernel actions gated by strictly nearer positive keys.

    ``fixed_prefix`` detaches prior episode keys and deltas, matching the native
    optimizer's frozen-memory contract; early keys are always detached.
    """
    memory._matrix(keys, memory.KEY_WIDTH, "query keys")
    h = memory._bandwidth(bandwidth)
    if (isinstance(fixed_prefix, (bool, np.bool_))
            or not isinstance(fixed_prefix, (int, np.integer))
            or not 0 <= fixed_prefix <= len(episodes)):
        raise ValueError("fixed prefix must identify existing episodes")
    if not episodes:
        return keys.new_zeros((len(keys), memory.ACTION_WIDTH))

    positives, early, deltas, n_pos, n_early = [], [], [], [], []
    for i, episode in enumerate(episodes):
        if not isinstance(episode, PhaseEpisode):
            raise ValueError("PhaseEpisode objects required")
        memory._matrix(episode.keys, memory.KEY_WIDTH, "episode keys", nonempty=True)
        memory._matrix(episode.early_keys, memory.KEY_WIDTH, "early keys", nonempty=True)
        if (not isinstance(episode.delta, torch.Tensor)
                or not episode.delta.is_floating_point()
                or episode.delta.shape != (memory.ACTION_WIDTH,)):
            raise ValueError("episode delta must be a floating 128-vector")
        memory._same_type(keys, episode.keys)
        memory._same_type(keys, episode.early_keys)
        memory._same_type(keys, episode.delta)
        if (not torch.isfinite(episode.delta).all()
                or episode.delta.detach().norm() > memory.EPISODE_RADIUS):
            raise ValueError("finite bounded episode delta required")
        positives.append(episode.keys.detach() if i < fixed_prefix else episode.keys)
        early.append(episode.early_keys.detach())
        deltas.append(episode.delta.detach() if i < fixed_prefix else episode.delta)
        n_pos.append(len(episode.keys))
        n_early.append(len(episode.early_keys))
    if not len(keys):
        return keys.new_zeros((0, memory.ACTION_WIDTH))

    def pack(parts, lengths):
        values = torch.cat(parts)
        groups = torch.repeat_interleave(
            torch.arange(len(parts), device=keys.device),
            torch.as_tensor(lengths, device=keys.device),
            output_size=len(values),
        )
        return values, groups

    pos, pos_group = pack(positives, n_pos)
    reference, early_group = pack(early, n_early)
    dp, weights = _nearest_and_kernel(keys, pos, pos_group, len(episodes), h, with_kernel=True)
    de, _ = _nearest_and_kernel(keys, reference, early_group, len(episodes), h, with_kernel=False)
    weights = weights * (dp < de).to(dtype=keys.dtype)
    result = (weights @ torch.stack(deltas)) / weights.sum(dim=1, keepdim=True).clamp_min(1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite phase memory action")
    return result


def fit_phase(base, ctx, fast, keys, episodes, bandwidth, indices, disruptive, early_keys):
    """Fit via the original 20-step optimizer with a scoped private gate patch.

    The private hook is process-global, like the underlying memory function;
    callers must not run concurrent fits or memory lookups during this scope.
    """
    memory._matrix(keys, memory.KEY_WIDTH, "anchor keys")
    memory._matrix(early_keys, memory.KEY_WIDTH, "early keys", nonempty=True)
    memory._same_type(keys, early_keys)
    early = early_keys.detach().clone()

    def phase_memory_actions(query, current, h, *, fixed_prefix=0):
        if fixed_prefix != len(episodes) or len(current) != fixed_prefix + 1:
            raise RuntimeError("native optimizer supplied an unexpected episode prefix")
        candidate = current[-1]
        wrapped = PhaseEpisode(candidate.keys, candidate.delta, early)
        return phase_actions(query, [*current[:-1], wrapped], h, fixed_prefix=fixed_prefix)

    with patch.object(memory, "_memory_actions", phase_memory_actions):
        fitted, receipt = _ORIGINAL_FIT_EPISODE(
            base, ctx, fast, keys, episodes, bandwidth, indices, disruptive
        )
    return PhaseEpisode(fitted.keys, fitted.delta, early), receipt
