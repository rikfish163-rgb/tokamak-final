"""Select evaluated, output-bounded symmetric steps at reached time indices."""

import math
import numbers

import numpy as np
import torch

SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
LIMIT = 0.10


def select_symmetric(shifts, available, *, scales=SCALES, limit=LIMIT):
    """Choose the largest evaluated feasible radius shared by adjacent signs.

    ``shifts`` contains the actual counterfactual deltas, with shape
    ``[batch, action, grid, 3]``. The grid is strictly descending, and adjacent
    actions are opposite signs of the same proposal. Nonfinite deltas are
    permitted only for pairs marked unavailable. The returned radius uses
    float64 to preserve the supplied grid precision; the per-action maxima
    retain the dtype of ``shifts``. Both outputs stay on the input device.
    """
    if not isinstance(shifts, torch.Tensor) or not shifts.is_floating_point():
        raise TypeError("shifts must be a real floating-point Torch tensor")
    if shifts.ndim != 4:
        raise ValueError("shifts must have shape [B, A, G, 3]")
    batch, actions, grid_size, clocks = shifts.shape
    if actions % 2 or clocks != 3:
        raise ValueError("actions must be paired and there must be three clocks")

    if not isinstance(available, torch.Tensor) or available.dtype != torch.bool:
        raise TypeError("available must be a boolean Torch tensor")
    if available.shape != (batch, actions):
        raise ValueError("available must have shape [B, A]")
    if available.device != shifts.device:
        raise ValueError("shifts and available must be on the same device")
    if not torch.equal(available[:, 0::2], available[:, 1::2]):
        raise ValueError("availability must be symmetric within each action pair")

    try:
        raw_scales = torch.as_tensor(scales, device=shifts.device)
        if raw_scales.dtype == torch.bool or raw_scales.is_complex():
            raise TypeError("scales must be real numbers, not booleans")
        grid = torch.as_tensor(scales, dtype=torch.float64, device=shifts.device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("scales must be a real numeric sequence") from exc
    if grid.ndim != 1 or grid.numel() == 0 or grid.numel() != grid_size:
        raise ValueError("scales must be a nonempty vector matching the grid size")
    if not bool(torch.isfinite(grid).all()) or not bool((grid > 0).all()):
        raise ValueError("scales must be finite and positive")
    if not bool((grid[:-1] > grid[1:]).all()):
        raise ValueError("scales must be strictly descending")

    if isinstance(limit, bool) or not isinstance(limit, numbers.Real):
        raise TypeError("limit must be a finite nonnegative real number")
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("limit must be a finite nonnegative real number")

    finite_action = torch.isfinite(shifts).all(dim=(-1, -2))
    if not bool((finite_action | ~available).all()):
        raise ValueError("every shift for an available action must be finite")

    action_max = shifts.abs().amax(dim=-1)
    action_feasible = (action_max <= limit) & available.unsqueeze(-1)
    pair_feasible = action_feasible[:, 0::2] & action_feasible[:, 1::2]
    has_step = pair_feasible.any(dim=-1)
    first_feasible = pair_feasible.to(torch.int8).argmax(dim=-1)
    pair_radius = torch.where(has_step, grid[first_feasible], 0.0)
    radius = pair_radius.repeat_interleave(2, dim=1)

    action_indices = first_feasible.repeat_interleave(2, dim=1).unsqueeze(-1)
    selected_max = action_max.gather(dim=2, index=action_indices).squeeze(-1)
    largest_abs = torch.where(radius > 0, selected_max, torch.zeros_like(selected_max))
    return radius, largest_abs


def reached_indices(maturity):
    """Return the three reached indices (m-16, m-8, m) for each maturity."""
    values = np.asarray(maturity)
    if values.ndim != 1:
        raise ValueError("maturity must be a one-dimensional batch")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("maturity must contain integer indices")
    if np.any(values < 63) or np.any(values > np.iinfo(np.int64).max):
        raise ValueError("maturity must fit int64 and be at least 63")
    values = values.astype(np.int64, copy=False)
    return np.stack((values - 16, values - 8, values), axis=1)
