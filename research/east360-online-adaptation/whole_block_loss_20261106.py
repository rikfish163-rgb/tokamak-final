"""Pure offline role construction and differentiable block-peak hinge loss.

The caller supplies completed-shot outcomes and native logits. Nothing in this
module loads data or performs an online update; these roles are training-only.
"""

import math
import numbers

import numpy as np
import torch


SAFE = 0
PROTECT = 1
RECOVER = 2

_LEAD_MS = 10.0
_TOLERANCE_MS = 1e-6
_MARGIN = 0.025


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _first_pre_outcome_crossing(raw_lead_ms, logits, threshold_logit):
    """Return the first crossing allowing the raw post-outcome tolerance."""
    hits = np.flatnonzero((raw_lead_ms >= -_TOLERANCE_MS) & (logits >= threshold_logit))
    return int(hits[0]) if hits.size else None


def role_masks(
    times, logits, slots, support, disruptive, down, horizon, threshold_logit
):
    """Build offline training rows for a single completed shot.

    ``times`` and disruptive ``down`` are seconds; ``horizon`` is milliseconds.
    One-dimensional, strictly increasing ``times`` and native ``logits`` have
    one value per endpoint. Integer ``slots`` maps each endpoint to an index in
    the one-dimensional boolean ``support`` array; -1 denotes an unchanged
    endpoint. Safe shots do not use ``down``. The horizon must exceed 10 ms.

    Returns (slot_index, role, ascending int64 endpoint indices) tuples, one
    per nonempty supported block. First alarms are measured over the entire
    original trajectory, including unchanged and unsupported endpoints.
    """
    try:
        times = np.asarray(times)
        if np.iscomplexobj(times):
            raise ValueError("complex times")
        times = times.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must be a finite, strictly increasing vector") from exc
    if (
        times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or np.any(times[1:] <= times[:-1])
    ):
        raise ValueError("times must be a nonempty, finite, strictly increasing vector")

    try:
        logits = np.asarray(logits)
        if np.iscomplexobj(logits):
            raise ValueError("complex logits")
        logits = logits.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("logits must be a finite vector aligned with times") from exc
    if logits.shape != times.shape or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite vector aligned with times")

    slots = np.asarray(slots)
    if slots.shape != times.shape or slots.dtype.kind not in "iu":
        raise ValueError("slots must be an integer vector aligned with times")
    support = np.asarray(support)
    if support.ndim != 1 or support.dtype != np.bool_:
        raise ValueError("support must be a one-dimensional boolean array")
    if np.any(slots < -1) or np.any(slots >= support.size):
        raise ValueError("slots must be -1 or a valid support index")

    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError("disruptive must be a boolean outcome")
    horizon = _finite_scalar(horizon, "horizon")
    if horizon <= _LEAD_MS:
        raise ValueError("horizon must exceed 10 ms")
    threshold_logit = _finite_scalar(threshold_logit, "threshold_logit")

    if not disruptive:
        eligible = np.ones(times.size, dtype=bool)
        role = SAFE
    else:
        down = _finite_scalar(down, "down")
        with np.errstate(over="ignore", invalid="ignore"):
            raw_lead_ms = (down - times) * 1000.0
            lead_ms = np.round(raw_lead_ms, decimals=9)
        if not np.isfinite(raw_lead_ms).all() or not np.isfinite(lead_ms).all():
            raise ValueError("lead times must be finite")
        first = _first_pre_outcome_crossing(raw_lead_ms, logits, threshold_logit)
        if first is not None and lead_ms[first] > _LEAD_MS:
            slot = int(slots[first])
            if slot == -1 or not support[slot]:
                return []
            eligible = (np.arange(times.size) <= first) & (lead_ms > _LEAD_MS)
            indices = np.flatnonzero((slots == slot) & eligible).astype(
                np.int64, copy=False
            )
            return [(slot, PROTECT, indices)] if indices.size else []
        eligible = (lead_ms > _LEAD_MS) & (lead_ms <= horizon + _TOLERANCE_MS)
        role = RECOVER

    rows = []
    for slot in np.flatnonzero(support):
        indices = np.flatnonzero((slots == slot) & eligible).astype(
            np.int64, copy=False
        )
        if indices.size:
            rows.append((int(slot), role, indices))
    return rows


def peak_loss(z, native, mask, role, threshold_logit, *, point=False):
    """Return an unweighted hinge loss for each row of a padded batch.

    ``z`` and frozen ``native`` are floating tensors [B, L], ``mask`` is a
    boolean tensor [B, L], and ``role`` is an integer tensor [B], all on the
    same device. Every row must contain an eligible point. Masked padding is
    ignored, including nonfinite padding values. ``point=False`` uses the
    updated maximum; ``point=True`` reads the updated value at the first
    native maximum among eligible points. Only ``z`` receives gradients.
    """
    if (
        not isinstance(z, torch.Tensor)
        or z.ndim != 2
        or not z.is_floating_point()
        or z.shape[0] == 0
        or z.shape[1] == 0
    ):
        raise ValueError("z must be a nonempty floating tensor with shape [B, L]")
    if (
        not isinstance(native, torch.Tensor)
        or native.shape != z.shape
        or not native.is_floating_point()
        or native.device != z.device
    ):
        raise ValueError("native must be a floating tensor [B, L] on z's device")
    if (
        not isinstance(mask, torch.Tensor)
        or mask.shape != z.shape
        or mask.dtype != torch.bool
        or mask.device != z.device
        or not bool(mask.any(dim=1).all())
    ):
        raise ValueError(
            "mask must be boolean [B, L], on z's device, with a point per row"
        )
    if (
        not isinstance(role, torch.Tensor)
        or role.shape != (z.shape[0],)
        or role.dtype
        not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        or role.device != z.device
        or not bool(((role == SAFE) | (role == PROTECT) | (role == RECOVER)).all())
    ):
        raise ValueError(
            "role must be an integer [B] tensor of valid roles on z's device"
        )
    if not bool(torch.isfinite(z.masked_select(mask)).all()):
        raise ValueError("z must be finite at eligible points")
    if not bool(torch.isfinite(native.masked_select(mask)).all()):
        raise ValueError("native must be finite at eligible points")
    threshold_logit = _finite_scalar(threshold_logit, "threshold_logit")
    if not isinstance(point, (bool, np.bool_)):
        raise ValueError("point must be boolean")

    if point:
        native_index = (
            native.detach().masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
        )
        peak = z.gather(1, native_index).squeeze(1)
    else:
        peak = z.masked_fill(~mask, -torch.inf).max(dim=1).values
    return torch.where(
        role == SAFE,
        torch.relu(peak - (threshold_logit - _MARGIN)),
        torch.relu((threshold_logit + _MARGIN) - peak),
    )
