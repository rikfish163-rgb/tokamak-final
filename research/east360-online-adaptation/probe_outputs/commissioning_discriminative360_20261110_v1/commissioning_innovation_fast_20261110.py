"""Matured observation features and temporary lagged internal state."""

import numpy as np
import torch


def lagged_forward(base, values, available, state):
    """Run a source model with per-example state inserted after its first layer.

    Data patches are in time order followed by CLS.  A patch receives the
    lag-four correction only when both it and its predecessor have at least
    one finite observation.  Calls sharing ``base`` must not overlap.
    """
    cfg = base.cfg
    if (cfg.input_view != "source" or cfg.window != 128 or cfg.patch != 8
            or cfg.layers != 2 or len(base.encoder.layers) != 2):
        raise ValueError("source-view W128 model with two layers and eight-point patches required")
    if values.ndim != 3 or values.shape[1:] != (128, 23):
        raise ValueError("values must have shape [batch, 128, 23]")
    if available.shape != values.shape or available.dtype != torch.bool:
        raise ValueError("matching boolean availability required")
    if (state.shape != (len(values), cfg.dim) or state.device != values.device
            or state.dtype != values.dtype):
        raise ValueError("state must match values with shape [batch, dim]")
    valid = (available & torch.isfinite(values)).reshape(len(values), 16, 8 * 23).any(-1)
    eligible = valid[:, 4:] & valid[:, :-4]

    def insert(_module, _inputs, hidden):
        patches = hidden[:, :-1]
        later = patches[:, 4:]
        correction = 0.5 * state.tanh()[:, None, :] * (later - patches[:, :-4])
        changed = torch.where(eligible[:, :, None], later + correction, later)
        return torch.cat((patches[:, :4], changed, hidden[:, -1:]), dim=1)

    handle = base.encoder.layers[0].register_forward_hook(insert)
    try:
        return base(values, available)
    finally:
        handle.remove()


def arrival_inputs(values, available, center, scale, origins):
    """Return matured H25 innovations in source asinh coordinates.

    Each origin uses its own last valid per-channel observation in W128 as
    reference.  The first half is the H25 masked mean relative to that
    reference; the second is the last-eight minus first-eight masked mean.
    Missing reference or required observations produce zero in that component.
    Only the past window and the 25 points following each origin are read.
    """
    values = np.asarray(values)
    available = np.asarray(available)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    origins = np.asarray(origins)
    if (values.ndim != 2 or values.shape[1] != 23
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)):
        raise ValueError("values must be a real numeric array with shape [time, 23]")
    if available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("matching boolean availability required")
    if (center.shape != (23,) or scale.shape != (23,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all()
            or not (scale > 0).all()):
        raise ValueError("finite source center and positive scale required")
    if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("origins must be a one-dimensional integer array")
    if np.any(origins < 127) or np.any(origins >= len(values) - 25):
        raise ValueError("each origin requires W128 past and H25 arrived observations")
    if origins.size == 0:
        return np.empty((0, 46), dtype=np.float32)

    origins = origins.astype(np.int64, copy=False)
    past_indices = origins[:, None] + np.arange(-127, 1)
    future_indices = origins[:, None] + np.arange(1, 26)
    past = values[past_indices].astype(np.float64, copy=False)
    future = values[future_indices].astype(np.float64, copy=False)
    past_valid = available[past_indices] & np.isfinite(past)
    future_valid = available[future_indices] & np.isfinite(future)
    last = np.where(past_valid, np.arange(128)[None, :, None], -1).max(axis=1)
    has_reference = last >= 0
    reference = np.take_along_axis(past, last.clip(min=0)[:, None, :], axis=1)[:, 0, :]
    reference = np.where(has_reference, reference, center)
    reference = np.arcsinh((reference - center) / scale)
    future = np.where(future_valid, future, center)
    relative = np.arcsinh((future - center) / scale) - reference[:, None, :]
    relative = np.where(future_valid, relative, 0.0)

    def masked_mean(start, stop):
        count = future_valid[:, start:stop].sum(axis=1)
        total = relative[:, start:stop].sum(axis=1)
        return total / np.maximum(count, 1), count > 0

    mean, has_future = masked_mean(0, 25)
    early, has_early = masked_mean(0, 8)
    late, has_late = masked_mean(17, 25)
    mean = np.where(has_reference & has_future, mean, 0.0)
    slope = np.where(has_reference & has_early & has_late, late - early, 0.0)
    return np.concatenate((mean, slope), axis=1).astype(np.float32)
