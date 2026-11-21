"""Stable composition of a source-asinh map and the unchanged source network.

Avoid the unnecessary sinh -> float32 raw -> asinh round trip. No coordinate is
clipped, no input is replaced by a baseline, and all original classifier layers
and weights run as before. The scoped embedding hook changes only selected,
observed input coordinates and is removed even when the forward call fails.
"""

import numpy as np
import torch

from .core import _signal, _normalizer, covariance_matrix


def coordinate_signal(signal, fit, center, scale):
    x, m = _signal(signal)
    center, scale = _normalizer(center, scale)
    z = np.arcsinh((np.where(m, x, center) - center) / scale)
    out = z.copy()
    active = np.asarray(fit["active"], bool)
    sm, ss = np.asarray(fit["source"]["mean"]), np.asarray(fit["source"]["std"])
    tm, ts = np.asarray(fit["target"]["mean"]), np.asarray(fit["target"]["std"])
    cs, ct = (
        np.asarray(fit["source"]["correlation"]),
        np.asarray(fit["target"]["correlation"]),
    )
    patterns, inverse = np.unique(m & active, axis=0, return_inverse=True)
    if fit["mode"] not in ("diagonal", "correlation"):
        raise ValueError("unknown map")
    for k, pattern in enumerate(patterns):
        cols = np.flatnonzero(pattern)
        if not len(cols):
            continue  # No observed, identifiable coordinate in this pattern.
        points = np.flatnonzero(inverse == k)
        ix = np.ix_(points, cols)
        transformed = (z[ix] - tm[cols]) / ts[cols]
        if fit["mode"] == "correlation":
            block = np.ix_(cols, cols)
            transformed = transformed @ covariance_matrix(cs[block], ct[block])
        out[ix] = transformed * ss[cols] + sm[cols]
    if not np.isfinite(out[m]).all() or (abs(out[m]) > np.finfo(np.float32).max).any():
        raise FloatingPointError("nonfinite or unrepresentable mapped model coordinate")
    return dict(values=out.astype(np.float32), available=m.copy())


def forward_coordinates(base, values, available, levels, active):
    if base.cfg.input_view != "source":
        raise ValueError("fixed source-coordinate model required")
    if (
        values.ndim != 3
        or values.shape[1:] != (base.cfg.window, 23)
        or levels.shape != values.shape
        or available.shape != values.shape
    ):
        raise ValueError(
            "matching original windows, mapped coordinates and mask required"
        )
    if (
        available.dtype != torch.bool
        or active.dtype != torch.bool
        or active.shape != (23,)
    ):
        raise ValueError("boolean observations and explicit channel selection required")
    observed = available & torch.isfinite(values)
    if not torch.isfinite(levels[observed]).all():
        raise ValueError("mapped coordinates must be finite")
    if (
        levels.dtype != values.dtype
        or levels.device != values.device
        or active.device != values.device
    ):
        raise ValueError("same dtype/device required")
    calls = 0

    def replace_selected(module, inputs):
        nonlocal calls
        calls += 1
        original = inputs[0]
        if original.shape != (
            len(values),
            base.cfg.window // base.cfg.patch,
            base.cfg.patch * 46,
        ):
            raise ValueError("source embedding interface changed")
        features = original.reshape(len(values), base.cfg.window, 46).clone()
        features[:, :, :23] = torch.where(
            observed & active, levels, features[:, :, :23]
        )
        return (features.reshape_as(original),)

    handle = base.embedding.register_forward_pre_hook(replace_selected)
    try:
        result = base(values, available)
        if calls != 1:
            raise AssertionError("exactly one source input embedding expected")
        if not torch.isfinite(result["logit"]).all():
            raise FloatingPointError("nonfinite aligned risk")
        return result
    finally:
        handle.remove()
