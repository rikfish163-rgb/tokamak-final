"""Apply a temporary source-coordinate embedding bias without editing the parent."""

import numpy as np
import torch
from torch.nn.functional import linear

from within_shot_coherence_20261106 import BLOCK, C


W, PATCH = 128, 8


def offsets(times, disturbance, correction=None, *, oracle=False):
    """Gather causal perturbations and subtract one past-fitted local correction."""
    t, d = np.asarray(times), np.asarray(disturbance)
    if t.ndim != 1 or t.dtype.kind not in "iu" or (t < 0).any() or (t >= len(d)).any():
        raise ValueError("valid integer endpoint clocks required")
    if d.ndim != 2 or d.shape[1] != C or d.dtype.kind != "f":
        raise ValueError("disturbance must be a floating [time,23] field")
    if oracle and correction is not None:
        raise ValueError("oracle may not consume an estimated correction")
    ix = t.astype(np.int64, copy=False)[:, None] + np.arange(1 - W, 1)
    out = np.where((ix >= 0)[:, :, None], d[np.maximum(ix, 0)], 0.0)
    if oracle:
        # Exactly the same four-token write footprint; this knows the perturbation by design.
        out[:, -BLOCK:] = 0.0
    elif correction is not None:
        q = np.asarray(correction)
        if q.shape != d.shape or q.dtype.kind != "f":
            raise ValueError(
                "one known-at-endpoint correction per observed row required"
            )
        out[:, -BLOCK:] -= q[t, None, :] * (ix[:, -BLOCK:, None] >= 0)
    return out


@torch.no_grad()
def shifted_logits(base, x, available, shift):
    if (
        (base.cfg.window, base.cfg.patch, base.cfg.input_view) != (W, PATCH, "source")
        or base.embedding.in_features != PATCH * 2 * C
        or base.training
        or any(p.requires_grad for p in base.parameters())
    ):
        raise ValueError("frozen source-view patch Transformer required")
    if (
        x.ndim != 3
        or x.shape[1:] != (W, C)
        or shift.shape != x.shape
        or available.shape != x.shape
        or available.dtype != torch.bool
        or not (x.device == available.device == shift.device)
        or x.dtype != shift.dtype
        or x.dtype != base.embedding.weight.dtype
    ):
        raise ValueError(
            "matching causal windows and source-coordinate shifts required"
        )
    obs = available & torch.isfinite(x)
    delta = shift.masked_fill(~obs, 0.0)
    if not torch.isfinite(delta).all():
        raise ValueError("observed perturbations must be finite")
    if not delta.any():
        return base(x, available)["logit"]
    features = torch.cat((delta, torch.zeros_like(delta)), dim=-1)
    bias = linear(
        features.reshape(len(x), W // PATCH, PATCH * 2 * C), base.embedding.weight
    )
    seen = 0

    def append_bias(_m, _args, output):
        nonlocal seen
        if output.shape != bias.shape:
            raise RuntimeError("unexpected source embedding shape")
        seen += 1
        return output + bias

    handle = base.embedding.register_forward_hook(append_bias)
    try:
        result = base(x, available)["logit"]
    finally:
        handle.remove()
    if seen != 1:
        raise RuntimeError("source embedding bias was not applied exactly once")
    return result


def scalar_control(static, coherent):
    """Give a score-only comparator the true change at each interval's first point."""
    s, c = np.asarray(static), np.asarray(coherent)
    if (
        s.ndim != 1
        or s.shape != c.shape
        or s.dtype.kind != "f"
        or c.dtype.kind != "f"
        or not np.isfinite(s).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("paired finite contiguous logit traces required")
    i = (np.arange(len(s)) // BLOCK) * BLOCK
    out = s.astype(np.float64) + c[i].astype(np.float64) - s[i].astype(np.float64)
    out[::BLOCK] = c[::BLOCK]
    return out
