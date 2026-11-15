"""Pure signal/time helpers for the trajectory-teaching development replay."""

import numpy as np
import torch
from torch.nn import functional as F

CHANNELS = tuple(i for i in range(23) if i not in (2, 10))
FIRST, STRIDE, RADIUS = 135, 32, 0.5
LEAD_BANDS = (30.0, 125.0, 500.0, 1500.0, 3000.0)


def window_views(values, mask):
    if (
        values.ndim != 2
        or values.shape[1] != 23
        or not len(values)
        or values.shape != mask.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matched 23-channel signals and boolean availability required")
    x = F.pad(values, (0, 0, 127, 0)).unfold(0, 128, 1).transpose(1, 2)
    a = F.pad(mask, (0, 0, 127, 0)).unfold(0, 128, 1).transpose(1, 2)
    return x, a


def contrast_features(level, available, gate):
    """Two motion scales and a lagged same-shot control, at arrival only."""
    if (
        level.ndim != 3
        or level.shape[1:] != (128, 23)
        or available.shape != level.shape
        or available.dtype != torch.bool
        or gate.shape != (len(level),)
    ):
        raise ValueError("fixed arrival feature shape required")
    x = level[:, :, CHANNELS].reshape(len(level), 8, 16, len(CHANNELS))
    a = available[:, :, CHANNELS].reshape_as(x)
    count = a.sum(dim=2)
    safe = torch.where(a, x, 0.0)
    half = safe.sum(dim=2) / count.clamp_min(1)
    band = safe.reshape(len(level), 4, 32, len(CHANNELS)).sum(dim=2) / count.reshape(
        len(level), 4, 2, len(CHANNELS)
    ).sum(dim=2).clamp_min(1)
    # Common coverage prevents the control from gaining/losing channels
    # simply because its change was measured one arrival earlier.
    supported = (count >= 8).all(dim=1)
    last16 = half[:, 7] - half[:, 6]
    previous16 = half[:, 5] - half[:, 4]
    last32 = band[:, 3] - 2 * band[:, 2] + band[:, 1]
    previous32 = band[:, 2] - 2 * band[:, 1] + band[:, 0]

    def describe(u, v):
        u, v = torch.where(supported, u, 0.0), torch.where(supported, v, 0.0)
        g = gate[:, None]
        return torch.cat((u, v, u * g, v * g), dim=1)

    return {
        "real": describe(last16, last32),
        "lagged": describe(previous16, previous32),
        "state": describe(band[:, 2], band[:, :3].mean(1)),
    }


def chronological_queries(scores, lead, *, disrupted):
    """For ended teaching shots only: preserve peak and midpoint per time cell."""
    z, lead = np.asarray(scores), np.asarray(lead)
    if (
        z.ndim != 1
        or z.shape != lead.shape
        or not len(z)
        or not np.isfinite(z).all()
        or not np.isfinite(lead).all()
        or (disrupted and (np.diff(lead) > 0).any())
    ):
        raise ValueError("ordered ended-shot teaching curve and clock required")
    idx = np.arange(len(z))
    blocks = np.where(idx >= FIRST, (idx - FIRST) // STRIDE, -1)
    bands = (
        np.searchsorted(LEAD_BANDS, lead, side="left")
        if disrupted
        else np.zeros(len(z), dtype=int)
    )
    boundaries = np.r_[
        0, np.flatnonzero((np.diff(blocks) != 0) | (np.diff(bands) != 0)) + 1, len(z)
    ]
    selected = set()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        points = idx[start:end]
        if disrupted:
            points = points[lead[points] >= -1e-6]
        if len(points):
            selected.update(
                (int(points[len(points) // 2]), int(points[np.argmax(z[points])]))
            )
    if not selected:
        raise ValueError("no valid ended-shot teaching points")
    return np.array(sorted(selected), dtype=np.int64)


def routed_pair(base, x, a, weights, delta):
    if (
        base.cfg.dim != 64
        or base.cfg.layers != 2
        or weights.shape != (len(x), 2, 17)
        or delta.shape != (len(x), 256)
    ):
        raise ValueError("two scoped 128D temporary states and token masks required")
    handles, calls = [], []

    def hook(i):
        def apply(norm, args, original):
            calls.append(i)
            gain, bias = delta[:, 128 * i : 128 * (i + 1)].chunk(2, dim=1)
            normalized = F.layer_norm(args[0], norm.normalized_shape, eps=norm.eps)
            change = normalized * gain[:, None] + bias[:, None]
            return original + weights[:, i, :, None] * change

        return apply

    try:
        for i, block in enumerate(base.encoder.layers):
            handles.append(block.norm1.register_forward_hook(hook(i)))
        z = base(x, a)["logit"]
    finally:
        for handle in handles:
            handle.remove()
    if calls != [0, 1]:
        raise RuntimeError("both pre-attention affine paths must execute exactly once")
    return z


def effect_gradient(base, x, a, weights):
    delta = x.new_zeros((len(x), 256), requires_grad=True)
    z = routed_pair(base, x, a, weights, delta)
    (g,) = torch.autograd.grad(z.sum(), (delta,))
    if not torch.isfinite(g).all():
        raise FloatingPointError("nonfinite trajectory Jacobian")
    return z.detach(), g.detach()


def action(features, scale, matrix):
    d = (features / scale) @ matrix
    return RADIUS * d / (1 + d.square().sum(1, keepdim=True)).sqrt()


def match_norm(control, real):
    old = control.norm(dim=1, keepdim=True)
    new = real.norm(dim=1, keepdim=True)
    return control * (new / torch.where(old > 0, old, 1.0))


def write_fast(target):
    """One actual SGD step on a learned proximal objective; no outcome input."""
    state = torch.zeros_like(target, requires_grad=True)
    loss = (state - target.detach()).square().sum() / 2
    loss.backward()
    return -state.grad.detach()
