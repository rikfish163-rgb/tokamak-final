"""Exact frozen source risk path with cached tokens and real internal writes."""

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import paired_diagnostic_step_20261106 as old
from peak_contrast_forward_20261106 import AXES, DIRECTION_FLOOR, SITE_NORM


@torch.no_grad()
def encode(base, x, available):
    """Only factor out repeated fixed input embedding for the locked source."""
    if (
        base.cfg.input_view != "source"
        or base.cfg.fast_site != "tokens"
        or (base.cfg.window, base.cfg.patch, base.cfg.dim, base.cfg.layers)
        != (128, 8, 64, 2)
        or base.training
        or torch.backends.mha.get_fastpath_enabled()
        or x.ndim != 3
        or x.shape[1:] != (128, 23)
        or available.shape != x.shape
        or available.dtype != torch.bool
    ):
        raise ValueError("frozen source token/attention contract required")
    a = available & torch.isfinite(x)
    clean = torch.where(a, x, base.center)
    level = torch.asinh((clean - base.center) / base.scale)
    v = base.embedding(torch.cat((level, a.to(level)), -1).reshape(len(x), 16, 368))
    # Native input fast state is exactly the source origin, hence no input write.
    h = torch.cat((v, base.cls.expand(len(x), -1, -1)), 1) + base.position
    good = a.reshape(len(x), 16, -1).any(-1)
    valid = torch.cat((good, torch.ones_like(good[:, :1])), 1)
    return h, valid


def packed_risk(base, packed, state):
    """Same encoder and weights as the direct hook; unused forecast not executed."""
    h, valid = packed
    if state.shape != (len(h), old.WIDTH):
        raise ValueError("one temporary affine state per reached window")
    gain, bias = state.chunk(2, 1)
    hits = []

    def write(norm, args, orig):
        hits.append(True)
        v = F.layer_norm(args[0], norm.normalized_shape, eps=norm.eps)
        return orig + valid[..., None] * (v * gain[:, None] + bias[:, None])

    handle = base.encoder.layers[0].norm1.register_forward_hook(write)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            z = base.encoder(h, src_key_padding_mask=~valid)
    finally:
        handle.remove()
    if len(hits) != 1:
        raise RuntimeError(
            "first-layer temporary parameters must be applied exactly once"
        )
    gain, bias = base.fast_origin.chunk(2)
    v = base.norm(z[:, -1]) * (1 + 0.1 * gain.tanh()) + 0.1 * bias.tanh()
    return base.risk_head(v).squeeze(-1)


@torch.no_grad()
def batched_risk(base, packed, states, *, batch_size=8192):
    h, a = packed
    n, k, _ = states.shape
    if not n or n != len(h) or not k or batch_size < 1:
        raise ValueError("nonempty reached windows and temporary state axes required")
    flat = states.reshape(n * k, old.WIDTH)
    out = h.new_empty(n * k)
    for start in range(0, n * k, batch_size):
        stop = min(start + batch_size, n * k)
        ix = torch.arange(start, stop, device=h.device) // k
        out[start:stop] = packed_risk(base, (h[ix], a[ix]), flat[start:stop])
    return out.reshape(n, k)


def action_directions(basis):
    raw = torch.stack([basis[:, axes].sum(1) for _, axes in AXES], 1)
    n = raw.norm(dim=-1)
    support = n > DIRECTION_FLOOR
    u = (SITE_NORM * raw / n.clamp_min(DIRECTION_FLOOR)[..., None]).masked_fill(
        ~support[..., None], 0
    )
    return torch.stack((u, -u), dim=2).flatten(1, 2), support.repeat_interleave(2, 1)


def teaching_masks(times, native, maturity, disruptive, down, horizon, tau):
    """Offline-only complete block masks; original timely first is its own target."""
    t, z, m = map(np.asarray, (times, native, maturity))
    if (
        t.ndim != 1
        or not len(t)
        or z.shape != t.shape
        or not np.isfinite(t).all()
        or not np.isfinite(z).all()
        or (np.diff(t) <= 0).any()
        or m.ndim != 1
        or m.dtype.kind not in "iu"
        or (m < 63).any()
        or (m >= len(t)).any()
        or ((m - 63) % 32).any()
        or (np.diff(m) <= 0).any()
    ):
        raise ValueError("aligned finite maturity and native clocks required")
    ix = m[:, None] + np.arange(1, 33)
    present = ix < len(t)
    k = np.minimum(ix, len(t) - 1)
    lead = (down - t) * 1000 if disruptive else np.full(len(t), 1.0)
    hits = np.flatnonzero((z.astype(np.float64) >= tau) & (lead >= -1e-6))
    first = int(hits[0]) if len(hits) else -1
    positive = np.zeros_like(present)
    protect = np.zeros_like(present)
    if disruptive:
        if not np.isfinite(lead).all() or not np.isfinite(horizon) or horizon <= 10:
            raise ValueError("canonical HDF5 source clock and horizon required")
        eligible = lead.round(9)
        positive = present & (eligible[k] > 10) & (eligible[k] <= horizon + 1e-6)
        if first >= 0 and eligible[first] > 10:
            protect = (
                present & (ix <= first) & ((m < first) & (first <= m + 32))[:, None]
            )
    return positive, protect, first
