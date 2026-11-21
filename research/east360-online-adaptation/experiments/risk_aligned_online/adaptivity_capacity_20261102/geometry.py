"""Masked first-block representation shifts and endpoint-wise RMS budgets."""

import math
from collections.abc import Mapping

import torch

from ..dual_history_ttt_20261101.model import project_gates


_STATE_SHAPES = {"gates": (16, 2), "ffn_lora": (4, 64)}


def _finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"nonfinite {name}")


def _pack(pack):
    if not isinstance(pack, Mapping):
        raise ValueError("pack must be a mapping")
    for name in ("h", "valid", "long", "snapshot", "phi"):
        if name not in pack:
            raise KeyError(name)
        if not isinstance(pack[name], torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
    h = pack["h"]
    if h.ndim != 3 or h.shape[1:] != (17, 64):
        raise ValueError("h must have shape (N, 17, 64)")
    n = h.shape[0]
    for name, shape in (
        ("valid", (n, 17)),
        ("long", (n, 16, 64)),
        ("snapshot", (n, 16, 64)),
        ("phi", (n, 16, 4)),
    ):
        if pack[name].shape != shape or pack[name].device != h.device:
            raise ValueError(f"invalid shape or device for {name}")
    if pack["valid"].dtype != torch.bool:
        raise ValueError("valid must be boolean")
    for name in ("h", "long", "snapshot", "phi"):
        if not pack[name].is_floating_point():
            raise ValueError(f"{name} must be floating point")
        _finite(name, pack[name])
    return n


def _state(state, family, device):
    if family not in _STATE_SHAPES:
        raise ValueError(f"unknown family: {family}")
    if (
        not isinstance(state, torch.Tensor)
        or state.shape != _STATE_SHAPES[family]
        or not state.is_floating_point()
        or state.device != device
    ):
        raise ValueError(f"invalid {family} state shape, type, or device")
    _finite("state", state)


def _metric(metric):
    if not isinstance(metric, Mapping):
        raise ValueError("metric must be a mapping")
    for name in ("gate_gram", "lora_gram", "denom"):
        if name not in metric:
            raise KeyError(name)
        if not isinstance(metric[name], torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
    denom = metric["denom"]
    if denom.ndim != 1:
        raise ValueError("denom must have shape (K,)")
    k = denom.shape[0]
    for name, shape in (
        ("gate_gram", (k, 16, 2, 2)),
        ("lora_gram", (k, 4, 4)),
        ("denom", (k,)),
    ):
        value = metric[name]
        if value.shape != shape or value.dtype != torch.float64 or value.device != denom.device:
            raise ValueError(f"invalid shape, dtype, or device for {name}")
        _finite(name, value)
    if (denom < 1).any():
        raise ValueError("denom must be at least one")
    return k


def shift(pack, state, family):
    """Return the masked data-token shift; the CLS token is never modified."""
    _pack(pack)
    _state(state, family, pack["h"].device)
    invalid = ~pack["valid"][:, :16, None]
    if family == "gates":
        delta = (
            pack["long"].masked_fill(invalid, 0) * state[None, :, :1]
            + pack["snapshot"].masked_fill(invalid, 0) * state[None, :, 1:]
        )
    else:
        dtype = torch.promote_types(pack["phi"].dtype, state.dtype)
        phi = pack["phi"].masked_fill(invalid, 0).to(dtype=dtype)
        delta = phi @ state.to(dtype=dtype)
    _finite("shift", delta)
    return delta.masked_fill(invalid, 0)


def build_metric(pack, enabled):
    """Cache float64 Grams and denominators for enabled endpoints only."""
    n = _pack(pack)
    if (
        not isinstance(enabled, torch.Tensor)
        or enabled.shape != (n,)
        or enabled.dtype != torch.bool
        or enabled.device != pack["h"].device
    ):
        raise ValueError("enabled must be a boolean (N,) tensor on the pack device")
    valid = pack["valid"][enabled, :16]
    views = torch.stack((pack["long"][enabled], pack["snapshot"][enabled]), dim=2)
    views = views.to(torch.float64).masked_fill(~valid[:, :, None, None], 0)
    phi = pack["phi"][enabled].to(torch.float64).masked_fill(~valid[:, :, None], 0)
    metric = {
        "gate_gram": torch.einsum("nkic,nkjc->nkij", views, views),
        "lora_gram": torch.einsum("nki,nkj->nij", phi, phi),
        "denom": (valid.sum(dim=1) * 64).clamp_min(1).to(torch.float64),
    }
    _metric(metric)
    return metric


def max_rms(metric, state, family):
    """Return the greatest enabled endpoint RMS, or a float64 zero if empty."""
    k = _metric(metric)
    _state(state, family, metric["denom"].device)
    if k == 0:
        return state.reshape(-1)[0].to(torch.float64) * 0
    state64 = state.to(torch.float64)
    if family == "gates":
        gram = metric["gate_gram"]
        squared = torch.einsum("ki,nkij,kj->n", state64, gram, state64)
        magnitude = torch.einsum("ki,nkij,kj->n", state64.abs(), gram.abs(), state64.abs())
    else:
        gram = metric["lora_gram"]
        squared = torch.einsum("ic,nij,jc->n", state64, gram, state64)
        magnitude = torch.einsum("ic,nij,jc->n", state64.abs(), gram.abs(), state64.abs())
    _finite("quadratic", squared)
    _finite("quadratic magnitude", magnitude)
    # Gram contractions can lose a few ulps under cancellation. Scale the
    # float64 tolerance to the sum of absolute terms (64 is a conservative
    # accumulation allowance); genuinely negative energies remain errors.
    tolerance = 64 * torch.finfo(torch.float64).eps * magnitude
    if (squared < -tolerance).any():
        raise FloatingPointError("materially negative quadratic energy")
    squared_rms = squared.clamp_min(0) / metric["denom"]
    # Avoid sqrt's infinite derivative at zero, while preserving zero gradients.
    positive = squared_rms > 0
    safe = torch.where(positive, squared_rms, torch.ones_like(squared_rms))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(safe)).max()


def project_state(metric, state, family, radius=0.25):
    """Project gates first, then uniformly scale any over-budget state."""
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("radius must be finite and positive") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    _metric(metric)
    _state(state, family, metric["denom"].device)
    projected = project_gates(state) if family == "gates" else state
    rms = max_rms(metric, projected, family)
    if rms > radius:
        projected = projected * (radius / rms).to(dtype=projected.dtype)
    return projected
