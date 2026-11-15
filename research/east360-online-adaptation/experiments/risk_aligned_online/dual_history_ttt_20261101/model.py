"""Adapt internal tokens with two shot-local gates, never the risk score.

The snapshot repeats each channel's latest observed value within the original
availability mask. It is a representational view, not a second real history.
"""

import math

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..causal_span_20261101.core import span_windows
from ..context_memory_20261028.model import ContextAdapter
from ..delayed_recurrent_memory_20261025.learning import batch
from ..window_history_diagnostic_20261101.core import history_view


def _check_model(model):
    if not isinstance(model, ContextAdapter):
        raise ValueError("the fixed ContextAdapter is required")
    if torch.backends.mha.get_fastpath_enabled():
        raise RuntimeError(
            "disable the MHA fastpath to preserve the cold math-SDPA path"
        )
    cfg = model.base.cfg
    if (
        (cfg.window, cfg.patch, cfg.dim, cfg.layers, cfg.horizon) != (128, 8, 64, 2, 25)
        or cfg.fast_site != "tokens"
        or cfg.input_view != "source"
        or model.training
        or model.base.training
        or any(p.requires_grad for p in model.base.parameters())
        or any(p.requires_grad for p in model.slow.parameters())
    ):
        raise ValueError(
            "eval-mode, frozen W128/P8/dim64/two-block/H25 source and slow required"
        )


def _finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {name}")


def project_gates(beta):
    """Euclidean-project each final-dimension pair onto {x >= 0, sum(x) <= 1}."""
    if not isinstance(beta, torch.Tensor) or beta.ndim < 1 or beta.shape[-1] != 2:
        raise ValueError("floating gate pairs required")
    if not beta.is_floating_point():
        raise ValueError("floating gate pairs required")
    _finite("gates", beta)
    positive = beta.clamp_min(0)
    first = ((positive[..., 0] - positive[..., 1] + 1) * 0.5).clamp(0, 1)
    boundary = torch.stack((first, 1 - first), dim=-1)
    return torch.where(positive.sum(-1, keepdim=True) <= 1, positive, boundary)


@torch.no_grad()
def encode_views(model, signal, ends):
    """Return short tokens and validity-masked long/snapshot token deltas.

    The long view falls back to the short window before endpoint 254. The
    snapshot is made by the existing last-observed representational control.
    All three views are calculated without a gradient graph.
    """
    _check_model(model)
    ends = np.asarray(ends)
    long_x, long_available, _ = span_windows(signal, ends, 2)
    device = model.base.center.device
    x, available = batch(signal, ends, device)
    x = x.to(dtype=model.base.center.dtype)
    short, valid = model.tokens(x, available)
    long_h, long_valid = model.tokens(
        torch.as_tensor(long_x, dtype=x.dtype, device=device),
        torch.as_tensor(long_available, device=device),
    )
    snap_x, snap_available = history_view(x, available, ends, "last_observed")
    snap_h, snap_valid = model.tokens(snap_x, snap_available)
    long_delta = torch.where(
        (valid[:, :16] & long_valid[:, :16])[..., None],
        long_h[:, :16] - short[:, :16],
        0.0,
    )
    snapshot_delta = torch.where(
        (valid[:, :16] & snap_valid[:, :16])[..., None],
        snap_h[:, :16] - short[:, :16],
        0.0,
    )
    for name, tensor in (
        ("short tokens", short),
        ("long delta", long_delta),
        ("snapshot delta", snapshot_delta),
    ):
        _finite(name, tensor)
    return {"h": short, "valid": valid, "long": long_delta, "snapshot": snapshot_delta}


def _check_pack(model, pack, beta):
    _check_model(model)
    if not isinstance(pack, dict) or set(pack) != {"h", "valid", "long", "snapshot"}:
        raise ValueError("encoded short tokens and the two view deltas required")
    h, valid, long_delta, snapshot = (
        pack[k] for k in ("h", "valid", "long", "snapshot")
    )
    if (
        not all(
            isinstance(v, torch.Tensor) for v in (h, valid, long_delta, snapshot, beta)
        )
        or h.ndim != 3
        or h.shape[1:] != (17, 64)
        or not len(h)
        or valid.shape != h.shape[:2]
        or valid.dtype != torch.bool
        or long_delta.shape != (len(h), 16, 64)
        or snapshot.shape != long_delta.shape
        or beta.shape != (16, 2)
        or h.dtype != model.base.center.dtype
        or long_delta.dtype != h.dtype
        or snapshot.dtype != h.dtype
        or beta.dtype != h.dtype
        or any(v.device != h.device for v in (valid, long_delta, snapshot, beta))
        or h.device != model.base.center.device
        or not valid[:, -1].all()
    ):
        raise ValueError(
            "matching Bx17x64 tokens, masks, two view deltas and 16x2 gates required"
        )
    for name, tensor in (
        ("short tokens", h),
        ("long delta", long_delta),
        ("snapshot delta", snapshot),
        ("gates", beta),
    ):
        _finite(name, tensor)
    if (beta < 0).any() or (beta.sum(-1) > 1).any():
        raise ValueError(
            "gates must be nonnegative with each pair summing to at most one"
        )


def latent_forward(model, pack, beta, *, forecast=False):
    """Run fixed second-block risk/forecast heads after internal token shifts."""
    _check_pack(model, pack, beta)
    h, valid = pack["h"], pack["valid"]
    shift = beta[None, :, :1] * pack["long"] + beta[None, :, 1:] * pack["snapshot"]
    data = h[:, :16] + shift
    last = data[:, -1] + model.slow(data[:, -1]) * valid[:, -2, None]
    sequence = torch.cat((data[:, :-1], last[:, None], h[:, -1:]), dim=1)
    with sdpa_kernel(SDPBackend.MATH):
        sequence = model.base.encoder.layers[1](sequence, src_key_padding_mask=~valid)
    gain, bias = model.base.initial_fast(len(h)).chunk(2, -1)
    sequence = model.base.norm(sequence) * (1 + 0.1 * gain[:, None].tanh())
    sequence = sequence + 0.1 * bias[:, None].tanh()
    logit = model.base.risk_head(sequence[:, -1]).squeeze(-1)
    _finite("risk logits", logit)
    prediction = None
    if forecast:
        masked = sequence.masked_fill(~valid[:, :, None], 0.0)
        prediction = model.base.forecast_head(masked.flatten(1)).reshape(len(h), 25, 23)
        _finite("forecast", prediction)
    return {"logit": logit, "forecast": prediction, "shift": shift}


def step(model, pack, beta, truth, mask, *, step_size=0.1, anchor=0.01):
    """Take one normalized forecast-gradient step, accepting strict improvements.

    Targets must already have arrived; construction of target/control masks is
    the caller's responsibility. No observed targets yields an explicit
    no-target receipt, with null losses and no update.
    """
    _check_pack(model, pack, beta)
    if (
        not isinstance(truth, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or truth.shape != (len(pack["h"]), 25, 23)
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
        or truth.device != beta.device
        or mask.device != beta.device
        or not truth.is_floating_point()
    ):
        raise ValueError("matching Bx25x23 floating targets and boolean masks required")
    if (
        not math.isfinite(step_size)
        or step_size < 0
        or not math.isfinite(anchor)
        or anchor < 0
    ):
        raise ValueError("finite nonnegative step size and anchor required")
    _finite("observed targets", truth[mask])
    current = beta.detach().clone()
    observed = int(mask.sum().item())
    if not observed:
        return current, {
            "loss_before": None,
            "loss_proposal": None,
            "loss_retained": None,
            "gradient_norm": 0.0,
            "step_norm": 0.0,
            "observed": 0,
            "accepted": False,
            "reason": "no_target",
            "gradient_computed": False,
            "gradient32": [0.0] * 32,
            "beta32_after": current.flatten().cpu().tolist(),
        }
    with torch.enable_grad():
        variable = current.detach().requires_grad_(True)
        prediction = latent_forward(model, pack, variable, forecast=True)["forecast"]
        loss_before = (
            model.base.forecast_loss(prediction, truth, mask)
            + anchor * variable.square().mean()
        )
        _finite("objective before update", loss_before)
        gradient = torch.autograd.grad(loss_before, variable)[0].detach()
    _finite("gate gradient", gradient)
    norm = gradient.double().norm()
    _finite("gradient norm", norm)
    direction = (
        (gradient.double() / norm).to(current.dtype)
        if norm > 0
        else torch.zeros_like(current)
    )
    with torch.no_grad():
        proposal = project_gates(current - step_size * direction)
        prediction = latent_forward(model, pack, proposal, forecast=True)["forecast"]
        loss_proposal = (
            model.base.forecast_loss(prediction, truth, mask)
            + anchor * proposal.square().mean()
        )
        _finite("proposed objective", loss_proposal)
        accepted = bool(loss_proposal < loss_before.detach())
        retained = proposal if accepted else current
        retained_loss = loss_proposal if accepted else loss_before.detach()
        step_norm = (retained - current).double().norm()
        _finite("step norm", step_norm)
    return retained.detach(), {
        "loss_before": float(loss_before.detach()),
        "loss_proposal": float(loss_proposal),
        "loss_retained": float(retained_loss),
        "gradient_norm": float(norm),
        "step_norm": float(step_norm),
        "observed": observed,
        "accepted": accepted,
        "reason": "improved" if accepted else "not_improved",
        "gradient_computed": True,
        "gradient32": gradient.flatten().cpu().tolist(),
        "beta32_after": retained.flatten().cpu().tolist(),
    }
