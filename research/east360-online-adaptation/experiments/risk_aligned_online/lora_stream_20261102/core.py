"""Causal-window rank4 FFN changes, with an immutable source risk head."""

import math

import numpy as np
import torch

from ..centered_gate_stream_20261101.model import loss
from ..delayed_recurrent_memory_20261025.learning import batch
from ..dual_history_ttt_20261101.model import _check_model, latent_forward


def cap_shift(delta, valid):
    """Cap each currently observed window, never using another endpoint."""
    if (
        delta.ndim != 3
        or delta.shape[1:] != (16, 64)
        or valid.shape != delta.shape[:2]
        or valid.dtype != torch.bool
    ):
        raise ValueError("Bx16x64 shift and boolean valid-token mask required")
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite FFN shift")
    clean = delta.masked_fill(~valid[:, :, None], 0)
    denominator_squared = (valid.sum(1).clamp_min(1) * 64).double()
    squared_rms = clean.double().square().sum((1, 2)) / denominator_squared
    # Clamping before sqrt keeps the applied map flat inside the radius,
    # including finite mixed second derivatives at a fresh zero state.
    factor = (0.25 / squared_rms.clamp_min(0.25**2).sqrt()).to(clean.dtype)
    rms = torch.linalg.vector_norm(clean.double().flatten(1), dim=1) / denominator_squared.sqrt()
    return clean * factor[:, None, None], rms * factor.double()


@torch.no_grad()
def encode(model, signal, ends, basis):
    """Cache the real a+(b+delta) FFN operands from each causal input."""
    _check_model(model)
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only")
    if basis.shape != (4, 128) or not torch.isfinite(basis).all():
        raise ValueError("finite source4x128 basis required")
    layer = model.base.encoder.layers[0]
    if not layer.norm_first:
        raise ValueError("source pre-norm first block required")
    operands = {"residual": [], "ffn": [], "phi": []}

    def residual(_module, args):
        operands["residual"].append(args[0].detach())

    def ffn(_module, args, output):
        operands["ffn"].append(output.detach())
        operands["phi"].append((args[0][:, :16] @ basis.T / math.sqrt(4)).detach())

    hooks = [
        layer.norm2.register_forward_pre_hook(residual),
        layer.linear2.register_forward_hook(ffn),
    ]
    try:
        x, available = batch(signal, np.asarray(ends), model.base.center.device)
        h, valid = model.tokens(x, available)
    finally:
        for hook in hooks:
            hook.remove()
    if any(len(values) != 1 for values in operands.values()):
        raise RuntimeError("exactly one first-FFN pass required")
    return dict(
        h=h,
        valid=valid,
        residual=operands["residual"][0][:, :16],
        ffn=operands["ffn"][0][:, :16],
        phi=operands["phi"][0].masked_fill(~valid[:, :16, None], 0),
    )


def finish(model, h, valid, *, forecast):
    zeros = h.new_zeros(len(h), 16, 64)
    native = dict(h=h, valid=valid, long=zeros, snapshot=zeros)
    return latent_forward(model, native, h.new_zeros(16, 2), forecast=forecast)


def forward(model, pack, state, *, forecast=False):
    _check_model(model)
    if state.shape != (4, 64) or not torch.isfinite(state).all():
        raise ValueError("finite temporary4x64 state required")
    raw = pack["phi"] @ state
    delta, rms = cap_shift(raw, pack["valid"][:, :16])
    data = pack["residual"] + (pack["ffn"] + delta)
    h = torch.cat((data, pack["h"][:, -1:]), 1)
    out = finish(model, h, pack["valid"], forecast=forecast)
    return {
        **out,
        "h": h,
        "shift": delta,
        "rms": rms,
        "capped": (raw != delta).any((1, 2)),
    }


def direct(model, signal, ends, basis, state, *, forecast=False):
    """Uncached source-layer insertion for numerical-path certification."""
    _check_model(model)
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only")
    x, available = batch(signal, np.asarray(ends), model.base.center.device)
    valid_data = (available & torch.isfinite(x)).reshape(len(x), 16, -1).any(-1)

    def insert(_module, args, output):
        phi = args[0][:, :16] @ basis.T / math.sqrt(4)
        delta, _ = cap_shift(phi @ state, valid_data)
        return output + torch.cat((delta, output.new_zeros(len(output), 1, 64)), 1)

    hook = model.base.encoder.layers[0].linear2.register_forward_hook(insert)
    try:
        h, valid = model.tokens(x, available)
    finally:
        hook.remove()
    return dict(**finish(model, h, valid, forecast=forecast), h=h)


def step(model, pack, state, truth, mask):
    if (
        truth.shape != (len(pack["h"]), 25, 23)
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matching full25x23 target and availability required")
    if not torch.isfinite(truth[mask]).all():
        raise ValueError("nonfinite observed target")
    current = state.detach().clone()
    observed = int(mask.sum())
    if not observed:
        return current, dict(
            loss_before=None,
            loss_proposal=None,
            loss_retained=None,
            gradient_norm=0.0,
            step_norm=0.0,
            observed=0,
            accepted=False,
            reason="no_target",
            gradient_computed=False,
            gradient256=[0.0] * 256,
            state256_after=current.flatten().cpu().tolist(),
            max_action_rms=0.0,
        )
    with torch.enable_grad():
        variable = current.detach().requires_grad_(True)
        before = forward(model, pack, variable, forecast=True)
        objective = (
            loss(model, before["forecast"], truth, mask)
            + 0.01 * variable.square().mean()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite matured forecast objective")
        gradient = torch.autograd.grad(objective, variable)[0].detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite first-FFN update gradient")
    magnitude = gradient.double().norm()
    direction = (
        (gradient.double() / magnitude).to(current.dtype)
        if magnitude > 0
        else torch.zeros_like(current)
    )
    with torch.no_grad():
        proposal = current - 0.1 * direction
        proposed = forward(model, pack, proposal, forecast=True)
        proposed_loss = (
            loss(model, proposed["forecast"], truth, mask)
            + 0.01 * proposal.square().mean()
        )
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite proposed first-FFN objective")
        accepted = bool(proposed_loss < objective.detach())
        after = proposal if accepted else current
        retained = proposed_loss if accepted else objective.detach()
        rms = proposed["rms"] if accepted else before["rms"].detach()
    return after.detach(), dict(
        loss_before=float(objective.detach()),
        loss_proposal=float(proposed_loss),
        loss_retained=float(retained),
        gradient_norm=float(magnitude),
        step_norm=float((after - current).double().norm()),
        observed=observed,
        accepted=accepted,
        reason="improved" if accepted else "not_improved",
        gradient_computed=True,
        gradient256=gradient.flatten().cpu().tolist(),
        state256_after=after.flatten().cpu().tolist(),
        max_action_rms=float(rms.max()),
    )
