"""Source-native temporary token offsets; no labels or persistent updates.

Offsets have shape Bx128 and are relative to the learned source fast origin.
The native normalized-step geometry is not matched to an FFN-offset budget.
"""

import torch
from torch import nn

from ..fast_update import normalized_step
from ..gate_loss_audit_20261101.core import centered_loss
from ..simple23 import Simple23
from ..simple_online import FAST_RADIUS, FAST_STEP


def _check_source(base):
    if not isinstance(base, Simple23):
        raise TypeError("a frozen source Simple23 is required")
    cfg = base.cfg
    if (
        (cfg.window, cfg.patch, cfg.dim, cfg.layers, cfg.horizon) != (128, 8, 64, 2, 25)
        or (cfg.fast_site, cfg.fast_rule, cfg.input_view) != ("tokens", "normalized", "source")
    ):
        raise ValueError("source W128/P8/d64/two-block/H25 normalized token configuration required")
    if any(m.training for m in base.modules()) or any(p.requires_grad for p in base.parameters()):
        raise ValueError("the source and all its parameters must be eval and frozen")
    head = base.forecast_head
    if not isinstance(head, nn.Linear) or (head.in_features, head.out_features) != (1088, 575):
        raise ValueError("source forecast Linear1088->575 required")
    if not torch.isfinite(base.center).all() or not torch.isfinite(base.scale).all() or not (base.scale > 0).all():
        raise ValueError("finite source coordinates with positive scale required")


def _check_decoder(base, decoder):
    if decoder is None:
        return
    if (
        not isinstance(decoder, nn.Linear)
        or (decoder.in_features, decoder.out_features) != (1088, 575)
        or decoder.bias is None
        or any(m.training for m in decoder.modules())
        or any(p.requires_grad for p in decoder.parameters())
        or any(p.device != base.center.device or p.dtype != base.center.dtype for p in decoder.parameters())
    ):
        raise ValueError("independent frozen eval Linear1088->575 on the source dtype/device required")
    source_storage = {p.untyped_storage().data_ptr() for p in base.parameters()}
    if any(p.untyped_storage().data_ptr() in source_storage for p in decoder.parameters()):
        raise ValueError("the external decoder must not share source parameter storage")


def _check_inputs(base, values, available, delta, decoder):
    _check_source(base)
    _check_decoder(base, decoder)
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 3
        or values.shape[1:] != (128, 23)
        or not len(values)
        or values.dtype != base.center.dtype
        or values.device != base.center.device
        or not isinstance(available, torch.Tensor)
        or available.shape != values.shape
        or available.dtype != torch.bool
        or available.device != values.device
        or not isinstance(delta, torch.Tensor)
        or delta.shape != (len(values), 128)
        or delta.dtype != values.dtype
        or delta.device != values.device
    ):
        raise ValueError("matching source Bx128x23 values/mask and Bx128 offset required")
    if not torch.isfinite(values[available]).all() or not torch.isfinite(delta).all():
        raise ValueError("observed inputs and temporary offsets must be finite")


def forward(base, values, available, delta, decoder=None):
    """Return native risk and direct forecast, optionally with an external head.

    The external head reads the native forecast-head input without detaching it.
    This function does not mutate the source, input mask, or decoder.
    """
    _check_inputs(base, values, available, delta, decoder)
    fast = base.initial_fast(len(values)) + delta
    if decoder is None:
        result = base(values, available, fast=fast)
        forecast = result["forecast"]
    else:
        captured = []
        hook = base.forecast_head.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        try:
            result = base(values, available, fast=fast)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (len(values), 1088):
            raise RuntimeError("exactly one native Bx1088 forecast-head input required")
        forecast = decoder(captured[0]).reshape(len(values), 25, 23)
    logit = result["logit"]
    if not torch.isfinite(logit).all() or not torch.isfinite(forecast).all():
        raise FloatingPointError("nonfinite native logit or diagnostic forecast")
    return {"logit": logit, "forecast": forecast}


def _loss(base, prediction, truth, mask, kind):
    if kind == "absolute":
        return base.forecast_loss(prediction, truth, mask)
    clean = torch.where(mask, truth.detach(), base.center)
    target = torch.asinh((clean - base.center) / base.scale)
    return centered_loss(prediction, target, mask)


def _receipt(before, after, gradient, counts, loss_before, loss_after, *, computed):
    return dict(
        reason="normalized_step" if computed else "no_target",
        gradient_computed=computed,
        observed=int(counts.sum()),
        observed_per_sample=counts.detach().cpu().tolist(),
        loss_before=loss_before,
        loss_after=loss_after,
        gradient_norm=gradient.detach().double().norm(dim=1).cpu().tolist(),
        step_norm=(after - before).detach().double().norm(dim=1).cpu().tolist(),
        offset_norm_before=before.detach().double().norm(dim=1).cpu().tolist(),
        offset_norm_after=after.detach().double().norm(dim=1).cpu().tolist(),
        state_offset=after.detach().cpu().tolist(),
    )


def advance(base, x, a, state, truth, mask, decoder=None, loss_kind="absolute"):
    """One detached, unconditional native normalized step on arrived targets.

    Returns (new_offset, receipt). A row with no observed target remains exactly
    unchanged; an entirely unobserved batch returns without a model forward.
    """
    _check_inputs(base, x, a, state, decoder)
    if loss_kind not in ("absolute", "centered"):
        raise ValueError("loss_kind must be absolute or centered")
    if (
        not isinstance(truth, torch.Tensor)
        or truth.shape != (len(x), 25, 23)
        or truth.dtype != x.dtype
        or truth.device != x.device
        or not isinstance(mask, torch.Tensor)
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
        or mask.device != x.device
    ):
        raise ValueError("matching source Bx25x23 truth and boolean target mask required")
    if not torch.isfinite(truth[mask]).all():
        raise ValueError("observed targets must be finite")
    before = state.detach().clone()
    counts = mask.sum(dim=(1, 2))
    if not counts.any():
        return before, _receipt(before, before, torch.zeros_like(before), counts, None, None, computed=False)

    with torch.enable_grad():
        variable = before.clone().requires_grad_(True)
        prediction = forward(base, x.detach(), a, variable, decoder)["forecast"]
        loss_before = _loss(base, prediction, truth, mask, loss_kind)
        if not torch.isfinite(loss_before):
            raise FloatingPointError("nonfinite diagnostic loss before step")
        (gradient,) = torch.autograd.grad(loss_before, variable)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite native token gradient")
    with torch.no_grad():
        origin = base.initial_fast(len(x)).detach()
        proposed = normalized_step(origin + before, origin, gradient, step=FAST_STEP, radius=FAST_RADIUS) - origin
        after = torch.where(counts[:, None] > 0, proposed, before).detach()
        prediction_after = forward(base, x.detach(), a, after, decoder)["forecast"]
        loss_after = _loss(base, prediction_after, truth, mask, loss_kind)
        if not torch.isfinite(loss_after):
            raise FloatingPointError("nonfinite diagnostic loss after step")
    return after, _receipt(before, after, gradient, counts, float(loss_before.detach()), float(loss_after), computed=True)
