"""Causal H25 forecast-loss VJPs at the frozen first encoder affine site.

The caller supplies only already-arrived targets and a frozen Simple23 model.
This module neither updates model parameters nor chooses or applies an update.
"""

import numpy as np
import torch
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
from paired_diagnostic_features_20261106 import GROUPS


HORIZON, CHANNELS, WIDTH = 25, 23, 128
MIN_OBSERVED = 13


def arrived_targets(values, available, origins, maturity):
    """Extract o+1 through o+25 only after each inclusive maturity has arrived.

    Inputs are NumPy arrays for one supplied signal prefix. Origins and
    maturities are equally sized vectors of nonnegative integer indices. The
    returned Boolean mask excludes nonfinite values, without inspecting any
    signal samples outside each requested target window.
    """
    values = np.asarray(values)
    available = np.asarray(available)
    origins = np.asarray(origins)
    maturity = np.asarray(maturity)
    if (
        values.ndim != 2
        or values.shape[1] != CHANNELS
        or values.dtype.kind not in "fiu"
        or available.shape != values.shape
        or available.dtype != np.bool_
    ):
        raise ValueError(
            "numeric [time, 23] values and a matching Boolean mask required"
        )
    if (
        origins.ndim != 1
        or maturity.ndim != 1
        or origins.shape != maturity.shape
        or origins.dtype.kind not in "iu"
        or maturity.dtype.kind not in "iu"
    ):
        raise ValueError("matching integer origin and maturity vectors required")
    if (
        (origins < 0).any()
        or (maturity < 0).any()
        or (maturity >= len(values)).any()
        or any(
            int(m) - int(o) < HORIZON for o, m in zip(origins, maturity, strict=True)
        )
    ):
        raise ValueError("each target window must mature within the supplied prefix")

    raw = np.empty((len(origins), HORIZON, CHANNELS), dtype=values.dtype)
    mask = np.empty(raw.shape, dtype=bool)
    for row, origin in enumerate(origins):
        start = int(origin) + 1
        raw[row] = values[start : start + HORIZON]
        mask[row] = available[start : start + HORIZON] & np.isfinite(raw[row])
    return raw, mask


def _reverse_observed(raw, mask):
    """Reverse the observed time sequence independently within each channel."""
    reverse = raw.clone()
    for row in range(len(raw)):
        for _, channels in GROUPS:
            for channel in channels:
                indices = torch.nonzero(mask[row, :, channel], as_tuple=True)[0]
                reverse[row, indices, channel] = raw[row, indices.flip(0), channel]
    return reverse


def _family_losses(base, forecast, raw, mask):
    safe_raw = torch.where(mask, raw, base.center)
    target = torch.asinh((safe_raw - base.center) / base.scale).to(forecast.dtype)
    if not torch.isfinite(target[mask]).all():
        raise FloatingPointError("nonfinite normalized observed forecast target")
    prediction = torch.where(mask, forecast, torch.zeros_like(forecast))
    point_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    point_loss = point_loss.masked_fill(~mask, 0.0)
    losses = []
    for _, channels in GROUPS:
        selected = list(channels)
        counts = mask[:, :, selected].sum(dim=(1, 2))
        totals = point_loss[:, :, selected].sum(dim=(1, 2))
        losses.append(totals / counts.clamp_min(1))
    return torch.stack(losses, dim=1)


def gradients(base, x_origin, available_origin, raw_targets, available_targets):
    """Return true and time-reversed family loss gradients at temporary u=0.

    ``x_origin`` and ``available_origin`` are torch tensors [B,128,23];
    targets may be torch tensors or NumPy arrays [B,25,23]. The returned
    tensors stay on the input device and are detached. ``loss[:, 0]`` is the
    true target and ``loss[:, 1]`` is the within-channel reversed control.
    Gradients are signed derivatives, not descent directions. The caller must
    provide a frozen model; this function does not change its gradient flags.
    """
    if not isinstance(x_origin, torch.Tensor) or not x_origin.is_floating_point():
        raise ValueError("floating-point torch origin windows required")
    if (
        x_origin.ndim != 3
        or x_origin.shape[1:] != (128, CHANNELS)
        or not isinstance(available_origin, torch.Tensor)
        or available_origin.shape != x_origin.shape
        or available_origin.dtype != torch.bool
        or available_origin.device != x_origin.device
    ):
        raise ValueError(
            "matching [batch, 128, 23] origin windows and Boolean masks required"
        )
    if (
        base.cfg.window != 128
        or base.cfg.patch != 8
        or base.cfg.dim != 64
        or base.cfg.layers != 2
        or base.cfg.horizon != HORIZON
        or len(base.encoder.layers) != 2
    ):
        raise ValueError(
            "the Simple23 128/8/64/two-layer H25 configuration is required"
        )
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise ValueError("base model must be frozen before computing gradients")
    if base.center.device != x_origin.device or base.center.dtype != x_origin.dtype:
        raise ValueError("origin windows must match the base device and dtype")

    raw = torch.as_tensor(raw_targets, device=x_origin.device).detach()
    supplied_mask = torch.as_tensor(available_targets, device=x_origin.device).detach()
    expected = (len(x_origin), HORIZON, CHANNELS)
    if (
        raw.shape != expected
        or supplied_mask.shape != expected
        or supplied_mask.dtype != torch.bool
        or raw.is_complex()
        or raw.dtype == torch.bool
    ):
        raise ValueError("numeric [batch, 25, 23] targets and Boolean masks required")

    observed = supplied_mask & torch.isfinite(raw)
    enough = observed.sum(dim=1) >= MIN_OBSERVED
    selected = torch.zeros(CHANNELS, dtype=torch.bool, device=x_origin.device)
    for _, channels in GROUPS:
        selected[list(channels)] = True
    mask = observed & enough[:, None, :] & selected[None, None, :]
    present = torch.stack(
        [mask[:, :, list(channels)].any(dim=(1, 2)) for _, channels in GROUPS], dim=1
    )

    if not len(x_origin):
        empty_grad = x_origin.new_zeros((0, len(GROUPS), WIDTH))
        return dict(
            true=empty_grad,
            reverse=empty_grad.clone(),
            groups=present,
            mask=mask,
            loss=x_origin.new_zeros((0, 2, len(GROUPS))),
        )

    reversed_raw = _reverse_observed(raw, mask)
    routes = latent.routes_all(available_origin)[:, 0]
    calls = []
    with torch.enable_grad():
        u = x_origin.new_zeros((len(x_origin), WIDTH), requires_grad=True)

        def write(norm, args, original):
            calls.append(1)
            gain, bias = u.chunk(2, dim=1)
            normalized = F.layer_norm(args[0], norm.normalized_shape, eps=norm.eps)
            change = normalized * gain[:, None, :] + bias[:, None, :]
            return original + routes[:, :, None] * change

        handle = base.encoder.layers[0].norm1.register_forward_hook(write)
        try:
            forecast = base(x_origin.detach(), available_origin)["forecast"]
        finally:
            handle.remove()
        if calls != [1]:
            raise RuntimeError("first encoder norm1 must execute exactly once")
        if forecast.shape != expected:
            raise ValueError("base forecast must have shape [batch, 25, 23]")
        if not torch.isfinite(forecast[mask]).all():
            raise FloatingPointError("nonfinite forecast on an observed target")

        true_loss = _family_losses(base, forecast, raw, mask)
        reverse_loss = _family_losses(base, forecast, reversed_raw, mask)
        vectors = []
        for arm, loss in enumerate((true_loss, reverse_loss)):
            arm_vectors = []
            for group in range(len(GROUPS)):
                (vector,) = torch.autograd.grad(
                    loss[:, group].sum(),
                    u,
                    retain_graph=(arm == 0 or group < len(GROUPS) - 1),
                )
                arm_vectors.append(vector)
            vectors.append(torch.stack(arm_vectors, dim=1))

    true, reverse = (
        vector.detach().masked_fill(~present[:, :, None], 0.0) for vector in vectors
    )
    loss = torch.stack((true_loss, reverse_loss), dim=1).detach()
    if (
        not torch.isfinite(true).all()
        or not torch.isfinite(reverse).all()
        or not torch.isfinite(loss).all()
    ):
        raise FloatingPointError("nonfinite forecast loss or gradient")
    return dict(true=true, reverse=reverse, groups=present, mask=mask, loss=loss)
