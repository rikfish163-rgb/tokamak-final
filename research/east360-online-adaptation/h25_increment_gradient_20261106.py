"""Frozen H25 forecast-increment VJPs at the first encoder affine site.

The caller supplies already-arrived targets. This helper neither fits a head
nor changes model parameters, parameter flags, or the native forward path.
"""

import torch
from torch.nn import functional as F

import h25_arrived_gradient_20261106 as arrived


HORIZON = arrived.HORIZON
CHANNELS = arrived.CHANNELS
WIDTH = arrived.WIDTH
MIN_OBSERVED = arrived.MIN_OBSERVED
GROUPS = arrived.GROUPS
LAG = 8


def _family_losses(base, forecast, raw, mask):
    """Mean pair loss by family; ``mask`` selects the 25 observed levels."""
    safe_raw = torch.where(mask, raw, base.center)
    target = torch.asinh((safe_raw - base.center) / base.scale).to(forecast.dtype)
    if not torch.isfinite(target[mask]).all():
        raise FloatingPointError("nonfinite normalized observed forecast target")

    pairs = mask[:, LAG:] & mask[:, :-LAG]
    zero = torch.zeros_like(forecast[:, LAG:])
    predicted_delta = torch.where(pairs, forecast[:, LAG:], zero) - torch.where(
        pairs, forecast[:, :-LAG], zero
    )
    target_delta = torch.where(pairs, target[:, LAG:], zero) - torch.where(
        pairs, target[:, :-LAG], zero
    )
    point_loss = F.smooth_l1_loss(
        predicted_delta, target_delta, reduction="none", beta=1.0
    ).masked_fill(~pairs, 0.0)
    losses = []
    for _, channels in GROUPS:
        selected = list(channels)
        counts = pairs[:, :, selected].sum(dim=(1, 2))
        totals = point_loss[:, :, selected].sum(dim=(1, 2))
        losses.append(totals / counts.clamp_min(1))
    return torch.stack(losses, dim=1)


def gradients(base, x_origin, available_origin, raw_targets, available_targets):
    """Return signed true/reversed lag-8 loss gradients at temporary u=0.

    Origins are torch tensors [B,128,23]; targets may be torch tensors or NumPy
    arrays [B,25,23]. The 25-level mask follows the arrived-H25 contract, and
    the returned [B,17,23] mask requires both endpoints of a lag-8 pair. A
    family is present iff at least one such pair exists. Reversal is within
    each channel's observed level sequence, before forming the same pairs.
    ``loss[:, 0]`` is the true arm and ``loss[:, 1]`` the reversed arm.
    The frozen base and its forward behavior are not modified; all returned
    tensors are detached and remain on the origin device.
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
    level_mask = observed & enough[:, None, :] & selected[None, None, :]
    pair_mask = level_mask[:, LAG:] & level_mask[:, :-LAG]
    present = torch.stack(
        [pair_mask[:, :, list(channels)].any(dim=(1, 2)) for _, channels in GROUPS],
        dim=1,
    )

    if not len(x_origin):
        empty_grad = x_origin.new_zeros((0, len(GROUPS), WIDTH))
        return dict(
            true=empty_grad,
            reverse=empty_grad.clone(),
            groups=present,
            mask=pair_mask,
            loss=x_origin.new_zeros((0, 2, len(GROUPS))),
        )

    reversed_raw = arrived._reverse_observed(raw, level_mask)
    routes = arrived.latent.routes_all(available_origin)[:, 0]
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
        if not (
            torch.isfinite(forecast[:, LAG:][pair_mask]).all()
            and torch.isfinite(forecast[:, :-LAG][pair_mask]).all()
        ):
            raise FloatingPointError("nonfinite forecast on an observed target pair")

        true_loss = _family_losses(base, forecast, raw, level_mask)
        reverse_loss = _family_losses(base, forecast, reversed_raw, level_mask)
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
        raise FloatingPointError("nonfinite forecast-increment loss or gradient")
    return dict(true=true, reverse=reverse, groups=present, mask=pair_mask, loss=loss)
