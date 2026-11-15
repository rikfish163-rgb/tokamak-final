"""Signed lag-8 feedback VJPs from a caller-provided forecast function.

The caller supplies already-arrived targets and owns any model wrapping,
freezing, and subsequent update. This module neither imports a model nor
changes model parameters, their gradient flags, or their gradient buffers.
"""

import torch
from torch.nn import functional as F

from paired_diagnostic_features_20261106 import GROUPS


HORIZON, CHANNELS, WIDTH = 25, 23, 64
MIN_OBSERVED, LAG = 13, 8
SEGMENTS = (slice(0, 6), slice(6, 12), slice(12, 17))


def _tensor(value, device, name):
    try:
        return torch.as_tensor(value, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"invalid {name}") from error


def _reverse_observed(raw, mask):
    """Reverse values at observed positions, without moving those positions."""
    positions = torch.arange(HORIZON, device=raw.device)[None, :, None].expand_as(mask)
    descending = torch.where(mask, positions, -1).sort(dim=1, descending=True).values
    ranks = (mask.cumsum(dim=1) - 1).clamp_min(0)
    sources = descending.gather(1, ranks).clamp_min(0)
    return torch.where(mask, raw.gather(1, sources), raw)


def _losses(forecast, raw, levels, pairs, center, scale):
    safe = torch.where(levels, raw, center)
    target = torch.asinh((safe - center) / scale).to(forecast.dtype)
    if not torch.isfinite(target[levels]).all():
        raise FloatingPointError("nonfinite normalized observed forecast target")

    zero = torch.zeros_like(forecast[:, LAG:])
    prediction_delta = torch.where(pairs, forecast[:, LAG:], zero) - torch.where(
        pairs, forecast[:, :-LAG], zero
    )
    target_delta = torch.where(pairs, target[:, LAG:], zero) - torch.where(
        pairs, target[:, :-LAG], zero
    )
    point_loss = F.smooth_l1_loss(
        prediction_delta, target_delta, beta=1.0, reduction="none"
    ).masked_fill(~pairs, 0.0)
    losses = []
    for _, channels in GROUPS:
        for segment in SEGMENTS:
            count = pairs[:, segment, list(channels)].sum(dim=(1, 2))
            total = point_loss[:, segment, list(channels)].sum(dim=(1, 2))
            losses.append(total / count.clamp_min(1))
    return torch.stack(losses, dim=1)


def vjps(
    forward, width_reference, raw_targets, available_targets, center, scale,
    *, independent_rows=False,
):
    """Return per-sample true/reversed lag-8 loss derivatives at state zero.

    ``width_reference`` is a floating tensor [B,64], used only for shape,
    device, and dtype. ``forward`` is evaluated exactly once (except for an
    empty batch) on a fresh gradient-enabled zero state, and must return a
    floating forecast tensor [B,25,23]. Targets and fixed [23] source-center
    and source-scale vectors may be tensors or array-like numeric inputs.

    Eligibility matches the arrived-H25 contract: a selected family channel
    needs at least 13 finite, available levels. A pair is usable only when
    both endpoints, eight positions apart, are eligible. Reversal happens
    within each channel's observed levels before pairs are formed; the pair
    mask itself does not change. Each loss is the mean SmoothL1(beta=1)
    over observed pairs within one family and one start-index segment
    (0:6, 6:12, 12:17). Terms are ordered family-major, then segment.

    Returns detached ``true`` and ``reverse`` signed gradients [B,9,64],
    Boolean ``terms`` [B,9], ``loss`` [B,2,9] (true, then reverse), and
    Boolean pair ``mask`` [B,17,23]. Unsupported terms are exactly zero.
    By default the gradient for row b is d(loss[b])/d(state[b]), even if the
    callback couples rows. Opt into ``independent_rows=True`` only when the
    caller has established that each forecast row depends on its own state
    row alone. That faster path uses 18 batched loss-sum backward calls, and
    does not verify the independence assumption.
    """
    if (
        not isinstance(width_reference, torch.Tensor)
        or not width_reference.is_floating_point()
        or width_reference.ndim != 2
        or width_reference.shape[1] != WIDTH
    ):
        raise ValueError("floating-point [batch, 64] width reference required")
    if not callable(forward):
        raise ValueError("forward must be callable")
    if not isinstance(independent_rows, bool):
        raise ValueError("independent_rows must be a bool")

    device = width_reference.device
    raw = _tensor(raw_targets, device, "targets")
    supplied = _tensor(available_targets, device, "availability")
    center = _tensor(center, device, "center")
    scale = _tensor(scale, device, "scale")
    expected = (len(width_reference), HORIZON, CHANNELS)
    if (
        raw.shape != expected
        or supplied.shape != expected
        or supplied.dtype != torch.bool
        or raw.is_complex()
        or raw.dtype == torch.bool
    ):
        raise ValueError("numeric [batch, 25, 23] targets and Boolean masks required")
    if (
        center.shape != (CHANNELS,)
        or scale.shape != (CHANNELS,)
        or center.is_complex()
        or scale.is_complex()
        or center.dtype == torch.bool
        or scale.dtype == torch.bool
        or not torch.isfinite(center).all()
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("finite [23] center and positive finite [23] scale required")

    observed = supplied & torch.isfinite(raw)
    enough = observed.sum(dim=1) >= MIN_OBSERVED
    selected = torch.zeros(CHANNELS, dtype=torch.bool, device=device)
    for _, channels in GROUPS:
        selected[list(channels)] = True
    levels = observed & enough[:, None, :] & selected[None, None, :]
    pairs = levels[:, :-LAG] & levels[:, LAG:]
    present = torch.stack(
        [
            pairs[:, segment, list(channels)].any(dim=(1, 2))
            for _, channels in GROUPS
            for segment in SEGMENTS
        ],
        dim=1,
    )
    vectors = width_reference.new_zeros((2, len(width_reference), len(GROUPS) * 3, WIDTH))
    if not len(width_reference):
        return dict(
            true=vectors[0], reverse=vectors[1], terms=present, mask=pairs,
            loss=width_reference.new_zeros((0, 2, len(GROUPS) * 3)),
        )

    reverse = _reverse_observed(raw, levels)
    with torch.enable_grad():
        state = width_reference.new_zeros((len(width_reference), WIDTH), requires_grad=True)
        forecast = forward(state)
        if (
            not isinstance(forecast, torch.Tensor)
            or forecast.shape != expected
            or not forecast.is_floating_point()
            or forecast.device != device
        ):
            raise ValueError("forward must return a floating [batch, 25, 23] tensor on the same device")
        if not (
            torch.isfinite(forecast[:, LAG:][pairs]).all()
            and torch.isfinite(forecast[:, :-LAG][pairs]).all()
        ):
            raise FloatingPointError("nonfinite forecast on an observed target pair")
        losses = torch.stack(
            (
                _losses(forecast, raw, levels, pairs, center, scale),
                _losses(forecast, reverse, levels, pairs, center, scale),
            ),
            dim=1,
        )
        if losses.requires_grad:
            if independent_rows:
                # The caller explicitly guarantees the off-diagonal blocks
                # vanish, so one summed VJP recovers every row for a term.
                for arm in range(2):
                    for term in range(len(GROUPS) * 3):
                        (gradient,) = torch.autograd.grad(
                            losses[:, arm, term].sum(), state,
                            retain_graph=True, allow_unused=True,
                        )
                        if gradient is not None:
                            vectors[arm, :, term] = gradient.masked_fill(
                                ~present[:, term, None], 0.0
                            )
            else:
                # Summed VJPs would mix the off-diagonal blocks here.
                for row in range(len(width_reference)):
                    for term in range(len(GROUPS) * 3):
                        if not present[row, term]:
                            continue
                        for arm in range(2):
                            (gradient,) = torch.autograd.grad(
                                losses[row, arm, term], state,
                                retain_graph=True, allow_unused=True,
                            )
                            if gradient is not None:
                                vectors[arm, row, term] = gradient[row]

    loss = losses.detach()
    if not torch.isfinite(vectors).all() or not torch.isfinite(loss).all():
        raise FloatingPointError("nonfinite forecast-increment loss or gradient")
    return dict(true=vectors[0].detach(), reverse=vectors[1].detach(), terms=present,
                mask=pairs, loss=loss)
