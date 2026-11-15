"""Differentiable paired motion views for a matured 25-sample forecast.

All three views use the same observed past and future mask.  The future mask
is therefore only available once the entire forecast window has matured; it
must not be used to form or alter the forecast made at the origin.
"""

import torch


def _check_inputs(forecast, truth, observed, past, past_available, center, scale):
    items = (forecast, truth, observed, past, past_available, center, scale)
    if not all(isinstance(value, torch.Tensor) for value in items):
        raise TypeError("motion inputs must be torch tensors")
    if forecast.ndim != 3 or forecast.shape[0] == 0 or forecast.shape[1:] != (25, 23):
        raise ValueError("forecast must have nonempty shape B x 25 x 23")
    batch = forecast.shape[0]
    if (truth.shape != (batch, 25, 23) or observed.shape != truth.shape
            or past.shape != (batch, 63, 23) or past_available.shape != past.shape
            or center.shape != (23,) or scale.shape != (23,)):
        raise ValueError("expected H25 truth/mask, W63 past/mask and 23 source coordinates")
    if observed.dtype != torch.bool or past_available.dtype != torch.bool:
        raise TypeError("observation masks must be boolean")
    if not all(value.is_floating_point() for value in (forecast, truth, past, center, scale)):
        raise TypeError("forecast, values and source coordinates must be floating point")
    if any(value.device != forecast.device for value in items):
        raise ValueError("all inputs must be on the same device")


def _require_finite(condition):
    message = "observed values and transformed forecasts must be finite; source scale must be finite and positive"
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
    else:
        # Avoid a device-to-host synchronization in the online hot path.
        torch._assert_async(condition, message)


def _recent32(values, dim):
    cumulative = torch.cumsum(values, dim=dim)
    zero = torch.zeros_like(cumulative.narrow(dim, 0, 1))
    cumulative = torch.cat((zero, cumulative), dim=dim)
    return cumulative.narrow(dim, 32, 25) - cumulative.narrow(dim, 0, 25)


def motion_views(forecast, truth, observed, past, past_available, center, scale,
                 *, frozen_forecast=None):
    """Return forecast, truth and persistence coherence, plus shared validity.

    ``past`` is B x 63 x 23, ending at q; the remaining sequences are
    B x 25 x 23 for q+1 through q+25.  ``center`` and ``scale`` are the
    frozen source coordinates, each of length 23.  ``forecast`` is in
    asinh((x-center)/scale) coordinates, whereas past/truth are raw mapped
    values.  Outputs are float64; validity requires at least 16 observed
    pairs, independently of whether their energy is zero.

    With a frozen origin forecast, use its difference from the adapted
    forecast as a correction to persistence in asinh coordinates. This
    retains the forecaster derivative but removes its zero-state bias.

    Masked payloads may be nonfinite and are ignored.  On accelerators the
    single numerical assertion is asynchronous, so a bad input can surface
    on a later operation and invalidate that device context.
    """
    _check_inputs(forecast, truth, observed, past, past_available, center, scale)

    center = center.to(dtype=torch.float64)
    scale = scale.to(dtype=torch.float64)
    # A bad source scale must not silently reverse or erase the motion.
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.nan)
    past64 = torch.where(past_available, past.to(dtype=torch.float64), center)
    truth64 = torch.where(observed, truth.to(dtype=torch.float64), center)
    past64 = (past64 - center) / scale
    truth64 = (truth64 - center) / scale

    indices = torch.arange(63, device=past.device).view(1, 63, 1)
    last = torch.where(past_available, indices, -1).amax(dim=1, keepdim=True)
    hold = past64.gather(1, last.clamp_min(0)).expand(-1, 25, -1)
    standardized = forecast.to(dtype=torch.float64)
    if frozen_forecast is not None:
        if frozen_forecast.shape not in (forecast.shape, (1, 25, 23)):
            raise ValueError("matched frozen H25 forecast required")
        standardized = torch.asinh(hold) + (standardized - frozen_forecast.detach().double())
    prediction = torch.sinh(torch.where(observed, standardized, 0.0))
    _require_finite(torch.isfinite(torch.cat((past64, truth64, prediction), dim=1)).all())

    values = torch.stack((torch.cat((past64, prediction), dim=1),
                          torch.cat((past64, truth64), dim=1),
                          torch.cat((past64, hold), dim=1)))
    mask = torch.cat((past_available, observed), dim=1)
    paired = mask[:, 32:] & mask[:, :-32]
    differences = torch.where(paired.unsqueeze(0), values[:, :, 32:] - values[:, :, :-32], 0.0)
    count = _recent32(paired.to(dtype=torch.int64), 1)
    net = _recent32(differences, 2)
    energy = _recent32(differences.square(), 2)
    valid = count >= 16

    # Substitute before sqrt: a where applied only after sqrt(0) leaves an
    # infinite derivative and can poison an otherwise zero-gradient path.
    positive = energy > 0
    safe_energy = torch.where(positive, energy, 1.0)
    denominator = safe_energy.sqrt() * count.clamp_min(1).to(energy.dtype).sqrt().unsqueeze(0)
    result = torch.where(valid.unsqueeze(0) & positive, net / denominator, 0.0).clamp(-1.0, 1.0)
    # Arithmetic overflow should reach the caller's finite-output checks.
    result = torch.where(torch.isfinite(net) & torch.isfinite(energy), result, torch.nan)
    return result[0], result[1], result[2], valid
