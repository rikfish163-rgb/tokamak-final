"""Weighted linear transport from observed signals to task-directed gradients."""

import math
from numbers import Real

import torch


def _real_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_complex() or value.dtype == torch.bool:
        raise TypeError(f"{name} must contain real numbers")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _scalar(value, name, *, positive):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _unit_rows(value):
    """Normalize along the last dimension without overflow or losing zero rows."""
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / torch.where(scale == 0, torch.ones_like(scale), scale)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / torch.where(norm == 0, torch.ones_like(norm), norm)


def fit_rule(signal, task, weights, ridge=0.1):
    """Fit weighted, centered linear ridge regression in sample space.

    Nonzero signal and task rows are normalized independently. The intercept
    is unpenalized, and the returned tensors are float64 on signal's device.
    """
    _real_tensor(signal, "signal")
    _real_tensor(task, "task")
    _real_tensor(weights, "weights")
    ridge = _scalar(ridge, "ridge", positive=True)
    if signal.ndim != 2 or not all(signal.shape):
        raise ValueError("signal must have nonempty shape (N, D)")
    if task.shape != signal.shape:
        raise ValueError("task must have the same shape as signal")
    if weights.ndim != 1 or weights.shape[0] != signal.shape[0]:
        raise ValueError("weights must have shape (N,)")

    device = signal.device
    x = _unit_rows(signal.to(dtype=torch.float64))
    y = _unit_rows(task.to(device=device, dtype=torch.float64))
    w = weights.to(device=device, dtype=torch.float64)
    if bool((w < 0).any()) or not bool((w > 0).any()):
        raise ValueError("weights must be nonnegative and have positive sum")
    # Scaling first also handles finite weights whose direct sum overflows.
    w = w / w.max()
    w = w / w.sum()
    mx = (w[:, None] * x).sum(dim=0)
    my = (w[:, None] * y).sum(dim=0)
    sqrtw = torch.sqrt(w)[:, None]
    keys = sqrtw * (x - mx)
    coefficients = torch.linalg.solve(
        keys @ keys.T + ridge * torch.eye(signal.shape[0], dtype=torch.float64, device=device),
        sqrtw * (y - my),
    )
    return {"keys": keys, "coefficients": coefficients, "center": mx, "mean": my, "ridge": ridge}


def predict_rule(query, rule):
    """Apply the learned transport; a zero observation produces exact zero."""
    _real_tensor(query, "query")
    if not query.is_floating_point():
        raise TypeError("query must have floating-point dtype")
    if query.ndim == 0:
        raise ValueError("query must have a feature dimension")
    if not isinstance(rule, dict):
        raise TypeError("rule must be a dictionary")
    try:
        keys, coefficients, center, mean = (rule[name] for name in ("keys", "coefficients", "center", "mean"))
    except KeyError as exc:
        raise ValueError("rule is missing a required buffer") from exc
    for name, value in (("keys", keys), ("coefficients", coefficients), ("center", center), ("mean", mean)):
        _real_tensor(value, name)
    if (keys.ndim != 2 or not all(keys.shape) or coefficients.shape != keys.shape
            or center.shape != (keys.shape[1],) or mean.shape != center.shape
            or query.shape[-1] != keys.shape[1]):
        raise ValueError("rule buffers and query must have compatible dimensions")

    keys, coefficients, center, mean = (
        value.to(device=query.device, dtype=torch.float64)
        for value in (keys, coefficients, center, mean)
    )
    unit_query = _unit_rows(query.to(dtype=torch.float64))
    prediction = (unit_query - center) @ keys.T @ coefficients + mean
    prediction = torch.where((unit_query == 0).all(dim=-1, keepdim=True), torch.zeros_like(prediction), prediction)
    return prediction.to(dtype=query.dtype)


def bounded_step(mapped, radius=0.05):
    """Return -radius*mapped/max(1, ||mapped||), using the whole tensor norm."""
    _real_tensor(mapped, "mapped")
    if not mapped.is_floating_point():
        raise TypeError("mapped must have floating-point dtype")
    if mapped.numel() == 0:
        raise ValueError("mapped must not be empty")
    radius = _scalar(radius, "radius", positive=False)
    value = mapped.to(dtype=torch.float64)
    scale = value.abs().max()
    if bool(scale == 0):
        return torch.zeros_like(mapped)
    scaled = value / scale
    scaled_norm = torch.linalg.vector_norm(scaled)
    if bool(scale <= 1 / scaled_norm):
        step = -radius * value
    else:
        step = (-radius / scaled_norm) * scaled
    step = step.to(dtype=mapped.dtype)
    if not bool(torch.isfinite(step).all()):
        raise ValueError("step is not representable in mapped dtype")
    return step
