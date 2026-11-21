"""Unpromoted, standalone clipped SGD + physical-clock source anchoring.

No model, loss, labels, logits, proxy gates, calibration or clock bookkeeping.
"""
import math
from numbers import Real

import torch


def _norm(value):
    # Scaling avoids squaring tiny/large entries directly, including float64.
    scale = value.abs().amax()
    length = scale if scale == 0 else scale * torch.linalg.vector_norm(value / scale)
    if not torch.isfinite(length):
        raise FloatingPointError("nonfinite intermediate norm")
    return length


def clipped_proximal_step(
    delta: torch.Tensor,
    gradient: torch.Tensor | None,
    *,
    c: float,
    step: float,
    tau: float,
    dt: float,
    radius: float,
    decay_only: bool = False,
) -> torch.Tensor:
    """Return project_R(exp(-dt/tau) * (delta - step/c * clip_norm(g,c))).

    ``delta`` is one nonempty flat float32/float64 offset from shot start.
    ``gradient`` must match its shape, dtype and device. The caller must freeze
    positive source-calibrated ``c`` before replay; this helper cannot attest
    its provenance. ``step`` >= 0, ``tau``, ``dt`` and ``radius`` > 0; all are
    finite real scalars, not tensors. dt/tau use the same physical time unit.

    No observations is explicit: gradient=None AND decay_only=True. It applies
    decay and projection only, not a learning event. A valid zero gradient or
    step=0 also decays prior state. Nonfinite gradients are never skipped.

    Inputs and .grad fields are never mutated. Autograd through delta/gradient
    (including second derivatives away from clipping/projection kinks) is
    retained; scalar hyperparameters are deliberately not differentiable.
    Arithmetic uses float64, returning the original dtype/device. Unrepresentable
    intermediate norms/proposals raise rather than silently inventing a step.
    """
    for name, value in (("c", c), ("step", step), ("tau", tau),
                        ("dt", dt), ("radius", radius)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar, not a tensor or bool")
        if not math.isfinite(value) or value < 0 or (name != "step" and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'step' else 'positive'}")
    c, step, tau, dt, radius = map(float, (c, step, tau, dt, radius))
    if type(decay_only) is not bool:
        raise TypeError("decay_only must be a bool")
    if not isinstance(delta, torch.Tensor):
        raise TypeError("delta must be a tensor")
    if delta.dtype not in (torch.float32, torch.float64) or delta.layout != torch.strided:
        raise TypeError("delta must be a strided float32/float64 tensor")
    if delta.ndim != 1 or delta.numel() == 0 or delta.is_meta:
        raise ValueError("delta must be one nonempty flat vector with stored values")
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite delta")

    if decay_only:
        if gradient is not None:
            raise ValueError("decay_only=True requires gradient=None")
    elif gradient is None:
        raise ValueError("gradient=None requires explicit decay_only=True")
    else:
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("gradient must be a tensor")
        if (gradient.shape != delta.shape or gradient.dtype != delta.dtype
                or gradient.device != delta.device or gradient.layout != delta.layout):
            raise ValueError("gradient must match delta shape, dtype, device and layout")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite gradient")

    proposal = delta.double()
    if gradient is not None:
        g = gradient.double()
        length = _norm(g)
        # clip(g,c)/c, without forming a potentially overflowing step/c.
        direction = g / c if length <= c else g / length
        proposal = proposal - step * direction
    if not torch.isfinite(proposal).all():
        raise FloatingPointError("nonfinite intermediate proposal")
    anchored = math.exp(-dt / tau) * proposal
    length = _norm(anchored)
    if length > radius:
        anchored = (anchored / length) * radius
    result = anchored.to(dtype=delta.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite result in delta dtype")
    return result
