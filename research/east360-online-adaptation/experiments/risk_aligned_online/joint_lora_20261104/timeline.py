"""Shot-local, predict-before-feedback diagnostic LoRA replay.

No outcome is read. Predictions at origins are stored as issued, and only
eight arrived observations may produce a temporary LoRA update. A maturity
update at t is applied before the risk prediction at t, never retroactively.
"""

from copy import deepcopy
import math
from numbers import Real

import numpy as np
import torch

from ..direct_fast_memory_20261104.memory import direct_target
from ..direct_fast_memory_20261104.model import levels
from ..direct_fast_memory_20261104.timeline import window_batch
from .model import JointLoRA


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive scalar")


def _displacement(parameters, start):
    return float(
        torch.linalg.vector_norm(
            torch.cat(
                [
                    (p.detach().double() - p0.double()).reshape(-1)
                    for p, p0 in zip(parameters, start, strict=True)
                ]
            )
        ).item()
    )


def _project(parameters, start, radius):
    # Project the concatenation of ALL Q/V LoRA tensors about the shot start.
    # A tiny inward margin accommodates rounding when copying back to float32.
    with torch.no_grad():
        distance = _displacement(parameters, start)
        for _ in range(4):
            if distance <= radius:
                return distance
            factor = (radius * 0.9999) / distance
            for p, p0 in zip(parameters, start, strict=True):
                p.copy_((p0.double() + (p.double() - p0.double()) * factor).to(p.dtype))
            distance = _displacement(parameters, start)
    if distance > radius:
        raise FloatingPointError("could not enforce the joint LoRA displacement bound")
    return distance


def _masked_mse(prediction, target, observed):
    return (prediction - target).square().masked_fill(
        ~observed, 0
    ).sum() / observed.sum()


def _targets(model, shot, origin, *, device):
    x, a = window_batch(shot, [origin], device=device)
    future = torch.as_tensor(
        shot["values"][origin + 1 : origin + 9][None], device=device
    )
    future_mask = torch.as_tensor(
        shot["available"][origin + 1 : origin + 9][None], device=device
    )
    with torch.no_grad():
        target, observed = direct_target(
            levels(model.clone, x, a),
            a,
            levels(model.clone, future, future_mask),
            future_mask,
        )
    return x, a, target, observed


def _risk_segment(model, shot, start, end, probabilities, *, device):
    for lo in range(start, end, 32):
        hi = min(lo + 32, end)
        queries = list(range(lo, hi))
        # Every risk forward has batch size 32, including event/terminal tails.
        queries.extend([queries[-1]] * (32 - len(queries)))
        x, a = window_batch(shot, queries, device=device)
        with torch.no_grad():
            score = model(x, a)["logit"].sigmoid()
        if score.shape != (32,) or not torch.isfinite(score).all():
            raise FloatingPointError("finite 32-row risk batch required")
        probabilities[lo:hi] = (
            score[: hi - lo].detach().cpu().numpy().astype(np.float32)
        )


def run_shot(model, shot, mode, *, device, lr=1e-4, radius=0.5):
    """Return risk curve, matured-event history, and issued forecast records.

    Forecast ``released_at`` is its target maturity index, or -1 if pending.
    Pending target entries are NaN with all-false masks. For mature entries,
    target/mask always describe the real observation in every control arm.
    ``supported`` is the count of observed target coordinates; unsupported
    losses and uncomputed gradient norms are None, not fabricated zeros.
    """
    if mode not in ("none", "real", "hold", "reverse"):
        raise ValueError("mode must be none, real, hold, or reverse")
    _positive(lr, "lr")
    _positive(radius, "radius")
    if not isinstance(model, JointLoRA):
        raise TypeError("JointLoRA model required")
    if not isinstance(shot, dict) or "values" not in shot or "available" not in shot:
        raise ValueError("shot must contain signal values and availability")
    values, availability = shot["values"], shot["available"]
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != 23
        or not isinstance(availability, np.ndarray)
        or availability.dtype != np.bool_
        or availability.shape != values.shape
    ):
        raise ValueError(
            "float32 T-by-23 values and matching boolean availability required"
        )
    device = torch.device(device)
    local = deepcopy(model).to(device)
    local.eval().requires_grad_(False)
    for parameter in local.parameters():
        parameter.grad = None
    parameters = local.neural_parameters()
    if sum(p.numel() for p in parameters) != 512:
        raise ValueError("exactly 512 Q/V LoRA parameters required")
    for parameter in parameters:
        parameter.requires_grad_(True)
    start_params = tuple(p.detach().clone() for p in parameters)
    optimizer = None if mode == "none" else torch.optim.Adam(parameters, lr=lr)
    n = len(values)
    probabilities = np.empty(n, dtype=np.float32)
    origins = list(range(127, n, 32))
    issues = set(origins)
    maturities = {q + 8: q for q in origins if q + 8 < n}
    boundaries = sorted({0, n, *origins, *maturities})
    records = {}
    history = []
    for now, until in zip(boundaries, boundaries[1:]):
        if now in maturities:
            origin = maturities[now]
            record = records[origin]
            x, a, target, observed = _targets(local, shot, origin, device=device)
            record["released_at"] = now
            record["target"] = target[0].detach().cpu().numpy().copy()
            record["mask"] = observed[0].detach().cpu().numpy().copy()
            supported = int(observed.sum().item())
            before_actual = after_actual = hold_mse = gradient_norm = None
            did_update = False
            gap = None
            if supported:
                issued = torch.as_tensor(record["prediction"], device=device)[None]
                before_actual = float(_masked_mse(issued, target, observed).item())
                hold_mse = float(
                    _masked_mse(torch.zeros_like(target), target, observed).item()
                )
                after_actual = before_actual
                if optimizer is not None:
                    old_params = tuple(p.detach().clone() for p in parameters)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        predicted = local(x, a)["diagnostic"]
                        gap = float((predicted.detach() - issued).abs().max().item())
                        desired = (
                            target
                            if mode == "real"
                            else (
                                torch.zeros_like(target) if mode == "hold" else -target
                            )
                        )
                        loss = _masked_mse(predicted, desired, observed)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite masked diagnostic loss")
                        loss.backward()
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, 1.0, error_if_nonfinite=True
                        ).item()
                    )
                    optimizer.step()
                    _project(parameters, start_params, radius)
                    did_update = any(
                        not torch.equal(p, old)
                        for p, old in zip(parameters, old_params, strict=True)
                    )
                    if did_update:
                        with torch.enable_grad():
                            post = local(x, a)["diagnostic"].detach()
                        after_actual = float(_masked_mse(post, target, observed).item())
            history.append(
                dict(
                    arrival=now,
                    origin=origin,
                    loss_before_actual=before_actual,
                    postloss_actual=after_actual,
                    hold_mse=hold_mse,
                    gradient_norm=gradient_norm,
                    parameter_displacement=_displacement(parameters, start_params),
                    updated=did_update,
                    supported=supported,
                    issued_vs_recomputed_max_abs=gap,
                )
            )
        if now in issues:
            x, a = window_batch(shot, [now], device=device)
            # Enable grad for the same eval dispatch as the later differentiable
            # recomputation, but persist only a detached copy of the issued value.
            with torch.enable_grad():
                issued = local(x, a)["diagnostic"].detach().clone()
            if issued.shape != (1, 168) or not torch.isfinite(issued).all():
                raise FloatingPointError("finite issued 168-vector required")
            records[now] = dict(
                origin=now,
                released_at=-1,
                prediction=issued[0].cpu().numpy().copy(),
                target=np.full(168, np.nan, dtype=np.float32),
                mask=np.zeros(168, dtype=bool),
            )
        _risk_segment(local, shot, now, until, probabilities, device=device)

    forecasts = dict(
        origin=np.asarray(origins, dtype=np.int64),
        released_at=np.asarray(
            [records[q]["released_at"] for q in origins], dtype=np.int64
        ),
        prediction=np.stack([records[q]["prediction"] for q in origins])
        if origins
        else np.empty((0, 168), np.float32),
        target=np.stack([records[q]["target"] for q in origins])
        if origins
        else np.empty((0, 168), np.float32),
        mask=np.stack([records[q]["mask"] for q in origins])
        if origins
        else np.empty((0, 168), bool),
    )
    return probabilities, history, forecasts
