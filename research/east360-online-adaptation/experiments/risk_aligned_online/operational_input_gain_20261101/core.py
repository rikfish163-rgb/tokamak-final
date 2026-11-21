"""Positive raw-input gains and matched mean/peak supervision.

These trainable input factors are not certified unit conversions. Dimensionless
shape statistics and known operating/derived inputs stay unchanged. A shared SXR
response gain scales mean linearly and variance quadratically; this is a declared
common-response hypothesis, not a recovered instrument-calibration certificate.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from ..io23 import RAW23
from ..delayed_recurrent_memory_20261025.learning import batch

GROUPS = (
    ("density", {"ne_nG": 1}),
    ("CIII", {"CIII": 1}),
    ("Mir_VV", {"Mir_VV": 1}),
    ("Mir_amplitude", {"Mir_avg_amp": 1}),
    ("locked_amplitude", {"n=1 amplitude": 1}),
    ("radial_position", {"dx_a": 1}),
    ("vertical_position", {"dy_a": 1}),
    ("SXR_response", {"sxr_mean": 1, "sxr_var": 2}),
    ("XUV_response", {"xuv_var": 2}),
)


def gain_basis(scope):
    if scope not in ("density", "structured"):
        raise ValueError("explicit density or structured scope required")
    groups = GROUPS[:1] if scope == "density" else GROUPS
    result = torch.zeros((len(groups), 23), dtype=torch.float64)
    for i, (_, columns) in enumerate(groups):
        for name, power in columns.items():
            result[i, RAW23.index(name)] = power
    return result


def effective_gain(theta, prior, basis):
    if (
        theta.ndim != 1
        or prior.shape != (23,)
        or basis.shape != (len(theta), 23)
        or not torch.isfinite(theta).all()
        or not torch.isfinite(prior).all()
        or not torch.isfinite(basis).all()
        or (prior <= 0).any()
    ):
        raise ValueError(
            "finite parameter vector, positive prior and fixed flat23 basis required"
        )
    gain = prior.double() * torch.exp(theta.double() @ basis.double())
    if not torch.isfinite(gain).all() or (gain <= 0).any():
        raise FloatingPointError("unrepresentable positive input gain")
    return gain


def apply_gain(values, available, theta, prior, basis):
    if (
        values.ndim != 3
        or values.shape[-1] != 23
        or available.shape != values.shape
        or available.dtype != torch.bool
    ):
        raise ValueError("batched flat23 windows and boolean observation mask required")
    if not torch.isfinite(values[available]).all():
        raise ValueError("nonfinite observed input")
    gain = effective_gain(theta, prior, basis)
    converted = (values.double() * gain).to(values.dtype)
    if not torch.isfinite(converted[available]).all():
        raise FloatingPointError("unrepresentable mapped observed input")
    return torch.where(available, converted, values)


def logits_at(base, signal, ends, theta, prior, basis, device):
    x, m = batch(signal, ends, device)
    z = base(apply_gain(x, m, theta, prior, basis), m)["logit"]
    if z.ndim != 1 or not torch.isfinite(z).all():
        raise FloatingPointError("invalid training logits")
    return z


def backward_bag(
    base,
    signal,
    ends,
    disruptive,
    boundary,
    aggregation,
    theta,
    prior,
    basis,
    *,
    weight,
    device,
):
    ends = np.asarray(ends)
    if (
        ends.ndim != 1
        or not len(ends)
        or np.any(np.diff(ends) <= 0)
        or aggregation not in ("mean", "peak")
        or not math.isfinite(boundary)
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("ordered endpoints and valid objective required")
    best = -math.inf
    best_chunk = None
    best_offset = None
    total = 0.0
    for lo in range(0, len(ends), 32):
        chunk = ends[lo : lo + 32]
        # Both scan and replay use the gradient-enabled path and identical
        # batch shapes. Peak graphs are discarded, not accumulated per shot.
        with torch.enable_grad():
            z = logits_at(base, signal, chunk, theta, prior, basis, device)
            index = int(z.detach().argmax())
            maximum = float(z[index].detach())
            if maximum > best:
                best = maximum
                best_chunk = chunk.copy()
                best_offset = index
            if aggregation == "mean":
                loss = (
                    F.softplus(boundary - z if disruptive else z - boundary).mean()
                    * len(chunk)
                    / len(ends)
                )
                (weight * loss).backward()
                total += float(loss.detach())
        del z
    if aggregation == "peak":
        z = logits_at(base, signal, best_chunk, theta, prior, basis, device)
        if float(z[best_offset].detach()) != best:
            raise AssertionError("same-shape peak replay must be exact")
        loss = F.softplus(
            boundary - z[best_offset] if disruptive else z[best_offset] - boundary
        )
        (weight * loss).backward()
        total = float(loss.detach())
    return dict(
        candidates=len(ends),
        peak_end=int(best_chunk[best_offset]),
        peak_logit=best,
        loss=total,
        aggregation=aggregation,
    )
