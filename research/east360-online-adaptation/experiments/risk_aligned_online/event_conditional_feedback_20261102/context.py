"""Frozen, past-only soft context; A-F are not ordered hazard stages."""

import numpy as np
import torch

from .. import io23
from ..delayed_recurrent_memory_20261025.learning import batch
from ..event_prefix_probe import causal_summary


def select_variation_features(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 115 or not np.isfinite(raw).all():
        raise ValueError("finite 115-column feature rows required")
    selected = np.zeros_like(raw)
    selected[:, 46:] = raw[:, 46:]
    selected[:, [23 * b + io23.RAW23.index("MNM") for b in range(5)]] = 0
    return selected


def probabilities(raw, fitted):
    x = select_variation_features(raw).astype(np.float64)
    mean, scale, weight, bias = [
        np.asarray(fitted[k], np.float64) for k in ("mean", "scale", "weight", "bias")
    ]
    if (
        mean.shape != (115,)
        or scale.shape != (115,)
        or weight.shape != (6, 115)
        or bias.shape != (6,)
    ):
        raise ValueError("fitted115->6 context required")
    if (
        not all(np.isfinite(v).all() for v in (mean, scale, weight, bias))
        or not (scale > 0).all()
    ):
        raise ValueError("finite head and positive source scales required")
    score = ((x - mean) / scale).astype(np.float32) @ weight.astype(
        np.float32
    ).T + bias.astype(np.float32)
    if not np.isfinite(score).all():
        raise ValueError("nonfinite frozen context score")
    score = score.astype(np.float64)
    score -= score.max(1, keepdims=True)
    numerator = np.exp(score)
    return (numerator / numerator.sum(1, keepdims=True)).astype(np.float32)


@torch.no_grad()
def at_origin(model, signal, origin, fitted):
    if (
        set(signal) != {"values", "available"}
        or type(origin) is not int
        or origin < 127
    ):
        raise ValueError("full past-only signal window required")
    x, mask = batch(signal, np.asarray([origin]), model.base.center.device)
    mask = mask.clone()
    mask[:, :, io23.RAW23.index("MNM")] = False
    raw = causal_summary(x, mask, model.base.center, model.base.scale).cpu().numpy()
    result = probabilities(raw, fitted)[0]
    return torch.as_tensor(result, device=model.base.center.device)
