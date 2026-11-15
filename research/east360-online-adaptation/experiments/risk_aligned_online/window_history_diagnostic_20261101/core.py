"""Counterfactual views of an already arrived, padded input window."""

import numpy as np
import torch

VIEWS = ("original", "reverse_past", "last_observed", "mean_observed")


def history_view(values, available, ends, mode):
    if (
        values.ndim != 3
        or values.shape[1:] != (128, 23)
        or available.shape != values.shape
        or available.dtype != torch.bool
        or len(ends) != len(values)
        or any(int(e) != e or e < 0 for e in ends)
        or mode not in VIEWS
    ):
        raise ValueError("declared view of batched causal W128/flat23 windows required")
    if mode == "original":
        return values.clone(), available.clone()
    clock = torch.arange(128, device=values.device)[None, :, None]
    if mode == "reverse_past":
        # Only past real rows move. The current row and left padding stay.
        end = torch.as_tensor(ends, device=values.device)[:, None, None]
        start = (127 - end).clamp_min(0)
        source = torch.where(
            (clock >= start) & (clock < 127), start + 126 - clock, clock
        )
        indices = source.expand(-1, -1, 23).long()
        return values.gather(1, indices), available.gather(1, indices)
    if mode == "last_observed":
        latest = torch.where(available, clock, -1).amax(1).clamp_min(0)
        value = values.gather(1, latest[:, None]).squeeze(1)
    else:
        total = torch.where(available, values.double(), 0.0).sum(1)
        value = (total / available.sum(1).clamp_min(1)).to(values.dtype)
    return torch.where(available, value[:, None], values), available.clone()


def first_alarm_index(probability, threshold):
    probability = np.asarray(probability)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("finite one-dimensional probability curve required")
    indices = np.flatnonzero(probability >= threshold)
    return int(indices[0]) if len(indices) else None


def timing_bin(index):
    if index is None:
        return "no_alarm"
    if index < 0:
        raise ValueError("nonnegative observation index required")
    return "before16" if index < 16 else "16to127" if index < 128 else "from128"
