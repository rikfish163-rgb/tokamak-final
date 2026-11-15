"""Offline ended-shot task selectors, never used by online inference."""

import math
from numbers import Integral

import numpy as np
import torch


def _arrays(times, logits, disruptive, down):
    t, z = np.asarray(times, np.float64), np.asarray(logits, np.float64)
    if (
        t.ndim != 1
        or not len(t)
        or z.shape != t.shape
        or not np.isfinite(t).all()
        or not np.isfinite(z).all()
        or not isinstance(disruptive, (bool, np.bool_))
        or (disruptive and (down is None or not math.isfinite(float(down))))
    ):
        raise ValueError("finite matching times/logits and revealed outcome required")
    return t, z


def select_critical(times, logits, disruptive, down):
    t, z = _arrays(times, logits, disruptive, down)
    eligible = (
        np.flatnonzero(np.round((down - t) * 1000, 9) > 30)
        if disruptive
        else np.arange(len(t))
    )
    if not len(eligible):
        raise ValueError("no timely endpoint")
    return int(eligible[np.argmax(z[eligible])])


def last_update(query):
    if isinstance(query, bool) or not isinstance(query, Integral) or query < 0:
        raise ValueError("nonnegative integer query required")
    return 159 + ((int(query) - 160) // 128) * 128 if query >= 160 else None


def task_label(times, index, disruptive, down, kind):
    if kind not in ("point", "operational"):
        raise ValueError("known outer task required")
    t, _ = _arrays(times, np.zeros_like(times), disruptive, down)
    if (
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or not 0 <= index < len(t)
    ):
        raise ValueError("valid endpoint required")
    if not disruptive:
        return 0
    if kind == "operational":
        return 1
    lead = (float(down) - t[index]) * 1000
    if lead < -1e-6:
        raise ValueError("post-terminal point is not a training label")
    return int(lead <= 125 + 1e-6)


def margin_loss(logit, threshold, label):
    if (
        not math.isfinite(float(threshold))
        or not 0 < threshold < 1
        or label not in (0, 1)
    ):
        raise ValueError("finite probability threshold and binary label required")
    margin = logit.double() - math.log(threshold / (1 - threshold))
    signed = (1 - 2 * label) * margin
    return torch.logaddexp(torch.zeros_like(signed), signed).mean()
