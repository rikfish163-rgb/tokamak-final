"""Fit independent target clones from the same already revealed prefix."""

from copy import deepcopy
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data


def make_readout(base, mode):
    if mode not in ("bias", "head") or (
        base.risk_head.in_features,
        base.risk_head.out_features,
    ) != (64, 1):
        raise ValueError("qualified risk head and declared readout mode required")
    result = deepcopy(base).eval().requires_grad_(False)
    result.risk_head.bias.requires_grad_(True)
    if mode == "head":
        result.risk_head.weight.requires_grad_(True)
    return result


def batch(shots, records, indices, *, device):
    if not indices:
        raise ValueError("nonempty draw required")
    windows, labels = [], []
    for i in indices:
        row = records[i]
        if row["shot"] not in shots or row["label"] not in (0, 1):
            raise ValueError("training draws must remain within the revealed prefix")
        windows.append(simple_data.window(shots[row["shot"]], row["query"], 128))
        labels.append(row["label"])
    return (
        torch.as_tensor(np.stack([w[0] for w in windows]), device=device),
        torch.as_tensor(np.stack([w[1] for w in windows]), device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def fit_model(model, shots, records, schedule, *, device, mode):
    if not schedule or [r["step"] for r in schedule] != list(
        range(1, len(schedule) + 1)
    ):
        raise ValueError("fixed consecutive fitting schedule required")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("explicit trainable clone parameters required")
    initial = [p.detach().clone() for p in params]
    total = sum(p.numel() for p in params)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    model.eval()
    history = []
    started = time.monotonic()
    # Model fitting consumes only the in-memory, ended prefix. The real runner
    # fills these arrays after prediction seals and HDF outcome revelation.
    with (
        patch.object(
            io23,
            "read_signals",
            side_effect=RuntimeError("no new signals during prefix fit"),
        ),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("no new outcomes during prefix fit"),
        ),
    ):
        for draw in schedule:
            x, a, y = batch(shots, records, draw["indices"], device=device)
            optimizer.zero_grad(set_to_none=True)
            z = model(x, a)["logit"]
            bce = F.binary_cross_entropy_with_logits(z, y)
            penalty = (
                sum(
                    (p - anchor).square().sum()
                    for p, anchor in zip(params, initial, strict=True)
                )
                / total
            )
            loss = bce + 0.01 * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite prefix fitting objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                dict(
                    step=draw["step"],
                    loss=float(loss.detach()),
                    bce=float(bce.detach()),
                    penalty=float(penalty.detach()),
                    gradient_norm=float(norm),
                )
            )
            if draw["step"] % 128 == 0:
                print(
                    dict(
                        stage="cold20_fit",
                        mode=mode,
                        **history[-1],
                        seconds=time.monotonic() - started,
                    ),
                    flush=True,
                )
    optimizer.zero_grad(set_to_none=True)
    model.requires_grad_(False).eval()
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("fit left parameter gradients attached")
    return history
