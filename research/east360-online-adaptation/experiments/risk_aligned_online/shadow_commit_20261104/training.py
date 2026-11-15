"""Matched outer learning from the same already-ended twenty shots."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..direct_fast_memory_20261104.model import Injection, predict


def records_and_pools(shots, truth, ids):
    if (
        len(ids) != 20
        or len(set(ids)) != 20
        or set(shots) != set(ids)
        or set(truth) != set(ids)
    ):
        raise ValueError("exactly the same twenty ended shots required")
    records = []
    pools = {g: {} for g in ("positive", "safe", "earlier")}
    for sid in ids:
        flag, down = truth[sid]
        if type(flag) is not bool or (
            flag and (down is None or not math.isfinite(down))
        ):
            raise ValueError("HDF boolean and finite disruption time required")
        signal = shots[sid]
        y, valid = io23.labels_at(signal["times"], flag, down, 125.0)
        for q in range(135, len(y), 8):
            if not valid[q] or (
                flag and y[q] and (down - signal["times"][q]) * 1000 <= 30.0
            ):
                continue
            group = "safe" if not flag else ("positive" if y[q] else "earlier")
            pools[group].setdefault(sid, []).append(len(records))
            records.append(dict(shot=sid, query=q, y=float(y[q]), group=group))
    if any(not p for p in pools.values()):
        raise ValueError("all operational sampling strata must be supported")
    return records, pools


def fit_connection(
    base, shots, records, schedule, innovations, *, threshold, device, name
):
    if set(innovations) != set(shots) or not 0 < threshold < 1 or not schedule:
        raise ValueError("same ended prefix, fixed threshold, and schedule required")
    for sid, signal in shots.items():
        if innovations[sid].shape != (len(signal["values"]), 168):
            raise ValueError("one causal innovation per ended-prefix point required")
    connection = Injection().to(device)
    optimizer = torch.optim.Adam(connection.parameters(), lr=0.01)
    tau = math.log(threshold / (1 - threshold))
    history = []
    for draw in schedule:
        selected = [records[i] for i in draw["indices"]]
        windows = [
            simple_data.window(shots[r["shot"]], r["query"], 128) for r in selected
        ]
        x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        a = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        signal = torch.stack([innovations[r["shot"]][r["query"]] for r in selected])
        labels = torch.tensor([r["y"] for r in selected], device=device)
        optimizer.zero_grad(set_to_none=True)
        logit = predict(base, connection, x, a, signal)
        bce = F.binary_cross_entropy_with_logits(logit - tau, labels)
        penalty = connection(signal).square().mean()
        loss = bce + 0.01 * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite matched connection fit")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            connection.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            dict(
                step=draw["step"],
                margin_bce=float(bce.detach()),
                penalty=float(penalty.detach()),
                loss=float(loss.detach()),
                gradient_norm=float(norm),
            )
        )
        if draw["step"] % 128 == 0:
            print(
                dict(stage="shadow_connection_fit", policy=name, **history[-1]),
                flush=True,
            )
    optimizer.zero_grad(set_to_none=True)
    connection.eval().requires_grad_(False)
    return connection, history
