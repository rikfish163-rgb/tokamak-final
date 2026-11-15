"""Supervision from explicitly supplied, already-ended EAST shots only."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..direct_fast_memory_20261104.memory import direct_target
from ..direct_fast_memory_20261104.model import levels


def ended_records(shots, truth, ids):
    ids = list(ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(shots) != set(ids)
        or set(truth) != set(ids)
    ):
        raise ValueError("exactly the supplied ended shots and HDF truth required")
    records = []
    pools = {g: {} for g in ("positive", "safe", "earlier")}
    for sid in ids:
        flag, down = truth[sid]
        if (
            type(flag) is not bool
            or (flag and (down is None or not math.isfinite(down)))
            or (not flag and down is not None)
        ):
            raise ValueError("HDF boolean and disruption-only finite time required")
        shot = shots[sid]
        if shot["shot"] != sid or shot["device"] != "EAST":
            raise ValueError("matching EAST shot required")
        y, valid = io23.labels_at(shot["times"], flag, down, 125.0)
        for q in range(135, len(y) - 8, 8):
            if not valid[q] or (
                flag
                and y[q]
                and round(float((down - shot["times"][q]) * 1000), 9) <= 30
            ):
                continue
            group = "safe" if not flag else ("positive" if y[q] else "earlier")
            pools[group].setdefault(sid, []).append(len(records))
            records.append(dict(shot=sid, query=q, y=float(y[q]), group=group))
    return records, pools


def training_batch(model, shots, selected, *, device):
    xs, masks, futures, future_masks = [], [], [], []
    for row in selected:
        sid, q = row["shot"], row["query"]
        shot = shots[sid]
        if q + 8 >= len(shot["values"]):
            raise ValueError("eight real future samples required in an ended shot")
        x, a = simple_data.window(shot, q, 128)
        xs.append(x)
        masks.append(a)
        futures.append(shot["values"][q + 1 : q + 9])
        future_masks.append(shot["available"][q + 1 : q + 9])
    x = torch.as_tensor(np.stack(xs), device=device)
    a = torch.as_tensor(np.stack(masks), device=device)
    future = torch.as_tensor(np.stack(futures), device=device)
    mask = torch.as_tensor(np.stack(future_masks), device=device)
    with torch.no_grad():
        target, valid = direct_target(
            levels(model.clone, x, a), a, levels(model.clone, future, mask), mask
        )
    return (
        x,
        a,
        target,
        valid,
        torch.tensor([r["y"] for r in selected], device=device),
    )


def train_schedule(model, shots, records, schedule, *, threshold, device, lr):
    if (
        not records
        or not schedule
        or not 0 < threshold < 1
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError(
            "nonempty ended training and a fixed valid threshold/lr required"
        )
    scope = (*model.neural_parameters(), *model.decoder_parameters())
    allowed = {id(p) for p in scope}
    for p in model.parameters():
        p.requires_grad_(id(p) in allowed)
        p.grad = None
    model.eval()
    optimizer = torch.optim.Adam(scope, lr=lr)
    tau = math.log(threshold / (1 - threshold))
    history = []
    for draw in schedule:
        selected = [records[i] for i in draw["indices"]]
        x, a, target, mask, y = training_batch(model, shots, selected, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, a)
        risk = F.binary_cross_entropy_with_logits(output["logit"] - tau, y)
        diagnostic = (output["diagnostic"] - target).square().masked_fill(
            ~mask, 0
        ).sum() / mask.sum().clamp_min(1)
        loss = risk + 0.1 * diagnostic
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite joint risk/diagnostic training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(scope, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            dict(
                step=draw["step"],
                loss=float(loss.detach()),
                risk_loss=float(risk.detach()),
                diagnostic_loss=float(diagnostic.detach()),
                gradient_norm=float(norm),
            )
        )
        if draw["step"] % 128 == 0:
            print(dict(stage="joint_lora_supervision", **history[-1]), flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    return history


def eos_schedule(current, previous, *, seed, steps=8, batch_size=64):
    """Half current ended shot, half earlier replay; no fast-state promotion.

    Within each half use exactly 1/2 positives,1/4 safe,1/4 earlier if all
    groups occur; 1/2 positives,1/2 earlier for a disruptive current shot;
    or all records from its only observed group. Empty histories are errors.
    """
    if not current or not previous or steps < 1 or batch_size < 8 or batch_size % 8:
        raise ValueError(
            "current and previous records and divisible-by-eight budget required"
        )
    if {r["shot"] for r in current} & {r["shot"] for r in previous}:
        raise ValueError("current shot must not already be in replay")
    records = list(current) + list(previous)
    rng = np.random.default_rng(seed)

    def draw_half(rows, offset):
        pools = {}
        for i, r in enumerate(rows):
            pools.setdefault(r["group"], {}).setdefault(r["shot"], []).append(
                offset + i
            )
        groups = set(pools)
        half = batch_size // 2
        if groups == {"positive", "safe", "earlier"}:
            counts = {"positive": half // 2, "safe": half // 4, "earlier": half // 4}
        elif groups == {"positive", "earlier"}:
            counts = {"positive": half // 2, "earlier": half // 2}
        elif len(groups) == 1:
            counts = {next(iter(groups)): half}
        else:
            raise ValueError(
                "unexpected incomplete replay strata; declare a new sampling contract"
            )
        chosen = []
        for group, count in counts.items():
            members = sorted(pools[group])
            for sid in rng.choice(members, size=count):
                chosen.append(int(rng.choice(pools[group][int(sid)])))
        return chosen

    schedule = []
    for step in range(1, steps + 1):
        selected = draw_half(current, 0) + draw_half(previous, len(current))
        rng.shuffle(selected)
        schedule.append(dict(step=step, indices=selected))
    return records, schedule
