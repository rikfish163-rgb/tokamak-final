"""Retrospective matched-position comparison of attainable score movement."""

import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from . import simple_data
from .commissioning_intra_20261110 import (
    HORIZON, START, STARTUP, configure, contrast, fast_step, load_frozen, prepare_actual,
)
from .history_gradient_metric_20261102.run import read, write
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .two_band_fast_20261110 import banded_forward


def peak_positions(scores, valid):
    candidates = np.flatnonzero(valid)
    candidates = candidates[np.argsort(scores[candidates], kind="stable")[::-1]]
    selected = []
    for q in candidates:
        if all(abs(int(q) - old) >= 16 for old in selected):
            selected.append(int(q))
        if len(selected) == 4:
            break
    return selected


def measure(base, coef, windows, masks, motion, *, banded):
    """Known-outcome intervention at selected windows, not an online updater."""
    out = []
    hidden = None

    def capture(_module, args):
        nonlocal hidden
        hidden = args[0]

    forward = partial(banded_forward, base) if banded else base
    hook = base.risk_head.register_forward_pre_hook(capture)
    try:
        for offset in range(0, len(windows), 128):
            sl = slice(offset, offset + 128)
            x, m, change = (torch.as_tensor(z[sl], device=coef.device)
                            for z in (windows, masks, motion))
            initial = (coef.new_zeros(len(x), 2, 2 * base.cfg.dim) if banded else
                       base.initial_fast(len(x)).detach())
            state = initial.clone()

            def logits(parameters):
                forward(x, m, parameters)
                return torch.cat((hidden, change), dim=-1) @ coef[:-1] + coef[-1]

            with torch.no_grad():
                zero = logits(initial).sigmoid()
            best = zero.clone()
            for _ in range(16):
                variable = state.detach().requires_grad_(True)
                (gradient,) = torch.autograd.grad(logits(variable).sum(), variable)
                with torch.no_grad():
                    state = fast_step(state, initial, -gradient, step=.20)
                    best = torch.maximum(best, logits(state).sigmoid())
            out.append(torch.stack((zero, best), dim=-1).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(out)


def run(reference, output):
    start = time.monotonic()
    configure()
    inputs, coef, _, cutoffs, bound = load_frozen()
    base, _ = load_parent(DEFAULT_PARENT)
    base.cuda()
    source_state = state_digest(base)
    protocol = read(reference / "protocol.json")
    rows = read(reference / "per_shot.json")["0.2"]["static"]
    if (protocol["order"] != inputs["ids"] or protocol["suffix_thresholds"] != cutoffs
            or any(protocol[k] != v for k, v in bound.items())):
        raise ValueError("the same previously sealed static parent is required")
    groups, before, windows, masks, motion = [], [], [], [], []
    for i, (sid, row) in enumerate(zip(inputs["ids"], rows, strict=True), 1):
        if i <= STARTUP or (row["timely"] if row["disruptive"] else row["alarm"]):
            continue
        if sid != row["shot"]:
            raise ValueError("score order changed")
        with np.load(reference / "shots" / f"{i:04d}_{sid}.npz", allow_pickle=False) as arr:
            scores, times = arr["static"], arr["times"]
        valid = np.arange(len(scores)) >= START + HORIZON + 1
        if row["disruptive"]:
            valid &= np.round((row["down_s"] - times) * 1000., 9) > 30.
        peaks = peak_positions(scores, valid)
        if not peaks:
            continue
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        np.testing.assert_array_equal(raw["times"], times)
        signal = prepare_actual(raw, i, inputs["mapping"])
        features = contrast(signal["values"], signal["available"])
        for q in peaks:
            x, m = simple_data.window(signal, q, base.cfg.window)
            windows.append(x)
            masks.append(m)
            motion.append(features[q])
            before.append(float(scores[q]))
            groups.append((sid, q, bool(row["disruptive"])))
    args = tuple(np.stack(v) for v in (windows, masks, motion))
    measurements = {name: measure(base, coef, *args, banded=banded) for name, banded in
                    (("uniform", False), ("two_bands", True))}
    threshold = cutoffs["0.2"]
    selected = {}
    for j, (sid, q, disruptive) in enumerate(groups):
        row = selected.setdefault(sid, dict(shot=sid, disruptive=disruptive, positions=[]))
        row["positions"].append(dict(
            endpoint=q, baseline=before[j],
            **{name: float(table[j, 1]) for name, table in measurements.items()},
        ))
    result = {}
    for name, table in measurements.items():
        np.testing.assert_allclose(table[:, 0], before, atol=2e-6, rtol=0)
        for disruptive, key in ((True, "disruptive_missed"), (False, "safe_not_alarmed")):
            members = [r for r in selected.values() if r["disruptive"] == disruptive]
            gaps, movement, crosses = [], [], []
            for r in members:
                initial = max(v["baseline"] for v in r["positions"])
                best = max(v[name] for v in r["positions"])
                gaps.append(threshold - initial)
                movement.append(best - initial)
                if best >= threshold:
                    crosses.append(r["shot"])
            result.setdefault(name, {})[key] = dict(
                n=len(members), crossing_ids=crosses,
                median_gap=float(np.median(gaps)), median_score_increase=float(np.median(movement)),
            )
    if state_digest(base) != source_state:
        raise AssertionError("source parameters changed")
    write(output, dict(
        status="RETROSPECTIVE_SAMPLED_CAPACITY_NOT_ONLINE", threshold=threshold,
        steps=16, step_size=.20, positions=len(groups), static_reference=str(reference),
        result=result, selected=list(selected.values()), seconds=time.monotonic() - start,
    ))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reference.resolve(), args.output.resolve())
