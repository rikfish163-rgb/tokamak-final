"""TRAIN-only coverage and finite action probe for subthreshold precursors."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import run_whole_block_update_20261106 as run
from verify_confirmation_raw_replay_20261105 import Hashes


def candidate_rows(times, logits, slots, support, down, horizon, tau):
    """Canonical positives before any native crossing; never a live selector."""
    t, z, k, s = (
        np.asarray(times, dtype=np.float64),
        np.asarray(logits, dtype=np.float64),
        np.asarray(slots),
        np.asarray(support),
    )
    if (
        t.ndim != 1
        or not t.size
        or t.shape != z.shape
        or (np.diff(t) <= 0).any()
        or not np.isfinite(t).all()
        or not np.isfinite(z).all()
        or not np.isfinite([down, horizon, tau]).all()
        or horizon <= 10
    ):
        raise ValueError("finite ordered ended source clock and logits required")
    if (
        k.shape != t.shape
        or k.dtype.kind not in "iu"
        or s.ndim != 1
        or s.dtype != bool
        or (k < -1).any()
        or (k >= len(s)).any()
    ):
        raise ValueError("clock-aligned integer slots and Boolean support required")
    lead_raw = (down - t) * 1000.0
    if not np.isfinite(lead_raw).all():
        raise ValueError("finite ended-source lead clock required")
    lead = lead_raw.round(9)
    alarm = np.flatnonzero((lead_raw >= -1e-6) & (z >= tau))
    first = int(alarm[0]) if len(alarm) else None
    timely = bool(first is not None and lead[first] > 10)
    eligible = (lead > 10) & (lead <= horizon + 1e-6) & (z < tau)
    if first is not None:
        eligible[first:] = False
    rows = [
        (int(i), q)
        for i in np.flatnonzero(s)
        if len(q := np.flatnonzero((k == i) & eligible))
    ]
    return first, timely, rows


def collect(base, cfg, inventory, h):
    census = r.read(run.prior.SOURCE_CENSUS)["source"]
    if [q["shot"] for q in census] != cfg["roles"]["train"]:
        raise ValueError("TRAIN-only source census mismatch")
    adv = r.simple_data.advanced_times()
    tapes, masks, rows, case, start = [], [], [], {}, 0
    for rec, original in zip(inventory, census, strict=True):
        sid = rec["shot"]
        if h(r.io23.path_for("J-TEXT", sid)) != rec["sha256"]:
            raise ValueError("TRAIN raw bytes changed")
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        if flag != original["disruptive"] or rec["sha256"] != original["sha256"]:
            raise ValueError("TRAIN canonical source census changed")
        if not flag:
            continue
        raw = r.simple_data.read_shot("J-TEXT", sid)
        f = run.prior.signal_schedule(base, raw)
        x, a = run.prior.views(base, raw)
        z = run.prior.scan(base, x, a).cpu().numpy()
        slots = run.old.write_slots(len(z), f["maturity"])
        horizon = r.io23.effective_horizon_ms(sid, adv)[0]
        first, timely, windows = candidate_rows(
            raw["times"], z, slots, f["support"], down, horizon, run.TAU
        )
        case[sid] = dict(
            shot=sid,
            native_first=first,
            native_timely=timely,
            native_lead_ms=round(float((down - raw["times"][first]) * 1000), 9)
            if first is not None
            else None,
            horizon_ms=horizon,
        )
        if not windows:
            continue
        for slot, ix in windows:
            valid = np.arange(run.old.BLOCK) < len(ix)
            query = np.full(run.old.BLOCK, start + ix[0], dtype=np.int64)
            query[valid] = start + ix
            native = np.zeros(run.old.BLOCK, dtype=np.float32)
            native[valid] = z[ix]
            rows.append(
                dict(
                    shot=sid,
                    role=run.loss.RECOVER,
                    ox=start + f["origin"][slot],
                    mx=start + f["maturity"][slot],
                    query=query,
                    mask=valid,
                    native=native,
                    delta=f["delta"][slot],
                    valid=f["valid"][slot],
                )
            )
        tapes.append(np.pad(raw["values"], ((run.CONTEXT - 1, 0), (0, 0))))
        masks.append(np.pad(raw["available"], ((run.CONTEXT - 1, 0), (0, 0))))
        start += len(raw["times"]) + run.CONTEXT - 1
    if not rows:
        raise ValueError("no subthreshold TRAIN episodes have causal support")
    dev = base.center.device
    data = {k: run.field(np.asarray([q[k] for q in rows]), dev) for k in rows[0]}
    for k in ("ox", "mx", "query"):
        data[k] += run.CONTEXT - 1
    tape = dict(
        values=run.field(np.concatenate(tapes), dev),
        available=run.field(np.concatenate(masks), dev),
    )
    return tape, data, case


def probe(root, out):
    h, start = Hashes(), time.monotonic()
    r.configure()
    cfg, inventory = run.source_inputs(h)
    train_ref = root / "train_feasibility.json"
    ref = r.read(train_ref)
    source_protocol = root / "protocol.json"
    fit = r.read(root / "fit_completed.json")
    if (
        out.exists()
        or h(train_ref) != fit["train_feasibility_sha256"]
        or h(source_protocol) != fit["protocol_sha256"]
        or r.read(source_protocol)["roles"] != cfg["roles"]
    ):
        raise ValueError(
            "previous TRAIN-only feasibility result or output path differs"
        )
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        parent["threshold"] != r.THRESHOLD
        or r.state_digest(base) != cfg["source_state_sha256"]
    ):
        raise ValueError("qualified source identity required")
    base.cuda()
    tape, data, case = collect(base, cfg, inventory, h)
    old = [run.prior.load_fit(run.PREVIOUS / f"fit_{s}.npz", base) for s in run.SEEDS]
    banks = [run.bank_on_tape(base, tape, data, f) for f in old]
    common = torch.stack([q[2] for q in banks]).all(0)
    groups, results = ("timely_before_first", "miss_or_late"), {}
    for s, (_, bank, _) in zip(run.SEEDS, banks, strict=True):
        feasible = run.train_feasibility(base, tape, data, bank, common)
        probes = feasible["probes"]
        subsets = (
            [q for q in probes if case[q["shot"]]["native_timely"] is v]
            for v in (True, False)
        )
        values = {}
        for label, rows in zip(groups, subsets, strict=True):
            n = len({q["shot"] for q in rows})
            crossing = sorted(
                {q["shot"] for q in rows if q["before"] < run.TAU <= q["best"]}
            )
            if label == "miss_or_late" and (
                n != ref["seeds"][str(s)]["n_shots"]
                or crossing != ref["seeds"][str(s)]["crossing_shots"]
            ):
                raise ValueError("original missed-source stratum did not reproduce")
            values[label] = dict(n_shots=n, n_rows=len(rows), crossing_shots=crossing)
        results[str(s)] = dict(groups=values, probes=probes)
        print(s, values, flush=True)
    input_hashes = {
        str(p.resolve()): h(p)
        for p in (
            train_ref,
            source_protocol,
            root / "fit_completed.json",
            Path(__file__),
            Path(run.__file__),
        )
    }
    h.check()
    r.write_new(
        out,
        dict(
            status="TRAIN_ONLY_PRE_FIRST_CAPACITY",
            fitted_new_models=False,
            val_test_east_reads=0,
            input_sha256=input_hashes,
            source_disruptive=list(case.values()),
            seeds=results,
            threshold_logit=run.TAU,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_result", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    probe(a.source_result, a.output)
