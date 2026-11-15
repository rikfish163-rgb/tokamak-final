"""Offline finite capacity and exact first-alarm witnesses on the six TRAIN misses."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

import run_h25_direction_mechanism_20261106 as m
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


def match_late(rows, maturity, positive, support):
    expected = {
        int(k): (int(k) + 1 + np.flatnonzero(p)).tolist()
        for k, p, s in zip(maturity, positive, support, strict=True)
        if s and p.any()
    }
    actual = {r["maturity"]: r["eligible"] for _, r in rows}
    if actual != expected or len(rows) != len(actual):
        raise ValueError(
            "stored late rows differ from the complete canonical positive windows"
        )
    return {r["maturity"]: i for i, r in rows}


def pick_trial(trials, native_peak):
    best = None
    for trial in trials:
        for arm in ("projected", "optimized"):
            v = trial[arm]
            if v is not None and v["peak"] > native_peak:
                native_peak = v["peak"]
                best = {**v, "target": trial["target"], "source": arm}
    return native_peak, best


def gate(summary, reach):
    pos = summary["positive"]
    support = pos["n_windows"] > 0 and 10 * pos["n_supported"] >= 7 * pos["n_windows"]
    counts = Counter()
    for base, real, reverse in zip(
        reach["old"], reach["old_true"], reach["old_reverse"], strict=True
    ):
        counts.update(set(real) - (set(base) | set(reverse)))
    new = sorted(sid for sid, n in counts.items() if n >= 2)
    capture = pos.get("paired_capture_gain", {}).get("real_minus_reverse")
    peak = pos["raw_peak_delta"]
    peak_diff = (
        float((np.asarray(peak)[:, 2] - np.asarray(peak)[:, 3]).mean())
        if peak is not None
        else None
    )
    harm = {}
    for role in ("safe", "protect"):
        counts = np.asarray(summary[role]["raw_harm_by_seed"], dtype=np.int64)
        harm[role] = (
            int((counts[:, 2] - counts[:, 3]).sum()) if counts.ndim == 2 else None
        )
    broad = (
        capture is not None
        and capture >= 0.03
        and peak_diff is not None
        and peak_diff > 0
        and all(v is not None and v <= 0 for v in harm.values())
    )
    return dict(
        promote=bool(support and (new or broad)),
        common_support=bool(support),
        new_shot_two_seeds=new,
        broad_gate=bool(broad),
        capture_difference=capture,
        danger_peak_difference=peak_diff,
        harm_difference=harm,
    )


def load_receipts(root, h):
    cfg, done = m.r.read(root / "protocol.json"), m.r.read(root / "mechanism.json")
    npz = root / "geometry.npz"
    if (
        cfg["status"] != "LOCKED_H25_MECHANISM"
        or done["status"] != "MEASURED_TRAIN_ONLY_H25_GEOMETRY"
        or h(root / "protocol.json") != done["protocol_sha256"]
        or h(npz) != done["geometry_sha256"]
        or tuple(cfg["seeds"]) != m.SEEDS
        or tuple(cfg["modes"]) != m.MODES
        or cfg["threshold"] != m.r.THRESHOLD
        or any(x["val_test_east_reads"] for x in (cfg, done))
        or not all(h(p) == sha for p, sha in cfg["inputs"].items())
    ):
        raise ValueError("frozen H25 TRAIN geometry and exact bound artifacts required")
    with np.load(npz, allow_pickle=False) as v:
        data = {
            k: v[k] for k in ("old", "true", "reverse", "past", "future", "support")
        }
    if (
        any(
            len(v) != len(done["cases"]) or not np.isfinite(v).all()
            for v in data.values()
        )
        or data["support"].dtype != bool
        or data["support"].tolist() != [q["supported"] for q in done["cases"]]
    ):
        raise ValueError("sealed gradients do not align with the selected windows")
    return cfg, done, data


@torch.no_grad()
def replay_alarm(base, x, a, native, times, down, maturity, eligible, pick):
    whole = np.arange(maturity + 1, min(maturity + 33, len(native)))
    state = x.new_tensor(pick["state"])
    past = m.bank.tr.reached_indices(np.asarray([maturity]))[0]
    check = m.mix.verify_state(
        base,
        x[past].contiguous(),
        a[past].contiguous(),
        x[eligible].contiguous(),
        a[eligible].contiguous(),
        native[past],
        state,
    )
    if (
        check is None
        or check["peak"] < m.TAU
        or abs(check["peak"] - pick["peak"]) > 5e-5
    ):
        raise ValueError(
            "saved feasible state does not reproduce its canonical crossing"
        )
    trace = native.copy()
    trace[whole] = (
        m.bank.old.risk(
            base,
            x[whole].contiguous(),
            a[whole].contiguous(),
            state.expand(len(whole), -1),
        )
        .cpu()
        .numpy()
    )
    alarm = first(times, trace, True, down, m.TAU)
    if (
        not alarm["timely"]
        or alarm["first"] not in whole
        or not np.array_equal(trace[: maturity + 1], native[: maturity + 1])
        or float(trace[eligible].max()) < m.TAU
    ):
        raise ValueError(
            "full causal-time stitch did not produce a new timely first alarm"
        )
    return dict(
        **alarm,
        canonical_peak=float(trace[eligible].max()),
        checked_drift=check["reached_drift"],
    )


def replay_shot(base, diag, raw, down, horizon, cases, data):
    x, a = m.bank.prior.views(base, raw)
    native = m.bank.prior.scan(base, x, a).cpu().numpy()
    if first(raw["times"], native, True, down, m.TAU)["timely"]:
        raise ValueError("raw source is no longer late in the native model")
    schedule = m.bank.prior.signal_schedule(base, raw)
    clock = schedule["maturity"]
    p, _, _ = m.bank.fwd.teaching_masks(
        raw["times"], native, clock, True, down, horizon, m.TAU
    )
    matched = match_late(cases, clock, p, schedule["support"])
    ms = np.asarray(list(matched), dtype=np.int64)
    if not len(ms):
        raise ValueError("at least one actionable original late window expected")
    st = m.state_for(base, diag, x, a, raw, ms, schedule)
    reached = m.bank.tr.reached_indices(ms)
    before = m.mix.gradient(
        base, x[reached.ravel()].contiguous(), a[reached.ravel()].contiguous()
    ).reshape(len(ms), 3, 128)
    lookup = dict(cases)
    rows, checks = [], []
    for j, (maturity, i) in enumerate(matched.items()):
        case = lookup[i]
        eligible = np.asarray(case["eligible"], dtype=np.int64)
        top = eligible[np.argsort(-native[eligible], kind="stable")[:3]]
        future = m.mix.gradient(base, x[top].contiguous(), a[top].contiguous())
        if top[0] != case["target"]:
            raise ValueError("canonical leading target changed")
        u = {k: st[k][j] for k in ("old", "true", "reverse")}
        native_peak = float(native[eligible].max())
        if abs(native_peak - case["source_peak"]) > 5e-5:
            raise ValueError("raw CPU and frozen GPU native scores differ")
        check = {
            k: float(np.abs(v - data[k][i]).max())
            for k, v in dict(**u, past=before[j], future=future[0]).items()
        }
        support = bool(st["old_ok"][j] and st["groups"][j].sum() >= 2)
        if max(check.values()) > 1e-4 or support != case["supported"]:
            raise ValueError("CPU VJP differs from the geometry receipt")
        checks.append(dict(maturity=maturity, **check))
        for seed, b in zip(m.SEEDS, u["old"], strict=True):
            spaces, rank = m.spaces(b, u["true"], u["reverse"])
            if spaces[2].shape[1] != spaces[3].shape[1]:
                raise ValueError("true and reversed spaces must have the same rank")
            by_arm = {}
            for name, q in zip(m.MODES, spaces, strict=True):
                trials = (
                    [
                        m.mix.candidates(
                            base, x, a, reached[j], eligible, q, w, before[j], g, native
                        )
                        for w, g in zip(top, future, strict=True)
                    ]
                    if support and q.shape[1]
                    else []
                )
                peak, best = pick_trial(trials, native_peak)
                witness = None
                if peak >= m.TAU:
                    if best is None:
                        raise ValueError(
                            "native late positive window cannot already be timely"
                        )
                    witness = replay_alarm(
                        base, x, a, native, raw["times"], down, maturity, eligible, best
                    )
                by_arm[name] = dict(
                    best=best,
                    peak=peak,
                    witness=witness,
                    trials=trials,
                    rank=q.shape[1],
                )
            rows.append(
                dict(
                    shot=case["shot"],
                    maturity=maturity,
                    seed=seed,
                    extra_rank=rank,
                    arms=by_arm,
                )
            )
    return rows, checks


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    h, t = Hashes(), time.monotonic()
    for p in (Path(__file__), root / "mechanism.json", m.PRIOR / "fit_completed.json"):
        h(p)
    cfg, evidence, data = load_receipts(root, h)
    prior, originals = m.bank.src.source_inputs(h)
    index = {q["shot"]: q["sha256"] for q in originals}
    ids = cfg["late"]
    lock = m.r.read(m.PRIOR / "train_crossfit.json")["geometry"]["non_timely_shots"]
    if (
        len(ids) != 6
        or set(ids) != set(lock)
        or not set(ids) <= set(prior["roles"]["train"])
    ):
        raise ValueError("original six TRAIN late cases required")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = m.r.load_parent(m.r.DEFAULT_PARENT)
    if m.r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("source state differs")
    diag, adv = m.bank.frozen_fits(base), m.r.simple_data.advanced_times()
    rows, check = [], {}
    for sid in ids:
        if h(m.r.io23.path_for("J-TEXT", sid)) != index[sid]:
            raise ValueError("source raw bytes changed")
        raw = m.r.simple_data.read_shot("J-TEXT", sid)
        flag, down = m.r.io23.read_outcome("J-TEXT", sid)
        if not flag:
            raise ValueError("late TRAIN case is not disruptive in HDF5")
        horizon = m.r.io23.effective_horizon_ms(sid, adv)[0]
        cases = [
            (i, q)
            for i, q in enumerate(evidence["cases"])
            if q["shot"] == sid and q["late"]
        ]
        part, check[str(sid)] = replay_shot(base, diag, raw, down, horizon, cases, data)
        rows.extend(part)
        print(f"H25 late capacity {sid}: {len(part)} windows × seed rows", flush=True)
    reach = {
        name: [
            sorted(
                {
                    q["shot"]
                    for q in rows
                    if q["seed"] == seed and q["arms"][name]["witness"] is not None
                }
            )
            for seed in m.SEEDS
        ]
        for name in m.MODES
    }
    decision = gate(evidence["summary"], reach)
    if m.r.state_digest(base) != cfg["source_state_sha256"] or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("frozen parent changed or accumulated a gradient")
    h.check()
    m.r.write_new(
        output,
        dict(
            status="MEASURED_OFFLINE_TRAIN_CAPACITY_NOT_POLICY",
            inputs={str(k): v[1] for k, v in h.cache.items()},
            source_late_ids=ids,
            seed_order=m.SEEDS,
            reachable=reach,
            gate=decision,
            cpu_gradient_parity=check,
            rows=rows,
            seconds=time.monotonic() - t,
            val_test_east_reads=0,
        ),
    )
    print(dict(reachable=reach, gate=decision), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("geometry", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.geometry, args.output)
