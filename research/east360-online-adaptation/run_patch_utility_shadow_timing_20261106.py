"""Measure how much fixed-action capacity survives observing more of a shot."""

import argparse
from numbers import Integral
from pathlib import Path
import time

import numpy as np

import run_patch_utility_multiview_20261106 as m
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis


b, sel = m.sel.bank, m.sel
PLAN = b.r.ROOT / "PATCH_UTILITY_SHADOW_TIMING_PROTOCOL_20261106.md"
DELAYS = (0, 4, 8, 16)


def wait(curves, delay):
    if (not isinstance(delay, Integral) or isinstance(delay, bool)
            or curves.ndim != 3 or curves.shape[1] < 1
            or not 0 <= delay <= curves.shape[-1]):
        raise ValueError("invalid delay or action curves")
    output = curves.copy()
    # Action index zero is the sealed static curve. Never rewrite reached risk.
    output[:, :, :delay] = curves[:, :1, :delay]
    return output


def one_view(tape):
    keys = ("late_single", "late_any", "false_remove_single", "false_remove_any", "false_add_single")
    summary = {str(delay): {k: {q: [] for q in keys} for k in ("real", "scalar")}
               for delay in DELAYS}
    witnesses = {}
    for q in tape.shots:
        before = q.teacher["baseline"]
        if q.disruptive and before["timely"]:
            continue
        m0, allowed, actual, effect = (
            tape.data[k][q.blocks] for k in ("clock", "allowed", "curves", "effect")
        )
        scalar = sel.scalar_curves(actual, effect, m0, len(q.native))
        notes = {}
        for delay in DELAYS:
            row = summary[str(delay)]
            note = {}
            for name, curves in (("real", actual), ("scalar", scalar)):
                delayed = wait(curves, delay)
                replay = menu_analysis(q, m0, delayed, allowed, b.TAU)
                y = replay["hard"]
                if delay == 0 and name == "real":
                    if not np.array_equal(y, tape.hard[q.blocks]) or any(
                        replay[k] != q.teacher[k] for k in
                        ("baseline", "reachable_first", "oracle_timely", "oracle_no_false_alarm")
                    ):
                        raise ValueError("zero delay must reproduce the original real menu")
                if replay["baseline"] != before:
                    raise ValueError("waiting changed the frozen baseline")
                if q.disruptive:
                    good = np.flatnonzero((y > 0).any(axis=1))
                    if len(good):
                        row[name]["late_single"].append(q.sid)
                    if replay["oracle_timely"]:
                        row[name]["late_any"].append(q.sid)
                    note[name] = dict(
                        latest_commit=int(m0[good[-1]]+delay) if len(good) else None,
                        action_blocks=int(len(good)),
                        available_actions=int((y > 0).sum()),
                    )
                elif before["false_alarm"]:
                    if (y > 0).any():
                        row[name]["false_remove_single"].append(q.sid)
                    if replay["oracle_no_false_alarm"]:
                        row[name]["false_remove_any"].append(q.sid)
                elif (y < 0).any():
                    row[name]["false_add_single"].append(q.sid)
            if q.disruptive:
                notes[str(delay)] = note
        if q.disruptive:
            witnesses[str(q.sid)] = dict(original=before, delayed=notes)
    for name in ("real", "scalar"):
        for d0,d1 in zip(DELAYS[:-1], DELAYS[1:], strict=True):
            for key in ("late_single","late_any"):
                if not set(summary[str(d1)][name][key]) <= set(summary[str(d0)][name][key]):
                    raise ValueError("a later commit cannot create a new timely crossing")
    return dict(capacity=summary, late_witnesses=witnesses)


def run(native, masked, native_result, phase_rank, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    rank = b.r.read(phase_rank)
    if rank["rank_gate"] or rank["first_alarm_replays"] or rank["new_val_test_east_reads"]:
        raise ValueError("diagnostic continuation requires unsuccessful rank-only probe")
    for p in (Path(__file__), Path(m.__file__), Path(sel.__file__), PLAN, phase_rank):
        h(p)
    tapes, _, _ = m.load(native, masked, native_result, h)
    views = {}
    for name in m.VIEWS:
        views[name] = one_view(tapes[name])
        print(name, {d: len(x["real"]["late_single"])
                     for d,x in views[name]["capacity"].items()}, flush=True)
    merged = {}
    for delay in DELAYS:
        s = str(delay)
        q = {k: {i for v in views.values() for i in v["capacity"][s][k]["late_single"]}
             for k in ("real", "scalar")}
        merged[s] = dict(real=sorted(q["real"]), scalar=sorted(q["scalar"]),
                         real_only=sorted(q["real"]-q["scalar"]))
    h.check()
    result = dict(
        status="SCORED_FIXED_SHADOW_TIMING_CAPACITY_NOT_ONLINE", views=views,
        timely_unique=merged, shadow_time_support_gate=len(merged["8"]["real_only"]) >= 2,
        no_policy_fit=True, no_new_waveforms=True, new_val_test_east_reads=0,
        input_sha256={str(p): v[1] for p,v in h.cache.items()},
        elapsed_seconds=time.monotonic()-start,
    )
    b.r.write_new(out, result)
    print(merged,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("native", "masked", "native_result", "phase_rank", "output"):
        p.add_argument(a, type=Path)
    q = p.parse_args()
    run(q.native,q.masked,q.native_result,q.phase_rank,q.output)
