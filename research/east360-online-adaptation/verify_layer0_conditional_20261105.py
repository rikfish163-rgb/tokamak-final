"""Independent HDF5 recount and opportunity audit for the frozen first-layer policy."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    RAW_ROOT,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)
from verify_moving_peak_replay_20261105 import check_curve, parity


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / "probe_outputs/confirmation_raw_conditional835_20261105_v1"
ROSTER = ROOT / "probe_outputs/conditional_confirmation_roster_20261105_v1.json"
ARMS = ["static", "intra", "lagged", "first_direction"]
FIT_SHA = "eaf4c97f7aa4a829db8058ea879ff0677c092435e3cd30d38a1ab22127639eac"
SOURCE = "99affd49f71f41129b87b03624201ef94ef6dc44e652eaef411b2924f8186dc8"
FIRST, STRIDE = 135, 32


def failure_headroom(clock, static, active, down, tau):
    """Separate effect size from alignment over eligible strict-timely endpoints."""
    select = (np.arange(len(clock)) >= FIRST) & (
        np.round((down - clock) * 1000, 9) > 30
    )
    if not select.any():
        return dict(mechanism="no_eligible_endpoint", gap=None, max_shift=None)
    a, b = static[select].astype(np.float64), active[select].astype(np.float64)
    gap, shift = float(tau - np.max(a)), float(np.max(b - a))
    need(gap > 0, "failure headroom requires an actually missed native trajectory")
    reason = (
        "crossed"
        if np.max(b) >= tau
        else ("insufficient_shift" if shift < gap else "shift_at_other_endpoint")
    )
    return dict(mechanism=reason, gap=gap, max_shift=shift)


def verify(run, output):
    need(not output.exists(), "output already exists")
    h = Hashes()
    cfg, done = [read(run / name) for name in ("protocol.json", "completed.json")]
    old_cfg, old = [read(PRIOR / name) for name in ("protocol.json", "completed.json")]
    roster = read(ROSTER)
    order, tau = cfg["order"], cfg["threshold"]
    need(
        len(order) == len(set(order)) == 835
        and order == roster["ids"] == old_cfg["order"]
        and old["status"] == "COMPLETE_RAW_CONFIRMATION_REPLAY"
        and cfg["raw_sha256"] == roster["raw_sha256"] == old_cfg["raw_sha256"]
        and cfg["independent_validation"] is False
        and roster["independent_validation"] is False
        and tau == old_cfg["threshold"] == 0.7258946895599365
        and cfg["arms"] == ARMS
        and (cfg["first"], cfg["stride"]) == (FIRST, STRIDE)
        and cfg["mapped_from_first"] is True
        and cfg["added_fit_labels"]
        == cfg["inter_updates"]
        == cfg["outcome_reads_during_replay"]
        == 0
        and cfg["fit_sha256"] == FIT_SHA
        and cfg["source_state_sha256"] == SOURCE,
        "wrong conditional cohort, freeze, or timing contract",
    )
    need(
        done["status"] == "COMPLETE_LAYER0_CONDITIONAL_835"
        and done["n_shots"] == len(done["sealed"]) == 835
        and done["n_curves"] == 3340
        and done["protocol_sha256"] == h(run / "protocol.json")
        and h(PRIOR / "protocol.json") == old["artifact_sha256"]["protocol.json"]
        and all(h(p) == sha for p, sha in cfg["input_sha256"].items()),
        "incomplete or changed conditional replay",
    )
    rows, mechanisms = {k: [] for k in ARMS}, []
    immutable_fp = []
    previous, last, max_error = None, cfg["initial_ns"], 0.0
    for rank, sid in enumerate(order, 1):
        stem = f"{rank:04d}_{sid}"
        path = run / "shots" / (stem + ".npz")
        seal = run / "shots" / (stem + "_seal.json")
        prior = PRIOR / "shots" / (stem + ".npz")
        note = read(seal)
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(
            done["sealed"][rank - 1] == dict(note, seal_sha256=h(seal))
            and (note["rank"], note["shot"]) == (rank, sid)
            and note["raw_sha256"] == h(raw) == cfg["raw_sha256"][str(sid)]
            and note["curve_sha256"] == h(path)
            and h(prior) == old["sealed"][rank - 1]["curve_sha256"]
            and note["previous_seal_sha256"] == previous
            and note["seal_ns"] >= last,
            f"conditional sealed input changed: {sid}",
        )
        previous, last = h(seal), note["seal_ns"]
        truth, clock = raw_truth(raw, sid)
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(prior, allow_pickle=False) as b,
        ):
            need(
                np.array_equal(a["times"], clock)
                and np.array_equal(b["times"], clock)
                and set(a.files) == {"times", *ARMS},
                "conditional time or schema mismatch",
            )
            native = a["static"]
            max_error = max(
                max_error, parity(native, b["static"], tau, f"{sid}/static")
            )
            for k in ARMS:
                p = a[k]
                check_curve(p, clock, k)
                need(
                    np.array_equal(p[:FIRST], native[:FIRST]),
                    "action before first causal write",
                )
                rows[k].append(score(clock, p, truth, tau))
            native_row = rows["static"][-1]
            if native_row["status"] == "fp" and native_row["first_index"] < FIRST:
                need(
                    all(
                        rows[k][-1]["first_index"] == native_row["first_index"]
                        for k in ARMS
                    ),
                    "immutable pre-update false alarm changed",
                )
                immutable_fp.append(sid)
            if truth["disruptive"] and native_row["status"] != "tp":
                result = failure_headroom(clock, native, a["intra"], truth["down"], tau)
                need(
                    (result["mechanism"] == "crossed")
                    == (rows["intra"][-1]["status"] == "tp"),
                    "eligible endpoint must agree with strict shot scoring",
                )
                mechanisms.append(dict(shot=sid, **result))
        actions = note["stats"]
        need(set(actions) == set(ARMS[1:]), "missing action receipts")
        n = len(range(FIRST, len(clock), STRIDE))
        first = actions["intra"]
        for stats in actions.values():
            need(
                0 <= stats["nonzero"] == first["nonzero"] <= n
                and 0 <= stats["max_norm"] <= 2.000001
                and abs(stats["max_norm"] - first["max_norm"]) <= 1e-6,
                "unequal or out-of-bound same-site action",
            )
    need(
        max_error == done["static_max_parity_error"]
        and done["completed_ns"] >= last
        and sum(v["disruptive"] for v in rows["static"]) == 510,
        "wrong terminal parity",
    )
    summary = {k: counts(v) for k, v in rows.items()}
    paired = {
        k: dict(
            strict=pair(rows["intra"], rows[k]),
            capped=pair(rows["intra"], rows[k], True),
        )
        for k in ARMS
        if k != "intra"
    }
    bins = [
        dict(
            start_rank=i + 1,
            end_rank=min(i + 200, len(order)),
            metrics={k: counts(v[i : i + 200]) for k, v in rows.items()},
        )
        for i in range(0, len(order), 200)
    ]
    eligible = [x for x in mechanisms if x["gap"] is not None]
    h.check()
    report = dict(
        status="PASS_LAYER0_CONDITIONAL_835_HDF5_RECOUNT",
        n_curves=3340,
        n_shots=835,
        conditional_only=True,
        full=summary,
        paired_intra=paired,
        sorted_rank_bands=bins,
        static_prewrite_false_alarms=immutable_fp,
        headroom=dict(
            counts=dict(Counter(x["mechanism"] for x in mechanisms)),
            median_gap=float(np.median([x["gap"] for x in eligible])),
            median_max_shift=float(np.median([x["max_shift"] for x in eligible])),
            cases=mechanisms,
        ),
        static_curve_max_error=max_error,
        verifier_sha256=h(__file__),
        producer_sha256=h(run / "completed.json"),
        scope="Conditional retrospective roster; no model fit or future-validation claim.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(report["status"], summary, report["headroom"]["counts"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    verify(args.run, args.output)
