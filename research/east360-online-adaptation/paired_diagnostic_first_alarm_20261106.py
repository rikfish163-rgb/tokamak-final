"""Diagnostic capacity of switching between already sealed block trajectories."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import logit

from run_paired_diagnostic_update_20261106 import (
    ARMS,
    SEEDS,
    checked_fit,
    checked_protocol,
)
import moving_peak_replay_20261105 as r
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_paired_diagnostic_update_20261106 import verify_prediction


def attainable(scores, threshold, before=None):
    """Exact feasible first alarms when one option is selected per whole block.

    Offline oracle only. Reaching a later block requires an option that did not
    cross anywhere in every earlier block; a pulse cannot undo an earlier alarm.
    """
    if scores.ndim != 2 or not len(scores) or not scores.shape[1]:
        raise ValueError("aligned nonempty block trajectories required")
    n = len(scores)
    if before is None:
        before = np.ones(n, dtype=bool)
    if before.dtype != bool or before.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("finite scores with a matched eligibility mask required")
    edges = np.r_[0, np.arange(64, n, 32), n]
    hits = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        cross = (scores[lo:hi] >= threshold) & before[lo:hi, None]
        triggered = cross.any(0)
        if triggered.any():
            hits.extend((lo + cross[:, triggered].argmax(0)).tolist())
        if triggered.all():
            return sorted(set(hits)), False
    return sorted(set(hits)), True


def shifted_peak(native, real, first, threshold):
    """Use the exact native peak-mining rule on the affected safe block."""
    if native.shape != real.shape or native.ndim != 1 or not 64 <= first < len(native):
        raise ValueError("a new post-maturity first alarm is required")
    start = first // 32 * 32
    stop = min(start + 32, len(native))
    original = int(start + np.argmax(native[start:stop]))
    updated = int(start + np.argmax(real[start:stop]))
    return dict(
        block_start=start,
        block_end_exclusive=stop,
        static_peak_index=original,
        real_peak_index=updated,
        native_peak_logit=float(native[original]),
        updated_at_original_peak_logit=float(real[original]),
        new_peak_logit=float(real[updated]),
        native_at_new_first_logit=float(native[first]),
        real_at_new_first_logit=float(real[first]),
        original_peak_stays_below_threshold=bool(real[original] < threshold),
    )


OPTIONS = {
    "static_real": (0, 1),
    "static_score": (0, 5),
    "all_observed_arms": tuple(range(6)),
}


def replay_row(t, z, flag, down, threshold):
    if flag:
        raw = (down - t) * 1000
        lead = np.round(raw, 9)
        before = raw >= -1e-6
    else:
        lead = np.zeros_like(t)
        before = np.ones_like(t, dtype=bool)
    rows = {}
    for j, seed in enumerate(SEEDS):
        q = z[:, j, :]
        first = []
        for k in range(len(ARMS)):
            hits = np.flatnonzero(before & (q[:, k] >= threshold))
            first.append(int(hits[0]) if len(hits) else None)
        tp = [flag and i is not None and lead[i] > 10 for i in first]
        fp = [not flag and i is not None for i in first]
        capacity = {}
        for name, columns in OPTIONS.items():
            possible, silent = attainable(q[:, columns], threshold, before)
            capacity[name] = dict(
                best_timely=bool(flag and any(lead[i] > 10 for i in possible)),
                best_reasonable=bool(
                    flag and any(10 < lead[i] <= 3000 for i in possible)
                ),
                minimum_fp=bool(not flag and not silent),
                possible_first=possible,
            )
        event = None
        if tp[0] != tp[1] or fp[0] != fp[1]:
            event = dict(
                gained_timely=bool(tp[1] and not tp[0]),
                lost_timely=bool(tp[0] and not tp[1]),
                new_fp=bool(fp[1] and not fp[0]),
                removed_fp=bool(fp[0] and not fp[1]),
                first=dict(zip(ARMS, first, strict=True)),
                lead=dict(
                    zip(
                        ARMS,
                        [
                            float(lead[i]) if flag and i is not None else None
                            for i in first
                        ],
                        strict=True,
                    )
                ),
            )
            if event["new_fp"]:
                event["block_peak"] = shifted_peak(
                    q[:, 0], q[:, 1], first[1], threshold
                )
                event["equal_score_at_new_first"] = float(q[first[1], 5])
        rows[str(seed)] = dict(
            static_tp=bool(tp[0]),
            real_tp=bool(tp[1]),
            static_fp=fp[0],
            real_fp=fp[1],
            capacity=capacity,
            event=event,
        )
    return rows


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    start, h = time.monotonic(), Hashes()
    cfg, _ = checked_protocol(root, h), checked_fit(root, h)
    audit_path, seal_path = root / "val_audit.json", root / "val" / "completed.json"
    audit, seal = r.read(audit_path), r.read(seal_path)
    audit_sha, seal_sha = h(audit_path), h(seal_path)
    if (
        audit["status"] != "RECOUNT_PAIRED_DIAGNOSTIC_UPDATE"
        or audit["role"] != "val"
        or audit["seal_sha256"] != seal_sha
        or seal["held_outcome_reads"] != 0
        or [q["shot"] for q in seal["rows"]] != cfg["roles"]["val"]
    ):
        raise ValueError(
            "a completed, independently scored source VAL seal is required"
        )
    for rank, q in enumerate(seal["rows"], 1):
        sid = q["shot"]
        if (q["rank"], q["file"]) != (rank, f"{rank:04d}_{sid}.npz") or (
            h(root / "val" / q["file"]) != q["sha256"]
            or h(r.io23.path_for("J-TEXT", sid)) != q["raw_sha256"]
        ):
            raise ValueError("source VAL predictions or HDF5 bytes changed")
    tau = float(logit(cfg["threshold"]))
    rows = []
    for q in seal["rows"]:
        sid = q["shot"]
        raw = r.simple_data.read_shot("J-TEXT", sid)
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        with np.load(root / "val" / q["file"], allow_pickle=False) as a:
            verify_prediction(a, raw["times"], raw["available"], enabled=True)
            rows.append(
                dict(
                    shot=sid,
                    seeds=replay_row(raw["times"], a["logits"], flag, down, tau),
                )
            )
    report = {}
    for s in map(str, SEEDS):
        z = [q["seeds"][s] for q in rows]
        counts = {
            k: sum(q[k] for q in z)
            for k in ("static_tp", "real_tp", "static_fp", "real_fp")
        }
        for arm in ("static", "real"):
            original = audit["aggregate"]["per_seed"][s][arm]
            if (counts[f"{arm}_tp"], counts[f"{arm}_fp"]) != (
                original["raw_tp"],
                original["fp"],
            ):
                raise ValueError("independent first alarm count differs")
        report[s] = dict(
            counts=counts,
            options={
                a: {
                    k: sum(q["capacity"][a][k] for q in z)
                    for k in ("best_timely", "best_reasonable", "minimum_fp")
                }
                for a in OPTIONS
            },
            flips=[
                dict(shot=q["shot"], **q["seeds"][s]["event"])
                for q in rows
                if q["seeds"][s]["event"] is not None
            ],
        )
    h.check()
    r.write_new(
        output,
        dict(
            status="SOURCE_VAL_OFFLINE_FIRST_ALARM_ATTRIBUTION_NO_FIT",
            audit_sha256=audit_sha,
            seal_sha256=seal_sha,
            code_sha256=h(Path(__file__)),
            results=report,
            all_shots=rows,
            seconds=time.monotonic() - start,
        ),
    )
    print(
        {
            s: {"counts": v["counts"], "options": v["options"]}
            for s, v in report.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.root, a.out)
