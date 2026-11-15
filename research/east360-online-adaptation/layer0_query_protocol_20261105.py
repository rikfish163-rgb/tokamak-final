"""Outcome-blind query selection and full-curve eligibility for local EAST memory."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import score


FIRST, STRIDE = 135, 32
BLOCK = 20
PERSISTENT_ARMS = ("inter", "both")
ARMS = ("static", "intra", *PERSISTENT_ARMS)


def _curve(a):
    if (
        a.ndim != 1
        or not len(a)
        or not np.isfinite(a).all()
        or (a < 0).any()
        or (a > 1).any()
    ):
        raise ValueError("finite probability vector required")


def choose_query(shots, threshold):
    """Each row is rank, ID, sealed static probability trace, sealed intra trace."""
    rows = []
    for rank, sid, a, b in shots:
        _curve(a)
        _curve(b)
        if (
            a.shape != b.shape
            or len(a) <= FIRST
            or not np.array_equal(a[:FIRST], b[:FIRST])
        ):
            raise ValueError("aligned traces with identical pre-update prefix required")
        p, q = float(np.max(a[FIRST:])), float(np.max(b[FIRST:]))
        prefix = bool((a[:FIRST] >= threshold).any())
        rows.append(
            dict(
                rank=rank,
                shot=sid,
                static_peak=p,
                intra_peak=q,
                prefix_alarm=prefix,
                disagree=(p >= threshold) != (q >= threshold),
                threshold_distance=min(abs(p - threshold), abs(q - threshold)),
                peak_difference=abs(p - q),
            ),
        )
    if (
        not rows
        or len({r["shot"] for r in rows}) != len(rows)
        or sorted({r["rank"] for r in rows}) != [r["rank"] for r in rows]
    ):
        raise ValueError("nonempty ordered distinct query candidates required")
    selected = min(
        rows,
        key=lambda r: (
            r["prefix_alarm"],
            not r["disagree"],
            r["threshold_distance"],
            -r["peak_difference"],
            r["rank"],
        ),
    )
    return selected["shot"], rows


def reveal_block(path, ids, oracle):
    """The sole in-stream read of labels, after a durable query commitment."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    commit = json.loads(data)
    if list(ids) != commit["selected"] or len(set(ids)) != len(ids):
        raise ValueError("different selection from durable commitment")
    last = commit["committed_ns"]
    if type(last) is not int:
        raise ValueError("invalid commitment clock")
    outcomes = []
    for sid in ids:
        start = time.monotonic_ns()
        if start < last:
            raise RuntimeError("label access before commitment")
        flag, down = oracle("EAST", sid)
        stop = time.monotonic_ns()
        if (
            type(flag) is not bool
            or (flag and (down is None or not np.isfinite(down)))
            or (not flag and down is not None)
        ):
            raise ValueError("invalid authoritative HDF5 outcome")
        outcomes.append(
            dict(
                shot=sid,
                disruptive=flag,
                down=down,
                read_start_ns=start,
                read_end_ns=stop,
            )
        )
        last = stop
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("query commitment changed during reveal")
    return dict(commit_sha256=digest, outcomes=outcomes)


def _score(times, trace, disruptive, down, threshold):
    return score(
        times, trace, dict(shot=None, disruptive=disruptive, down=down), threshold
    )


def _capped(row):
    return row["status"] == "tp" and row["lead_ms"] <= 3000


def guard_violations(times, old, new, disruptive, down, threshold):
    left, right = (
        _score(times, trace, disruptive, down, threshold) for trace in (old, new)
    )
    if not disruptive:
        return (
            ["new_false_alarm"]
            if left["status"] == "tn" and right["status"] == "fp"
            else []
        )
    if left["status"] == "tp" and right["status"] != "tp":
        return ["lost_timely"]
    return ["lost_30_to_3000"] if _capped(left) and not _capped(right) else []


def _positive_points(times, down):
    lead = np.round((down - times) * 1000, 9)
    return (lead > 30) & (lead <= 125)


def teaching_indices(times, curves, disruptive, down, threshold):
    eligible = np.arange(FIRST, len(times))
    if disruptive:
        eligible = eligible[_positive_points(times, down)[eligible]]
        if not len(eligible):
            return eligible
        n = min(8, len(eligible))
        return eligible[np.linspace(0, len(eligible) - 1, n, dtype=np.int64)]
    p = np.maximum(curves["inter"], curves["both"])
    chosen = {
        int(piece[np.argmax(p[piece])])
        for piece in np.array_split(eligible, 8)
        if len(piece)
    }
    for a in curves.values():
        hits = np.flatnonzero(a >= threshold)
        if len(hits) and hits[0] >= FIRST:
            chosen.add(int(hits[0]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def query_progress(times, old, new, disruptive, down, threshold):
    """Require a real reduction of an existing error, not just a training loss drop."""
    if set(old) != set(PERSISTENT_ARMS) or set(new) != set(PERSISTENT_ARMS):
        raise ValueError("both persistent contexts required")
    mask = _positive_points(times, down) if disruptive else np.ones(len(times), bool)
    mask[:FIRST] = False
    if not mask.any():
        return False, {}
    diagnostics = {}
    ok, moved = True, False
    for arm in PERSISTENT_ARMS:
        a, b = (
            _score(times, src[arm], disruptive, down, threshold) for src in (old, new)
        )
        violations = guard_violations(
            times, old[arm], new[arm], disruptive, down, threshold
        )
        wrong = a["status"] in ("miss", "tardy", "fp")
        rescued = wrong and b["status"] in ("tp", "tn")
        shift = float(np.max(new[arm][mask]) - np.max(old[arm][mask]))
        gain = shift if disruptive else -shift
        if violations or (wrong and not rescued and gain < -1e-6):
            ok = False
        if wrong and (rescued or gain >= 0.02):
            moved = True
        diagnostics[arm] = dict(
            before=a,
            after=b,
            peak_gain=gain,
            corrected=rescued,
            guard_violations=violations,
        )
    return ok and moved, diagnostics
