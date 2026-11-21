"""Re-score the sealed §33 curves on CPU; print the postseal report as JSON."""

import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[1]))

from experiments.risk_aligned_online.commissioning_coincident_support_audit_20261111 import _raw_truth
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha


LENGTHS = (1, 2, 3, 5)
MAIN = "0.2"
INTRA, RAW, TAUGHT = "intra_innovation", "joint_raw", "joint_taught"


def confirmations(above, length):
    """Mark the endpoint of every completed run of length consecutive hits."""
    done = np.zeros_like(above)
    if len(above) >= length:
        done[length - 1:] = np.lib.stride_tricks.sliding_window_view(above, length).all(axis=1)
    return done


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def window_row(sid, rank, tau, times, lead, curves, near):
    idx = np.flatnonzero(near)
    intra, raw, taught = (logit(curves[k][idx]) for k in (INTRA, RAW, TAUGHT))
    peak = int(intra.argmax())
    uplift = taught - intra
    return dict(
        shot=sid, rank=rank, endpoints=len(idx),
        intra_peak_index=int(idx[peak]), intra_peak_time_s=float(times[idx[peak]]),
        intra_peak_lead_ms=float(lead[idx[peak]]),
        min_logit_gap=float(logit(tau) - intra[peak]),
        max_taught_minus_intra_logit=float(uplift.max()),
        max_taught_minus_raw_logit=float((taught - raw).max()),
        taught_minus_intra_at_intra_peak_logit=float(uplift[peak]),
    )


def flip_detail(sid, activity, y, down, t, z, tau, arms, starts, leads):
    records = {}
    first = [int(i[0]) for i in starts.values() if len(i)]
    # Breakout around the decisive crossing(s), including the next two points.
    indices = range(min(first), min(len(t), max(first) + 3))
    for arm in arms:
        above = starts[arm]
        end = np.flatnonzero(np.diff(above) != 1)
        run_length = (int(end[0]) + 1 if len(end) else len(above))
        records[arm] = dict(
            first_index=int(above[0]) if len(above) else None,
            first_consecutive_points=run_length,
            confirmation_lead_ms=leads[arm],
            whole_curve_peak=float(z[arm].max()),
        )
    points = []
    for i in indices:
        release = int(z["release"][np.searchsorted(z["release"], i, side="right") - 1])
        points.append(dict(index=i, time_s=float(t[i]),
                           lead_ms=round(float((down - t[i]) * 1000), 9) if y else None,
                           dose=float(z["dose"][i]), release_index=release,
                           points_since_release=i - release,
                           probabilities={arm: float(z[arm][i]) for arm in arms}))
    return dict(shot=sid, disruptive=y, down_s=down, threshold=tau,
                timeline=activity, arms=records, points=points)


def analyze():
    p, s, timeline, saved, result = (
        read(RUN / name) for name in
        ("protocol.json", "seal.json", "timeline.json", "per_shot.json", "result.json")
    )
    order, arms = p["order"], p["arms"]
    names = [f"{rank:04d}_{sid}.npz" for rank, sid in enumerate(order, 1)]
    if len(set(order)) != 360 or set(names) != set(s["curve_sha256"]):
        raise ValueError("incomplete sealed cohort")
    for name, key in (("executed_probe.py", "executed_sha256"), ("executed_group.py", "group_sha256"),
                      ("reference", "reference_seal"), ("historical_reference", "historical_seal")):
        path = Path(p[name]) / "seal.json" if name in p else RUN / name
        if sha(path) != p[key]:
            raise ValueError(f"sealed input changed: {path}")
    for key in ("reference", "historical_reference"):
        ref = read(Path(p[key]) / "protocol.json")
        if any(ref[k] != p[k] for k in ("order", "startup", "original", "calibration")):
            raise ValueError(f"protocol differs from {key}")

    metrics = {b: {n: {k: dict(tp=0, fp=0) for k in arms} for n in LENGTHS}
               for b in p["calibration"]}
    rows, absent, details = [], [], {}
    misses = n_disruptive = n_exact = n_active = 0
    for rank, (sid, name, activity) in enumerate(zip(order, names, timeline, strict=True), 1):
        latest = activity["latest_teacher_rank"]
        if (activity["rank"], activity["shot"]) != (rank, sid) or (latest is not None and latest >= rank):
            raise ValueError(f"invalid teacher timeline: {sid}")
        n_active += activity["teacher_active"]
        path = RUN / "shots" / name
        if sha(path) != s["curve_sha256"][name]:
            raise ValueError(f"sealed prediction changed: {name}")
        y, down, t = _raw_truth(sid)
        n_disruptive += y
        valid = (down - t) * 1000 >= -1e-6 if y else np.ones(len(t), dtype=bool)
        with np.load(path, allow_pickle=False) as file:
            z = dict(file)
        starts, leads = {}, {}
        for band, fixed_tau in p["calibration"].items():
            tau = p["original"] if rank <= p["startup"] else fixed_tau
            for arm in arms:
                prob = z[arm]
                if prob.shape != t.shape or not np.isfinite(prob).all():
                    raise ValueError(f"invalid curve: {sid} {arm}")
                above = valid & (prob >= tau)
                for length in LENGTHS:
                    done = confirmations(above, length)
                    score = score_curve(t, done, y, down, 0.5)
                    if length == 1:
                        row = dict(shot=sid, disruptive=y, down_s=down, **score)
                        if row != saved[band][arm][rank - 1]:
                            raise ValueError(f"one-point scoring differs: {band} {sid} {arm}")
                        n_exact += 1
                    counts = metrics[band][length][arm]
                    counts["tp"] += score["timely"]
                    counts["fp"] += score["alarm"] and not y
                    if band == MAIN and sid in (73405, 73743):
                        starts[arm] = np.flatnonzero(above)
                        leads.setdefault(arm, {})[length] = score["lead_ms"]
        tau = p["calibration"][MAIN]
        if sid in (73405, 73743):
            details[sid] = flip_detail(sid, activity, y, down, t, z, tau, arms, starts, leads)
        if rank > p["startup"] and y and not saved[MAIN][INTRA][rank - 1]["timely"]:
            misses += 1
            lead = np.round((down - t) * 1000, 9)
            near = (z["dose"] > 0) & (lead > 30) & (lead <= 125)
            if near.any():
                rows.append(window_row(sid, rank, tau, t, lead, z, near))
            else:
                absent.append(sid)

    if (n_disruptive, len(order) - n_disruptive, n_active, n_exact) != (
        180, 180, result["active_shots"], result["raw_verified"]
    ):
        raise ValueError("cohort, timeline activity or exact scoring count differs")
    for band in metrics:
        for arm, counts in metrics[band][1].items():
            if any(result["metrics"][band][arm][k] != v for k, v in counts.items()):
                raise ValueError(f"one-point summary differs: {band} {arm}")
    median_keys = ("min_logit_gap", "max_taught_minus_intra_logit", "max_taught_minus_raw_logit",
                   "taught_minus_intra_at_intra_peak_logit")
    return dict(
        scope="SEALED_DEVELOPMENT_CURVE_ATTRIBUTION",
        input_seal_sha256=sha(RUN / "seal.json"),
        verification=dict(sealed_curves=len(names), hdf5_clocks_and_outcomes=len(order),
                          exact_one_point_rows=n_exact, teacher_active_shots=n_active,
                          n_disruptive=n_disruptive, n_safe=len(order) - n_disruptive),
        thresholds=dict(startup_shots=p["startup"], startup=p["original"], after_startup=p["calibration"]),
        scoring="Consecutive scores >= original shot threshold; first alarm at confirmation completion; strict lead >30 ms.",
        confirmed_point_sensitivity=metrics,
        near_window=dict(
            selection="Post-startup original intra misses or late alarms at band 0.2; dose >0 and lead in (30,125] ms.",
            definitions="Gap = threshold logit minus max intra logit; maximum uplifts are paired at the same endpoint.",
            late_or_missed=misses, with_legal_window=len(rows), without_legal_window=absent,
            medians={k: float(np.median([row[k] for row in rows])) for k in median_keys},
            per_shot=rows,
        ),
        flip_details=details,
    )


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
