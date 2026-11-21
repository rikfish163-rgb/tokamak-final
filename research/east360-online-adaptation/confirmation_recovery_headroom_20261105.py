"""Post-reveal attribution of why the locked update did or did not recover shots."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from joint_lora_hindsight_capacity_20261104 import read, sha, write_new

GUARD = "intra_confirm_past32"


def classify_miss(native, active, confirmed, valid, threshold):
    """Split shortfall of the *observed* update from refusal by its gate."""
    if not valid.any():
        return dict(
            category="no_eligible_time", native_gap=None, largest_upward_change=None
        )
    p, a, c = (v[valid].astype(np.float64) for v in (native, active, confirmed))
    gap = float(threshold - p.max())
    if gap <= 0:
        raise ValueError("a native timely crossing is not a miss")
    uplift = max(0.0, float((a - p).max()))
    if (c >= threshold).any():
        if not (a >= threshold).any():
            raise ValueError(
                "confirmation cannot add a crossing the active model did not propose"
            )
        category = "confirmed_recovery"
    elif (a >= threshold).any():
        category = "reached_but_not_confirmed"
    elif uplift < gap:
        category = "unreached_uplift_smaller_than_gap"
    else:
        category = "unreached_uplift_not_at_crossing"
    return dict(category=category, native_gap=gap, largest_upward_change=uplift)


def analyze(run, verification, output):
    completed = run / "completed.json"
    receipt = read(verification)
    if receipt["status"] != "PASS_RAW_CONFIRMATION_RECOUNT" or receipt[
        "producer_completion_sha256"
    ] != sha(completed):
        raise ValueError("independently verified completion required")
    cfg, per, done = [
        read(run / name)
        for name in ("protocol.json", "per_shot.json", "completed.json")
    ]
    tau, first = cfg["threshold"], cfg["first_anchor"]
    if any(
        sha(run / name) != done["artifact_sha256"][name]
        for name in ("protocol.json", "per_shot.json")
    ):
        raise ValueError("verified protocol or shot outcomes changed")
    prefix, classified, safe_fps, changes = 0, [], [], []
    for i, (base, active, guard, seal) in enumerate(
        zip(per["static"], per["point_real"], per[GUARD], done["sealed"], strict=True),
        1,
    ):
        sid = base["shot"]
        enabled = i > cfg["fixed_prefix_shots"]
        if base["status"] == "fp":
            safe_fps.append(
                dict(
                    shot=sid,
                    before_any_fast_update=not enabled or base["first_index"] < first,
                )
            )
        if base["status"] == "tp" and guard["status"] == "tp":
            changes.append(guard["lead_ms"] - base["lead_ms"])
        if not base["disruptive"] or base["status"] == "tp":
            continue
        if not enabled:
            prefix += 1
            continue
        path = run / "shots" / f"{i:04d}_{sid}.npz"
        if sha(path) != seal["curve_sha256"]:
            raise ValueError(f"sealed curve changed: {sid}")
        with np.load(path, allow_pickle=False) as f:
            t = f["times"]
            lead = np.round((base["down"] - t) * 1000, 9)
            valid = (np.arange(len(t)) >= first) & (lead > 30)
            result = classify_miss(f["static"], f["point_real"], f[GUARD], valid, tau)
        if (result["category"] == "confirmed_recovery") != (guard["status"] == "tp"):
            raise ValueError(f"confirmation does not match strict lead outcome: {sid}")
        proposed = result["category"] in (
            "confirmed_recovery",
            "reached_but_not_confirmed",
        )
        if proposed != (active["status"] == "tp"):
            raise ValueError(
                f"active crossing does not match strict lead outcome: {sid}"
            )
        classified.append(dict(shot=sid, initial_status=base["status"], **result))
    counts = dict(Counter(row["category"] for row in classified))
    gaps = [r["native_gap"] for r in classified if r["native_gap"] is not None]
    effects = [
        r["largest_upward_change"]
        for r in classified
        if r["largest_upward_change"] is not None
    ]
    stats = dict(
        fixed_prefix_untimely=prefix,
        enabled_untimely=len(classified),
        categories=counts,
        median_native_gap=float(np.median(gaps)) if gaps else None,
        median_maximum_observed_positive_move=float(np.median(effects))
        if effects
        else None,
        native_false_alarms=len(safe_fps),
        native_false_alarms_before_any_enabled_fast_step=sum(
            r["before_any_fast_update"] for r in safe_fps
        ),
        safe_false_alarm_cases=safe_fps,
        common_timely=len(changes),
        lead_advanced=sum(x > 0 for x in changes),
        lead_delayed=sum(x < 0 for x in changes),
        median_common_lead_delta_ms=float(np.median(changes)) if changes else None,
    )
    write_new(
        output,
        dict(
            status="COMPLETE_DESCRIPTIVE_UPDATE_HEADROOM",
            verification_sha256=sha(verification),
            mode=cfg["mode"],
            stats=stats,
            scope="Observed moves, not a causal or theoretical bound for another update.",
            per_baseline_untimely_shot=classified,
        ),
    )
    print({k: v for k, v in stats.items() if k != "safe_false_alarm_cases"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "verification", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.verification, args.output)
