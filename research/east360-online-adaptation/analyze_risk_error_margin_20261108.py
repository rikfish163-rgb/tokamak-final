"""CPU-only diagnostics for the sealed fixed360 risk-teacher curves."""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import run_joint_trajectory_20261107 as joint


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "probe_outputs/risk_positive_pool360_20261108_v2"
OUTPUT = RUN / "error_margin_diagnostic_20261108.json"
THRESHOLD = 0.7258946895599365
FIRST_EFFECT = 153
ARMS = ("static", "recorded_sampling_real")
STATUSES = ("tp", "tardy", "miss", "fp", "tn")
WINDOWS = ("eligible_history_or_safe_full", "early_positive")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def distribution(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {key: None if key != "n" else 0 for key in
                ("n", "min", "p25", "median", "p75", "max")}
    quantiles = np.quantile(values, [0, .25, .5, .75, 1], method="linear")
    return dict(zip(("n", "min", "p25", "median", "p75", "max"),
                    [len(values), *map(float, quantiles)], strict=True))


def first_alarm_relation(first, indices):
    if not len(indices):
        return None
    if first is None:
        return "no_alarm"
    if first < indices[0]:
        return "before_window"
    if first > indices[-1]:
        return "after_window"
    need(first in indices, "first alarm lies in a gap of a contiguous window")
    return "within_window"


def peak_relation(peak_index, first_alarm):
    if first_alarm is None:
        return "no_alarm"
    if peak_index < first_alarm:
        return "before_first_alarm"
    if peak_index == first_alarm:
        return "at_first_alarm"
    return "after_first_alarm"


def window_summary(indices, times, leads, static, intra, static_row, intra_row,
                   first_release):
    indices = np.asarray(indices, dtype=np.int64)
    result = {
        "point_count": int(len(indices)),
        "first_index": int(indices[0]) if len(indices) else None,
        "last_index": int(indices[-1]) if len(indices) else None,
        "static": None,
        "intra": None,
        "intra_at_static_max_probability": None,
        "intra_minus_static_at_static_max": None,
        "intra_max_minus_static_max": None,
        "static_max_vs_first_actual_release": None,
        "points_at_or_after_first_actual_release": (
            int(np.count_nonzero(indices >= first_release))
            if first_release is not None else None),
        "static_first_alarm_vs_window": first_alarm_relation(static_row["first_index"], indices),
        "intra_first_alarm_vs_window": first_alarm_relation(intra_row["first_index"], indices),
    }
    if not len(indices):
        return result

    for label, values, row in (("static", static, static_row), ("intra", intra, intra_row)):
        index = int(indices[np.argmax(values[indices])])
        probability = float(values[index])
        result[label] = {
            "max_probability": probability,
            "max_index": index,
            "max_time_s": float(times[index]),
            "max_lead_ms": float(leads[index]) if leads is not None else None,
            "margin_to_fixed_threshold": probability - THRESHOLD,
            "max_vs_first_alarm": peak_relation(index, row["first_index"]),
        }
    si = result["static"]["max_index"]
    result["intra_at_static_max_probability"] = float(intra[si])
    result["intra_minus_static_at_static_max"] = float(intra[si]) - float(static[si])
    result["intra_max_minus_static_max"] = (
        result["intra"]["max_probability"] - result["static"]["max_probability"])
    result["static_max_vs_first_actual_release"] = (
        "no_actual_release" if first_release is None else
        "before_release" if si < first_release else "at_or_after_release")
    return result


def directions(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "up": sum(v > 0 for v in values),
            "unchanged_exactly": sum(v == 0 for v in values),
            "down": sum(v < 0 for v in values)}


def grouped_window(rows, window_name):
    applicable = [row["windows"][window_name] for row in rows
                  if row["windows"][window_name] is not None]
    present = [window for window in applicable if window["point_count"] > 0]
    fields = {
        "point_count": lambda w: w["point_count"],
        "static_max_probability": lambda w: w["static"]["max_probability"],
        "static_max_index": lambda w: w["static"]["max_index"],
        "static_max_time_s": lambda w: w["static"]["max_time_s"],
        "static_max_lead_ms": lambda w: w["static"]["max_lead_ms"],
        "static_margin": lambda w: w["static"]["margin_to_fixed_threshold"],
        "intra_max_probability": lambda w: w["intra"]["max_probability"],
        "intra_max_index": lambda w: w["intra"]["max_index"],
        "intra_max_time_s": lambda w: w["intra"]["max_time_s"],
        "intra_max_lead_ms": lambda w: w["intra"]["max_lead_ms"],
        "intra_margin": lambda w: w["intra"]["margin_to_fixed_threshold"],
        "intra_at_static_max_probability": lambda w: w["intra_at_static_max_probability"],
        "intra_minus_static_at_static_max": lambda w: w["intra_minus_static_at_static_max"],
        "intra_max_minus_static_max": lambda w: w["intra_max_minus_static_max"],
        "points_at_or_after_first_actual_release": lambda w: w["points_at_or_after_first_actual_release"],
    }
    stats = {name: distribution([getter(w) for w in (applicable if name == "point_count" else present)])
             for name, getter in fields.items()}
    near = {}
    for arm in ("static", "intra"):
        margins = [w[arm]["margin_to_fixed_threshold"] for w in present]
        near[arm] = {
            "n_valid": len(margins),
            "at_or_above_threshold": sum(value >= 0 for value in margins),
            "below_within_inclusive": {
                str(distance): sum(-distance <= value < 0 for value in margins)
                for distance in (.01, .05, .10)},
            "below_more_than_0.10": sum(value < -.10 for value in margins),
        }
    active = [w for w in present if w["static_max_vs_first_actual_release"] == "at_or_after_release"]
    return {
        "group_n": len(rows),
        "not_applicable_n": len(rows) - len(applicable),
        "empty_window_n": len(applicable) - len(present),
        "valid_window_n": len(present),
        "distributions": stats,
        "fixed_threshold_distance_counts": near,
        "static_max_vs_first_actual_release": dict(Counter(w["static_max_vs_first_actual_release"] for w in present)),
        "static_first_alarm_vs_window": dict(Counter(w["static_first_alarm_vs_window"] for w in present)),
        "intra_first_alarm_vs_window": dict(Counter(w["intra_first_alarm_vs_window"] for w in present)),
        "static_max_vs_static_first_alarm": dict(Counter(w["static"]["max_vs_first_alarm"] for w in present)),
        "intra_max_vs_intra_first_alarm": dict(Counter(w["intra"]["max_vs_first_alarm"] for w in present)),
        "delta_at_static_max_directions": directions([w["intra_minus_static_at_static_max"] for w in present]),
        "delta_at_static_max_after_release_directions": directions([w["intra_minus_static_at_static_max"] for w in active]),
        "delta_at_static_max_after_release_distribution": distribution([w["intra_minus_static_at_static_max"] for w in active]),
    }


def witness(row, times, leads, source, static, intra):
    window = row["windows"]["eligible_history_or_safe_full"]
    if not window["point_count"]:
        return None
    maximum = window["static"]["max_index"]
    first = row["static_score"]["first_index"]
    chosen = {maximum, window["last_index"]}
    if first is not None:
        chosen.add(first)
    for index in (maximum - 1, maximum + 1):
        if 0 <= index < len(times):
            chosen.add(index)
    points = [{"index": int(i), "time_s": float(times[i]), "lead_ms": float(leads[i]),
               "source_probability": float(source[i]), "static_probability": float(static[i]),
               "intra_probability": float(intra[i])} for i in sorted(chosen)]
    need(float(static[maximum]) == window["static"]["max_probability"], "witness max probability")
    need(leads[maximum] > 30, "witness max is not a legal lead")
    return {"shot": row["shot"], "rank": row["rank"],
            "static_status": row["static_score"]["status"],
            "intra_status": row["intra_score"]["status"],
            "hdf_is_disrupt": True, "hdf_down_time_s": row["hdf_down_time_s"],
            "static_window_max_index": maximum, "static_first_alarm_index": first,
            "points": points}


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    need(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "bytecode must be disabled")
    need(os.environ.get("OPENBLAS_NUM_THREADS") == "1" and os.environ.get("OMP_NUM_THREADS") == "1",
         "CPU thread settings must be one")
    need(not OUTPUT.exists(), "output already exists; do not overwrite")
    protocol = read_json(RUN / "protocol.json")
    completed = read_json(RUN / "completed.json")
    saved = read_json(RUN / "result.json")
    need(protocol["status"] == "RISK_TEACHER_INTRA360", "unexpected protocol")
    need(completed["status"] == "SEALED_RISK_TEACHER360", "run is not sealed")
    need(saved["status"] == "HDF5_RISK_TEACHER360", "result is not complete")
    need(protocol["threshold"] == THRESHOLD and protocol["first_affected"] == FIRST_EFFECT,
         "threshold or first effect changed")
    reference, config, archive, (real, stale, warm) = joint.cohort("fixed360")
    order = protocol["order"]
    need(warm == protocol["warm"] == 20, "warm-up mismatch")
    need(len(order) == len(set(order)) == 360 and order == config["order"], "cohort order mismatch")
    need([r["shot"] for r in completed["records"]] == order, "sealed records mismatch")
    expected_files = {f"{rank:04d}_{shot}.npz" for rank, shot in enumerate(order, 1)}
    need({p.name for p in (RUN / "shots").glob("*.npz")} == expected_files,
         "input shot file coverage mismatch")
    need(all(len(saved["rows"][arm]) == 360 for arm in ARMS), "saved rows incomplete")

    rows, all_scores, witnesses = [], {"source": [], "static": [], "intra": []}, {}
    max_source_static_error = 0.
    for rank, (shot, note, record) in enumerate(zip(order, archive["sealed"],
                                                     completed["records"], strict=True), 1):
        need(note["shot"] == record["shot"] == shot and record["rank"] == rank,
             f"shot order mismatch at {rank}")
        times, source, _, _ = joint.recorded(reference, note, rank, real, stale)
        truth, clock = joint.raw_truth(joint.r.io23.path_for("EAST", shot), shot)
        np.testing.assert_array_equal(times, clock, err_msg=f"HDF clock mismatch for {shot}")
        with np.load(RUN / "shots" / f"{rank:04d}_{shot}.npz", allow_pickle=False) as data:
            static, intra = (data[arm] for arm in ARMS)
        need(all(a.ndim == 1 and len(a) == len(clock) for a in (source, static, intra)),
             f"curve/clock length mismatch for {shot}")
        need(all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
                 for a in (source, static, intra)), f"invalid probabilities for {shot}")
        max_source_static_error = max(max_source_static_error, joint.require_parity(static, source))
        scores = {key: joint.score(clock, values, truth, THRESHOLD)
                  for key, values in (("source", source), ("static", static), ("intra", intra))}
        for key in all_scores:
            all_scores[key].append(scores[key])
        need(scores["source"] == scores["static"], f"source/static score mismatch for {shot}")
        for key, arm in (("static", ARMS[0]), ("intra", ARMS[1])):
            need(scores[key] == saved["rows"][arm][rank - 1], f"saved {key} row mismatch for {shot}")
        releases = record["recorded_sampling"]["releases"]
        first_release = int(releases[0]) if releases else None
        need(all(type(i) is int and FIRST_EFFECT <= i < len(clock) for i in releases),
             f"invalid release indices for {shot}")
        need(not releases if rank <= warm or len(clock) <= FIRST_EFFECT else first_release == FIRST_EFFECT,
             f"first release mismatch for {shot}")
        need(len(releases) == record["recorded_sampling"]["updates"]["real"],
             f"recorded update count mismatch for {shot}")

        leads = np.round((truth["down"] - clock) * 1000., 9) if truth["disruptive"] else None
        legal = np.flatnonzero(leads > 30.) if leads is not None else np.arange(len(clock))
        early = np.flatnonzero((leads > 30.) & (leads <= 125.)) if leads is not None else None
        if truth["disruptive"]:
            need(bool(np.any(static[legal] >= THRESHOLD)) == (scores["static"]["status"] == "tp"),
                 f"legal window disagrees with source score for {shot}")
            need(bool(np.any(intra[legal] >= THRESHOLD)) == (scores["intra"]["status"] == "tp"),
                 f"legal window disagrees with intra score for {shot}")
        last_legal = int(legal[-1]) if truth["disruptive"] and len(legal) else None
        make = lambda ix: window_summary(ix, clock, leads, static, intra,
                                         scores["static"], scores["intra"], first_release)
        row = {
            "rank": rank, "shot": shot, "hdf_is_disrupt": truth["disruptive"],
            "hdf_down_time_s": truth["down"], "clock_points": len(clock),
            "clock_start_s": float(clock[0]), "clock_end_s": float(clock[-1]),
            "static_score": scores["static"], "intra_score": scores["intra"],
            "first_actual_release_index": first_release, "recorded_update_count": len(releases),
            "last_legal_warning_index": last_legal,
            "last_legal_before_index_153": last_legal < FIRST_EFFECT if last_legal is not None else None,
            "last_legal_before_actual_first_release": (
                last_legal < first_release if last_legal is not None and first_release is not None else None),
            "windows": {"eligible_history_or_safe_full": make(legal),
                        "early_positive": make(early) if early is not None else None},
        }
        rows.append(row)
        status = scores["static"]["status"]
        if status in ("tardy", "miss") and status not in witnesses:
            witnesses[status] = witness(row, clock, leads, source, static, intra)
        if rank % 60 == 0:
            print(f"validated {rank}/360 sealed shots", flush=True)

    counts = {key: joint.counts(value) for key, value in all_scores.items()}
    need(counts["source"] == counts["static"], "source/static aggregate mismatch")
    for key, arm in (("static", ARMS[0]), ("intra", ARMS[1])):
        need(all(saved["metrics"][arm][k] == v for k, v in counts[key].items()),
             f"saved {key} aggregate mismatch")
    need((counts["static"]["n_disruptive"], counts["static"]["n_safe"],
          counts["static"]["tp"], counts["static"]["fp"],
          counts["intra"]["tp"], counts["intra"]["fp"]) == (180, 180, 116, 29, 118, 30),
         "expected source/static/intra full-stream counts changed")
    need(Counter(row["static_score"]["status"] for row in rows) ==
         Counter({"tp": 116, "tardy": 38, "miss": 26, "fp": 29, "tn": 151}),
         "static categories changed")
    need(all(witnesses.get(key) is not None for key in ("tardy", "miss")),
         "missing two direct error-shot witnesses")

    groups = {}
    for status in STATUSES:
        members = [row for row in rows if row["static_score"]["status"] == status]
        groups[status] = {
            "n": len(members),
            "intra_status_counts": dict(Counter(row["intra_score"]["status"] for row in members)),
            "no_actual_release_n": sum(row["first_actual_release_index"] is None for row in members),
            "last_legal_before_index_153": dict(Counter(
                "null" if row["last_legal_before_index_153"] is None else
                str(row["last_legal_before_index_153"]).lower() for row in members)),
            "last_legal_before_actual_first_release": dict(Counter(
                "null" if row["last_legal_before_actual_first_release"] is None else
                str(row["last_legal_before_actual_first_release"]).lower() for row in members)),
            "windows": {name: grouped_window(members, name) for name in WINDOWS},
        }

    result = {
        "status": "SEALED_FIXED360_CPU_ERROR_MARGIN_DIAGNOSTIC_COMPLETE",
        "inputs": {"run": str(RUN), "sealed_reference": str(reference),
                   "cohort": "fixed360", "scored_arms": ["source_archived_static", *ARMS]},
        "definitions": {
            "threshold": THRESHOLD,
            "threshold_was_not_tuned": True,
            "indexing": "zero-based, on the shared original HDF ip clock",
            "lead_ms": "(raw HDF DownTime - raw HDF clock) * 1000, rounded to 9 decimal places as in score",
            "eligible_history_or_safe_full": "disruptive: all lead > 30 ms; safe: full observed clock with no invented DownTime",
            "early_positive": "disruptive only: 30 < lead <= 125 ms; safe: null",
            "peak_ties": "first index at the maximum probability",
            "margin": "max probability in the window minus the fixed threshold",
            "fixed_distance_counts": "cumulative: -distance <= margin < 0, distances 0.01, 0.05, 0.10",
            "delta_at_static_peak": "intra probability minus static probability at the static arm's peak index",
            "first_effect_index_153": "first theoretically affected zero-based index; first 20 warm shots receive no update",
            "empty_window": "point_count is zero and all peak/margin values are null; safe positive window is null",
            "percentiles": "numpy linear interpolation, over shots with a defined quantity",
            "interpretation": "descriptive paired diagnostic, not threshold selection, causal mechanism, or independent validation",
        },
        "verification": {
            "n_shots": len(rows), "n_disruptive": counts["static"]["n_disruptive"],
            "n_safe": counts["static"]["n_safe"], "expected_input_npz": len(expected_files),
            "clock_exactly_equals_raw_hdf_for_all": True,
            "source_static_intra_curve_lengths_match_clock_for_all": True,
            "recorded_reference_hashes_validated_by_existing_recorded": True,
            "source_static_row_scores_identical": True,
            "saved_static_intra_rows_and_counts_match": True,
            "max_source_static_probability_difference": max_source_static_error,
            "full_stream_counts": counts,
            "static_status_counts": dict(Counter(row["static_score"]["status"] for row in rows)),
            "two_error_shot_direct_points": witnesses,
        },
        "groups_by_static_status": groups,
        "shots": rows,
    }
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with OUTPUT.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(f"wrote {OUTPUT}", flush=True)
    print(json.dumps({"counts": counts, "witnesses": witnesses}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
