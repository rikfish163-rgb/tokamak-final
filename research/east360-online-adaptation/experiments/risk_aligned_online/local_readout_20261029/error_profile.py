"""Retrospective, CPU-only error profile of the sealed five-arm EAST stream.

This reads revealed outcomes only after checking completion and seal hashes. It
does not change the predictor, its feedback, the alarm, or the fixed threshold.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import io23

ARMS = ("cold", "frozen_readout", "local_readout", "local_reverse", "local_hold")
CONTROLS = ("cold", "frozen_readout", "local_reverse", "local_hold")
CANDIDATE = "local_readout"
FIXED_THRESHOLD = 0.7258946895599365
REFERENCE_ATOL = 2e-6


def cosine_stats(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("aligned finite delta vectors required")
    dot = float(a @ b)
    aa, bb = float(a @ a), float(b @ b)
    return {"n": len(a), "dot": dot, "candidate_energy": aa,
            "control_energy": bb,
            "cosine": float(dot / np.sqrt(aa * bb)) if aa and bb else None}


def _feedback(receipt):
    return receipt.get("updates", {}).get("feedback", [])


def causal_feedback_before(receipt, arm, index):
    """Bounds on actual updates applied by index; individual events are not saved."""
    events = _feedback(receipt)
    before = sum(item["clock"] <= index for item in events)
    total = 0 if arm == "cold" else receipt.get("updates", {}).get(arm, {}).get("updates", 0)
    lower, upper = max(0, total - (len(events) - before)), min(total, before)
    return {"opportunities": before, "min_updates": lower, "max_updates": upper,
            "had_update": True if lower else (False if not upper else None)}


def _validate_receipt(receipt, n):
    if receipt.get("outcome_used") is not False:
        raise ValueError("receipt does not establish outcome-free prediction")
    events = _feedback(receipt)
    if not receipt.get("adaptation_active") and (events or receipt.get("updates")):
        raise ValueError("inactive shot has feedback")
    previous = 0
    for e in events:
        clock = e["clock"]
        if (type(clock) is not int or not previous < clock < n
                or not 0 <= e["calibration_last_target"] < e["gradient_first_target"]
                <= e["gradient_last_target"] < clock):
            raise ValueError("non-causal or unordered feedback receipt")
        previous = clock
    for arm in ARMS[1:]:
        summary = receipt.get("updates", {}).get(arm, {})
        opportunities, updates = summary.get("opportunities", 0), summary.get("updates", 0)
        if (type(opportunities) is not int or type(updates) is not int
                or opportunities != len(events) or not 0 <= updates <= opportunities):
            raise ValueError("inconsistent feedback/update counts")


def _reference_guard(cold, other, threshold, end):
    if (np.any(np.abs(cold[:end] - other[:end]) > REFERENCE_ATOL)
            or np.any((cold[:end] >= threshold) != (other[:end] >= threshold))):
        raise ValueError("no-update reference scores or threshold decisions differ")


def _category(disrupt, alarm, timely):
    if not disrupt:
        return "FP" if alarm else "TN"
    return "TP" if timely else ("tardy" if alarm else "FN")


def profile_shot(times, curves, threshold, disruptive, down, receipt, records=None):
    """Profile one sealed shot; down is explicitly retrospective outcome data."""
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(threshold)):
        raise ValueError("finite, strictly increasing times and threshold required")
    if set(curves) != set(ARMS):
        raise ValueError("all five arms required")
    curves = {a: np.asarray(curves[a], dtype=np.float64) for a in ARMS}
    if any(v.shape != times.shape or not np.isfinite(v).all()
           or np.any((v < 0) | (v > 1)) for v in curves.values()):
        raise ValueError("aligned finite probability curves required")
    if disruptive and (down is None or not np.isfinite(down)):
        raise ValueError("disruptive shot requires finite retrospective down time")
    _validate_receipt(receipt, len(times))
    events = _feedback(receipt)
    for arm in ARMS[1:]:
        count = receipt.get("updates", {}).get(arm, {}).get("updates", 0)
        end = (events[0]["clock"] if events and count else len(times))
        _reference_guard(curves["cold"], curves[arm], threshold, end)

    if disruptive:
        raw_lead = (float(down) - times) * 1000.0
        valid = raw_lead >= -1e-6  # matches labels_at's preterminal mask
        lead = np.fromiter((round(float(x), 9) for x in raw_lead), float, len(times))
        timely = valid & (lead > 30.0)
    else:
        valid = np.ones(len(times), dtype=bool)
        lead, timely = None, None
    if not valid.any():
        raise ValueError("no valid preterminal scores")

    def describe(index, arm):
        if index is None:
            return None
        return {"index": int(index), "time_s": float(times[index]),
                "score": float(curves[arm][index]),
                "margin": float(curves[arm][index] - threshold),
                "lead_ms": float(lead[index]) if disruptive else None,
                "feedback_before": causal_feedback_before(receipt, arm, index),
                "candidate_feedback_before": causal_feedback_before(receipt, CANDIDATE, index)}

    def peak(arm, mask):
        indices = np.flatnonzero(mask)
        return describe(int(indices[np.argmax(curves[arm][indices])]), arm) if len(indices) else None

    arms = {}
    for arm in ARMS:
        above = np.flatnonzero(valid & (curves[arm] >= threshold))
        first = describe(int(above[0]), arm) if len(above) else None
        is_timely = bool(disruptive and first is not None and first["lead_ms"] > 30.0)
        item = {"category": _category(disruptive, first is not None, is_timely),
                "timely": is_timely, "first_crossing": first,
                "valid_peak": peak(arm, valid),
                "timely_peak": peak(arm, timely) if disruptive else None}
        if records is not None:
            record = records[arm]
            actual = (receipt["shot"], bool(disruptive), first is not None, is_timely,
                      first["lead_ms"] if first else None)
            saved = (record["shot"], record["is_disrupt"], record["alarm"], record["timely"],
                     record["lead_ms"])
            if actual != saved:
                raise ValueError(f"recorded decision mismatch for shot {receipt['shot']} arm {arm}")
        arms[arm] = item

    comparisons = {}
    candidate_delta = curves[CANDIDATE] - curves["cold"]
    for control in CONTROLS:
        other_delta = curves[control] - curves["cold"]
        a, b = arms[CANDIDATE], arms[control]
        ap = a["timely_peak"] if disruptive else a["valid_peak"]
        bp = b["timely_peak"] if disruptive else b["valid_peak"]
        ac, bc = a["first_crossing"], b["first_crossing"]
        comparisons[control] = {
            "transition": f"{b['category']}->{a['category']}",
            "candidate_minus_control_peak_margin": ap["margin"] - bp["margin"] if ap and bp else None,
            "crossing_relation": ("both" if ac and bc else "candidate_only" if ac else
                                  "control_only" if bc else "neither"),
            "candidate_minus_control_first_crossing_ms":
                float((ac["time_s"] - bc["time_s"]) * 1000) if ac and bc else None,
            "candidate_minus_control_at_control_crossing":
                float(curves[CANDIDATE][bc["index"]] - curves[control][bc["index"]]) if bc else None,
            "valid_delta_cosine": cosine_stats(candidate_delta[valid], other_delta[valid]),
            "timely_delta_cosine": cosine_stats(candidate_delta[timely], other_delta[timely])
                if disruptive else None,
        }
    return {"shot": receipt["shot"], "index": receipt["index"],
            "is_disrupt": bool(disruptive), "down_time_s": float(down) if disruptive else None,
            "baseline_category": arms["cold"]["category"],
            "feedback_opportunities": len(events), "arms": arms, "comparisons": comparisons}


def aggregate(rows):
    categories = ("FP", "FN", "tardy", "TP", "TN")
    strata = {k: {"count": 0, "candidate_peak_feedback": Counter(),
                   "candidate_crossing_feedback": Counter(), "transitions": Counter(),
                   "margin_by_peak_feedback": {}}
              for k in categories}
    pooled = {c: {domain: {"n": 0, "shots": 0, "dot": 0.0,
                            "candidate_energy": 0.0, "control_energy": 0.0}
                  for domain in ("valid", "timely")} for c in CONTROLS}
    for row in rows:
        s = strata[row["baseline_category"]]
        s["count"] += 1
        arm = row["arms"][CANDIDATE]
        for event, counter, missing in ((arm["timely_peak"] if row["is_disrupt"] else arm["valid_peak"],
                                          "candidate_peak_feedback", "no_peak"),
                                         (arm["first_crossing"], "candidate_crossing_feedback", "no_crossing")):
            value = event["feedback_before"]["had_update"] if event else None
            key = missing if not event else ("yes" if value is True else "no" if value is False else "unknown")
            s[counter][key] += 1
            if counter == "candidate_peak_feedback" and event:
                group = s["margin_by_peak_feedback"].setdefault(key, {c: [] for c in CONTROLS})
                for control in CONTROLS:
                    change = row["comparisons"][control]["candidate_minus_control_peak_margin"]
                    if change is not None:
                        group[control].append(change)
        s["transitions"][row["comparisons"]["cold"]["transition"]] += 1
        for control in CONTROLS:
            for domain in ("valid", "timely"):
                part = row["comparisons"][control][domain + "_delta_cosine"]
                if part is None:
                    continue
                dst = pooled[control][domain]
                dst["shots"] += 1
                for field in ("n", "dot", "candidate_energy", "control_energy"):
                    dst[field] += part[field]
    for values in pooled.values():
        for p in values.values():
            aa, bb = p["candidate_energy"], p["control_energy"]
            p["cosine"] = p["dot"] / np.sqrt(aa * bb) if aa and bb else None
    for s in strata.values():
        for group in s["margin_by_peak_feedback"].values():
            for control, values in group.items():
                group[control] = {"count": len(values),
                                  "mean": float(np.mean(values)) if values else None,
                                  "median": float(np.median(values)) if values else None,
                                  "min": float(min(values)) if values else None,
                                  "max": float(max(values)) if values else None}
    return {"shots": len(rows), "baseline_categories": {k: strata[k]["count"] for k in categories},
            "strata": strata, "pooled_delta_cosine": pooled}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_sha(path, expected):
    if _sha(path) != expected:
        raise ValueError(f"sealed hash mismatch: {path}")


def load_sealed_run(directory):
    directory = Path(directory)
    complete_path = directory / "complete.json"
    if not complete_path.is_file():
        raise ValueError("complete.json required before retrospective analysis")
    complete = _read(complete_path)
    if complete.get("complete") is not True:
        raise ValueError("run is not complete")
    for filename, key in (("protocol.json", "protocol_sha256"),
                          ("per_shot.json", "per_shot_sha256"),
                          ("receipts.json", "receipts_sha256")):
        _require_sha(directory / filename, complete[key])
    protocol, decisions, receipts = (_read(directory / filename) for filename in
                                     ("protocol.json", "per_shot.json", "receipts.json"))
    order = protocol["order"]
    if (protocol["arms"] != list(ARMS) or len(order) != 360 or len(set(order)) != 360
            or protocol["warmup"] != 20 or protocol["threshold"] != FIXED_THRESHOLD
            or protocol["inter_updates_after_cold20"] != 0 or protocol["alarm_updates"] != 0
            or set(decisions) != set(ARMS) or len(receipts) != 360
            or any(len(decisions[a]) != 360 for a in ARMS)):
        raise ValueError("unexpected five-arm full360 fixed protocol")
    previous = Path(protocol["previous"])
    _require_sha(previous / "complete.json", protocol["previous_complete_sha256"])
    prior_complete = _read(previous / "complete.json")
    if prior_complete.get("complete") is not True:
        raise ValueError("prior cold reference is not complete")
    _require_sha(previous / "receipts.json", prior_complete["receipts_sha256"])
    prior_receipts = _read(previous / "receipts.json")
    if len(prior_receipts) != len(order):
        raise ValueError("incomplete prior cold reference receipts")
    root = Path(__file__).resolve().parents[3]
    ref = root / protocol["reference"]
    hashes_path = ref / "input_hashes.json"
    _require_sha(hashes_path, protocol["reference_hashes"]["input_hashes.json"])
    return complete, protocol, decisions, receipts, previous, prior_receipts, _read(hashes_path)


def run(directory):
    directory = Path(directory)
    complete, protocol, decisions, receipts, previous, prior_receipts, raw_hashes = load_sealed_run(directory)
    rows, max_ref_error = [], 0.0
    for i, sid in enumerate(protocol["order"]):
        receipt = receipts[i]
        if (receipt["shot"] != sid or receipt["index"] != i + 1
                or receipt["adaptation_active"] != (i >= 20)):
            raise ValueError("shot order or cold20 activation mismatch")
        name = f"{i + 1:04d}_{sid}"
        path, seal = directory / "shots" / f"{name}.npz", directory / "shots" / f"{name}_seal.json"
        if _read(seal) != receipt:
            raise ValueError(f"individual receipt mismatch: {name}")
        _require_sha(path, receipt["sha256"])
        old_path = previous / "shots" / f"{name}.npz"
        old_receipt = _read(previous / "shots" / f"{name}_seal.json")
        if old_receipt != prior_receipts[i] or old_receipt["shot"] != sid or old_receipt["index"] != i + 1:
            raise ValueError(f"prior cold receipt mismatch: {name}")
        _require_sha(old_path, old_receipt["sha256"])
        # Stream the raw file hash; only the small outcome fields are then read.
        _require_sha(io23.path_for("EAST", sid), raw_hashes[str(sid)])
        disruptive, down = io23.read_outcome("EAST", sid)
        with np.load(path, allow_pickle=False) as z, np.load(old_path, allow_pickle=False) as old:
            times = z["times"]
            curves = {arm: z[arm] for arm in ARMS}
            if not np.array_equal(times, old["times"]):
                raise ValueError(f"cold reference times differ: {name}")
            _reference_guard(old["cold"], curves["cold"], FIXED_THRESHOLD, len(times))
            err = float(np.max(np.abs(old["cold"].astype(float) - curves["cold"].astype(float))))
            if abs(err - receipt["cold_max_abs_error"]) > 1e-12:
                raise ValueError(f"cold reference error mismatch: {name}")
            max_ref_error = max(max_ref_error, err)
            records = {a: decisions[a][i] for a in ARMS}
            rows.append(profile_shot(times, curves, FIXED_THRESHOLD, disruptive, down, receipt, records))
    for arm in ARMS:
        expected = complete["results"][arm]["full360"]
        tp = sum(row["arms"][arm]["category"] == "TP" for row in rows)
        fp = sum(row["arms"][arm]["category"] == "FP" for row in rows)
        nd = sum(row["is_disrupt"] for row in rows)
        if (tp, fp, nd, len(rows) - nd) != (expected["tp"], expected["fp"], expected["n_disrupt"], expected["n_safe"]):
            raise ValueError(f"full360 saved metrics mismatch for {arm}")
    return {
        "scope": "RETROSPECTIVE DEVELOPMENT-EXPOSED ANALYSIS; NOT INDEPENDENT CONFIRMATION",
        "outcome_access": "Only after completed stream, per-shot seals and raw-file hash checks; no outcome enters predictions or feedback.",
        "limitations": "Feedback timing follows receipt clocks and aggregate update counts; individual per-clock validity is not independently recorded. Cosines compare risk deltas relative to cold; zero norm is undefined. Pooled sample cosines are descriptive, not independent-shot inference.",
        "method": {"threshold": FIXED_THRESHOLD, "threshold_optimized": False,
                   "lead": "EAST strict >30 ms, first crossing; values rounded to 9 decimal ms as in scorer",
                   "baseline_categories": "FP/TN=safe crossing/no crossing; TP=timely disruptive crossing; tardy=disruptive crossing at <=30ms; FN=disruptive no crossing",
                   "peak": "maximum over strict timely-valid samples for disruption; maximum over valid samples for safe shots",
                   "cosine": "candidate-minus-cold vs control-minus-cold on identical valid or disruption-timely samples",
                   "feedback": "update at clock k is first applicable to index k; causal targets precede clock; partial aggregate counts produce bounded/unknown status"},
        "inputs": {"complete_sha256": _sha(directory / "complete.json"),
                   "protocol_sha256": complete["protocol_sha256"],
                   "per_shot_sha256": complete["per_shot_sha256"],
                   "receipts_sha256": complete["receipts_sha256"],
                   "prior_complete_sha256": protocol["previous_complete_sha256"],
                   "input_hashes_sha256": protocol["reference_hashes"]["input_hashes.json"],
                   "profiler_sha256": _sha(Path(__file__))},
        "guards": {"shot_and_raw_hashes_checked": len(rows), "prior_cold_curves_checked": len(rows),
                   "cold_reference_max_abs_difference": max_ref_error,
                   "pre_feedback_and_no_update_reference_checked": len(rows),
                   "recorded_decisions_and_full360_counts_checked": True},
        "saved_full360_results": {arm: complete["results"][arm]["full360"] for arm in ARMS},
        "summary": aggregate(rows), "per_shot": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.run)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Wrote {len(result['per_shot'])} retrospective shot profiles to {args.output}")
