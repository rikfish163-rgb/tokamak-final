"""CPU-only posthoc analysis of already sealed source probabilities and HDF metadata.

Run from any directory with ``python -B headroom.py``; add ``--write-receipt``
once to save a reproducibility receipt alongside this file. Never opens raw HDF.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "probe_outputs/qualified_source_static360_20261104_v1"
INVENTORY = ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v5"
LOCKED_ORDER = ROOT / "probe_outputs/east_decoder_complete360_v1/protocol.json"
HERE = Path(__file__).resolve().parent
ARMS = ("unmapped", "map_after20", "retrospective_map_all")
FIRST_ADAPTATION = 135  # zero-based; update occurs before risk at this index
EXPECTED_THRESHOLD = 0.7258946895599365


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(times, values, disruptive, down, threshold):
    valid = (down - times) * 1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    hits = np.flatnonzero(valid & (values >= threshold))
    first = int(hits[0]) if len(hits) else None
    lead = round(float((down - times[first]) * 1000), 9) if disruptive and first is not None else None
    return {"alarm": first is not None, "lead_ms": lead, "timely": bool(lead is not None and lead > 30)}, first


def aggregate(rows):
    d = [r for r in rows if r["disruptive"]]
    s = [r for r in rows if not r["disruptive"]]
    tp = sum(r["timely"] for r in d)
    fp = sum(r["alarm"] for r in s)
    return {
        "shots": len(rows), "n_disruptive": len(d), "n_safe": len(s),
        "tp": tp, "fp": fp, "tardy": sum(r["alarm"] and not r["timely"] for r in d),
        "miss": sum(not r["alarm"] for r in d), "tpr": tp / len(d), "fpr": fp / len(s),
        "J": tp / len(d) - fp / len(s),
        "early_over3000": sum(r["timely"] and r["lead_ms"] > 3000 for r in d),
        "diagnostic_oracle_advance_ms": {
            str(h): sum(r["alarm"] and r["lead_ms"] + h > 30 for r in d)
            for h in (0, 5, 10, 15, 20, 25, 30, 50)
        },
        "oracle_is_deployable": False,
    }


def margin_stats(rows, key):
    margins = [r[key] for r in rows if r[key] is not None]
    assert all(x > 0 for x in margins)
    if not margins:
        return {"n": 0}
    return {
        "n": len(margins), "min": min(margins), "median": float(np.median(margins)),
        "max": max(margins),
        "at_most": {str(b): sum(x <= b for x in margins) for b in (0.001, 0.01, 0.05, 0.1, 0.2)},
    }


def group_stats(rows):
    safe_fp = [r for r in rows if not r["disruptive"] and r["alarm"]]
    early = [r for r in rows if r["disruptive"] and r["timely"] and r["lead_ms"] > 3000]
    misses = [r for r in rows if r["disruptive"] and not r["alarm"]]
    tardy = [r for r in rows if r["disruptive"] and r["alarm"] and not r["timely"]]
    not_timely = misses + tardy
    def prefix(collection):
        return sum(r["first_index"] is not None and r["first_index"] < FIRST_ADAPTATION for r in collection)
    return {
        "baseline": aggregate(rows),
        "safe_fp_first_before_135": prefix(safe_fp),
        "safe_fp_first_at_or_after_135": len(safe_fp) - prefix(safe_fp),
        "early_over3000_first_before_135": prefix(early),
        "early_over3000_first_at_or_after_135": len(early) - prefix(early),
        "tp_first_before_135": prefix([r for r in rows if r["disruptive"] and r["timely"]]),
        "tardy_first_before_135": prefix(tardy),
        "no_alarm_miss": {
            "count": len(misses),
            "eligible_margin_any_timely": margin_stats(misses, "margin_any_timely"),
            "eligible_margin_30_to_3000ms": margin_stats(misses, "margin_30_to_3000ms"),
        },
        "tardy": {
            "count": len(tardy),
            "eligible_margin_any_timely": margin_stats(tardy, "margin_any_timely"),
            "eligible_margin_30_to_3000ms": margin_stats(tardy, "margin_30_to_3000ms"),
        },
        "all_not_timely": {
            "count": len(not_timely),
            "eligible_margin_any_timely": margin_stats(not_timely, "margin_any_timely"),
            "eligible_margin_30_to_3000ms": margin_stats(not_timely, "margin_30_to_3000ms"),
        },
    }


def main():
    assert sys.argv[1:] in ([], ["--write-receipt"])
    inv_done = load(INVENTORY / "completed.json")
    assert inv_done["status"] == "PASS"
    for name, expected in inv_done["artifact_sha256"].items():
        assert sha(INVENTORY / name) == expected, ("inventory", name)
    audit = load(INVENTORY / "audit.json")
    assert audit["status"] == "PASS"
    metadata = load(INVENTORY / "hdf5_metadata.json")
    assert len(metadata["EAST"]) == 2953
    east = {r["shot"]: r for r in metadata["EAST"]}
    assert len(east) == len(metadata["EAST"])

    done = load(RUN / "completed.json")
    assert done["status"] == "COMPLETED_STATIC_BASELINE_DIAGNOSTIC"
    assert done["curves"] == 1080 and done["neural_updates"] == 0
    for name, expected in done["artifact_sha256"].items():
        assert sha(RUN / name) == expected, ("sealed run", name)
    protocol = load(RUN / "protocol.json")
    per = load(RUN / "per_shot.json")
    summary = load(RUN / "summary.json")
    order = [r["shot"] for r in protocol["order"]]
    assert len(order) == len(set(order)) == 360
    assert protocol["threshold"] == EXPECTED_THRESHOLD
    assert protocol["arms"] == list(ARMS)
    assert sha(LOCKED_ORDER) == protocol["input_sha256"][str(LOCKED_ORDER)]
    assert order == [r["shot"] for r in load(LOCKED_ORDER)["order"]]
    assert len(done["seals"]) == 360
    assert all([r["shot"] for r in per[arm]] == order for arm in ARMS)

    rows = {arm: [] for arm in ARMS}
    points, ties, checked = 0, 0, 0
    for stream_index, shot in enumerate(order):
        truth = east[shot]
        assert type(truth["disruptive"]) is bool
        disruptive, down = truth["disruptive"], truth["down"]
        assert (isinstance(down, (int, float)) and math.isfinite(down)) if disruptive else down is None
        # stat only: no HDF file contents or raw signal arrays are opened.
        stat = Path(truth["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (truth["size"], truth["mtime_ns"])
        base = RUN / "shots" / f"{stream_index + 1:04d}_{shot}"
        curve_path = base.with_suffix(".npz")
        seal = load(base.with_name(base.name + "_seal.json"))
        assert seal == done["seals"][stream_index]
        assert seal["shot"] == shot and seal["current_outcome_used_in_prediction"] is False
        assert sha(curve_path) == seal["curve_sha256"]
        with np.load(curve_path, allow_pickle=False) as data:
            assert set(data.files) == {"times", *ARMS}
            times = data["times"]
            assert times.ndim == 1 and len(times) == truth["samples"] and len(times) > 0
            assert np.isfinite(times).all() and (np.diff(times) > 0).all()
            assert np.array_equal(times, truth["start"] + np.arange(len(times)) / truth["rate"])
            assert np.array_equal(data["map_after20"], data["unmapped" if stream_index < 20 else "retrospective_map_all"])
            points += len(times)
            for arm in ARMS:
                values = data[arm]
                assert values.shape == times.shape and np.isfinite(values).all()
                assert ((values >= 0) & (values <= 1)).all()
                ties += int(np.count_nonzero(values == EXPECTED_THRESHOLD))
                scored, first = score(times, values, disruptive, down, EXPECTED_THRESHOLD)
                expected = {"shot": shot, "disruptive": disruptive, **scored}
                assert per[arm][stream_index] == expected, (arm, shot, per[arm][stream_index], expected)
                checked += 1
                row = {**expected, "stream_index_1based": stream_index + 1, "first_index": first,
                       "margin_any_timely": None, "margin_30_to_3000ms": None,
                       "eligible_points_any_timely": 0, "eligible_points_30_to_3000ms": 0,
                       "eligible_points_after_135_any_timely": 0}
                if disruptive and not scored["timely"]:
                    raw_lead = (down - times) * 1000
                    lead = np.round(raw_lead, 9)
                    # Check Python's established score-rounding at both decision boundaries.
                    for boundary in (30, 3000):
                        near = np.flatnonzero(np.abs(raw_lead - boundary) < 1e-5)
                        assert all(bool(lead[i] > boundary) == (round(float(raw_lead[i]), 9) > boundary) for i in near)
                    indices = np.arange(len(times))
                    eligible = (indices >= FIRST_ADAPTATION) & (lead > 30)
                    in_3s = eligible & (lead <= 3000)
                    row["eligible_points_any_timely"] = int(eligible.sum())
                    row["eligible_points_30_to_3000ms"] = int(in_3s.sum())
                    row["eligible_points_after_135_any_timely"] = int(((indices > FIRST_ADAPTATION) & (lead > 30)).sum())
                    for mask, key in ((eligible, "margin_any_timely"), (in_3s, "margin_30_to_3000ms")):
                        if mask.any():
                            row[key] = EXPECTED_THRESHOLD - float(np.max(values[mask]))
                            assert row[key] > 0, (arm, shot, key, row[key])
                rows[arm].append(row)

    assert ties == 0 and checked == 1080
    for arm in ARMS:
        for label, subset in (("full360", rows[arm]), ("startup20", rows[arm][:20]), ("suffix340", rows[arm][20:])):
            actual = aggregate(subset)
            assert actual == summary[arm][label], (arm, label, actual, summary[arm][label])

    # Synthetic checks for index 134 vs 135 and 30/3000-ms strict boundaries.
    st = np.arange(137, dtype=float) / 1000
    sv = np.zeros(137, dtype=float)
    sv[134], sv[135] = EXPECTED_THRESHOLD, EXPECTED_THRESHOLD
    result, first = score(st, sv, False, None, EXPECTED_THRESHOLD)
    assert result["alarm"] and first == 134
    sv[134] = 0
    assert score(st, sv, False, None, EXPECTED_THRESHOLD)[1] == 135
    assert round((0.165 - st[135]) * 1000, 9) == 30.0
    assert round((3.135 - st[135]) * 1000, 9) == 3000.0

    by_arm = {}
    for arm in ARMS:
        these = rows[arm]
        by_arm[arm] = {"full360": group_stats(these), "startup20": group_stats(these[:20]), "suffix340": group_stats(these[20:])}
    receipt = {
        "status": "PASS",
        "scope": "POSTHOC_FIXED_SOURCE_THRESHOLD_HEADROOM_ONLY_NOT_PERFORMANCE",
        "threshold_probability": EXPECTED_THRESHOLD,
        "first_adaptation_zero_based": FIRST_ADAPTATION,
        "prefix_fixed_indices": "0..134; risk at 135 may change if supported update occurs",
        "inputs": {
            "source_completed_sha256": sha(RUN / "completed.json"),
            "inventory_completed_sha256": sha(INVENTORY / "completed.json"),
            "inventory_metadata_sha256": sha(INVENTORY / "hdf5_metadata.json"),
            "locked_order_sha256": sha(LOCKED_ORDER),
            "script_sha256": sha(Path(__file__).resolve()),
        },
        "verification": {
            "sealed_shot_files": 360, "curves_scored_and_matched": checked,
            "points_per_arm": points, "threshold_equalities_across_all_arms": ties,
            "all_artifact_and_curve_hashes_match": True,
            "all_cached_clocks_match_inventory": True, "all_outcomes_and_scores_match": True,
            "all_nine_saved_summaries_exact": True, "hdf_stat_matches_inventory": 360,
            "synthetic_boundary_checks": "PASS", "raw_hdf_opened": False,
            "raw_signal_arrays_read": False, "model_or_gpu_used": False,
        },
        "by_arm": by_arm,
        "primary_map_after20_shots": rows["map_after20"],
        "notes": [
            "Margins are threshold minus maximum baseline probability over eligible points; they are not achievable LoRA changes.",
            "All eligibility and outcome use is retrospective; alarm times and prefixes cannot be revised by a causal update.",
            "Existing hashes and recorded seal flags establish artifact consistency, not independent proof of historical execution.",
        ],
    }
    print(json.dumps({"verification": receipt["verification"], "by_arm": by_arm}, indent=2))
    if sys.argv[1:]:
        with (HERE / "receipt.json").open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
