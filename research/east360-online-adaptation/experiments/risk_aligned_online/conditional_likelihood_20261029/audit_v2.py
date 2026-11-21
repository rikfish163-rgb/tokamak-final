"""Read-only, CPU-only independent audit of the archived cold20 neural results.

This module deliberately does not import the experiment's evaluation code.
The optional --write switch creates only the specified audit output directory.
It does not read raw shot files, load a neural model, or run an experiment.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARMS = ("real", "misplaced", "identical")
RADIUS = 0.05


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare_npz(left, right):
    a, b = load_npz(left), load_npz(right)
    require(set(a) == set(b), f"archive keys differ: {left} vs {right}")
    for key in a:
        require(a[key].shape == b[key].shape and a[key].dtype == b[key].dtype,
                f"archive shape/dtype differs: {key} in {left}")
        require(np.array_equal(a[key], b[key]), f"archive array differs: {key} in {left}")
    left_hash, right_hash = sha256(left), sha256(right)
    require(left_hash == right_hash, f"archive bytes differ: {left} vs {right}")
    return b, {"sha256": left_hash, "array_count": len(a), "rows": len(b["ids"])}


def future_groups(times, outcome):
    """Labels/masks from outcome times, including the archived 1e-6 ms tolerance."""
    times = np.asarray(times, dtype=np.float64)
    require(times.ndim == 2 and np.isfinite(times).all(), "future times must be a finite matrix")
    if not outcome["disruptive"]:
        return np.zeros(times.shape, dtype=np.int8), {"safe": np.ones(times.shape, dtype=bool)}
    down = float(outcome["down"])
    require(np.isfinite(down), "invalid disruption time")
    lead_ms = (down - times) * 1000.0
    valid = lead_ms >= -1e-6
    positive = (lead_ms <= 125.0 + 1e-6) & valid
    timely = np.round(lead_ms, 9) > 30.0
    return positive.astype(np.int8), {
        "danger_timely": positive & timely,
        "early_normal": valid & ~positive,
        "danger_late": positive & ~timely,
    }


def origin_timely_eligible(origin, arrival, outcome):
    """Plan's diagnostic endpoint rule, distinct from future-point grouping."""
    require(outcome["disruptive"], "origin danger eligibility requires a disruptive shot")
    origin = np.asarray(origin, dtype=np.float64)
    arrival = np.asarray(arrival, dtype=np.float64)
    require(origin.shape == arrival.shape and np.isfinite(origin).all() and np.isfinite(arrival).all(),
            "invalid origin/arrival shape or time")
    down = float(outcome["down"])
    lead, arrival_lead = (down - origin) * 1000.0, (down - arrival) * 1000.0
    return ((lead >= -1e-6) & (lead <= 125.0 + 1e-6) & (arrival_lead >= -1e-6)
            & (np.round(arrival_lead, 9) > 30.0))


def binary_cross_entropy(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    require(logits.shape == labels.shape, "BCE shape mismatch")
    require(np.isfinite(logits).all() and np.isin(labels, (0, 1)).all(), "invalid BCE inputs")
    return np.logaddexp(0.0, logits) - labels * logits


def aggregate_rows(rows):
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for group in sorted({r["group"] for r in rows if r["arm"] == arm}):
            selected = [r for r in rows if r["arm"] == arm and r["group"] == group]
            ids = [r["shot"] for r in selected]
            require(len(ids) == len(set(ids)), "duplicate shot in group")
            delta = np.array([r["delta_bce"] for r in selected], dtype=np.float64)
            groups[arm][group] = {
                "shots": len(selected),
                "windows": sum(r["windows"] for r in selected),
                "timepoints": sum(r["timepoints"] for r in selected),
                "mean_delta_bce": float(delta.mean()),
                "median_delta_bce": float(np.median(delta)),
                "improved": int((delta < 0).sum()),
                "worsened": int((delta > 0).sum()),
                "unchanged": int((delta == 0).sum()),
            }
    return groups


def direction_gate(groups):
    failures = []
    for name in ("safe", "danger_timely"):
        real = groups.get("real", {}).get(name)
        control = groups.get("misplaced", {}).get(name)
        if not (real is not None and control is not None and real["shots"] >= 2
                and real["mean_delta_bce"] < 0
                and real["improved"] * 2 >= real["shots"]
                and real["mean_delta_bce"] < control["mean_delta_bce"]):
            failures.append(name)
    return not failures, failures


def validate_folds(folds, all_ids):
    universe = set(all_ids)
    require(len(all_ids) == 20 and len(universe) == 20, "expected 20 distinct outcomes")
    require(len(folds) == 5 and {f["fold"] for f in folds} == set(range(5)), "expected five folds")
    held_all = []
    for spec in folds:
        fit, held = spec["fit"], spec["held"]
        require(len(fit) == 16 and len(held) == 4, "fold must have 16 fit and 4 held shots")
        require(len(set(fit)) == len(fit) and len(set(held)) == len(held), "duplicate fold shot")
        require(not (set(fit) & set(held)), "fit/held overlap")
        require(set(fit) | set(held) == universe, "fold does not cover outcome universe")
        held_all.extend(held)
    require(Counter(held_all) == Counter(all_ids), "held coverage is not exactly once")


def validate_identical_control(sealed, neural):
    identical = sealed["identical_logs"]
    require(identical.ndim == 2 and identical.shape[1] == 2
            and np.array_equal(identical[:, 0], identical[:, 1]), "identical densities differ")
    require(np.array_equal(sealed["identical_counts"], sealed["real_counts"]),
            "identical channel counts differ")
    if "real_logs" in sealed:
        require(np.array_equal(identical[:, 0], sealed["real_logs"][:, 0]),
                "identical class is not the real class-zero density")
    require(np.array_equal(neural["identical_logits"], neural["baseline_logits"]),
            "identical predictions changed")
    for suffix in ("state_norm", "evidence_gradient", "changed"):
        require(np.count_nonzero(neural["identical_" + suffix]) == 0,
                "identical control has nonzero " + suffix)


def validate_norms(neural):
    counts, maxima = {}, {}
    for arm in ARMS:
        norm, changed = neural[arm + "_state_norm"], neural[arm + "_changed"]
        require(norm.ndim == 1 and norm.shape == changed.shape and changed.dtype == np.bool_,
                arm + " norm/changed shape mismatch")
        require(np.isfinite(norm).all() and (norm >= 0).all() and (norm <= RADIUS).all(),
                arm + " norm outside [0, 0.05]")
        require(np.array_equal(changed, norm > 0), arm + " changed flags disagree with norm")
        counts[arm] = int(np.count_nonzero(norm))
        maxima[arm] = float(norm.max(initial=0))
    return counts, maxima


def validate_episode_alignment(sealed, neural):
    """Join episode origins to sealed endpoints; check maturity and next horizon."""
    pairs = [(int(sid), int(end)) for sid, end in zip(sealed["ids"], sealed["ends"])]
    require(len(pairs) == len(set(pairs)), "duplicate sealed endpoint")
    lookup = {pair: i for i, pair in enumerate(pairs)}
    neural_pairs = [(int(sid), int(end)) for sid, end in zip(neural["ids"], neural["ends"])]
    require(len(neural_pairs) == len(set(neural_pairs)), "duplicate neural endpoint")
    expected = {pair for pair in pairs if (pair[0], pair[1] + 25) in lookup}
    require(set(neural_pairs) == expected, "neural endpoints do not match complete subsequent horizons")
    indices = np.array([lookup[pair] for pair in neural_pairs], dtype=int)
    next_indices = np.array([lookup[(sid, end + 25)] for sid, end in neural_pairs], dtype=int)
    arrival = sealed["arrival_time"][indices]
    future = neural["future_times"]
    require(future.shape == (len(indices), 25) and np.isfinite(future).all(), "bad future time shape")
    require(np.array_equal(arrival, sealed["origin_time"][next_indices]), "arrival/next origin mismatch")
    require((future[:, 0] > arrival).all() and (np.diff(future, axis=1) > 0).all(),
            "future time is not strictly after maturity or not increasing")
    require(np.array_equal(future[:, -1], sealed["arrival_time"][next_indices]),
            "future horizon does not end at next arrival")
    return indices


def _score_shot(data, selection, shot, fold, outcome):
    times = data["future_times"][selection]
    labels, masks = future_groups(times, outcome)
    before = data["baseline_logits"][selection].astype(np.float64)
    before_loss = binary_cross_entropy(before, labels)
    rows = []
    for arm in ARMS:
        after = data[arm + "_logits"][selection].astype(np.float64)
        after_loss = binary_cross_entropy(after, labels)
        for group, mask in masks.items():
            if not mask.any():
                continue
            diff = after - before
            rows.append({"shot": int(shot), "fold": int(fold), "group": group, "arm": arm,
                         "windows": int(mask.any(axis=1).sum()), "timepoints": int(mask.sum()),
                         "before_bce": float(before_loss[mask].mean()),
                         "after_bce": float(after_loss[mask].mean()),
                         "delta_bce": float((after_loss - before_loss)[mask].mean()),
                         "mean_logit_change": float(diff[mask].mean()),
                         "max_abs_logit_change": float(np.abs(diff)[mask].max())})
    return rows, masks


def _compare_published_rows(rows, published):
    def key(row):
        return row["shot"], row["fold"], row["arm"], row["group"]
    left, right = {key(r): r for r in rows}, {key(r): r for r in published}
    require(len(left) == len(rows) and len(right) == len(published), "duplicate published/per-shot key")
    require(left.keys() == right.keys(), "published/per-shot coverage differs")
    maximum = 0.0
    for k in left:
        require(left[k].keys() == right[k].keys(), "published/per-shot fields differ")
        for field in ("windows", "timepoints"):
            require(left[k][field] == right[k][field], "published per-shot count differs")
        for field in ("before_bce", "after_bce", "delta_bce", "mean_logit_change", "max_abs_logit_change"):
            diff = abs(left[k][field] - right[k][field])
            maximum = max(maximum, diff)
            require(diff <= 1e-12, "published per-shot numerical result differs")
    return maximum


def _validate_density_training_ids(path, fit_ids, held_ids):
    # Only inspect model provenance metadata; no density calculation or diagnosis.
    import torch
    model = torch.load(path, map_location="cpu", weights_only=True)
    require(len(model["channels"]) == 23, "unexpected number of density channels")
    classes = 0
    for channel in model["channels"]:
        require(len(channel["classes"]) == 2, "unexpected density class count")
        for entry in channel["classes"]:
            shot_ids = entry["shot_ids"]
            require(len(shot_ids) == entry["nshots"] and len(shot_ids) == len(set(shot_ids)),
                    "density model shot provenance count mismatch")
            require(set(shot_ids) <= set(fit_ids) and not (set(shot_ids) & set(held_ids)),
                    "held shot appears in fitted density metadata")
            classes += 1
    return classes


def audit(v1, v2):
    v1, v2 = Path(v1), Path(v2)
    p1, p2 = read_json(v1 / "protocol.json"), read_json(v2 / "protocol.json")
    require(p1["folds"] == p2["folds"], "v1/v2 fold specifications differ")
    outcomes = read_json(v2 / "outcomes.json")
    require(read_json(v1 / "outcomes.json") == outcomes, "v1/v2 outcomes differ")
    all_ids = [int(sid) for sid in outcomes]
    validate_folds(p2["folds"], all_ids)
    observations, observations_equality = compare_npz(v1 / "observations.npz", v2 / "observations.npz")
    require(set(map(int, np.unique(observations["ids"]))) == set(all_ids), "observation coverage differs")
    require(read_json(v2 / "collection.json")["observations_sha256"] == observations_equality["sha256"],
            "collection observations hash mismatch")
    require(read_json(v2 / "collection.json")["outcome_reads"] == 0, "collection reports outcome reads")
    expected_fit_access, expected_eval_access = [], []
    for spec in p2["folds"]:
        expected_fit_access.extend((spec["fold"], sid, "fit") for sid in spec["fit"])
        expected_eval_access.extend((spec["fold"], sid, "evaluation_after_all_seals") for sid in spec["held"])
    for filename, expected in (("training_access.json", expected_fit_access),
                               ("evaluation_access.json", expected_eval_access)):
        accesses = read_json(v2 / filename)
        require([(x["fold"], x["shot"], x["role"]) for x in accesses] == expected,
                filename + " does not match fold roles")

    arrays, fold_details, rows, restricted_rows = {}, [], [], []
    totals = Counter()
    origin_over_shots = set()
    maxima = {arm: 0.0 for arm in ARMS}
    metadata_classes = 0
    min_after_arrival, max_after_arrival = float("inf"), float("-inf")
    evidence_max_error, norm_formula_max_error = 0.0, 0.0
    origin_logit_max_difference, timing_max_error = 0.0, 0.0
    for spec in p2["folds"]:
        fold = spec["fold"]
        d1, d2 = v1 / f"fold{fold}", v2 / f"fold{fold}"
        sealed, sealed_equal = compare_npz(d1 / "sealed.npz", d2 / "sealed.npz")
        neural, neural_equal = compare_npz(d1 / "neural.npz", d2 / "neural.npz")
        arrays[f"fold{fold}/sealed.npz"] = sealed_equal
        arrays[f"fold{fold}/neural.npz"] = neural_equal
        for d in (d1, d2):
            for filename, info in (("sealed.json", sealed_equal), ("neural_seal.json", neural_equal)):
                seal = read_json(d / filename)
                require(seal["sha256"] == info["sha256"], "stored seal hash mismatch")
                require(seal["fit_ids"] == spec["fit"] and seal["held_ids"] == spec["held"],
                        "stored seal split mismatch")
                require(seal["held_outcome_reads"] == 0, "seal reports held outcome reads")
            excluded = read_json(d / "neural_seal.json")["exclusions"]
            require(len(excluded) == len(spec["held"])
                    and {x["shot"] for x in excluded} == set(spec["held"]),
                    "neural exclusion coverage differs")
            for item in excluded:
                sid = item["shot"]
                actual = int((sealed["ids"] == sid).sum() - (neural["ids"] == sid).sum())
                require(item["reason"] == "incomplete next25" and item["windows"] == actual,
                        "neural incomplete-horizon exclusion differs")
        require(set(map(int, np.unique(sealed["ids"]))) == set(spec["held"]), "sealed held coverage differs")
        require(set(map(int, np.unique(neural["ids"]))) == set(spec["held"]), "neural held coverage differs")
        expected = np.isin(observations["ids"], spec["held"])
        for field in ("ids", "ends", "origin_time", "arrival_time", "logit"):
            require(np.array_equal(sealed[field], observations[field][expected]),
                    "sealed field differs from collected observations: " + field)
        for sid in spec["held"]:
            ends = sealed["ends"][sealed["ids"] == sid]
            require(len(ends) >= 2 and ends[0] == 127 and (np.diff(ends) == 25).all(),
                    "unexpected endpoint sequence")
        indices = validate_episode_alignment(sealed, neural)
        validate_identical_control(sealed, neural)
        counts, fold_maxima = validate_norms(neural)
        totals.update(counts)
        maxima = {arm: max(maxima[arm], fold_maxima[arm]) for arm in ARMS}
        first_gap = neural["future_times"][:, 0] - sealed["arrival_time"][indices]
        min_after_arrival = min(min_after_arrival, float(first_gap.min()))
        max_after_arrival = max(max_after_arrival, float(first_gap.max()))
        maturity_gap = sealed["arrival_time"][indices] - sealed["origin_time"][indices]
        timing_max_error = max(timing_max_error, float(np.max(np.abs(maturity_gap - 0.025))),
                               float(np.max(np.abs(first_gap - 0.001))),
                               float(np.max(np.abs(np.diff(neural["future_times"], axis=1) - 0.001))))
        require(timing_max_error <= 1e-11, "saved timeline differs from 1 ms cadence / H25 maturity")
        origin = neural["origin_logit"].astype(np.float64)
        sealed_origin = sealed["logit"][indices].astype(np.float64)
        origin_logit_max_difference = max(origin_logit_max_difference, float(np.max(np.abs(origin - sealed_origin))))
        totals["origin_logits_exact"] += int((origin == sealed_origin).sum())
        probability = np.exp(-np.logaddexp(0.0, -origin))
        for arm in ARMS:
            logs = sealed[arm + "_logs"][indices]
            counts_for_episode = sealed[arm + "_counts"][indices]
            totals["no_channels_" + arm] += int((counts_for_episode == 0).sum())
            totals["future_logit_changes_" + arm] += int(np.count_nonzero(neural[arm + "_logits"] != neural["baseline_logits"]))
            posterior = np.exp(-np.logaddexp(0.0, -(origin + logs[:, 1] - logs[:, 0])))
            gradient = np.where(counts_for_episode > 0, probability - posterior, 0.0)
            evidence_error = float(np.max(np.abs(gradient - neural[arm + "_evidence_gradient"])))
            evidence_max_error = max(evidence_max_error, evidence_error)
            require(evidence_error <= 1e-12, "saved evidence gradient differs from independent p-r")
            norms = neural[arm + "_state_norm"].astype(np.float64)
            require((norms[counts_for_episode == 0] == 0).all(), "nonzero state without observed channels")
            active = norms > 0
            if active.any():
                norm_error = float(np.max(np.abs(norms[active] - RADIUS * np.abs(gradient[active]))))
                norm_formula_max_error = max(norm_formula_max_error, norm_error)
                require(norm_error <= 1e-7, "nonzero state norm differs from bounded feedback formula")
        model_hashes = {}
        for arm in ("real", "misplaced"):
            model_hashes[arm] = sha256(d2 / (arm + "_density.pt"))
            require(model_hashes[arm] == sha256(d1 / (arm + "_density.pt")), "v1/v2 density model bytes differ")
            metadata_classes += _validate_density_training_ids(d2 / (arm + "_density.pt"), spec["fit"], spec["held"])
        for sid in spec["held"]:
            selected = neural["ids"] == sid
            shot_rows, masks = _score_shot(neural, selected, sid, fold, outcomes[str(sid)])
            rows.extend(shot_rows)
            combined = np.logical_or.reduce(list(masks.values()))
            totals["episodes"] += int(selected.sum())
            totals["future_points"] += int(combined.size)
            totals["valid_future_points"] += int(combined.sum())
            totals["postterminal_future_points"] += int((~combined).sum())
            totals["episodes_no_valid_future"] += int((~combined.any(axis=1)).sum())
            totals["episodes_partial_valid_future"] += int((combined.any(axis=1) & ~combined.all(axis=1)).sum())
            if outcomes[str(sid)]["disruptive"]:
                down = float(outcomes[str(sid)]["down"])
                arrival = sealed["arrival_time"][indices[selected]]
                totals["episodes_arrival_postterminal"] += int((((down - arrival) * 1000) < -1e-6).sum())
                origin = sealed["origin_time"][indices[selected]]
                eligible = origin_timely_eligible(origin, arrival, outcomes[str(sid)])
                over_125 = ((down - origin) * 1000) > 125.0 + 1e-6
                timely = masks["danger_timely"]
                totals["future_timely_timepoints"] += int(timely.sum())
                totals["future_timely_points_origin_over_125_ms"] += int(timely[over_125].sum())
                totals["future_timely_points_origin_eligible"] += int(timely[eligible].sum())
                totals["future_timely_episodes_origin_over_125_ms"] += int(timely[over_125].any(axis=1).sum())
                totals["future_timely_episodes_origin_eligible"] += int(timely[eligible].any(axis=1).sum())
                totals["future_timely_points_origin_other_ineligible"] += int(timely[~eligible & ~over_125].sum())
                if timely[over_125].any():
                    origin_over_shots.add(int(sid))
                mask = timely & eligible[:, None]
                if mask.any():
                    labels, _ = future_groups(neural["future_times"][selected], outcomes[str(sid)])
                    before = neural["baseline_logits"][selected].astype(np.float64)
                    before_loss = binary_cross_entropy(before, labels)
                    for arm in ARMS:
                        after = neural[arm + "_logits"][selected].astype(np.float64)
                        delta = binary_cross_entropy(after, labels) - before_loss
                        restricted_rows.append({"shot": int(sid), "fold": fold, "group": "danger_timely",
                                                "arm": arm, "windows": int(mask.any(axis=1).sum()),
                                                "timepoints": int(mask.sum()), "delta_bce": float(delta[mask].mean())})
        fold_details.append({"fold": fold, "fit": spec["fit"], "held": spec["held"],
                             "sealed_rows": len(sealed["ids"]), "neural_episodes": len(neural["ids"]),
                             "nonzero_state_norm": counts, "max_state_norm": fold_maxima,
                             "density_model_sha256": model_hashes})
    groups = aggregate_rows(rows)
    gate, failed = direction_gate(groups)
    published = read_json(v2 / "completed.json")["neural"]
    maximum_row_error = _compare_published_rows(rows, read_json(v2 / "neural_per_shot.json"))
    require(not read_json(v2 / "neural_exclusions.json"), "published neural exclusions are nonempty")
    for arm in ARMS:
        require(groups[arm].keys() == published["groups"][arm].keys(), "published group coverage differs")
        for name, values in groups[arm].items():
            for field, actual in published["groups"][arm][name].items():
                require(field in values and (abs(values[field] - actual) <= 1e-12),
                        "published group statistic differs: " + arm + "/" + name + "/" + field)
    require(gate == published["eligible_for_full_stream"], "published gate differs")
    require(published["reasons"] == [name + " did not pass paired direction screen" for name in failed],
            "published gate reasons differ")
    nonzero = {arm: totals[arm] for arm in ARMS}
    require(nonzero == published["actual_nonzero_episode_updates"], "published update counts differ")
    restricted_groups = aggregate_rows([row for row in rows if row["group"] == "safe"] + restricted_rows)
    restricted_gate, restricted_failures = direction_gate(restricted_groups)
    return {
        "arrays": {"all_v1_v2_equal": True, "observations": observations_equality, "fold_files": arrays},
        "folds": {"unique_held": len(set(all_ids)), "details": fold_details,
                  "fit_access_records": len(expected_fit_access), "held_evaluation_access_records": len(expected_eval_access),
                  "density_class_metadata_records": metadata_classes},
        "counts": {"nonzero_state_norm": nonzero,
                   "future_logit_changes": {a: totals["future_logit_changes_" + a] for a in ARMS},
                   "no_supported_channels": {a: totals["no_channels_" + a] for a in ARMS},
                   **{k: totals[k] for k in
                    ("episodes", "future_points", "valid_future_points", "postterminal_future_points",
                     "episodes_no_valid_future", "episodes_partial_valid_future", "episodes_arrival_postterminal",
                     "origin_logits_exact")}},
        "bounds": {"state_norm_maxima": maxima, "max_evidence_gradient_abs_error": evidence_max_error,
                   "max_nonzero_norm_formula_abs_error": norm_formula_max_error,
                   "max_origin_logit_abs_difference": origin_logit_max_difference,
                   "max_timing_cadence_abs_error_seconds": timing_max_error,
                   "first_future_minus_arrival_seconds_min": min_after_arrival,
                   "first_future_minus_arrival_seconds_max": max_after_arrival},
        "groups": groups,
        "gate": {"eligible": gate, "failed_groups": failed},
        "origin_boundary": {**{k: totals[k] for k in ("future_timely_timepoints",
                            "future_timely_points_origin_over_125_ms", "future_timely_points_origin_eligible",
                            "future_timely_episodes_origin_over_125_ms", "future_timely_episodes_origin_eligible",
                            "future_timely_points_origin_other_ineligible")},
                            "shots_with_future_timely_points_origin_over_125_ms": sorted(origin_over_shots),
                            "restricted_groups": restricted_groups,
                            "restricted_gate": {"eligible": restricted_gate, "failed_groups": restricted_failures},
                            "note": "Exploratory intersection of the plan's origin/arrival rule and the archived future-time group; not the published gate."},
        "comparisons": {"published_neural_matches": True, "published_per_shot_rows": len(rows),
                        "maximum_per_shot_float_abs_error": maximum_row_error},
        "per_shot": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--write", action="store_true", help="create the fixed authorized audit output directory")
    writes.add_argument("--write-boundary", action="store_true", help="add origin-boundary analysis without overwriting prior audit output")
    writes.add_argument("--write-final", action="store_true", help="add the final complete summary without overwriting prior output")
    args = parser.parse_args()
    output = ROOT / "probe_outputs/conditional_likelihood_cold20_audit"
    report = audit(ROOT / "probe_outputs/conditional_likelihood_cold20_v1",
                   ROOT / "probe_outputs/conditional_likelihood_cold20_v2")
    if args.write_final:
        require(output.is_dir(), "audit directory must exist for the final summary")
        report.pop("per_shot")
        with (output / "audit_final.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    elif args.write_boundary:
        require(output.is_dir(), "audit directory must exist for the boundary sidecar")
        with (output / "origin_boundary.json").open("x", encoding="utf-8") as stream:
            json.dump(report["origin_boundary"], stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report["origin_boundary"], indent=2, allow_nan=False))
    elif args.write:
        output.mkdir(exist_ok=False)
        rows = report.pop("per_shot")
        for name, value in (("audit.json", report), ("per_shot.json", rows)):
            with (output / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
