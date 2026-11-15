#!/usr/bin/env python3
"""Two-prefix, shot-disjoint coverage audit of the frozen §63/§65 artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
WEAK = ROOT / "outputs/east360_weak_signal_probe_20261114_v2"
SHARED = ROOT / "outputs/east360_shared_motion_20261114_v1"
OUT = Path(__file__).with_name("result.json")
OFFSET = 32
CALIPERS = np.array([0.50, 0.35, 0.35, 0.50, 0.35, 0.25])
VIEWS = {"raw_trajectory": 1, "frozen_cls": 3}
CONTEXT = ["parent_logit_margin", "ip", "bt", "ip_change", "log_elapsed", "excitation"]


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_candidates(shot, group, rank, attributed, sealed):
    name = f"{rank:04d}_{shot}.npz"
    if name not in sealed[0] or name not in sealed[1]:
        raise ValueError(f"{name}: missing from an input seal inventory")
    weak = load_npz(WEAK / "shots" / name)
    shared = load_npz(SHARED / "shots" / name)
    ends, dose, valid = weak["ends"], weak["dose"], weak["valid"]
    releases = shared["compare_at_release"][:, 0]
    if not np.all(np.isfinite(releases)) or not np.all(releases == releases.astype(int)):
        raise ValueError(f"{name}: invalid release indices")
    releases = releases.astype(int)
    if np.any(releases < 0) or np.any(releases >= len(shared["dose"])):
        raise ValueError(f"{name}: release outside saved dose curve")
    if group == "never_alarm_safe":
        selected = releases[shared["dose"][releases] > 0]
        reason = "all saved nonzero-dose handoffs; original scorer recorded no alarm"
    else:
        a = attributed[shot]
        if a["kind"] != ("late" if group in ("no_first_alarm", "late_first_alarm") else "false"):
            raise ValueError(f"{name}: attribution class does not match scorer")
        indices = np.asarray(a["endpoints"], dtype=int)
        active = np.asarray(a["nonzero_dose"], dtype=bool)
        if len(indices) != len(active) or not np.isin(indices, releases).all():
            raise ValueError(f"{name}: attribution endpoints do not match sealed handoffs")
        if not np.array_equal(active, shared["dose"][indices] > 0):
            raise ValueError(f"{name}: stored attribution dose flags disagree")
        selected = indices[active]
        reason = ("§65 handoff in strict (30,125] ms; original scorer missed" if a["kind"] == "late"
                  else "§65 handoff at or before original first false alarm")
    rows = []
    for u in selected:
        i = int(np.searchsorted(ends, u, side="right")) - 1
        j = i - 8  # §63 records every four samples; the other anchor is exactly 32 samples earlier.
        if j < 0 or ends[i] - ends[j] != OFFSET:
            continue
        if not 0 <= u - ends[i] < 4:
            raise ValueError(f"{name}: handoff is not adjacent to the saved §63 grid")
        if not (valid[i] and valid[j] and dose[i] > 0 and dose[j] > 0):
            continue
        if not (np.isfinite(weak["context"][[i, j]]).all() and
                np.isfinite(weak["score"][[i, j]]).all()):
            raise ValueError(f"{name}: non-finite eligible context or probe score")
        rows.append({"shot": shot, "group": group, "release": int(u),
                     "q": int(ends[i]), "earlier": int(ends[j]),
                     "context": weak["context"][i].astype(float),
                     "context_earlier": weak["context"][j].astype(float),
                     "scores": weak["score"][i].astype(float),
                     "scores_earlier": weak["score"][j].astype(float)})
    return rows, {"shot": shot, "group": group, "selection": reason,
                  "nonzero_handoffs": int(len(selected)), "two_prefix_windows": len(rows)}


def candidate_edges(positive, safe, two_prefixes):
    """Collapse eligible endpoint pairs to the least-distance pair per shot pair."""
    scale = np.tile(CALIPERS, 2 if two_prefixes else 1)
    def vector(row):
        return np.r_[row["context"], row["context_earlier"]] if two_prefixes else row["context"]
    if not positive or not safe:
        return {}
    px = np.array([vector(x) for x in positive])
    sx = np.array([vector(x) for x in safe])
    tree = cKDTree(sx / scale)
    edges = {}
    for i, neighbors in enumerate(tree.query_ball_point(px / scale, r=1 + 1e-12, p=np.inf)):
        for j in neighbors:
            difference = np.abs(px[i] - sx[j])
            if np.any(difference > scale):
                continue
            distance = float(np.square(difference / scale).sum())
            p, s = positive[i], safe[j]
            key = (p["shot"], s["shot"])
            # The final two terms make endpoint selection deterministic; probe scores are never used.
            choice = (distance, p["release"], s["release"])
            if key not in edges or choice < edges[key][0]:
                edges[key] = (choice, i, j)
    return edges


def assignment(positive_shots, safe_shots, edges, excluded=None):
    """Maximize disjoint shots, then minimize total 12-coordinate distance."""
    positive_shots, safe_shots = sorted(positive_shots), sorted(safe_shots)
    n, m = len(positive_shots), len(safe_shots)
    pi = {shot: i for i, shot in enumerate(positive_shots)}
    si = {shot: i for i, shot in enumerate(safe_shots)}
    dummy = 13.0 * (n + 1)  # every feasible two-prefix distance is at most 12
    cost = np.full((n, m + n), 2 * dummy)
    cost[:, m:] = dummy
    for (p, s), (choice, _, _) in edges.items():
        if (p, s) != excluded:
            cost[pi[p], si[s]] = choice[0]
    rows, cols = linear_sum_assignment(cost)
    pairs = [(positive_shots[i], safe_shots[j]) for i, j in zip(rows, cols)
             if j < m and cost[i, j] < dummy]
    return pairs, float(sum(edges[key][0][0] for key in pairs))


def scores(row, earlier=False):
    data = row["scores_earlier"] if earlier else row["scores"]
    return {name: float(data[index]) for name, index in VIEWS.items()}


def make_pair(key, edges, positive, safe):
    choice, i, j = edges[key]
    a, b = positive[i], safe[j]
    result = {"positive_shot": a["shot"], "positive_group": a["group"],
              "safe_shot": b["shot"], "safe_group": b["group"],
              "match_distance": choice[0], "anchors": {}}
    for name, earlier in (("earlier_32", True), ("current", False)):
        ac = a["context_earlier"] if earlier else a["context"]
        bc = b["context_earlier"] if earlier else b["context"]
        av, bv = scores(a, earlier), scores(b, earlier)
        result["anchors"][name] = {
            "positive_index": a["earlier"] if earlier else a["q"],
            "safe_index": b["earlier"] if earlier else b["q"],
            "positive_scores": av, "safe_scores": bv,
            "positive_minus_safe": {view: av[view] - bv[view] for view in VIEWS},
            "absolute_context_difference": dict(zip(CONTEXT, np.abs(ac - bc).tolist())),
        }
        if np.any(np.abs(ac - bc) > CALIPERS):
            raise ValueError("returned pair violates a matching tolerance")
    result["positive_handoff"] = a["release"]
    result["safe_handoff"] = b["release"]
    return result


def summarize(pairs, group):
    rows = [r for r in pairs if r["positive_group"] == group]
    out = {"matched_shots": len(rows)}
    for view in VIEWS:
        margins = {anchor: [r["anchors"][anchor]["positive_minus_safe"][view] for r in rows]
                   for anchor in ("earlier_32", "current")}
        out[view] = {anchor: {"correct": sum(x > 0 for x in values),
                              "ties": sum(x == 0 for x in values)}
                     for anchor, values in margins.items()}
        out[view]["correct_at_both"] = sum(x > 0 and y > 0 for x, y in
                                           zip(margins["earlier_32"], margins["current"]))
    return out


def audit(control, positive, negative, positive_info, negative_info):
    pshots = [r["shot"] for r in positive_info]
    sshots = [r["shot"] for r in negative_info]
    only_now = candidate_edges(positive, negative, False)
    both = candidate_edges(positive, negative, True)
    selected, cost = assignment(pshots, sshots, both)
    if len({x[0] for x in selected}) != len(selected) or len({x[1] for x in selected}) != len(selected):
        raise ValueError("assignment reused a shot")
    pairs = [make_pair(key, both, positive, negative) for key in selected]
    p_assigned = {p for p, _ in selected}
    s_assigned = {s for _, s in selected}
    for row, index, matched in ((row, 0, p_assigned) for row in positive_info):
        row["current_candidate_shots"] = len({key[1] for key in only_now if key[index] == row["shot"]})
        row["two_prefix_candidate_shots"] = len({key[1] for key in both if key[index] == row["shot"]})
        row["status"] = ("matched" if row["shot"] in matched else
                         "no_two_prefix_windows" if not row["two_prefix_windows"] else
                         "no_current_context_match" if not row["current_candidate_shots"] else
                         "no_two_prefix_context_match" if not row["two_prefix_candidate_shots"] else
                         "unmatched_due_to_shot_competition")
    for row in negative_info:
        row["current_candidate_shots"] = len({key[0] for key in only_now if key[1] == row["shot"]})
        row["two_prefix_candidate_shots"] = len({key[0] for key in both if key[1] == row["shot"]})
        row["status"] = ("matched" if row["shot"] in s_assigned else
                         "no_two_prefix_windows" if not row["two_prefix_windows"] else
                         "no_current_context_match" if not row["current_candidate_shots"] else
                         "no_two_prefix_context_match" if not row["two_prefix_candidate_shots"] else
                         "unmatched_due_to_shot_competition")
    alternatives = []
    for key in selected:
        alternative, alt_cost = assignment(pshots, sshots, both, excluded=key)
        item = {"excluded_shot_pair": list(key), "matched_shots": len(alternative),
                "coverage_falls": len(alternative) < len(selected)}
        if len(alternative) == len(selected):
            alt_pairs = [make_pair(k, both, positive, negative) for k in alternative]
            item["additional_distance"] = alt_cost - cost
            item["strata"] = {g: summarize(alt_pairs, g) for g in ("no_first_alarm", "late_first_alarm")}
        alternatives.append(item)
    strata = {}
    for group in ("no_first_alarm", "late_first_alarm"):
        info = [x for x in positive_info if x["group"] == group]
        stats = summarize(pairs, group)
        stats.update({"denominator_shots": len(info),
                      "shots_with_two_prefix_windows": sum(x["two_prefix_windows"] > 0 for x in info),
                      "shots_with_current_context_match": sum(x["current_candidate_shots"] > 0 for x in info),
                      "shots_with_two_prefix_context_match": sum(x["two_prefix_candidate_shots"] > 0 for x in info),
                      "unmatched_reasons": {status: sum(x["status"] == status for x in info)
                                            for status in sorted({x["status"] for x in info if x["status"] != "matched"})}})
        strata[group] = stats
    return {"control": control, "positive_candidates": len(positive), "safe_candidates": len(negative),
            "safe_denominator_shots": len(negative_info),
            "safe_shots_with_two_prefix_windows": sum(x["two_prefix_windows"] > 0 for x in negative_info),
            "safe_shots_with_two_prefix_context_match": sum(x["two_prefix_candidate_shots"] > 0 for x in negative_info),
            "maximum_disjoint_pairs": len(pairs), "total_matching_distance": cost,
            "strata": strata, "pairs": pairs,
            "positive_coverage": positive_info, "safe_coverage": negative_info,
            "single_pair_exclusion_sensitivity": alternatives}


def run():
    weak_protocol = read_json(WEAK / "protocol.json")
    weak_result = read_json(WEAK / "result.json")
    shared_protocol = read_json(SHARED / "protocol.json")
    source = read_json(SHARED / "per_shot.json")["0.2"]["current_5"]
    attribution = {r["shot"]: r for r in read_json(SHARED / "result.json")["attribution"]}
    sealed = (read_json(WEAK / "seal.json")["curves"], read_json(SHARED / "seal.json")["curves"])
    order = weak_protocol["order"]
    if order != shared_protocol["order"] or order != [x["shot"] for x in source]:
        raise ValueError("sealed input orders differ")
    if weak_result["calipers"] != CALIPERS.tolist() or weak_protocol["views"][1] != "raw_trajectory" or weak_protocol["views"][3] != "risk_state":
        raise ValueError("§63 matching or probe contract changed")
    rank = {s: i + 1 for i, s in enumerate(order)}
    groups = {"no_first_alarm": [], "late_first_alarm": [], "false_alarm_safe": [], "never_alarm_safe": []}
    for row in source[20:]:
        if row["disruptive"] and not row["timely"]:
            groups["late_first_alarm" if row["alarm"] else "no_first_alarm"].append(row["shot"])
        elif not row["disruptive"]:
            groups["false_alarm_safe" if row["alarm"] else "never_alarm_safe"].append(row["shot"])
    late = groups["no_first_alarm"] + groups["late_first_alarm"]
    if set(late) != set(weak_result["old_late"]) or set(groups["false_alarm_safe"]) != set(weak_result["old_false"]):
        raise ValueError("§63 and §65 scored error sets differ")
    candidates, coverage = {}, {}
    for group, shots in groups.items():
        candidates[group], coverage[group] = [], []
        for shot in sorted(shots):
            rows, info = make_candidates(shot, group, rank[shot], attribution, sealed)
            candidates[group].extend(rows)
            coverage[group].append(info)
    positives = candidates["no_first_alarm"] + candidates["late_first_alarm"]
    # Each comparison gets a fresh copy: the two safe strata are separate analyses.
    comparisons = {}
    for group in ("false_alarm_safe", "never_alarm_safe"):
        pinfo = [dict(x) for g in ("no_first_alarm", "late_first_alarm") for x in coverage[g]]
        sinfo = [dict(x) for x in coverage[group]]
        comparisons[group] = audit(group, positives, candidates[group], pinfo, sinfo)
    return {
        "scope": "retrospective frozen-score observability audit; no new alarm, model fit, raw HDF5 or GPU",
        "inputs": {"weak": str(WEAK.relative_to(ROOT)), "shared": str(SHARED.relative_to(ROOT)),
                   "fields": ["§63 protocol/result/seal and shots: ends, valid, dose, context, score",
                              "§65 protocol/per_shot/result/seal and shots: compare_at_release, dose"]},
        "selection": {
            "reference": "§65 per_shot 0.2.current_5; preparation shots 1-20 excluded from this diagnostic",
            "positive": "§65 stored handoff u in (30,125] ms and nonzero §65 dose",
            "false_alarm_safe": "§65 stored handoff u at or before original first false alarm, nonzero §65 dose",
            "never_alarm_safe": "all §65 stored handoffs u with nonzero §65 dose from shots without an original alarm",
            "anchors": "q is the last §63 probe index at or before u; q-32 is the earlier anchor; both require valid §63 inputs and positive §63 dose",
            "clock_note": "q is no later than the verified handoff; q and q-32 can precede the positive 125 ms window; 32 is a sample offset, not a verified physical receipt delay",
            "selection_clock": "q is selected retrospectively using the subsequently completed handoff u; each saved prefix score is causal, but this selection is not an online gate at q",
            "comparison": "each safe group matched independently, all 33 positives jointly; max cardinality then minimum summed squared tolerance-normalized context distance over both anchors; one shot per side",
            "endpoint_tiebreak": "for each shot pair choose minimum context distance, then earliest positive handoff, then earliest safe handoff; assignment rows/columns sorted by shot ID",
            "match_coordinates": CONTEXT, "calipers_each_anchor": CALIPERS.tolist(),
            "probe_selection": "no probe score enters eligibility, matching cost, or tie-break",
            "sensitivity": "remove each selected shot-pair edge in turn and recompute the same maximum-cardinality minimum-distance assignment; no alternative replaces the primary result",
        },
        "counts": {name: len(ids) for name, ids in groups.items()},
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    result = run()
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != encoded:
            raise SystemExit("recomputation differs from saved result")
        print("recomputation matches saved result")
    elif not sys.argv[1:]:
        OUT.write_text(encoded, encoding="utf-8")
        for name, audit_result in result["comparisons"].items():
            print(name, "pairs", audit_result["maximum_disjoint_pairs"],
                  "strata", json.dumps(audit_result["strata"], ensure_ascii=False))
    else:
        raise SystemExit("usage: python script.py [--check]")
