"""Post-seal development scoring, separate from label-free neural computation."""

import math

import numpy as np

STEMS = ("original", "reference_bt")
MODES = ("static", "real", "reverse", "hold")


def coordinates(records, ids):
    """Reuse only identities and clocks, never old-cold outputs or outcomes."""
    if len(ids) != 20 or len(set(ids)) != 20 or len(records) != 100:
        raise ValueError("fixed20 shots and100 retrospective episodes required")
    selected = []
    for record in records:
        r = {k: record[k] for k in ("shot", "origin", "query")}
        if (
            any(type(v) is not int for v in r.values())
            or r["shot"] not in ids
            or r["origin"] < 127
            or r["query"] != r["origin"] + 26
        ):
            raise ValueError("allowed shot and causal matured25/query clock required")
        selected.append(r)
    if len({(r["shot"], r["query"]) for r in selected}) != 100 or any(
        sum(r["shot"] == sid for r in selected) != 5 for sid in ids
    ):
        raise ValueError("five distinct queries for every fixed shot required")
    return selected


def score(predictions, outcomes, threshold):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("fixed finite source probability threshold required")
    threshold_logit = math.log(threshold / (1 - threshold))
    rows = []
    for r in predictions:
        if set(r) & {
            "label",
            "y",
            "is_disrupt",
            "disruptive",
            "down",
            "actual_lead_ms",
        }:
            raise ValueError("prediction record must be outcome-free until seal")
        flag, down = outcomes[r["shot"]]
        if type(flag) is not bool or (
            flag and (down is None or not math.isfinite(down))
        ):
            raise ValueError("HDF boolean outcome and finite disruptive time required")
        if (
            r["stem"] not in STEMS
            or r["mode"] not in MODES
            or not all(
                math.isfinite(r[k])
                for k in ("query_time", "before_logit", "after_logit")
            )
        ):
            raise ValueError("declared finite prediction required")
        lead = round((down - r["query_time"]) * 1000, 9) if flag else None
        if flag and lead <= 30:
            raise ValueError("this fixed development screen requires strict lead>30ms")
        y = int(flag)
        before, after = [
            float(np.logaddexp(0.0, z) - y * z)
            for z in (
                r["before_logit"] - threshold_logit,
                r["after_logit"] - threshold_logit,
            )
        ]
        rows.append(
            dict(
                **r,
                disruptive=flag,
                actual_lead_ms=lead,
                point_label125=int(flag and lead <= 125 + 1e-6),
                margin_target=y,
                before_bce=before,
                after_bce=after,
                delta_bce=after - before,
            )
        )
    return rows


def aggregate(rows):
    """Average within shots before group averaging; queries are not replicates."""
    result = {}
    for group in ("safe", "disruptive", "danger_window", "early_window"):
        selected = [
            r
            for r in rows
            if (
                not r["disruptive"]
                if group == "safe"
                else r["disruptive"]
                and (
                    group == "disruptive"
                    or (
                        30 < r["actual_lead_ms"] <= 125
                        if group == "danger_window"
                        else r["actual_lead_ms"] > 125
                    )
                )
            )
        ]
        ids = sorted({r["shot"] for r in selected})
        result[group] = dict(n_shots=len(ids), n_points=len(selected))
        for field in ("before_bce", "after_bce", "delta_bce"):
            means = [
                np.mean([r[field] for r in selected if r["shot"] == sid]) for sid in ids
            ]
            result[group]["mean_" + field] = float(np.mean(means)) if means else None
        result[group]["helpful_points"] = sum(r["delta_bce"] < 0 for r in selected)
        result[group]["harmful_points"] = sum(r["delta_bce"] > 0 for r in selected)
    if any(not result[g]["n_shots"] for g in ("safe", "disruptive", "danger_window")):
        raise ValueError("predeclared safety, danger and30-125ms groups required")
    result["balanced"] = (
        result["safe"]["mean_delta_bce"] + result["disruptive"]["mean_delta_bce"]
    ) / 2
    return result


def gate(summary):
    selected = summary["reference_bt_real"]
    result = {
        g + "_improved": selected[g]["mean_delta_bce"] < -1e-6
        for g in ("safe", "disruptive", "danger_window")
    }
    result.update(
        {
            "better_than_" + mode: selected["balanced"]
            < summary["reference_bt_" + mode]["balanced"] - 1e-6
            for mode in ("reverse", "hold")
        }
    )
    result["passed"] = all(result.values())
    return result


def analyze(rows):
    arms = {
        stem + "_" + mode: aggregate(
            [r for r in rows if r["stem"] == stem and r["mode"] == mode]
        )
        for stem in STEMS
        for mode in MODES
    }
    old = {
        (r["shot"], r["query"]): r
        for r in rows
        if r["stem"] == "original" and r["mode"] == "static"
    }
    differences = []
    for r in rows:
        if r["stem"] == "reference_bt" and r["mode"] == "static":
            before = old[r["shot"], r["query"]]["after_bce"]
            differences.append(
                dict(r, before_bce=before, delta_bce=r["after_bce"] - before)
            )
    return dict(
        arms=arms,
        static_representation_delta=aggregate(differences),
        candidate_gate=gate(arms),
        operational_online_gain_measured=False,
    )
