"""Post-hoc analysis for ``representation_online`` sealed artifacts.

This module deliberately has no model or runner dependency.  It reads the
sealed prediction trace, verifies its receipt before reading any outcome, and
then uses :func:`simple_data.attach_outcome` only to make an offline report.
The report never changes a checkpoint, a trace, a model, or an online result.

The threshold scan is explicitly a test-driven oracle diagnostic: it uses the
revealed outcomes from the same run and must not be described as a threshold
that was deployed online in advance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import simple_data


LEAD_MS = 30.0
THRESHOLD_CAPS = (0.20, 0.15, 0.10)
STAGE_BOUNDARIES = (1, 2, 4, 8, 16, 30)
CLASSES = ("TP", "FN", "TARDY", "TN", "FP")
_SEAL_RE = re.compile(r"^(?P<index>\d+)_(?P<shot>\d+)_seal\.json$")


def rounded_lead_ms(down: float, prediction_time: float) -> float:
    """Return the nine-decimal lead used by the online metric contract."""
    if down is None or prediction_time is None:
        raise ValueError("lead requires both DownTime and prediction time")
    value = float(round((float(down) - float(prediction_time)) * 1000.0, 9))
    if not math.isfinite(value):
        raise ValueError("non-finite lead")
    return value


def classify_prediction(is_disrupt: bool, alarm: bool, lead_ms: float | None,
                        *, lead: float = LEAD_MS) -> str:
    """Map one evaluated shot to the five mutually exclusive classes.

    ``TP`` means a strict timely alarm.  A disruptive shot that alarms but has
    a lead at or below the boundary is ``TARDY``; a disruptive shot without an
    alarm is ``FN``.  For safe shots, alarm/no-alarm are ``FP``/``TN``.
    """
    if bool(is_disrupt):
        if not bool(alarm):
            return "FN"
        return "TP" if lead_ms is not None and float(lead_ms) > float(lead) else "TARDY"
    return "FP" if bool(alarm) else "TN"


def _empty_class_counts() -> dict[str, int]:
    return {name: 0 for name in CLASSES}


def _safe_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _metrics_from_class_counts(class_counts: Mapping[str, int], *, threshold: float,
                               predictions: Sequence[dict[str, Any]] | None = None,
                               timely_leads: Sequence[float] = ()) -> dict[str, Any]:
    counts = {name: int(class_counts.get(name, 0)) for name in CLASSES}
    n_disrupt = counts["TP"] + counts["FN"] + counts["TARDY"]
    n_safe = counts["TN"] + counts["FP"]
    tpr = _rate(counts["TP"], n_disrupt)
    fpr = _rate(counts["FP"], n_safe)
    result: dict[str, Any] = {
        "tp": counts["TP"],
        "fp": counts["FP"],
        "n_disrupt": n_disrupt,
        "n_safe": n_safe,
        "tpr": tpr,
        "fpr": fpr,
        "J": float(tpr - fpr) if tpr is not None and fpr is not None else None,
        "threshold": float(threshold),
        "median_lead_ms": (float(np.median(np.asarray(timely_leads, dtype=float)))
                            if timely_leads else None),
        "class_counts": counts,
    }
    if predictions is not None:
        result["predictions"] = list(predictions)
    return result


def _row_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        times = np.asarray(row["times"], dtype=float)
        scores = np.asarray(row["scores"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"row missing {exc.args[0]}") from exc
    if times.ndim != 1 or scores.ndim != 1 or len(times) != len(scores):
        raise ValueError("sealed times/scores must be aligned one-dimensional arrays")
    if not np.isfinite(times).all() or not np.isfinite(scores).all():
        raise ValueError("sealed times/scores must be finite")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("sealed scores must be probabilities in [0, 1]")
    return times, scores


def metrics_for_rows(rows: Iterable[Mapping[str, Any]], threshold: float,
                     *, lead: float = LEAD_MS) -> dict[str, Any]:
    """Recompute simple-train-compatible metrics from post-hoc rows.

    Rows are expected to contain only the valid pre-terminal part of a
    disruptive trace.  ``>=`` is intentional: a safe score exactly equal to
    the threshold is an alarm, matching ``simple_train.metrics``.
    """
    threshold = _safe_float(threshold, "threshold")
    # A value just above one is the bounded no-alarm sentinel used by the
    # post-hoc scan.  Negative thresholds are never meaningful probabilities.
    if threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    predictions: list[dict[str, Any]] = []
    class_counts = _empty_class_counts()
    timely_leads: list[float] = []
    for raw in rows:
        is_disrupt = bool(raw.get("is_disrupt", False))
        down = raw.get("down")
        times, scores = _row_arrays(raw)
        indices = np.flatnonzero(scores >= threshold)
        alarm = bool(len(indices))
        lead_ms = None
        if is_disrupt and alarm:
            if down is None:
                raise ValueError(f"disruptive shot {raw.get('shot')} has no DownTime")
            lead_ms = rounded_lead_ms(float(down), float(times[int(indices[0])]))
        category = classify_prediction(is_disrupt, alarm, lead_ms, lead=lead)
        class_counts[category] += 1
        if category == "TP":
            timely_leads.append(float(lead_ms))
        predictions.append({
            "shot": int(raw["shot"]),
            "is_disrupt": is_disrupt,
            "alarm": alarm,
            "lead_ms": lead_ms,
            "timely": category == "TP",
            "class": category,
        })
    return _metrics_from_class_counts(class_counts, threshold=threshold,
                                      predictions=predictions,
                                      timely_leads=timely_leads)


def _prediction_class(row: Mapping[str, Any], *, lead: float = LEAD_MS) -> str:
    if "class" in row and row["class"] in CLASSES:
        return str(row["class"])
    return classify_prediction(bool(row.get("is_disrupt", False)),
                               bool(row.get("alarm", False)), row.get("lead_ms"),
                               lead=lead)


def _flip_classification(before: str, after: str) -> str:
    if before == after:
        return "unchanged"
    # The paired comparison is intentionally about strict >30 ms performance:
    # FN<->TARDY does not change the timely/safe contract and is therefore
    # reported as neutral rather than being sold as a gain.
    if ((before in {"FN", "TARDY"} and after == "TP")
            or (before == "FP" and after == "TN")):
        return "helpful"
    if ((before == "TP" and after in {"FN", "TARDY"})
            or (before == "TN" and after == "FP")):
        return "harmful"
    return "neutral"


def paired_flips(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
                 *, before_arm: str = "before", after_arm: str = "after",
                 lead: float = LEAD_MS) -> dict[str, Any]:
    """Classify same-roster before/after transitions without pooling arms."""
    before_by_shot = {int(row["shot"]): row for row in before}
    after_by_shot = {int(row["shot"]): row for row in after}
    if len(before_by_shot) != len(before) or len(after_by_shot) != len(after):
        raise ValueError("paired predictions contain duplicate shots")
    if set(before_by_shot) != set(after_by_shot):
        raise ValueError("paired predictions have different shot rosters")

    transitions: Counter[tuple[str, str]] = Counter()
    transition_shots: dict[tuple[str, str], list[int]] = {}
    for shot in before_by_shot:
        left = before_by_shot[shot]
        right = after_by_shot[shot]
        if bool(left.get("is_disrupt")) != bool(right.get("is_disrupt")):
            raise ValueError(f"paired outcome differs for shot {shot}")
        transition = (_prediction_class(left, lead=lead),
                      _prediction_class(right, lead=lead))
        transitions[transition] += 1
        transition_shots.setdefault(transition, []).append(shot)

    flips = []
    helpful = harmful = neutral = 0
    for before_class in CLASSES:
        for after_class in CLASSES:
            count = int(transitions.get((before_class, after_class), 0))
            if not count or before_class == after_class:
                continue
            classification = _flip_classification(before_class, after_class)
            flips.append({"from": before_class, "to": after_class,
                          "classification": classification, "count": count,
                          "shots": sorted(transition_shots[(before_class, after_class)])})
            if classification == "helpful":
                helpful += count
            elif classification == "harmful":
                harmful += count
            else:
                neutral += count

    def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        result = _empty_class_counts()
        for row in rows:
            result[_prediction_class(row, lead=lead)] += 1
        return result

    before_counts = counts(before)
    after_counts = counts(after)
    before_disrupt = before_counts["TP"] + before_counts["FN"] + before_counts["TARDY"]
    after_disrupt = after_counts["TP"] + after_counts["FN"] + after_counts["TARDY"]
    before_tpr = _rate(before_counts["TP"], before_disrupt)
    after_tpr = _rate(after_counts["TP"], after_disrupt)
    timely_tp_gained = sum(
        transitions.get((source, "TP"), 0) for source in ("FN", "TARDY"))
    timely_tp_lost = sum(
        transitions.get(("TP", target), 0) for target in ("FN", "TARDY"))

    return {
        "before_arm": before_arm,
        "after_arm": after_arm,
        "shot_count": len(before),
        "flips": flips,
        "helpful_count": helpful,
        "harmful_count": harmful,
        "neutral_count": neutral,
        "class_counts_before": before_counts,
        "class_counts_after": after_counts,
        "fixed_threshold": {
            "before_tp": before_counts["TP"],
            "after_tp": after_counts["TP"],
            "timely_tp_gained": int(timely_tp_gained),
            "timely_tp_lost": int(timely_tp_lost),
            "tpr_before": before_tpr,
            "tpr_after": after_tpr,
            "tpr_delta": (float(after_tpr - before_tpr)
                           if before_tpr is not None and after_tpr is not None else None),
            "tpr_declined": (bool(after_tpr < before_tpr)
                              if before_tpr is not None and after_tpr is not None else None),
        },
    }


def _summary_for_predictions(predictions: Sequence[Mapping[str, Any]], *, start: int,
                             stop: int) -> dict[str, Any]:
    subset = list(predictions[start:stop])
    counts = _empty_class_counts()
    for row in subset:
        counts[_prediction_class(row)] += 1
    metrics = _metrics_from_class_counts(counts, threshold=0.0)
    metrics.pop("threshold", None)
    metrics.pop("median_lead_ms", None)
    return {
        "first_index": start + 1 if subset else None,
        "last_index": start + len(subset) if subset else None,
        "shot_count": len(subset),
        "class_counts": counts,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "n_disrupt": metrics["n_disrupt"],
        "n_safe": metrics["n_safe"],
        "tpr": metrics["tpr"],
        "fpr": metrics["fpr"],
        "J": metrics["J"],
    }


def feedback_stages(predictions: Sequence[Mapping[str, Any]], *,
                    boundaries: Sequence[int] = STAGE_BOUNDARIES) -> list[dict[str, Any]]:
    """Report common-prefix cumulative and following-ten-shot composition."""
    result = []
    total = len(predictions)
    for boundary in boundaries:
        boundary = int(boundary)
        cumulative = (_summary_for_predictions(predictions, start=0, stop=boundary)
                       if total >= boundary else None)
        next10 = (_summary_for_predictions(predictions, start=boundary,
                                            stop=min(total, boundary + 10))
                  if total > boundary else None)
        result.append({"after_feedback_shot": boundary,
                       "cumulative": cumulative,
                       "next10": next10})
    return result


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def threshold_diagnostic(rows: Sequence[Mapping[str, Any]], *, lead: float = LEAD_MS,
                         caps: Sequence[float] = THRESHOLD_CAPS) -> dict[str, Any]:
    """Scan post-hoc thresholds and report best TPR at each FPR cap.

    Candidate thresholds are the per-shot maximum scores plus a no-alarm
    sentinel.  This is sufficient because a shot-level alarm changes only when
    its maximum score is crossed, and keeps the JSON report bounded.
    """
    candidates = {float(np.nextafter(1.0, 2.0))}
    for row in rows:
        times, scores = _row_arrays(row)
        if len(scores):
            candidates.add(float(scores.max()))
        if bool(row.get("is_disrupt", False)) and row.get("down") is not None:
            rounded_leads = np.round((float(row["down"]) - times) * 1000.0, 9)
            early = scores[rounded_leads > float(lead)]
            if len(early):
                candidates.add(float(early.max()))
    curve: list[dict[str, Any]] = []
    for threshold in sorted(candidates, reverse=True):
        measured = metrics_for_rows(rows, threshold, lead=lead)
        curve.append({
            "threshold": float(threshold),
            "tp": measured["tp"],
            "fp": measured["fp"],
            "n_disrupt": measured["n_disrupt"],
            "n_safe": measured["n_safe"],
            "tpr": measured["tpr"],
            "fpr": measured["fpr"],
            "J": measured["J"],
            "class_counts": measured["class_counts"],
        })

    best_by_cap: dict[str, Any] = {}
    for cap in caps:
        cap = _safe_float(cap, "FPR cap")
        if not 0.0 <= cap <= 1.0:
            raise ValueError("FPR cap must be in [0, 1]")
        feasible = [point for point in curve
                    if point["fpr"] is not None and point["tpr"] is not None
                    and point["fpr"] <= cap]
        if not feasible:
            best = None
        else:
            # Maximize TPR, then prefer lower FPR and finally the most
            # conservative threshold.  All ties still use score >= threshold.
            best = max(feasible, key=lambda point: (
                float(point["tpr"]), -float(point["fpr"]), float(point["threshold"])))
            best = dict(best)
            best["fpr_cap"] = cap
            best["constraint_satisfied"] = bool(best["fpr"] <= cap)
        best_by_cap[f"{cap:.2f}"] = best

    return {
        "kind": "TEST_DRIVEN_ORACLE_DIAGNOSTIC",
        "not_online_predeployed": True,
        "warning": "Uses revealed outcomes from this run; not an online pre-deployed threshold result.",
        "lead_rule": "round((DownTime-time)*1000, 9) > 30.0",
        "caps": best_by_cap,
        "curve": curve,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}") from exc


def _summary_metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("metrics", "metrics_full", "metrics_prefix"):
        candidate = result.get(key)
        if isinstance(candidate, Mapping) and isinstance(candidate.get("predictions"), list):
            return candidate
    raise ValueError("arm summary has no metrics.predictions")


def _arm_results(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source: Any = summary.get("arms", summary)
    if not isinstance(source, Mapping):
        raise ValueError("summary must contain arm results")
    arms = {
        str(name): value for name, value in source.items()
        if isinstance(value, Mapping)
        and any(key in value for key in ("metrics", "metrics_full", "metrics_prefix"))
    }
    if not arms:
        raise ValueError("summary contains no representation arms")
    return arms


def _load_sealed_arm(arm_dir: Path) -> list[dict[str, Any]]:
    """Load traces only after verifying every receipt hash in the arm."""
    receipts = sorted(arm_dir.glob("*_seal.json"))
    if not receipts:
        raise ValueError(f"arm has no seal receipts: {arm_dir}")
    traces: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_shots: set[int] = set()
    resolved_dir = arm_dir.resolve()
    for receipt_path in receipts:
        match = _SEAL_RE.match(receipt_path.name)
        if not match:
            raise ValueError(f"invalid seal filename: {receipt_path.name}")
        index = int(match.group("index"))
        filename_shot = int(match.group("shot"))
        receipt = _read_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"seal receipt is not an object: {receipt_path}")
        npz_name = receipt.get("npz")
        expected_sha = receipt.get("sha256")
        if not isinstance(npz_name, str) or not isinstance(expected_sha, str):
            raise ValueError(f"seal receipt lacks npz/sha256: {receipt_path}")
        npz_path = (arm_dir / npz_name).resolve()
        if npz_path.parent != resolved_dir:
            raise ValueError(f"sealed trace escapes arm directory: {receipt_path}")
        if not npz_path.is_file():
            raise ValueError(f"sealed trace is missing: {npz_path}")
        actual_sha = _sha256(npz_path)
        if actual_sha.lower() != expected_sha.lower():
            raise ValueError(f"sealed trace SHA mismatch: {npz_path}")

        # The hash check above is intentionally before opening the NPZ or
        # consulting any outcome.  This ordering is part of the audit contract.
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if "times" not in archive or "scores" not in archive:
                    raise ValueError(f"sealed trace lacks times/scores: {npz_path}")
                times = np.asarray(archive["times"], dtype=float).copy()
                scores = np.asarray(archive["scores"], dtype=float).copy()
                npz_shot = int(np.asarray(archive["shot"]).reshape(-1)[0]) if "shot" in archive else filename_shot
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("sealed trace lacks"):
                raise
            raise ValueError(f"cannot read sealed trace {npz_path}") from exc
        if npz_shot != filename_shot or int(receipt.get("shot", filename_shot)) != filename_shot:
            raise ValueError(f"sealed shot identity mismatch: {receipt_path}")
        if index in seen_indices or filename_shot in seen_shots:
            raise ValueError(f"duplicate sealed index or shot in {arm_dir}")
        seen_indices.add(index)
        seen_shots.add(filename_shot)
        audit_path = arm_dir / f"{npz_path.name.removesuffix('_sealed.npz')}_audit.json"
        if not audit_path.is_file():
            raise ValueError(f"sealed audit is missing: {audit_path}")
        _row_arrays({"times": times, "scores": scores})
        traces.append({"index": index, "shot": filename_shot, "times": times,
                       "scores": scores, "receipt": dict(receipt)})
    traces.sort(key=lambda row: row["index"])
    return traces


def _outcome_rows(traces: Sequence[Mapping[str, Any]],
                  outcome_loader: Callable[[dict[str, Any], str, Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        times = np.asarray(trace["times"], dtype=float)
        source = {"shot": int(trace["shot"]), "device": "EAST", "times": times}
        revealed = outcome_loader(source, "EAST", {})
        if not isinstance(revealed, Mapping):
            raise ValueError(f"outcome loader returned non-object for shot {trace['shot']}")
        shot = int(revealed.get("shot", trace["shot"]))
        if shot != int(trace["shot"]):
            raise ValueError(f"outcome shot mismatch for {trace['shot']}")
        disruptive = bool(revealed.get("is_disrupt", False))
        down = revealed.get("down")
        if disruptive and down is None:
            raise ValueError(f"disruptive shot {shot} has no DownTime")
        valid = np.asarray(revealed.get("valid", np.ones(len(times), dtype=bool)), dtype=bool)
        if valid.shape != times.shape:
            raise ValueError(f"outcome validity shape mismatch for shot {shot}")
        if disruptive:
            valid &= times <= float(down)
        rows.append({"shot": shot, "is_disrupt": disruptive,
                     "down": None if down is None else float(down),
                     "times": times[valid],
                     "scores": np.asarray(trace["scores"], dtype=float)[valid]})
    return rows


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9))
    except (TypeError, ValueError):
        return False


def _verify_summary(result: Mapping[str, Any], summary_metrics: Mapping[str, Any],
                    measured: Mapping[str, Any], trace_count: int) -> None:
    predictions = summary_metrics.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != trace_count:
        raise ValueError("summary prediction count does not match sealed traces")
    if "observed_count" in result and int(result["observed_count"]) != trace_count:
        raise ValueError("summary observed_count does not match sealed traces")
    for key in ("tp", "fp", "n_disrupt", "n_safe"):
        if key in summary_metrics and int(summary_metrics[key]) != int(measured[key]):
            raise ValueError(f"summary {key} does not match sealed evaluation")
    for key in ("tpr", "fpr", "J", "threshold", "median_lead_ms"):
        if key in summary_metrics and not _same_number(summary_metrics[key], measured[key]):
            raise ValueError(f"summary {key} does not match sealed evaluation")
    measured_predictions = measured["predictions"]
    for expected, actual in zip(measured_predictions, predictions):
        if int(actual.get("shot", -1)) != expected["shot"]:
            raise ValueError("summary prediction shot roster does not match sealed traces")
        for key in ("is_disrupt", "alarm", "timely"):
            if bool(actual.get(key)) != bool(expected[key]):
                raise ValueError(f"summary prediction {key} does not match sealed evaluation")
        if not _same_number(actual.get("lead_ms"), expected["lead_ms"]):
            raise ValueError("summary prediction lead_ms does not match sealed evaluation")


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def analyze_run(run: str | Path, output: str | Path | None = None, *,
                outcome_loader: Callable[[dict[str, Any], str, Mapping[str, Any]], Mapping[str, Any]] | None = None
                ) -> dict[str, Any]:
    """Analyze one representation run and optionally write a new JSON file."""
    run_path = Path(run).resolve()
    if not run_path.is_dir():
        raise ValueError(f"run directory does not exist: {run_path}")
    summary_path = run_path / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"run summary is missing: {summary_path}")
    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError("run summary must be an object")
    arm_results = _arm_results(summary)

    # Phase 1: load every arm and verify every sealed SHA.  No outcome access
    # happens before this phase completes successfully.
    traces_by_arm = {}
    for arm in arm_results:
        traces_by_arm[arm] = _load_sealed_arm(run_path / arm)

    # Counts are a receipt-level invariant and can be checked before any
    # outcome/DownTime access.  Detailed metric and row checks happen after
    # the offline outcome join below.
    for arm, traces in traces_by_arm.items():
        summary_metrics = _summary_metrics(arm_results[arm])
        if len(summary_metrics["predictions"]) != len(traces):
            raise ValueError("summary prediction count does not match sealed traces")
        if ("observed_count" in arm_results[arm]
                and int(arm_results[arm]["observed_count"]) != len(traces)):
            raise ValueError("summary observed_count does not match sealed traces")

    loader = outcome_loader or simple_data.attach_outcome
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    for arm, traces in traces_by_arm.items():
        result = arm_results[arm]
        summary_metrics = _summary_metrics(result)
        rows = _outcome_rows(traces, loader)
        threshold = _safe_float(summary_metrics.get("threshold", traces[0]["receipt"].get("threshold")),
                                f"{arm} threshold")
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError(f"{arm} online threshold must be in [0, 1]")
        for trace in traces:
            receipt_threshold = trace["receipt"].get("threshold")
            if receipt_threshold is not None and not _same_number(receipt_threshold, threshold):
                raise ValueError(f"{arm} seal threshold differs from summary")
        measured = metrics_for_rows(rows, threshold)
        _verify_summary(result, summary_metrics, measured, len(traces))
        rows_by_arm[arm] = rows
        metrics_by_arm[arm] = measured

    rosters = {arm: [int(row["shot"]) for row in metrics["predictions"]]
               for arm, metrics in metrics_by_arm.items()}
    roster_values = list(rosters.values())
    if roster_values and any(roster != roster_values[0] for roster in roster_values[1:]):
        raise ValueError("arms do not share one sealed shot roster")

    base_arm = ("stats" if "stats" in arm_results else
                "contract" if "contract" in arm_results else
                "static" if "static" in arm_results else None)
    paired: dict[str, Any] = {}
    if base_arm is not None and "intra" in metrics_by_arm:
        paired["intra_vs_base"] = paired_flips(
            metrics_by_arm[base_arm]["predictions"], metrics_by_arm["intra"]["predictions"],
            before_arm=base_arm, after_arm="intra")
    if base_arm is not None and "inter" in metrics_by_arm:
        paired["inter_vs_base"] = paired_flips(
            metrics_by_arm[base_arm]["predictions"], metrics_by_arm["inter"]["predictions"],
            before_arm=base_arm, after_arm="inter")
    if "inter" in metrics_by_arm and "both" in metrics_by_arm:
        paired["both_vs_inter"] = paired_flips(
            metrics_by_arm["inter"]["predictions"], metrics_by_arm["both"]["predictions"],
            before_arm="inter", after_arm="both")

    arms_report = {}
    for arm, metrics in metrics_by_arm.items():
        arms_report[arm] = {
            "observed_count": len(metrics["predictions"]),
            "online_threshold": _public_metrics(metrics),
            "threshold_diagnostic": threshold_diagnostic(rows_by_arm[arm]),
            "feedback_stages": feedback_stages(metrics["predictions"]),
        }

    report = {
        "format": "representation-analysis-v1",
        "run": str(run_path),
        "analysis_only": True,
        "model_write": False,
        "gpu_used": False,
        "lead_rule": "round((DownTime-time)*1000, 9) > 30.0",
        "base_arm": base_arm,
        "arms": arms_report,
        "paired_flips": paired,
        "verification": {
            "sealed_sha256_verified": True,
            "summary_prediction_counts_verified": True,
            "summary_metrics_verified": True,
            "outcome_source": "simple_data.attach_outcome",
            "threshold_diagnostic_kind": "TEST_DRIVEN_ORACLE_DIAGNOSTIC",
            "threshold_diagnostic_not_online_predeployed": True,
        },
    }
    if output is not None:
        _write_json_once(Path(output).resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze_run(args.run, args.output)
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    main()
