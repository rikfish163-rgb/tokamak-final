#!/usr/bin/env python3
"""Build fail-closed evidence for the selected dynamic-source mainline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


class EvidenceContractError(RuntimeError):
    """An immutable evidence or statistical contract was violated."""


_OUTCOME_KEYS = frozenset({
    "advancedtime",
    "advancedtimems",
    "downtime",
    "downtimes",
    "isdisrupt",
    "label",
    "leadms",
    "outcome",
    "target",
    "timely",
})


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _outcome_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(key) in _OUTCOME_KEYS:
                return str(key)
            nested = _outcome_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            nested = _outcome_key(child)
            if nested is not None:
                return nested
    return None


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject outcome-bearing data in an outcome-free prediction seal."""

    for index, row in enumerate(rows):
        key = _outcome_key(row)
        if key is not None:
            raise EvidenceContractError(
                f"prediction row {index} contains outcome-bearing key {key!r}"
            )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceContractError(
                    f"invalid JSONL row {line_number} in {path}"
                ) from error
            if not isinstance(row, dict):
                raise EvidenceContractError(
                    f"prediction row {line_number} is not an object"
                )
            rows.append(row)
    return rows


def load_prediction_seal(
    prediction_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Load a complete outcome-free seal and verify its immutable bindings."""

    prediction_path = Path(prediction_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not prediction_path.is_file() or not manifest_path.is_file():
        raise EvidenceContractError("prediction seal files are incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvidenceContractError("prediction seal manifest is unreadable") from error
    if not isinstance(manifest, dict):
        raise EvidenceContractError("prediction seal manifest is not an object")
    expected_sha = manifest.get("prediction_sha256", manifest.get("predictions_sha256"))
    if not _is_sha256(expected_sha):
        raise EvidenceContractError("prediction seal lacks a complete prediction SHA")
    actual_sha = _sha256_file(prediction_path)
    if actual_sha != expected_sha:
        raise EvidenceContractError("prediction seal SHA does not match prediction bytes")
    expected_count = manifest.get("row_count", manifest.get("shot_count"))
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise EvidenceContractError("prediction seal lacks a complete row count")
    if expected_count <= 0:
        raise EvidenceContractError("prediction seal row count must be positive")
    if manifest.get("outcome_fields_present") is True:
        raise EvidenceContractError("prediction seal declares outcome fields present")
    explicit_seal_sha = manifest.get("seal_sha256")
    if explicit_seal_sha is not None and not _is_sha256(explicit_seal_sha):
        raise EvidenceContractError("prediction seal has an incomplete seal SHA")

    rows = _read_jsonl(prediction_path)
    if len(rows) != expected_count:
        raise EvidenceContractError(
            f"prediction seal row count mismatch: {len(rows)} != {expected_count}"
        )
    validate_prediction_rows(rows)
    shots: list[int] = []
    for index, row in enumerate(rows):
        if "shot" not in row:
            raise EvidenceContractError(f"prediction row {index} lacks shot identity")
        shot = int(row["shot"])
        shots.append(shot)
        stream_index = row.get("stream_index", index)
        if int(stream_index) != index:
            raise EvidenceContractError("prediction seal stream order is not contiguous")
    if len(shots) != len(set(shots)):
        raise EvidenceContractError("prediction seal contains duplicate shots")
    return {
        "status": "available",
        "schema_version": manifest.get("schema_version"),
        "rows": rows,
        "row_count": len(rows),
        "prediction_sha256": actual_sha,
        "seal_manifest_sha256": _sha256_file(manifest_path),
        "declared_seal_sha256": explicit_seal_sha,
        "shot_order_sha256": _sha256_json(shots),
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
    }


def build_postseal_outcome_join(
    seal: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    evaluation_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Join authoritative outcomes only after validating the complete seal."""

    if seal.get("status") != "available" or not _is_sha256(
        seal.get("seal_manifest_sha256")
    ):
        raise EvidenceContractError("post-seal join requires a complete seal SHA")
    prediction_rows = list(seal.get("rows", ()))
    validate_prediction_rows(prediction_rows)
    outcome_by_shot: dict[int, dict[str, Any]] = {}
    for source in outcomes:
        if "shot" not in source or "is_disrupt" not in source:
            raise EvidenceContractError("post-seal outcome row lacks required fields")
        shot = int(source["shot"])
        if shot in outcome_by_shot:
            raise EvidenceContractError("post-seal outcomes contain duplicate shots")
        is_disrupt = bool(source["is_disrupt"])
        down_time = source.get("down_time_s")
        if is_disrupt and (
            down_time is None or not math.isfinite(float(down_time))
        ):
            raise EvidenceContractError(
                f"disruptive shot {shot} lacks a finite post-seal DownTime"
            )
        outcome_by_shot[shot] = dict(source)

    shots = [int(row["shot"]) for row in prediction_rows]
    if set(outcome_by_shot) != set(shots):
        missing = sorted(set(shots) - set(outcome_by_shot))
        extra = sorted(set(outcome_by_shot) - set(shots))
        raise EvidenceContractError(
            f"post-seal outcome roster mismatch: missing={missing}, extra={extra}"
        )
    mask = [True] * len(shots) if evaluation_mask is None else list(evaluation_mask)
    if len(mask) != len(shots) or any(type(value) is not bool for value in mask):
        raise EvidenceContractError("evaluation mask must contain one boolean per shot")
    joined_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for prediction, shot in zip(prediction_rows, shots):
        outcome = outcome_by_shot[shot]
        truth = {
            "shot": shot,
            "is_disrupt": bool(outcome["is_disrupt"]),
            "down_time_s": outcome.get("down_time_s"),
        }
        for optional in ("advanced_time_ms", "outcome_source_sha256"):
            if optional in outcome:
                truth[optional] = outcome[optional]
        truth_rows.append(truth)
        joined_rows.append({**prediction, **truth})
    return {
        "status": "available",
        "prediction_seal_sha256": seal["seal_manifest_sha256"],
        "prediction_sha256": seal["prediction_sha256"],
        "shot_order_sha256": _sha256_json(shots),
        "truth_join_sha256": _sha256_json(truth_rows),
        "evaluation_mask_sha256": _sha256_json(mask),
        "evaluation_mask": mask,
        "rows": joined_rows,
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Wilson score interval for one seed's physical-shot denominator."""

    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or successes < 0
        or successes > total
    ):
        raise EvidenceContractError("Wilson counts require 0 <= successes <= total")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise EvidenceContractError("Wilson alpha must lie in (0, 1)")
    z = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if math.isclose(lower, 0.0, rel_tol=0.0, abs_tol=1e-15):
        lower = 0.0
    if math.isclose(upper, 1.0, rel_tol=0.0, abs_tol=1e-15):
        upper = 1.0
    return {
        "status": "available",
        "method": "wilson_score",
        "alpha": float(alpha),
        "numerator": int(successes),
        "denominator": int(total),
        "interval": [lower, upper],
        "uncertainty_unit": "physical_shots_within_seed",
    }


def _available_rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        return {
            "status": "unavailable",
            "reason": "zero_denominator",
            "numerator": int(numerator),
            "denominator": int(denominator),
            "value": None,
        }
    return {
        "status": "available",
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator),
        "wilson_95": wilson_interval(numerator, denominator),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "status": "unavailable",
            "reason": "no_eligible_shots",
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
        }
    return {
        "status": "available",
        "n": len(values),
        "median": _percentile(values, 0.5),
        "q1": _percentile(values, 0.25),
        "q3": _percentile(values, 0.75),
    }


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> dict[str, Any]:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return {
            "status": "unavailable",
            "reason": "auc_requires_both_classes",
            "value": None,
            "n_disruptions": len(positives),
            "n_safe": len(negatives),
        }
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return {
        "status": "available",
        "value": wins / (len(positives) * len(negatives)),
        "n_disruptions": len(positives),
        "n_safe": len(negatives),
    }


def _validate_threshold_receipt(
    receipt: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    try:
        threshold = float(receipt["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceContractError("threshold receipt lacks a finite value") from error
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise EvidenceContractError("threshold receipt value must lie in (0, 1)")
    source = str(receipt.get("source", ""))
    expected_source = "J-TEXT VAL" if device == "jtext" else "EAST support"
    if expected_source.lower() not in source.lower():
        raise EvidenceContractError(
            f"{device} threshold is not bound to {expected_source}"
        )
    digest = receipt.get("sha256")
    if not _is_sha256(digest):
        raise EvidenceContractError("threshold receipt lacks a complete SHA")
    return {"value": threshold, "source": source, "sha256": digest}


def compute_operational_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    threshold_receipt: Mapping[str, Any],
    evaluation_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Recompute all endpoint metrics from per-shot post-seal evidence."""

    normalized_device = str(device).lower()
    if normalized_device not in {"jtext", "east"}:
        raise EvidenceContractError("device must be 'jtext' or 'east'")
    threshold = _validate_threshold_receipt(
        threshold_receipt, device=normalized_device,
    )
    mask = [True] * len(rows) if evaluation_mask is None else list(evaluation_mask)
    if len(mask) != len(rows) or any(type(value) is not bool for value in mask):
        raise EvidenceContractError("evaluation mask must contain one boolean per row")
    selected_rows = [row for row, included in zip(rows, mask) if included]
    if not selected_rows:
        return {
            "status": "unavailable",
            "reason": "empty_evaluation_mask",
            "threshold": threshold,
        }

    seen_shots: set[int] = set()
    labels: list[bool] = []
    scores: list[float] = []
    per_shot: list[dict[str, Any]] = []
    raw_leads: list[float] = []
    operational_leads: list[float] = []
    for row in selected_rows:
        required = {"shot", "score", "is_disrupt", "down_time_s"}
        missing = sorted(required - set(row))
        if missing:
            raise EvidenceContractError(
                f"post-seal metric row lacks fields: {', '.join(missing)}"
            )
        shot = int(row["shot"])
        if shot in seen_shots:
            raise EvidenceContractError("post-seal metrics contain duplicate shots")
        seen_shots.add(shot)
        score = float(row["score"])
        if not math.isfinite(score):
            raise EvidenceContractError(f"shot {shot} has a non-finite score")
        is_disrupt = bool(row["is_disrupt"])
        alarmed = bool(score >= threshold["value"])
        alarm_time = row.get("alarm_time_s")
        if alarmed and (
            alarm_time is None or not math.isfinite(float(alarm_time))
        ):
            raise EvidenceContractError(
                f"alarmed shot {shot} lacks a finite sealed alarm_time_s"
            )
        down_time = row.get("down_time_s")
        if is_disrupt and (
            down_time is None or not math.isfinite(float(down_time))
        ):
            raise EvidenceContractError(
                f"disruptive shot {shot} lacks a finite DownTime"
            )

        lead_ms: float | None = None
        operational_tp = False
        tardy = False
        if is_disrupt and alarmed:
            lead_ms = (float(down_time) - float(alarm_time)) * 1000.0
            if normalized_device == "jtext":
                operational_tp = lead_ms > 10.0 + 1e-9
            else:
                operational_tp = lead_ms >= 30.0 - 1e-9
            tardy = not operational_tp
            raw_leads.append(lead_ms)
            if operational_tp:
                operational_leads.append(lead_ms)
        labels.append(is_disrupt)
        scores.append(score)
        per_shot.append({
            "shot": shot,
            "is_disrupt": is_disrupt,
            "score": score,
            "alarmed": alarmed,
            "lead_ms": lead_ms,
            "raw_tp": bool(is_disrupt and alarmed),
            "operational_tp": bool(operational_tp),
            "tardy": bool(tardy),
            "false_positive": bool((not is_disrupt) and alarmed),
        })

    n_disruptions = sum(labels)
    n_safe = len(labels) - n_disruptions
    raw_tp = sum(row["raw_tp"] for row in per_shot)
    raw_fn = n_disruptions - raw_tp
    operational_tp = sum(row["operational_tp"] for row in per_shot)
    tardy = sum(row["tardy"] for row in per_shot)
    operational_fn = n_disruptions - operational_tp
    fp = sum(row["false_positive"] for row in per_shot)
    tn = n_safe - fp
    if raw_tp != operational_tp + tardy or operational_fn != raw_fn + tardy:
        raise EvidenceContractError("raw/operational confusion identities failed")

    raw_tpr = _available_rate(raw_tp, n_disruptions)
    operational_tpr = _available_rate(operational_tp, n_disruptions)
    fpr = _available_rate(fp, n_safe)
    if operational_tpr["status"] == "available" and fpr["status"] == "available":
        j_rate: dict[str, Any] = {
            "status": "available",
            "value": operational_tpr["value"] - fpr["value"],
        }
    else:
        j_rate = {
            "status": "unavailable",
            "reason": "j_requires_operational_tpr_and_fpr",
            "value": None,
        }
    very_early = sum(
        row["operational_tp"] and float(row["lead_ms"]) > 3000.0
        for row in per_shot
    )
    return {
        "status": "available",
        "device": normalized_device,
        "threshold": threshold,
        "alarm_score_comparator": "ge",
        "operational_lead_contract": (
            {"value_ms": 10.0, "comparator": "gt"}
            if normalized_device == "jtext"
            else {"value_ms": 30.0, "comparator": "ge"}
        ),
        "counts": {
            "n_shots": len(selected_rows),
            "n_disruptions": n_disruptions,
            "n_safe": n_safe,
            "raw_TP": raw_tp,
            "raw_FN": raw_fn,
            "TARDY": tardy,
            "operational_TP": operational_tp,
            "operational_FN": operational_fn,
            "FP": fp,
            "TN": tn,
        },
        "rates": {
            "raw_tpr": raw_tpr,
            "operational_tpr": operational_tpr,
            "fpr": fpr,
            "j": j_rate,
            "auc": _auc(labels, scores),
        },
        "lead_distributions": {
            "raw_alarm_lead_ms": _distribution(raw_leads),
            "operational_tp_lead_ms": _distribution(operational_leads),
        },
        "very_early_alarm": {
            "threshold_ms": 3000.0,
            "count": very_early,
            "rate_among_disruptions": _available_rate(very_early, n_disruptions),
            "rate_among_operational_tp": _available_rate(
                very_early, operational_tp,
            ),
        },
        "per_shot": per_shot,
    }


def summarize_seed_rate(
    per_seed: Mapping[str, Mapping[str, Any]],
    rate_name: str,
) -> dict[str, Any]:
    """Describe model-seed variation without pooling repeated physical shots."""

    values: dict[str, float] = {}
    denominators: dict[str, int] = {}
    unavailable_seeds: dict[str, str] = {}
    for seed, metrics in sorted(per_seed.items(), key=lambda item: int(item[0])):
        rate = metrics.get("rates", {}).get(rate_name)
        if not isinstance(rate, Mapping) or rate.get("status") != "available":
            unavailable_seeds[str(seed)] = (
                str(rate.get("reason", "rate_unavailable"))
                if isinstance(rate, Mapping)
                else "rate_missing"
            )
            continue
        values[str(seed)] = float(rate["value"])
        denominators[str(seed)] = int(rate["denominator"])
    if unavailable_seeds:
        return {
            "status": "unavailable",
            "reason": "one_or_more_seed_rates_unavailable",
            "unavailable_seeds": unavailable_seeds,
            "available_values": values,
            "aggregation": "equal_weight_over_model_seeds",
            "uncertainty": "per_seed_wilson_only",
        }
    if not values:
        return {
            "status": "unavailable",
            "reason": "no_seed_rates",
            "aggregation": "equal_weight_over_model_seeds",
            "uncertainty": "per_seed_wilson_only",
        }
    ordered_values = list(values.values())
    mean = sum(ordered_values) / len(ordered_values)
    sample_sd = (
        math.sqrt(
            sum((value - mean) ** 2 for value in ordered_values)
            / (len(ordered_values) - 1)
        )
        if len(ordered_values) > 1
        else None
    )
    return {
        "status": "available",
        "aggregation": "equal_weight_over_model_seeds",
        "model_seed_count": len(ordered_values),
        "values": values,
        "mean": mean,
        "sample_sd": sample_sd,
        "minimum": min(ordered_values),
        "maximum": max(ordered_values),
        "per_seed_denominators": denominators,
        "uncertainty": "per_seed_wilson_only",
        "seed_times_shot_pooling": False,
    }


def _paired_binary_change(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise EvidenceContractError("paired binary endpoints require equal non-empty rows")
    baseline_only = sum(left and not right for left, right in zip(baseline, candidate))
    candidate_only = sum(right and not left for left, right in zip(baseline, candidate))
    both = sum(left and right for left, right in zip(baseline, candidate))
    neither = len(baseline) - baseline_only - candidate_only - both
    return {
        "status": "available",
        "value": (sum(candidate) - sum(baseline)) / len(baseline),
        "denominator": len(baseline),
        "direction": f"{candidate_name}_minus_{baseline_name}",
        "discordance": {
            f"{baseline_name}_only": baseline_only,
            f"{candidate_name}_only": candidate_only,
            "both": both,
            "neither": neither,
        },
    }


def compute_shot_paired_delta(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    static_hashes: Mapping[str, Any],
    online_hashes: Mapping[str, Any],
    baseline_name: str = "static",
    candidate_name: str = "online",
) -> dict[str, Any]:
    """Compute within-seed static/online deltas on identical physical shots."""

    hash_keys = (
        "shot_order_sha256",
        "truth_join_sha256",
        "evaluation_mask_sha256",
    )
    for key in hash_keys:
        static_value = static_hashes.get(key)
        online_value = online_hashes.get(key)
        if not _is_sha256(static_value) or not _is_sha256(online_value):
            raise EvidenceContractError(f"paired delta lacks complete {key}")
        if static_value != online_value:
            raise EvidenceContractError(f"paired delta {key} mismatch")
    if (
        baseline_metrics.get("status") != "available"
        or candidate_metrics.get("status") != "available"
    ):
        return {
            "status": "unavailable",
            "reason": "paired_arm_metrics_unavailable",
        }
    baseline_rows = list(baseline_metrics.get("per_shot", ()))
    candidate_rows = list(candidate_metrics.get("per_shot", ()))
    if len(baseline_rows) != len(candidate_rows) or not baseline_rows:
        raise EvidenceContractError("paired arm rows differ in length")
    for left, right in zip(baseline_rows, candidate_rows):
        if int(left["shot"]) != int(right["shot"]):
            raise EvidenceContractError("paired arm shot order differs")
        if bool(left["is_disrupt"]) != bool(right["is_disrupt"]):
            raise EvidenceContractError("paired arm truth differs")

    disruptive_indices = [
        index for index, row in enumerate(baseline_rows) if row["is_disrupt"]
    ]
    safe_indices = [
        index for index, row in enumerate(baseline_rows) if not row["is_disrupt"]
    ]
    if not disruptive_indices or not safe_indices:
        return {
            "status": "unavailable",
            "reason": "paired_delta_requires_both_classes",
        }
    operational_delta = _paired_binary_change(
        [bool(baseline_rows[index]["operational_tp"]) for index in disruptive_indices],
        [bool(candidate_rows[index]["operational_tp"]) for index in disruptive_indices],
        baseline_name=baseline_name,
        candidate_name=candidate_name,
    )
    fpr_delta = _paired_binary_change(
        [bool(baseline_rows[index]["false_positive"]) for index in safe_indices],
        [bool(candidate_rows[index]["false_positive"]) for index in safe_indices],
        baseline_name=baseline_name,
        candidate_name=candidate_name,
    )
    j_delta = operational_delta["value"] - fpr_delta["value"]

    baseline_distribution = baseline_metrics["lead_distributions"][
        "operational_tp_lead_ms"
    ]
    candidate_distribution = candidate_metrics["lead_distributions"][
        "operational_tp_lead_ms"
    ]
    if (
        baseline_distribution.get("status") == "available"
        and candidate_distribution.get("status") == "available"
    ):
        armwise_difference: dict[str, Any] = {
            "status": "available",
            "value": (
                float(candidate_distribution["median"])
                - float(baseline_distribution["median"])
            ),
            "paired": False,
            "composition_sensitive": True,
        }
    else:
        armwise_difference = {
            "status": "unavailable",
            "reason": "one_or_both_operational_lead_distributions_unavailable",
            "value": None,
            "paired": False,
            "composition_sensitive": True,
        }
    common_lead_deltas = [
        float(candidate_rows[index]["lead_ms"])
        - float(baseline_rows[index]["lead_ms"])
        for index in disruptive_indices
        if baseline_rows[index]["operational_tp"]
        and candidate_rows[index]["operational_tp"]
    ]
    common_distribution = _distribution(common_lead_deltas)
    if common_distribution["status"] == "available":
        common_timing: dict[str, Any] = {
            "status": "available",
            "n_both": common_distribution["n"],
            "median": common_distribution["median"],
            "q1": common_distribution["q1"],
            "q3": common_distribution["q3"],
            "paired": True,
            "direction": f"{candidate_name}_minus_{baseline_name}",
        }
    else:
        common_timing = {
            "status": "unavailable",
            "reason": "no_common_operational_true_positives",
            "n_both": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "paired": True,
            "direction": f"{candidate_name}_minus_{baseline_name}",
        }
    return {
        "status": "available",
        "comparison_unit": "physical_shot_within_seed",
        "direction": f"{candidate_name}_minus_{baseline_name}",
        "pairing_hashes": {key: static_hashes[key] for key in hash_keys},
        "binary_deltas": {
            "operational_tpr": operational_delta,
            "fpr": fpr_delta,
            "j": {
                "status": "available",
                "value": j_delta,
                "direction": f"{candidate_name}_minus_{baseline_name}",
            },
        },
        "timing_deltas": {
            "armwise_operational_median_difference_ms": armwise_difference,
            "common_operational_tp_lead_delta_ms": common_timing,
        },
    }


def audit_query_update_resources(
    query_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    *,
    expected_shots: int,
    block_size: int,
) -> dict[str, Any]:
    """Recompute query, update, window, replay, and FLOP resources row by row."""

    if expected_shots <= 0 or block_size <= 0:
        raise EvidenceContractError("resource audit sizes must be positive")
    if len(query_rows) != expected_shots:
        raise EvidenceContractError(
            f"query ledger length mismatch: {len(query_rows)} != {expected_shots}"
        )
    selected_queries: dict[int, tuple[int, int]] = {}
    seen_shots: set[int] = set()
    selected_per_block: dict[str, int] = {}
    for expected_index, row in enumerate(query_rows):
        try:
            stream_index = int(row["stream_index"])
            shot = int(row["shot"])
            block = int(row["block"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceContractError("query receipt lacks identity fields") from error
        if stream_index != expected_index:
            raise EvidenceContractError("query ledger stream order is not contiguous")
        if shot in seen_shots:
            raise EvidenceContractError("query ledger contains duplicate shots")
        seen_shots.add(shot)
        expected_block = stream_index // block_size
        if block != expected_block:
            raise EvidenceContractError("query ledger block assignment mismatch")
        selected_per_block.setdefault(str(block), 0)
        if bool(row.get("selected")):
            selected_per_block[str(block)] += 1
            selected_queries[stream_index] = (shot, block)

    if len(update_rows) != len(selected_queries):
        raise EvidenceContractError(
            "update receipts do not match the selected query count"
        )
    optimizer_attempts = 0
    accepted_updates = 0
    rollbacks = 0
    steps_committed = 0
    current_windows = 0
    replay_windows = 0
    guard_windows = 0
    reported_flops = 0
    replay_sizes: list[int] = []
    flop_inputs_rows: list[dict[str, int]] = []
    update_streams: set[int] = set()
    for expected_attempt, row in enumerate(update_rows):
        try:
            stream_index = int(row["stream_index"])
            shot = int(row["shot"])
            steps = int(row["steps_attempted"])
            committed = int(row["steps_committed"])
            compute = row["compute"]
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceContractError("update receipt lacks resource fields") from error
        if stream_index in update_streams:
            raise EvidenceContractError("duplicate update receipt for one stream position")
        update_streams.add(stream_index)
        if selected_queries.get(stream_index, (None, None))[0] != shot:
            raise EvidenceContractError("update receipt is not bound to a selected query")
        if not isinstance(compute, Mapping):
            raise EvidenceContractError("update compute receipt is not an object")
        try:
            attempt_index = int(compute["attempt_index"])
            current = int(compute["current_windows"])
            replay = int(compute["replay_windows"])
            guard = int(compute["guard_windows"])
            compute_attempts = int(compute["optimizer_attempts"])
            replay_size = int(compute["replay_size"])
            reported = int(compute["reported_flops"])
            flop_inputs = compute["flop_inputs"]
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceContractError("compute receipt lacks required resources") from error
        if attempt_index != expected_attempt:
            raise EvidenceContractError("compute attempt indices are not contiguous")
        if min(current, replay, guard, compute_attempts, replay_size, reported) < 0:
            raise EvidenceContractError("compute resources cannot be negative")
        if steps != compute_attempts:
            raise EvidenceContractError("optimizer attempts disagree with update receipt")
        if not isinstance(flop_inputs, Mapping):
            raise EvidenceContractError("FLOP inputs are not an object")
        expected_inputs = {
            "current_windows": current,
            "replay_windows": replay,
            "guard_windows": guard,
            "optimizer_attempts": compute_attempts,
            "online_parameter_count": int(flop_inputs.get("online_parameter_count", -1)),
        }
        if expected_inputs["online_parameter_count"] < 0:
            raise EvidenceContractError("FLOP inputs lack online_parameter_count")
        if {key: int(flop_inputs.get(key, -1)) for key in expected_inputs} != expected_inputs:
            raise EvidenceContractError("FLOP inputs disagree with compute receipt")
        recomputed_flops = (
            2
            * expected_inputs["online_parameter_count"]
            * (current + replay + guard)
            * compute_attempts
        )
        if reported != recomputed_flops:
            raise EvidenceContractError("reported FLOPs do not recompute from receipt inputs")
        optimizer_attempts += steps
        steps_committed += committed
        accepted = bool(row.get("accepted"))
        accepted_updates += int(accepted)
        rollbacks += int(not accepted)
        current_windows += current
        replay_windows += replay
        guard_windows += guard
        reported_flops += reported
        replay_sizes.append(replay_size)
        flop_inputs_rows.append(expected_inputs)

    return {
        "status": "available",
        "queries": {
            "decision_count": len(query_rows),
            "selected_count": len(selected_queries),
            "selected_per_block": selected_per_block,
        },
        "updates": {
            "receipt_count": len(update_rows),
            "optimizer_attempts": optimizer_attempts,
            "accepted_updates": accepted_updates,
            "rollbacks": rollbacks,
            "steps_committed": steps_committed,
        },
        "resources": {
            "window_totals": {
                "current": current_windows,
                "replay": replay_windows,
                "guard": guard_windows,
            },
            "replay_size_sequence": replay_sizes,
            "flop_inputs_sha256": _sha256_json(flop_inputs_rows),
            "reported_flops": reported_flops,
        },
    }


def compare_matched_resources(
    active: Mapping[str, Any],
    random: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate the selector estimand on exact realized active/random resources."""

    if active.get("status") != "available" or random.get("status") != "available":
        return {
            "status": "unavailable",
            "selector_estimand_status": "suppressed",
            "reason": "one_or_both_resource_audits_unavailable",
        }
    comparable = {
        "query_count": (
            active["queries"]["selected_count"],
            random["queries"]["selected_count"],
        ),
        "selected_per_block": (
            active["queries"]["selected_per_block"],
            random["queries"]["selected_per_block"],
        ),
        "update_receipt_count": (
            active["updates"]["receipt_count"],
            random["updates"]["receipt_count"],
        ),
        "optimizer_attempts": (
            active["updates"]["optimizer_attempts"],
            random["updates"]["optimizer_attempts"],
        ),
        "window_totals": (
            active["resources"]["window_totals"],
            random["resources"]["window_totals"],
        ),
        "replay_size_sequence": (
            active["resources"]["replay_size_sequence"],
            random["resources"]["replay_size_sequence"],
        ),
        "flop_inputs_sha256": (
            active["resources"]["flop_inputs_sha256"],
            random["resources"]["flop_inputs_sha256"],
        ),
        "reported_flops": (
            active["resources"]["reported_flops"],
            random["resources"]["reported_flops"],
        ),
    }
    mismatches = [key for key, (left, right) in comparable.items() if left != right]
    if mismatches:
        return {
            "status": "mismatch",
            "selector_estimand_status": "suppressed",
            "mismatched_fields": mismatches,
            "matched": False,
        }
    return {
        "status": "available",
        "selector_estimand_status": "available",
        "mismatched_fields": [],
        "matched": True,
    }
