"""Fail-closed lineage contract for the selected J-TEXT -> EAST mainline.

Legacy source and parent schemas remain owned by their archival runners.  This
module accepts only the selected ``advanced_floor`` J-TEXT source and the
explicit ``fixed125`` EAST parent.  It never infers either policy from a
missing path or from a legacy payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
JTEXT_ADVANCED_TIME = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"

SELECTED_PARENT_SCHEMA = "dynamic-source-east-fixed-parent-v1"
SELECTED_LABEL_ARM = "source_dynamic__east_fixed"
SOURCE_LABEL_POLICY = "advanced_floor"
EAST_TARGET_LABEL_POLICY = "fixed125"
EAST_TARGET_HORIZON_MS = 125.0
CLAIM_STATUS = "TEST_DRIVEN_EXPLORATION"
SELECTION_ROLE = "post_adjudication_user_selected_development_candidate"
HISTORICAL_ADMISSION = "FAIL"
CONFIRMATION_ELIGIBLE = False
CORRECTED_ABLATION_REPORT_SHA256 = (
    "896584467254d2ae1df894f6f4a0e4ed3a024ec556dcc1ffeb7e083f04db3d04"
)
SELECTION_METRIC = "mean balanced360 operational_TPR - FPR"
HISTORICAL_FAILURE_REASON = "seed133 TPR@30 < 0.80"
THRESHOLD_RECEIPT_SCHEMA = "selected-threshold-receipt-v1"
THRESHOLD_SELECTION_RULE = (
    "meet target TPR, then minimize FPR, then maximize finite threshold"
)
STATIC_SEAL_SCHEMA = "selected-static-seal-v1"
ALLOWED_SEEDS = (131, 133, 137)
EXPECTED_SPLIT_COUNTS = {"train": 1353, "val": 160, "test": 220}
SOURCE_INPUT_SCHEMA = "jtext-rebuilt17-prefix-causal-v2"
SOURCE_RAW_SIGNAL_NAMES = (
    "Bt", "ip", "v_loop", "sxr_kurt", "sxr_skew", "sxr_var",
    "xuv_kurt", "xuv_skew", "xuv_var", "Mir_avg_fre",
    "n=1 amplitude", "mode_number_n", "dx_a", "dy_a",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_JSON_CONTENT_CACHE: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    # Parent construction may receive a hash-only immutable contract while the
    # corresponding receipt/scaler objects were just computed in this process.
    # Keep a content-addressed copy; persisted startup never relies on this
    # cache and still validates every embedded object from bytes on disk.
    _JSON_CONTENT_CACHE[digest] = json.loads(encoded.decode("utf-8"))
    return digest


def resolve_json_content(digest: str, *, name: str) -> Any:
    """Resolve recently computed content by digest or fail closed."""

    _require_sha256(digest, name)
    if digest not in _JSON_CONTENT_CACHE:
        raise RuntimeError(f"{name} content is required, not only its SHA256")
    value = _JSON_CONTENT_CACHE[digest]
    if sha256_json(value) != digest:
        raise RuntimeError(f"{name} content cache SHA256 mismatch")
    return json.loads(json.dumps(value, allow_nan=False))


def _state_items(state: Mapping[str, Any]) -> list[tuple[str, torch.Tensor]]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state dictionary must be a non-empty mapping")
    items: list[tuple[str, torch.Tensor]] = []
    for name, raw_tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(raw_tensor, torch.Tensor):
            raise ValueError("state dictionary must contain named tensors only")
        items.append((name, raw_tensor.detach().cpu().contiguous()))
    return items


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash one state with the same codec used by all runtime module helpers."""

    return state_bundle_sha256(state)


def state_bundle_sha256(*states: Mapping[str, Any]) -> str:
    if not states:
        raise ValueError("state bundle cannot be empty")
    digest = hashlib.sha256()
    for state_index, state in enumerate(states):
        for name, tensor in _state_items(state):
            value = tensor.numpy()
            digest.update(
                f"{state_index}:{name}:{value.dtype}:{value.shape}".encode("utf-8")
            )
            digest.update(value.tobytes())
    return digest.hexdigest()


def scaler_sha256(center: torch.Tensor, scale: torch.Tensor) -> str:
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("source scaler tensors are required")
    return state_dict_sha256({"center": center, "scale": scale})


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _normalize_threshold_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in predictions:
        if not isinstance(raw, Mapping) or set(raw) != {"shot", "score", "outcome"}:
            raise ValueError("threshold receipt predictions have an invalid schema")
        shot = int(raw["shot"])
        score = float(raw["score"])
        outcome = raw["outcome"]
        if isinstance(raw["shot"], bool) or not math.isfinite(score):
            raise ValueError("threshold receipt predictions must be finite")
        if not isinstance(outcome, (bool, np.bool_)):
            raise ValueError("threshold receipt outcomes must be boolean")
        normalized.append({"shot": shot, "score": score, "outcome": bool(outcome)})
    shots = [row["shot"] for row in normalized]
    if not normalized or len(shots) != len(set(shots)):
        raise ValueError("threshold receipt roster must be non-empty and unique")
    outcomes = np.asarray([row["outcome"] for row in normalized], dtype=bool)
    if outcomes.all() or not outcomes.any():
        raise ValueError("threshold receipt requires both support outcomes")
    return normalized


def _threshold_candidates(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    scores = np.asarray([float(row["score"]) for row in predictions], dtype=np.float64)
    outcomes = np.asarray([bool(row["outcome"]) for row in predictions], dtype=bool)
    candidates: list[dict[str, float]] = []
    for threshold in sorted(
        {float(score) for score in scores if 0.0 < float(score) < 1.0},
        reverse=True,
    ):
        candidates.append({
            "threshold": threshold,
            "tpr": float(np.mean(scores[outcomes] >= threshold)),
            "fpr": float(np.mean(scores[~outcomes] >= threshold)),
        })
    if not candidates:
        raise ValueError("threshold receipt has no finite candidate in 0 < t < 1")
    return candidates


def build_threshold_receipt(
    *, scope: str, predictions: Sequence[Mapping[str, Any]],
    data_manifest_sha256: str, outcome_manifest_sha256: str,
    target_tpr: float,
) -> dict[str, Any]:
    """Build the only selected threshold decision receipt."""

    if scope not in {"J-TEXT VAL160", "EAST support110"}:
        raise ValueError("threshold receipt scope is not selected")
    normalized = _normalize_threshold_predictions(predictions)
    data_sha = _require_sha256(data_manifest_sha256, "threshold data manifest")
    outcome_sha = _require_sha256(
        outcome_manifest_sha256, "threshold outcome manifest",
    )
    target = float(target_tpr)
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("threshold receipt target TPR is invalid")
    candidates = _threshold_candidates(normalized)
    eligible = [row for row in candidates if row["tpr"] >= target]
    if not eligible:
        raise RuntimeError(f"threshold receipt cannot meet target TPR={target}")
    chosen = min(
        eligible,
        key=lambda row: (row["fpr"], -row["threshold"], -row["tpr"]),
    )
    roster = [int(row["shot"]) for row in normalized]
    body: dict[str, Any] = {
        "schema_version": THRESHOLD_RECEIPT_SCHEMA,
        "scope": scope,
        "roster": roster,
        "roster_sha256": sha256_json(roster),
        "data_manifest_sha256": data_sha,
        "outcome_manifest_sha256": outcome_sha,
        "selection_rule": THRESHOLD_SELECTION_RULE,
        "target_tpr": target,
        "predictions": normalized,
        "candidates": candidates,
        "chosen_threshold": float(chosen["threshold"]),
    }
    return {**body, "receipt_sha256": sha256_json(body)}


def validate_threshold_receipt(
    receipt: Mapping[str, Any], *, expected_scope: str,
    expected_roster: Sequence[int] | None = None,
    expected_threshold: float | None = None,
) -> dict[str, Any]:
    """Recompute roster, candidates, choice, and receipt digest fail closed."""

    def fail(message: str) -> RuntimeError:
        return RuntimeError(f"threshold receipt {message}")

    if not isinstance(receipt, Mapping):
        raise fail("is missing")
    body = dict(receipt)
    receipt_sha = body.pop("receipt_sha256", None)
    try:
        _require_sha256(receipt_sha, "threshold receipt SHA256")
    except ValueError as exc:
        raise fail("SHA256 is invalid") from exc
    if receipt_sha != sha256_json(body):
        raise fail("SHA256 mismatch")
    if body.get("schema_version") != THRESHOLD_RECEIPT_SCHEMA:
        raise fail("schema mismatch")
    if body.get("scope") != expected_scope:
        raise fail("scope mismatch")
    if body.get("selection_rule") != THRESHOLD_SELECTION_RULE:
        raise fail("selection rule mismatch")
    try:
        normalized = _normalize_threshold_predictions(body.get("predictions", []))
        _require_sha256(body.get("data_manifest_sha256"), "threshold data manifest")
        _require_sha256(
            body.get("outcome_manifest_sha256"), "threshold outcome manifest",
        )
    except ValueError as exc:
        raise fail(str(exc)) from exc
    roster = [row["shot"] for row in normalized]
    if body.get("roster") != roster or body.get("roster_sha256") != sha256_json(roster):
        raise fail("roster mismatch")
    if expected_roster is not None and roster != [int(shot) for shot in expected_roster]:
        raise fail("expected roster mismatch")
    target = float(body.get("target_tpr", float("nan")))
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise fail("target TPR mismatch")
    all_candidates = _threshold_candidates(normalized)
    candidate_by_threshold = {row["threshold"]: row for row in all_candidates}
    serialized_candidates = body.get("candidates")
    if not isinstance(serialized_candidates, list) or not serialized_candidates:
        raise fail("candidates are missing")
    for candidate in serialized_candidates:
        if not isinstance(candidate, Mapping):
            raise fail("candidate schema mismatch")
        threshold = float(candidate.get("threshold", float("nan")))
        expected = candidate_by_threshold.get(threshold)
        if expected is None or any(
            not math.isclose(
                float(candidate.get(name, float("nan"))), float(expected[name]),
                rel_tol=0.0, abs_tol=1e-12,
            )
            for name in ("threshold", "tpr", "fpr")
        ):
            raise fail("candidate recomputation mismatch")
    eligible = [row for row in all_candidates if row["tpr"] >= target]
    if not eligible:
        raise fail("cannot meet target TPR")
    chosen = min(
        eligible,
        key=lambda row: (row["fpr"], -row["threshold"], -row["tpr"]),
    )["threshold"]
    actual_chosen = float(body.get("chosen_threshold", float("nan")))
    if (
        not math.isfinite(actual_chosen)
        or not 0.0 < actual_chosen < 1.0
        or not math.isclose(actual_chosen, chosen, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise fail("chosen threshold mismatch")
    if expected_threshold is not None and not math.isclose(
        actual_chosen, float(expected_threshold), rel_tol=0.0, abs_tol=1e-7,
    ):
        raise fail("expected threshold mismatch")
    return {**body, "receipt_sha256": str(receipt_sha)}


@dataclass(frozen=True)
class SelectedMainlineContract:
    """Immutable source/target lineage shared by every selected stage."""

    seed: int
    selection_role: str
    historical_admission: str
    confirmation_eligible: bool
    corrected_ablation_report_sha256: str
    selection_metric: str
    historical_failure_reason: str
    source_checkpoint_sha256: str
    source_label_policy: str
    source_label_map_sha256: str
    source_split_sha256: str
    source_scaler_sha256: str
    source_input_contract_sha256: str
    source_preprocessing_sha256: str
    source_val_threshold_receipt_sha256: str
    east_target_label_policy: str
    east_target_horizon_ms: float
    east_support_roster_sha256: str
    east_support_data_manifest_sha256: str
    east_support_outcome_manifest_sha256: str
    east_threshold_receipt_sha256: str
    target_scaler_sha256: str
    raw8_scaler_sha256: str
    physics_scales_sha256: str
    reconstruction_constants_sha256: str
    claim_status: str
    jtext_test_roster_sha256: str | None = None
    east_test_roster_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or int(self.seed) not in ALLOWED_SEEDS:
            raise ValueError(f"selected mainline seed must be one of {ALLOWED_SEEDS}")
        if self.selection_role != SELECTION_ROLE:
            raise ValueError("selected mainline selection role mismatch")
        if self.historical_admission != HISTORICAL_ADMISSION:
            raise ValueError("selected mainline historical admission mismatch")
        if self.confirmation_eligible is not CONFIRMATION_ELIGIBLE:
            raise ValueError("selected mainline confirmation eligibility mismatch")
        if (
            self.corrected_ablation_report_sha256
            != CORRECTED_ABLATION_REPORT_SHA256
        ):
            raise ValueError("selected mainline corrected ablation SHA256 mismatch")
        if self.selection_metric != SELECTION_METRIC:
            raise ValueError("selected mainline selection metric mismatch")
        if self.historical_failure_reason != HISTORICAL_FAILURE_REASON:
            raise ValueError("selected mainline historical failure reason mismatch")
        if self.source_label_policy != SOURCE_LABEL_POLICY:
            raise ValueError("selected source label policy must be advanced_floor")
        if self.east_target_label_policy != EAST_TARGET_LABEL_POLICY:
            raise ValueError("selected EAST target label policy must be fixed125")
        if not math.isclose(
            float(self.east_target_horizon_ms), EAST_TARGET_HORIZON_MS,
            rel_tol=0.0, abs_tol=0.0,
        ):
            raise ValueError("selected EAST target horizon must be exactly 125.0 ms")
        if self.claim_status != CLAIM_STATUS:
            raise ValueError("selected mainline claim status mismatch")
        for name in (
            "source_checkpoint_sha256", "source_label_map_sha256",
            "source_split_sha256", "source_scaler_sha256",
            "source_input_contract_sha256", "source_preprocessing_sha256",
            "source_val_threshold_receipt_sha256",
            "east_support_roster_sha256", "east_support_data_manifest_sha256",
            "east_support_outcome_manifest_sha256",
            "east_threshold_receipt_sha256", "target_scaler_sha256",
            "raw8_scaler_sha256", "physics_scales_sha256",
            "reconstruction_constants_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("jtext_test_roster_sha256", "east_test_roster_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectedMainlineContract":
        if not isinstance(value, Mapping):
            raise ValueError("selected mainline contract must be a mapping")
        field_names = {field.name for field in fields(cls)}
        optional = {"jtext_test_roster_sha256", "east_test_roster_sha256"}
        missing = sorted(field_names.difference(optional).difference(value))
        if missing:
            raise ValueError(
                "selected mainline contract is missing: " + ", ".join(missing)
            )
        unknown = sorted(set(value).difference(field_names))
        if unknown:
            raise ValueError(
                "selected mainline contract has unknown fields: "
                + ", ".join(unknown)
            )
        return cls(**{name: value[name] for name in field_names if name in value})


_FORBIDDEN_ROSTER_FIELDS = {
    "is_disrupt", "isdisrupt", "outcome", "label", "class", "target",
    "downtime", "down_time", "down_time_s", "advancedtime", "advanced_time",
    "advanced_time_ms", "ttd", "ttd_ms",
}


def load_label_free_roster(path: Path | str, *, expected_count: int) -> list[int]:
    """Load ordered ``stream_index``/``shot`` rows without any label object."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"label-free roster does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"label-free roster line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict) or set(row) != {"stream_index", "shot"}:
            forbidden = sorted(
                str(name) for name in row
                if str(name).replace("-", "_").lower() in _FORBIDDEN_ROSTER_FIELDS
            ) if isinstance(row, dict) else []
            detail = f": {', '.join(forbidden)}" if forbidden else ""
            raise ValueError(f"label-free roster schema mismatch{detail}")
        rows.append(row)
    if len(rows) != int(expected_count):
        raise ValueError(
            f"label-free roster requires {int(expected_count)} rows, got {len(rows)}"
        )
    indices = [row["stream_index"] for row in rows]
    if any(isinstance(index, bool) for index in indices) or indices != list(
        range(int(expected_count))
    ):
        raise ValueError("label-free roster stream indices are not contiguous")
    shots = [int(row["shot"]) for row in rows]
    if any(isinstance(row["shot"], bool) for row in rows) or len(set(shots)) != len(shots):
        raise ValueError("label-free roster shot IDs must be unique integers")
    return shots


def read_required_timebase(signal: Any) -> tuple[float, float]:
    """Read selected-path metadata with no silent SampleRate/StartTime fallback."""

    attrs = getattr(signal, "attrs", None)
    if attrs is None:
        raise RuntimeError("selected metadata object has no attributes")
    for name in ("SampleRate", "StartTime"):
        if name not in attrs:
            raise RuntimeError(f"selected metadata is missing {name}")
    rate = float(np.asarray(attrs["SampleRate"]).reshape(-1)[0])
    start = float(np.asarray(attrs["StartTime"]).reshape(-1)[0])
    if not math.isfinite(rate) or rate <= 0.0:
        raise RuntimeError("selected SampleRate must be finite and positive")
    if not math.isfinite(start):
        raise RuntimeError("selected StartTime must be finite")
    return rate, start


def support_data_manifest_sha256(shots: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for shot in sorted(shots, key=lambda row: int(row["shot"])):
        row: dict[str, Any] = {"shot": int(shot["shot"])}
        for name in ("v", "a"):
            if name in shot:
                row[f"{name}_sha256"] = array_sha256(shot[name])
        for name in ("rate", "start"):
            if name in shot:
                value = float(shot[name])
                if not math.isfinite(value):
                    raise ValueError("support data manifest contains non-finite metadata")
                row[name] = value
        rows.append(row)
    if not rows or len({row["shot"] for row in rows}) != len(rows):
        raise ValueError("support data manifest requires unique shots")
    return sha256_json(rows)


def support_outcome_manifest_sha256(shots: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for shot in sorted(shots, key=lambda row: int(row["shot"])):
        if "is_disrupt" not in shot:
            raise ValueError("support outcome manifest is missing is_disrupt")
        down = shot.get("down", shot.get("down_time_s"))
        rows.append({
            "shot": int(shot["shot"]),
            "is_disrupt": bool(shot["is_disrupt"]),
            "down_time_s": float(down) if down is not None else None,
            "positive_horizon_ms": (
                float(shot["positive_horizon_ms"])
                if shot.get("positive_horizon_ms") is not None else None
            ),
        })
    if not rows or len({row["shot"] for row in rows}) != len(rows):
        raise ValueError("support outcome manifest requires unique shots")
    return sha256_json(rows)


def write_trace_file(path: Path, *, shot: int, risk_trace: Any) -> str:
    array = np.asarray(risk_trace, dtype=np.float32)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise RuntimeError(f"shot {int(shot)} has an invalid static risk trace")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"static trace already exists: {path}")
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            shot_id=np.asarray([int(shot)], dtype=np.int64),
            risk_trace=array,
        )
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def write_static_seal_manifest(
    path: Path, *, stage: str, prediction_path: Path,
    roster: Sequence[int], risk_traces: Sequence[Mapping[str, Any]],
    threshold_receipt_sha256: str, parent_fingerprint_sha256: str,
) -> tuple[dict[str, Any], str]:
    _require_sha256(threshold_receipt_sha256, "static threshold receipt")
    _require_sha256(parent_fingerprint_sha256, "static parent fingerprint")
    ordered = [int(shot) for shot in roster]
    if len(risk_traces) != len(ordered):
        raise RuntimeError("static seal risk-trace count mismatch")
    normalized_traces: list[dict[str, Any]] = []
    for expected_shot, raw in zip(ordered, risk_traces):
        trace_path = Path(str(raw["path"]))
        absolute = trace_path if trace_path.is_absolute() else path.parent / trace_path
        digest = _require_sha256(raw.get("sha256"), "static risk-trace SHA256")
        if int(raw.get("shot", -1)) != expected_shot:
            raise RuntimeError("static seal risk-trace roster mismatch")
        if not absolute.is_file() or sha256_file(absolute) != digest:
            raise RuntimeError(f"static risk trace hash mismatch: {expected_shot}")
        normalized_traces.append({
            "shot": expected_shot,
            "path": str(trace_path),
            "sha256": digest,
        })
    prediction_sha = sha256_file(prediction_path)
    body = {
        "schema_version": STATIC_SEAL_SCHEMA,
        "stage": stage,
        "row_count": len(ordered),
        "roster_sha256": sha256_json(ordered),
        "prediction_path": prediction_path.name,
        "prediction_sha256": prediction_sha,
        "risk_traces": normalized_traces,
        "threshold_receipt_sha256": threshold_receipt_sha256,
        "parent_fingerprint_sha256": parent_fingerprint_sha256,
    }
    if path.exists():
        raise FileExistsError(f"static seal manifest already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return body, sha256_file(path)


def validate_static_seal_manifest(
    path: Path | str, *, expected_count: int | None = None,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("static seal manifest is unreadable") from exc
    if manifest.get("schema_version") != STATIC_SEAL_SCHEMA:
        raise RuntimeError("static seal manifest schema mismatch")
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise RuntimeError("static seal manifest stage mismatch")
    row_count = int(manifest.get("row_count", -1))
    if expected_count is not None and row_count != int(expected_count):
        raise RuntimeError("static seal manifest row-count mismatch")
    prediction_path = resolved.parent / str(manifest.get("prediction_path", ""))
    if (
        not prediction_path.is_file()
        or sha256_file(prediction_path) != manifest.get("prediction_sha256")
    ):
        raise RuntimeError("static prediction seal hash mismatch")
    lines = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != row_count:
        raise RuntimeError("static prediction seal row-count mismatch")
    roster = [int(row.get("shot", -1)) for row in lines]
    if manifest.get("roster_sha256") != sha256_json(roster):
        raise RuntimeError("static prediction seal roster mismatch")
    traces = manifest.get("risk_traces")
    if not isinstance(traces, list) or len(traces) != row_count:
        raise RuntimeError("static seal risk-trace manifest is incomplete")
    for row, trace in zip(lines, traces):
        shot = int(row.get("shot", -1))
        if int(trace.get("shot", -2)) != shot:
            raise RuntimeError("static risk trace shot mismatch")
        trace_path = resolved.parent / str(trace.get("path", ""))
        digest = trace.get("sha256")
        if (
            row.get("risk_trace_sha256") != digest
            or not trace_path.is_file()
            or sha256_file(trace_path) != digest
        ):
            raise RuntimeError(f"static risk trace hash mismatch: {shot}")
    _require_sha256(
        manifest.get("threshold_receipt_sha256"), "static threshold receipt",
    )
    _require_sha256(
        manifest.get("parent_fingerprint_sha256"), "static parent fingerprint",
    )
    return manifest


def _expected_source_split_sha256() -> str:
    import train_jtext_source_transformer as source_trainer

    roles = source_trainer.load_canonical_jtext_roles(ROOT)
    return source_trainer.split_sha256(roles)


def _validate_training_label(payload: Mapping[str, Any]) -> str:
    training_label = payload.get("training_label")
    if not isinstance(training_label, Mapping):
        raise ValueError("selected source training label contract is missing")
    if training_label.get("policy") != SOURCE_LABEL_POLICY:
        raise ValueError("selected source training label policy must be advanced_floor")
    required = {
        "outcome_authority": "HDF5 meta/IsDisrupt",
        "down_time_authority": "HDF5 meta/DownTime",
        "advanced_time_role": "positive-window upper boundary only",
        "advanced_time_used": True,
        "formula": "upper_ms=max(25, AdvancedTime_ms); missing->25",
    }
    for name, expected in required.items():
        if training_label.get(name) != expected:
            raise ValueError(f"selected source label-map provenance mismatch: {name}")
    label_map_sha = _require_sha256(
        training_label.get("advanced_time_sha256"), "source label-map SHA256",
    )
    if not JTEXT_ADVANCED_TIME.is_file():
        raise FileNotFoundError(
            f"selected J-TEXT AdvancedTime label map does not exist: {JTEXT_ADVANCED_TIME}"
        )
    if label_map_sha != sha256_file(JTEXT_ADVANCED_TIME):
        raise ValueError("selected source label-map SHA256 mismatch")
    audit = training_label.get("audit")
    if not isinstance(audit, Mapping) or any(
        audit.get(name) != expected
        for name, expected in {
            "device": "J-TEXT", "policy": SOURCE_LABEL_POLICY,
            "unit": "ms", "floor_ms": 25.0,
        }.items()
    ):
        raise ValueError("selected source label-map audit provenance mismatch")
    return label_map_sha


def validate_source_input_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require the prefix-causal preprocessing contract added by revised Task 2."""

    value = payload.get("input_contract")
    if not isinstance(value, Mapping):
        raise ValueError("selected source input contract is missing")
    import grid_search as grid

    exact_fields: dict[str, Any] = {
        "schema": SOURCE_INPUT_SCHEMA,
        "raw_signal_names": list(SOURCE_RAW_SIGNAL_NAMES),
        "feature_names": list(grid.RH.REBUILD),
        "window_points": 128,
        "history_alignment": "right",
        "left_padding": "constant_zero",
        "valid_lengths_required": True,
        "valid_length_range": [1, 128],
        "patch_mask_rule": "patch_end <= window_points - valid_length",
        "availability": "finite-and-prefix-causal-seen-nonzero-v1",
    }
    for name, expected in exact_fields.items():
        if value.get(name) != expected:
            raise ValueError(f"selected source input contract mismatch: {name}")
    try:
        preprocessing_sha = _require_sha256(
            value.get("preprocessing_sha256"),
            "selected source input contract preprocessing_sha256",
        )
    except ValueError as exc:
        raise ValueError(
            "selected source input contract mismatch: preprocessing_sha256"
        ) from exc
    normalized = {name: value[name] for name in exact_fields}
    normalized["preprocessing_sha256"] = preprocessing_sha
    return normalized


def load_selected_source_checkpoint(
    path: Path | str | None, seed: int,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Load only an explicit, TRAIN/VAL-selected dynamic-label checkpoint."""

    if path is None:
        raise ValueError("selected mainline requires an explicit source checkpoint")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"selected J-TEXT source checkpoint does not exist: {resolved}"
        )

    import grid_search as grid

    encoder, head, raw_payload = grid.load_source_checkpoint(
        resolved, channels=len(grid.RH.REBUILD), expected_seed=int(seed),
        device="cpu", allow_fallback=False,
    )
    payload = dict(raw_payload)
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("selected source checkpoint seed mismatch")
    if payload.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("selected source split-count contract mismatch")
    expected_split = _expected_source_split_sha256()
    if payload.get("split_sha256") != expected_split:
        raise ValueError("selected source split SHA256 mismatch")
    if int(payload.get("test_shards_read", -1)) != 0:
        raise ValueError("selected source checkpoint read J-TEXT TEST220")
    input_contract = validate_source_input_contract(payload)
    label_map_sha = _validate_training_label(payload)
    import train_jtext_source_transformer as source_trainer

    roles = source_trainer.load_canonical_jtext_roles(ROOT)
    try:
        source_threshold = float(payload.get("validation_threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("selected source VAL threshold is missing") from exc
    source_threshold_receipt = validate_threshold_receipt(
        payload.get("validation_threshold_receipt", {}),
        expected_scope="J-TEXT VAL160",
        expected_roster=roles["val"],
        expected_threshold=source_threshold,
    )
    scaler_hash = scaler_sha256(
        payload.get("scaler_center"), payload.get("scaler_scale"),
    )
    embedded_scaler_hash = payload.get("scaler_sha256")
    if embedded_scaler_hash is not None and embedded_scaler_hash != scaler_hash:
        raise ValueError("selected source scaler SHA256 mismatch")
    embedded_label_hash = payload.get("label_map_sha256")
    if embedded_label_hash is not None and embedded_label_hash != label_map_sha:
        raise ValueError("selected source label-map SHA256 mismatch")
    payload.update({
        "source_checkpoint_sha256": sha256_file(resolved),
        "source_label_map_sha256": label_map_sha,
        "source_scaler_sha256": scaler_hash,
        "source_input_contract_sha256": sha256_json(input_contract),
        "source_preprocessing_sha256": input_contract["preprocessing_sha256"],
        "source_val_threshold_receipt_sha256": source_threshold_receipt[
            "receipt_sha256"
        ],
    })
    return encoder, head, payload


def contract_from_source(
    source_path: Path | str, source_payload: Mapping[str, Any], *, seed: int,
    east_support_roster_sha256: str,
    east_support_data_manifest_sha256: str,
    east_support_outcome_manifest_sha256: str,
    east_threshold_receipt_sha256: str,
    target_scaler_sha256: str,
    raw8_scaler_sha256: str,
    physics_scales_sha256: str,
    reconstruction_constants_sha256: str,
    jtext_test_roster_sha256: str | None = None,
    east_test_roster_sha256: str | None = None,
) -> SelectedMainlineContract:
    return SelectedMainlineContract(
        seed=int(seed),
        selection_role=SELECTION_ROLE,
        historical_admission=HISTORICAL_ADMISSION,
        confirmation_eligible=CONFIRMATION_ELIGIBLE,
        corrected_ablation_report_sha256=CORRECTED_ABLATION_REPORT_SHA256,
        selection_metric=SELECTION_METRIC,
        historical_failure_reason=HISTORICAL_FAILURE_REASON,
        source_checkpoint_sha256=sha256_file(Path(source_path)),
        source_label_policy=str(source_payload["training_label"]["policy"]),
        source_label_map_sha256=str(source_payload["source_label_map_sha256"]),
        source_split_sha256=str(source_payload["split_sha256"]),
        source_scaler_sha256=str(source_payload["source_scaler_sha256"]),
        source_input_contract_sha256=str(
            source_payload["source_input_contract_sha256"]
        ),
        source_preprocessing_sha256=str(
            source_payload["source_preprocessing_sha256"]
        ),
        source_val_threshold_receipt_sha256=str(
            source_payload["source_val_threshold_receipt_sha256"]
        ),
        east_target_label_policy=EAST_TARGET_LABEL_POLICY,
        east_target_horizon_ms=EAST_TARGET_HORIZON_MS,
        east_support_roster_sha256=east_support_roster_sha256,
        east_support_data_manifest_sha256=east_support_data_manifest_sha256,
        east_support_outcome_manifest_sha256=east_support_outcome_manifest_sha256,
        east_threshold_receipt_sha256=east_threshold_receipt_sha256,
        target_scaler_sha256=target_scaler_sha256,
        raw8_scaler_sha256=raw8_scaler_sha256,
        physics_scales_sha256=physics_scales_sha256,
        reconstruction_constants_sha256=reconstruction_constants_sha256,
        claim_status=CLAIM_STATUS,
        jtext_test_roster_sha256=jtext_test_roster_sha256,
        east_test_roster_sha256=east_test_roster_sha256,
    )


def _require_parent_state(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    state = payload.get(name)
    try:
        _state_items(state)
    except ValueError as exc:
        raise RuntimeError(f"selected parent is missing {name} state") from exc
    return state


def canonical_parent_fingerprint(payload: Mapping[str, Any]) -> str:
    """Recompute a selected parent fingerprint without trusting embedded hashes."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("selected parent fingerprint requires a mapping")
    encoder = _require_parent_state(payload, "encoder")
    head = _require_parent_state(payload, "head")
    shared = _require_parent_state(payload, "shared")
    private = _require_parent_state(payload, "private")
    preprocessing: dict[str, str] = {}
    for name in (
        "target_scaler", "raw8_scaler", "physics_scales",
        "reconstruction_constants",
    ):
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"selected parent is missing {name}")
        preprocessing[f"{name}_sha256"] = sha256_json(value)
    metadata_names = (
        "schema_version", "selected_label_arm", "lineage", "seed",
        "selection_role", "historical_admission", "confirmation_eligible",
        "corrected_ablation_report_sha256", "selection_metric",
        "historical_failure_reason", "source_checkpoint_sha256",
        "source_label_policy", "source_label_map_sha256", "source_split_sha256",
        "source_scaler_sha256", "source_input_contract_sha256",
        "source_preprocessing_sha256", "source_val_threshold_receipt_sha256",
        "east_target_label_policy", "east_target_horizon_ms",
        "east_support_roster_sha256", "east_support_data_manifest_sha256",
        "east_support_outcome_manifest_sha256", "east_threshold_receipt_sha256",
        "jtext_test_roster_sha256", "east_test_roster_sha256", "claim_status",
    )
    fingerprint = {
        "metadata": {name: payload.get(name) for name in metadata_names},
        "east_threshold": float(payload.get("east_threshold")),
        "states": {
            "target_encoder_sha256": state_dict_sha256(encoder),
            "target_head_sha256": state_dict_sha256(head),
            "target_base_sha256": state_bundle_sha256(encoder, head),
            "shared_parent_sha256": state_dict_sha256(shared),
            "private_parent_sha256": state_dict_sha256(private),
        },
        "preprocessing": preprocessing,
    }
    return sha256_json(fingerprint)


def validate_selected_parent(
    payload: Mapping[str, Any], contract: SelectedMainlineContract,
) -> dict[str, Any]:
    """Validate metadata and every serialized model state before online reset."""

    if not isinstance(contract, SelectedMainlineContract):
        raise TypeError("selected parent validation requires SelectedMainlineContract")
    if not isinstance(payload, Mapping):
        raise RuntimeError("selected parent payload must be a mapping")
    if payload.get("schema_version") != SELECTED_PARENT_SCHEMA:
        raise RuntimeError("selected parent schema mismatch")
    if payload.get("selected_label_arm") != SELECTED_LABEL_ARM or payload.get(
        "lineage"
    ) != {"source": "source_dynamic", "target": "east_fixed"}:
        raise RuntimeError("selected parent source_dynamic/east_fixed lineage mismatch")
    if int(payload.get("seed", -1)) != contract.seed:
        raise RuntimeError("selected parent seed mismatch")
    if payload.get("source_checkpoint_sha256") != contract.source_checkpoint_sha256:
        raise RuntimeError("selected parent source checkpoint SHA256 mismatch")
    exact_metadata = {
        "selection_role": SELECTION_ROLE,
        "historical_admission": HISTORICAL_ADMISSION,
        "confirmation_eligible": CONFIRMATION_ELIGIBLE,
        "corrected_ablation_report_sha256": CORRECTED_ABLATION_REPORT_SHA256,
        "selection_metric": SELECTION_METRIC,
        "historical_failure_reason": HISTORICAL_FAILURE_REASON,
        "source_label_policy": SOURCE_LABEL_POLICY,
        "east_target_label_policy": EAST_TARGET_LABEL_POLICY,
        "claim_status": CLAIM_STATUS,
    }
    metadata_messages = {
        "source_label_policy": "source label policy",
        "east_target_label_policy": "target label policy",
    }
    for name, expected in exact_metadata.items():
        if payload.get(name) != expected or getattr(contract, name) != expected:
            raise RuntimeError(
                f"selected parent {metadata_messages.get(name, name)} mismatch"
            )
    try:
        target_horizon = float(payload.get("east_target_horizon_ms"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("selected parent EAST target horizon is missing") from exc
    if not math.isclose(
        target_horizon, EAST_TARGET_HORIZON_MS, rel_tol=0.0, abs_tol=0.0,
    ):
        raise RuntimeError("selected parent EAST target horizon mismatch")
    lineage_fields = {
        "source_label_map_sha256": contract.source_label_map_sha256,
        "source_split_sha256": contract.source_split_sha256,
        "source_scaler_sha256": contract.source_scaler_sha256,
        "source_input_contract_sha256": contract.source_input_contract_sha256,
        "source_preprocessing_sha256": contract.source_preprocessing_sha256,
        "source_val_threshold_receipt_sha256": (
            contract.source_val_threshold_receipt_sha256
        ),
        "east_support_roster_sha256": contract.east_support_roster_sha256,
        "east_support_data_manifest_sha256": (
            contract.east_support_data_manifest_sha256
        ),
        "east_support_outcome_manifest_sha256": (
            contract.east_support_outcome_manifest_sha256
        ),
        "east_threshold_receipt_sha256": contract.east_threshold_receipt_sha256,
        "target_scaler_sha256": contract.target_scaler_sha256,
        "raw8_scaler_sha256": contract.raw8_scaler_sha256,
        "physics_scales_sha256": contract.physics_scales_sha256,
        "reconstruction_constants_sha256": (
            contract.reconstruction_constants_sha256
        ),
    }
    if contract.jtext_test_roster_sha256 is not None:
        lineage_fields["jtext_test_roster_sha256"] = (
            contract.jtext_test_roster_sha256
        )
    if contract.east_test_roster_sha256 is not None:
        lineage_fields["east_test_roster_sha256"] = contract.east_test_roster_sha256
    for name, expected in lineage_fields.items():
        if payload.get(name) != expected:
            raise RuntimeError(f"selected parent {name} mismatch")
    try:
        threshold = float(payload.get("east_threshold"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("selected parent EAST threshold is missing") from exc
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise RuntimeError("selected parent EAST threshold is invalid")

    try:
        source_receipt = validate_threshold_receipt(
            payload.get("source_val_threshold_receipt", {}),
            expected_scope="J-TEXT VAL160",
        )
        east_receipt = validate_threshold_receipt(
            payload.get("east_threshold_receipt", {}),
            expected_scope="EAST support110",
            expected_threshold=threshold,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"selected parent threshold receipt mismatch: {exc}") from exc
    if (
        source_receipt["receipt_sha256"]
        != contract.source_val_threshold_receipt_sha256
        or east_receipt["receipt_sha256"] != contract.east_threshold_receipt_sha256
        or east_receipt["roster_sha256"] != contract.east_support_roster_sha256
        or east_receipt["data_manifest_sha256"]
        != contract.east_support_data_manifest_sha256
        or east_receipt["outcome_manifest_sha256"]
        != contract.east_support_outcome_manifest_sha256
    ):
        raise RuntimeError("selected parent threshold receipt lineage mismatch")

    preprocessing_names = {
        "target_scaler": "target scaler",
        "raw8_scaler": "raw8 scaler",
        "physics_scales": "physics scales",
        "reconstruction_constants": "reconstruction constants",
    }
    for name, message in preprocessing_names.items():
        value = payload.get(name)
        expected = getattr(contract, f"{name}_sha256")
        if (
            not isinstance(value, Mapping)
            or sha256_json(value) != expected
            or payload.get(f"{name}_sha256") != expected
        ):
            raise RuntimeError(f"selected parent {message} hash mismatch")

    encoder = _require_parent_state(payload, "encoder")
    head = _require_parent_state(payload, "head")
    shared = _require_parent_state(payload, "shared")
    private = _require_parent_state(payload, "private")
    actual_hashes = {
        "target_encoder_sha256": state_dict_sha256(encoder),
        "target_head_sha256": state_dict_sha256(head),
        "target_base_sha256": state_bundle_sha256(encoder, head),
        "shared_parent_sha256": state_dict_sha256(shared),
        "private_parent_sha256": state_dict_sha256(private),
    }
    messages = {
        "target_encoder_sha256": "target encoder hash",
        "target_head_sha256": "target head hash",
        "target_base_sha256": "target base hash",
        "shared_parent_sha256": "shared parent hash",
        "private_parent_sha256": "private parent hash",
    }
    for name, actual in actual_hashes.items():
        if payload.get(name) != actual:
            raise RuntimeError(f"selected parent {messages[name]} mismatch")
    actual_fingerprint = canonical_parent_fingerprint(payload)
    if payload.get("canonical_parent_fingerprint_sha256") != actual_fingerprint:
        raise RuntimeError("selected parent canonical fingerprint mismatch")
    return {
        "schema_version": SELECTED_PARENT_SCHEMA,
        "threshold": threshold,
        "expected_base_sha256": actual_hashes["target_base_sha256"],
        "expected_encoder_sha256": actual_hashes["target_encoder_sha256"],
        "expected_head_sha256": actual_hashes["target_head_sha256"],
        "expected_shared_sha256": actual_hashes["shared_parent_sha256"],
        "expected_private_sha256": actual_hashes["private_parent_sha256"],
        "east_threshold_receipt_sha256": east_receipt["receipt_sha256"],
        "claim_status": contract.claim_status,
    }
