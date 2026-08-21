"""Fail-closed preregistration seals for a future chronological EAST stream."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProspectiveContractError(RuntimeError):
    """Raised when a prospective confirmation contract is not sealable."""


REQUIRED_ARMS = {
    "static",
    "active",
    "no_label",
    "causal_random_query",
    "label_frontload",
    "east_only",
}
REQUIRED_SEEDS = [131, 133, 137]
REQUIRED_LABEL_SOURCE = ["/meta/IsDisrupt", "/meta/DownTime"]
REQUIRED_PRIOR_EAST_SOURCES = {
    "config/splits/east_paper131_split_all_in.json": {
        "/train/disruption_shots",
        "/train/non_disruption_shots",
        "/test/disruption_shots",
        "/test/non_disruption_shots",
    },
    "config/splits/east_hdf5_authoritative_balanced360_v1.json": {
        "/test/disruption_shots",
        "/test/non_disruption_shots",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProspectiveContractError(f"git inspection failed: {exc}") from exc


def _read_shot_only_roster(path: Path) -> list[int]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProspectiveContractError(f"cannot read roster: {exc}") from exc
    if not lines:
        raise ProspectiveContractError("future roster must not be empty")
    shots: list[int] = []
    for line_number, raw in enumerate(lines, start=1):
        value = raw.strip()
        if not value or not value.isascii() or not value.isdigit():
            raise ProspectiveContractError(
                "future roster must contain shot IDs only, one integer per line; "
                f"invalid line {line_number}"
            )
        shot = int(value)
        if shot <= 0:
            raise ProspectiveContractError("future shot IDs must be positive")
        shots.append(shot)
    if any(current >= following for current, following in zip(shots, shots[1:])):
        raise ProspectiveContractError(
            "future shot IDs must be unique and strictly increasing"
        )
    return shots


def _read_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveContractError(f"cannot read preregistration spec: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProspectiveContractError("preregistration spec must be a JSON object")
    expected = {
        "schema_version": "prospective-east-confirmation-v1",
        "claim_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
        "seeds": REQUIRED_SEEDS,
        "label_source": REQUIRED_LABEL_SOURCE,
        "primary_metric": "TPR_at_lead_ge_30ms_minus_FPR",
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise ProspectiveContractError(f"invalid or missing frozen field: {key}")
    if set(payload.get("arms", [])) != REQUIRED_ARMS:
        raise ProspectiveContractError(
            "prospective arms must be exactly static, active, no_label, "
            "causal_random_query, label_frontload and east_only"
        )
    gates = payload.get("admission_gates")
    if not isinstance(gates, dict) or gates != {
        "tpr_at_lead_ge_30ms_min": 0.80,
        "fpr_max": 0.20,
        "median_lead_ms_min": 30.0,
        "worst_block_tpr_at_lead_ge_30ms_min": 0.80,
        "worst_block_fpr_max": 0.20,
        "absolute_shot_id_spearman_max": 0.20,
    }:
        raise ProspectiveContractError("prospective admission gates are incomplete")
    invariants = {
        "over_3000ms_counts_as_tpr": True,
        "retuning_prohibited": True,
        "complete_stream_reporting": True,
        "prediction_order": (
            "predict-seal-query-selected-reveal-later-shot-update"
        ),
    }
    for key, required in invariants.items():
        if payload.get(key) != required:
            raise ProspectiveContractError(f"invalid prospective invariant: {key}")
    artifacts = payload.get("frozen_artifacts")
    if not isinstance(artifacts, list) or not artifacts or not all(
        isinstance(item, str) and item for item in artifacts
    ):
        raise ProspectiveContractError("frozen_artifacts must be a non-empty path list")
    if payload.get("target_validation_completed") is not True:
        raise ProspectiveContractError("target validation must be completed before seal")
    for role in (
        "target_validation_artifact",
        "random_schedule_artifact",
        "execution_plan_artifact",
    ):
        artifact = payload.get(role)
        if not isinstance(artifact, str) or artifact not in artifacts:
            raise ProspectiveContractError(
                f"{role} must be included in frozen_artifacts"
            )
    excluded = payload.get("excluded_shots", [])
    if not isinstance(excluded, list) or not all(
        isinstance(item, int) and item > 0 for item in excluded
    ):
        raise ProspectiveContractError("excluded_shots must be a positive integer list")
    if len(excluded) != len(set(excluded)):
        raise ProspectiveContractError("excluded_shots contains duplicates")
    sources = payload.get("excluded_shot_sources", [])
    if not isinstance(sources, list):
        raise ProspectiveContractError("excluded_shot_sources must be a list")
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ProspectiveContractError("invalid excluded_shot_sources path")
        pointers = source.get("json_pointers")
        if not isinstance(pointers, list) or not pointers or not all(
            isinstance(pointer, str) and pointer.startswith("/")
            for pointer in pointers
        ):
            raise ProspectiveContractError(
                "excluded_shot_sources requires JSON pointer lists"
            )
    registered_sources = {
        source["path"]: set(source["json_pointers"])
        for source in sources
    }
    if any(
        registered_sources.get(path) != pointers
        for path, pointers in REQUIRED_PRIOR_EAST_SOURCES.items()
    ):
        raise ProspectiveContractError(
            "excluded_shot_sources must bind every canonical prior EAST roster"
        )
    if not excluded and not sources:
        raise ProspectiveContractError(
            "at least one prior-shot exclusion source is required"
        )
    return payload


def _repository_file(repo_root: Path, raw: str, role: str) -> tuple[Path, str]:
    root = repo_root.resolve()
    relative = Path(raw)
    if relative.is_absolute():
        raise ProspectiveContractError(f"{role} must be relative: {raw}")
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveContractError(f"{role} escapes repository: {raw}") from exc
    if not path.is_file():
        raise ProspectiveContractError(f"{role} missing: {normalized}")
    return path, normalized


def _json_pointer(document: Any, pointer: str) -> Any:
    value = document
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ProspectiveContractError(f"JSON pointer not found: {pointer}")
    return value


def _excluded_shots(
    spec: dict[str, Any], repo_root: Path,
) -> tuple[set[int], dict[str, str]]:
    shots = set(spec.get("excluded_shots", []))
    hashes: dict[str, str] = {}
    for source in spec.get("excluded_shot_sources", []):
        path, normalized = _repository_file(
            repo_root, source["path"], "excluded shot source",
        )
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ProspectiveContractError(
                f"cannot read excluded shot source {normalized}: {exc}"
            ) from exc
        for pointer in source["json_pointers"]:
            values = _json_pointer(document, pointer)
            if not isinstance(values, list) or not all(
                isinstance(item, int) and item > 0 for item in values
            ):
                raise ProspectiveContractError(
                    f"excluded shot pointer is not a positive integer list: {pointer}"
                )
            shots.update(values)
        hashes[normalized] = _sha256_file(path)
    return shots, dict(sorted(hashes.items()))


def _artifact_hashes(spec: dict[str, Any], repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in spec["frozen_artifacts"]:
        path, normalized = _repository_file(repo_root, raw, "frozen artifact")
        hashes[normalized] = _sha256_file(path)
    if len(hashes) != len(spec["frozen_artifacts"]):
        raise ProspectiveContractError("frozen_artifacts contains duplicate paths")
    return dict(sorted(hashes.items()))


def _validate_repository(spec: dict[str, Any], repo_root: Path) -> tuple[str, str]:
    sha = _git(repo_root, "rev-parse", "HEAD")
    if spec.get("frozen_git_sha") != sha:
        raise ProspectiveContractError(
            f"frozen_git_sha mismatch: spec={spec.get('frozen_git_sha')} actual={sha}"
        )
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProspectiveContractError("prospective seal requires a clean git worktree")
    return sha, status


def create_prospective_seal(
    *, spec_path: Path, validation_roster_path: Path, roster_path: Path,
    output_path: Path, repo_root: Path,
) -> dict[str, Any]:
    """Create an exclusive, outcome-free hash seal before future evaluation."""
    if output_path.exists():
        raise FileExistsError(output_path)
    spec = _read_spec(spec_path)
    validation_shots = _read_shot_only_roster(validation_roster_path)
    shots = _read_shot_only_roster(roster_path)
    excluded, exclusion_hashes = _excluded_shots(spec, repo_root)
    prior_overlap = sorted(
        (set(validation_shots) | set(shots)).intersection(excluded)
    )
    if prior_overlap:
        raise ProspectiveContractError(
            f"future roster overlaps excluded shots: {prior_overlap[:10]}"
        )
    validation_overlap = sorted(set(validation_shots).intersection(shots))
    if validation_overlap:
        raise ProspectiveContractError(
            "target validation and confirmation rosters overlap: "
            f"{validation_overlap[:10]}"
        )
    sha, _ = _validate_repository(spec, repo_root)
    artifact_hashes = _artifact_hashes(spec, repo_root)
    payload: dict[str, Any] = {
        "schema_version": "prospective-east-seal-v1",
        "status": "SEALED_BEFORE_OUTCOME_ACCESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
        "git_sha": sha,
        "git_worktree_clean_at_seal": True,
        "spec_sha256": _sha256_file(spec_path),
        "roster": {
            "count": len(shots),
            "first_shot": shots[0],
            "last_shot": shots[-1],
            "sha256": _sha256_file(roster_path),
        },
        "target_validation_roster": {
            "count": len(validation_shots),
            "first_shot": validation_shots[0],
            "last_shot": validation_shots[-1],
            "sha256": _sha256_file(validation_roster_path),
        },
        "outcome_fields_present": False,
        "excluded_prior_shot_count": len(excluded),
        "exclusion_source_sha256": exclusion_hashes,
        "frozen_artifact_sha256": artifact_hashes,
        "contract": spec,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def verify_prospective_seal(
    *, seal_path: Path, spec_path: Path, validation_roster_path: Path,
    roster_path: Path, repo_root: Path,
) -> dict[str, Any]:
    """Verify that a seal still binds the exact code, roster and artifacts."""
    errors: list[str] = []
    try:
        seal = json.loads(seal_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"cannot read seal: {exc}"]}
    try:
        spec = _read_spec(spec_path)
        validation_shots = _read_shot_only_roster(validation_roster_path)
        shots = _read_shot_only_roster(roster_path)
        excluded, exclusion_hashes = _excluded_shots(spec, repo_root)
        artifact_hashes = _artifact_hashes(spec, repo_root)
        current_sha = _git(repo_root, "rev-parse", "HEAD")
        current_status = _git(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all",
        )
    except ProspectiveContractError as exc:
        return {"passed": False, "errors": [str(exc)]}

    expected_pairs = {
        "git SHA": (seal.get("git_sha"), current_sha),
        "spec": (seal.get("spec_sha256"), _sha256_file(spec_path)),
        "roster": (seal.get("roster", {}).get("sha256"), _sha256_file(roster_path)),
        "roster count": (seal.get("roster", {}).get("count"), len(shots)),
        "target validation roster": (
            seal.get("target_validation_roster", {}).get("sha256"),
            _sha256_file(validation_roster_path),
        ),
        "target validation roster count": (
            seal.get("target_validation_roster", {}).get("count"),
            len(validation_shots),
        ),
        "contract": (seal.get("contract"), spec),
        "excluded prior shot count": (
            seal.get("excluded_prior_shot_count"), len(excluded),
        ),
        "exclusion source hashes": (
            seal.get("exclusion_source_sha256", {}), exclusion_hashes,
        ),
    }
    for name, (sealed, current) in expected_pairs.items():
        if sealed != current:
            errors.append(f"{name} differs from prospective seal")
    sealed_artifacts = seal.get("frozen_artifact_sha256", {})
    for path, digest in artifact_hashes.items():
        if sealed_artifacts.get(path) != digest:
            errors.append(f"frozen artifact differs from prospective seal: {path}")
    if set(sealed_artifacts) != set(artifact_hashes):
        errors.append("frozen artifact path set differs from prospective seal")
    if current_status:
        errors.append("git worktree is not clean")
    if seal.get("schema_version") != "prospective-east-seal-v1":
        errors.append("seal schema version is invalid")
    if seal.get("status") != "SEALED_BEFORE_OUTCOME_ACCESS":
        errors.append("seal status is invalid")
    if seal.get("claim_status") != "PROSPECTIVE_PREREGISTRATION_ONLY":
        errors.append("seal claim status is invalid")
    if seal.get("git_worktree_clean_at_seal") is not True:
        errors.append("seal does not attest a clean worktree")
    if seal.get("outcome_fields_present") is not False:
        errors.append("seal does not attest an outcome-free roster")
    return {"passed": not errors, "errors": errors}


__all__ = [
    "ProspectiveContractError",
    "create_prospective_seal",
    "verify_prospective_seal",
]
