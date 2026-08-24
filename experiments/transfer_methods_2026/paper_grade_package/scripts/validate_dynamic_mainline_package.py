#!/usr/bin/env python3
"""Fail-closed validator for the hash-bound dynamic-mainline package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


SCHEMA = "dynamic-mainline-package-manifest-v1"
REQUIRED_ROLES = (
    "evidence_report",
    "figure_manifest",
    "report_pdf",
    "effective_config",
    "code_manifest",
)
PDF_INPUT_ROLES = (
    "evidence_report",
    "figure_manifest",
    "effective_config",
    "code_manifest",
)
LEGACY_V3C_POINTS = (
    (0.8515, 0.2121),
    (0.8407, 0.1667),
    (0.8481, 0.1519),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MACHINE_PATH_RE = re.compile(r"(?:^|[\s:=\"'])/(?:home|media)(?:/|$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, failures: list[str], role: str) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        failures.append(f"{role} is missing or unreadable JSON: {error}")
        return None


def _resolve_repo_path(
    repo_root: Path, value: Any, failures: list[str], role: str,
) -> Path | None:
    if not isinstance(value, str) or not value:
        failures.append(f"{role} path is missing")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        failures.append(f"{role} path is not portable: {value!r}")
        return None
    candidate = (repo_root / relative).resolve()
    if candidate != repo_root and repo_root not in candidate.parents:
        failures.append(f"{role} path escapes repository: {value!r}")
        return None
    return candidate


def _machine_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_machine_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_machine_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str) and MACHINE_PATH_RE.search(value):
        found.append(f"{path}={value!r}")
    return found


def _metric(value: dict[str, Any], token: str) -> float | None:
    for key, candidate in value.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if normalized.endswith(token) and isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def _legacy_metric_paths(
    value: Any, path: tuple[str, ...] = (), archived: bool = False,
) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        local_archived = archived or any(
            marker in part.lower()
            for part in path
            for marker in ("legacy", "archive", "historical")
        )
        status = str(value.get("claim_status", value.get("status", ""))).lower()
        local_archived = local_archived or any(
            marker in status for marker in ("legacy", "archive", "historical")
        )
        tpr, fpr = _metric(value, "tpr"), _metric(value, "fpr")
        if not local_archived and tpr is not None and fpr is not None:
            for old_tpr, old_fpr in LEGACY_V3C_POINTS:
                if abs(tpr - old_tpr) <= 5e-5 and abs(fpr - old_fpr) <= 5e-5:
                    found.append("$." + ".".join(path) if path else "$")
                    break
        for key, child in value.items():
            found.extend(
                _legacy_metric_paths(child, (*path, str(key)), local_archived)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _legacy_metric_paths(child, (*path, f"[{index}]"), archived)
            )
    return found


def _pdf_info(path: Path, failures: list[str]) -> dict[str, Any]:
    info = {"path": str(path), "pages": 0, "bytes": 0, "parseable": False}
    if not path.is_file():
        failures.append(f"report PDF is missing: {path}")
        return info
    info["bytes"] = path.stat().st_size
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        failures.append("report PDF cannot be verified: pdfinfo is unavailable")
        return info
    result = subprocess.run(
        [pdfinfo, str(path)], capture_output=True, text=True, check=False,
    )
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
    pages = int(match.group(1)) if match else 0
    info.update({"pages": pages, "parseable": result.returncode == 0 and pages > 0})
    if not info["parseable"]:
        detail = (result.stderr or result.stdout).strip().splitlines()
        failures.append(
            "report PDF is not parseable or has zero pages"
            + (f": {detail[0]}" if detail else "")
        )
    return info


def validate_package(*, repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate package closure; every uncertainty is returned as a failure."""
    repo_root = Path(repo_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    failures: list[str] = []
    result: dict[str, Any] = {
        "schema_version": "dynamic-mainline-package-validation-v1",
        "passed": False,
        "manifest": str(manifest_path),
        "pdf": {"pages": 0, "bytes": 0, "parseable": False},
        "failures": failures,
    }
    manifest = _load_json(manifest_path, failures, "package manifest")
    if not isinstance(manifest, dict):
        return result
    if manifest.get("schema_version") != SCHEMA:
        failures.append(f"unexpected package manifest schema: {manifest.get('schema_version')!r}")

    for location in _machine_paths(manifest):
        failures.append(f"package manifest contains machine absolute path: {location}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("package manifest artifacts map is missing")
        return result

    paths: dict[str, Path] = {}
    payloads: dict[str, Any] = {}
    actual_hashes: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        row = artifacts.get(role)
        if not isinstance(row, dict):
            failures.append(f"required artifact is missing: {role}")
            continue
        candidate = _resolve_repo_path(repo_root, row.get("path"), failures, role)
        expected = row.get("sha256")
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            failures.append(f"{role} has no valid SHA-256")
        if candidate is None:
            continue
        paths[role] = candidate
        if not candidate.is_file():
            failures.append(f"required artifact is missing: {role}: {candidate}")
            continue
        actual = sha256_file(candidate)
        actual_hashes[role] = actual
        if actual != expected:
            failures.append(f"{role} hash mismatch: expected {expected}, got {actual}")
        if role != "report_pdf":
            payload = _load_json(candidate, failures, role)
            payloads[role] = payload
            if payload is not None:
                for location in _machine_paths(payload):
                    failures.append(
                        f"{role} contains machine absolute path: {location}"
                    )

    pdf_row = artifacts.get("report_pdf", {})
    bindings = pdf_row.get("input_sha256") if isinstance(pdf_row, dict) else None
    if not isinstance(bindings, dict):
        failures.append("report PDF has no hash-bound input map")
    else:
        for role in PDF_INPUT_ROLES:
            expected = actual_hashes.get(role)
            if expected is None or bindings.get(role) != expected:
                failures.append(
                    f"stale PDF binding for {role}: "
                    f"bound {bindings.get(role)!r}, current {expected!r}"
                )

    code_manifest = payloads.get("code_manifest")
    tracked = code_manifest.get("tracked_files") if isinstance(code_manifest, dict) else None
    code_rows = manifest.get("code_files")
    if not isinstance(tracked, list) or not all(isinstance(item, str) for item in tracked):
        failures.append("code manifest has no valid tracked_files inventory")
    if not isinstance(code_rows, list):
        failures.append("package manifest has no code_files hash closure")
    elif isinstance(tracked, list):
        rows = {
            row.get("path"): row
            for row in code_rows
            if isinstance(row, dict) and isinstance(row.get("path"), str)
        }
        if set(rows) != set(tracked):
            failures.append("code_files hash closure does not match tracked_files")
        for relative in tracked:
            row = rows.get(relative)
            if row is None:
                continue
            candidate = _resolve_repo_path(
                repo_root, relative, failures, f"code file {relative}",
            )
            if candidate is None or not candidate.is_file():
                failures.append(f"tracked code file is missing: {relative}")
                continue
            expected = row.get("sha256")
            actual = sha256_file(candidate)
            if SHA256_RE.fullmatch(str(expected)) is None or actual != expected:
                failures.append(f"tracked code file hash mismatch: {relative}")

    for role in ("evidence_report", "figure_manifest"):
        payload = payloads.get(role)
        if payload is not None:
            for location in _legacy_metric_paths(payload):
                failures.append(
                    f"{role} mixes a legacy v3c metric into current results: {location}"
                )

    if "report_pdf" in paths:
        result["pdf"] = _pdf_info(paths["report_pdf"], failures)
    result["passed"] = not failures
    return result


def main(argv: list[str] | None = None) -> int:
    default_repo = Path(__file__).resolve().parents[4]
    default_manifest = (
        Path(__file__).resolve().parents[1]
        / "technical_report_dynamic/build_manifest.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_package(
        repo_root=args.repo_root, manifest_path=args.manifest,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
