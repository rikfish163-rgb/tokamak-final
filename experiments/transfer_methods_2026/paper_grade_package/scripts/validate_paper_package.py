#!/usr/bin/env python3
"""Fail-closed validation for the paper-grade evidence and manuscript bundle."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
PACKAGE = ROOT / "experiments/transfer_methods_2026/paper_grade_package"
EVIDENCE = PACKAGE / "evidence/three_stage_results.json"
FIGURE_MANIFEST = PACKAGE / "figures/figure_manifest.json"
TABLE_MANIFEST = PACKAGE / "manuscript/tables/table_manifest.json"
MANUSCRIPT = PACKAGE / "manuscript"
REPORT = PACKAGE / "TECHNICAL_REPORT_zh.md"
EQUAL_BUDGET = (
    PACKAGE
    / "evidence/equal_label_budget_frontload_v2/common_mask_report/summary.json"
)
CAUSAL_RANDOM = (
    PACKAGE / "evidence/causal_random_query_online_v1/comparison.json"
)
EQUAL_COMPUTE = (
    PACKAGE / "evidence/equal_compute_selector_control_v1/report.json"
)
EAST_TARGET_ONLY = PACKAGE / "evidence/east_target_only_v1/comparison.json"
HDF5_CONTENT_MANIFEST = PACKAGE / "evidence/required_hdf5_content_manifest.json"
Q30_MANIFEST = PACKAGE / "evidence/fixed_query_low_budget30_seed131.json"
PROSPECTIVE_TEMPLATE = ROOT / "configs/prospective_east_confirmation.template.json"
EQUAL_COMPUTE_CODE_MANIFEST = (
    PACKAGE
    / "evidence/equal_compute_selector_control_v1/code_environment_manifest.json"
)
OUTPUT = PACKAGE / "validation_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(path: Path) -> Path:
    if path.is_file():
        return path
    compressed = Path(f"{path}.gz")
    if compressed.is_file():
        return compressed
    return path


def sha256_artifact(path: Path) -> str:
    resolved = resolve_artifact(path)
    if resolved.suffix != ".gz":
        return sha256_file(resolved)
    digest = hashlib.sha256()
    with gzip.open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, name: str, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0)


def pdf_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "pages": 0, "bytes": 0, "text_chars": 0}
    pages = 0
    text_chars = 0
    tool = shutil.which("pdfinfo")
    if tool:
        result = subprocess.run(
            [tool, str(path)], capture_output=True, text=True, check=False,
        )
        match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
        pages = int(match.group(1)) if match else 0
    text_tool = shutil.which("pdftotext")
    if text_tool:
        result = subprocess.run(
            [text_tool, str(path), "-"], capture_output=True, text=True, check=False,
        )
        text_chars = len(result.stdout.strip()) if result.returncode == 0 else 0
    return {
        "exists": True,
        "pages": pages,
        "bytes": path.stat().st_size,
        "text_chars": text_chars,
    }


def validate_manifest(path: Path, output_base: Path, key: str) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [f"missing manifest: {path}"]
    payload = json.loads(path.read_text())
    failures: list[str] = []
    for name, expected in payload.get("inputs", {}).items():
        relative = Path(name)
        candidate = (ROOT / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or ROOT not in candidate.parents:
            failures.append(f"invalid manifest input path: {name}")
        elif not candidate.is_file():
            failures.append(f"missing manifest input: {candidate}")
        elif sha256_file(candidate) != expected:
            failures.append(f"manifest input hash mismatch: {candidate}")
    if key == "figures":
        for figure in payload["figures"].values():
            for name, expected in figure["outputs"].items():
                candidate = output_base / name
                if not candidate.is_file():
                    failures.append(f"missing {candidate}")
                elif sha256_file(candidate) != expected:
                    failures.append(f"hash mismatch {candidate}")
    else:
        for name, expected in payload["tables"].items():
            candidate = output_base / name
            if not candidate.is_file():
                failures.append(f"missing {candidate}")
            elif sha256_file(candidate) != expected:
                failures.append(f"hash mismatch {candidate}")
    return not failures, failures


def validate_compact_artifact_manifest(path: Path) -> tuple[bool, list[str]]:
    """Verify every archived artifact, including decompressed ledger hashes."""
    if not path.is_file():
        return False, [f"missing manifest: {path}"]
    payload = json.loads(path.read_text())
    root = path.parent.resolve()
    failures: list[str] = []
    seen: set[str] = set()
    for row in payload.get("artifacts", []):
        relative = str(row.get("path", ""))
        candidate = (root / relative).resolve()
        if (
            not relative or relative in seen or root not in candidate.parents
            or candidate == root
        ):
            failures.append(f"invalid or duplicate artifact path: {relative!r}")
            continue
        seen.add(relative)
        if not candidate.is_file():
            failures.append(f"missing {candidate}")
            continue
        if candidate.stat().st_size != int(row.get("bytes", -1)):
            failures.append(f"size mismatch {candidate}")
        if sha256_file(candidate) != row.get("sha256"):
            failures.append(f"hash mismatch {candidate}")
        if "decompressed_sha256" in row:
            if candidate.suffix != ".gz":
                failures.append(f"decompressed hash declared for non-gzip {candidate}")
                continue
            digest = hashlib.sha256()
            decompressed_bytes = 0
            with gzip.open(candidate, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    decompressed_bytes += len(chunk)
            if digest.hexdigest() != row["decompressed_sha256"]:
                failures.append(f"decompressed hash mismatch {candidate}")
            if decompressed_bytes != int(row.get("decompressed_bytes", -1)):
                failures.append(f"decompressed size mismatch {candidate}")
    if not seen:
        failures.append("artifact manifest contains no artifacts")
    return not failures, failures


def validate_release_code_manifest(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [f"missing release code manifest: {path}"]
    payload = json.loads(path.read_text())
    failures: list[str] = []
    if payload.get("schema_version") != "equal-compute-release-code-manifest-v2":
        failures.append("unexpected release code manifest schema")
    if payload.get("provenance_status") != (
        "POST_RUN_RELEASE_SNAPSHOT_NOT_EXECUTION_TIME_SEAL"
    ):
        failures.append("release code provenance boundary is missing")
    rows = payload.get("source_files", [])
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        candidate = (ROOT / relative).resolve()
        if (
            not relative or relative in seen or ROOT not in candidate.parents
            or candidate == ROOT
        ):
            failures.append(f"invalid release source path: {relative!r}")
            continue
        seen.add(relative)
        if not candidate.is_file():
            failures.append(f"missing release source: {relative}")
            continue
        if candidate.stat().st_size != int(row.get("bytes", -1)):
            failures.append(f"release source size mismatch: {relative}")
        if sha256_file(candidate) != row.get("sha256"):
            failures.append(f"release source hash mismatch: {relative}")
    tracked_result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
        check=True, capture_output=True,
    )
    tracked = {item for item in tracked_result.stdout.decode().split("\0") if item}
    expected = {
        relative for relative in tracked if (
            Path(relative).suffix in {".py", ".pyi", ".sh"}
            or relative in {
                "pyproject.toml", "uv.lock", ".python-version",
                ".github/workflows/ci.yml",
            }
            or relative.startswith(("config/", "configs/"))
        )
    }
    if seen != expected:
        for relative in sorted(expected - seen):
            failures.append(f"release code manifest omits tracked source: {relative}")
        for relative in sorted(seen - expected):
            failures.append(f"release code manifest has out-of-scope source: {relative}")
    return not failures, failures


def validate_git_tracked(paths: list[Path]) -> tuple[bool, list[str]]:
    """Require release-critical artifacts to be present in the Git index."""
    if not (ROOT / ".git").exists():
        return True, ["git metadata unavailable; tracking gate not applicable"]
    relative = [str(path.resolve().relative_to(ROOT)) for path in paths]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *relative], cwd=ROOT,
        check=True, capture_output=True,
    )
    tracked = {value for value in result.stdout.decode().split("\0") if value}
    missing = sorted(set(relative) - tracked)
    return not missing, [f"untracked release artifact: {value}" for value in missing]


def validate_hdf5_content_manifest(path: Path) -> tuple[bool, list[str]]:
    """Validate the portable 2,203-file content-hash inventory itself."""
    if not path.is_file():
        return False, [f"missing manifest: {path}"]
    payload = json.loads(path.read_text())
    manifest = payload.get("content_manifest", {})
    rows = manifest.get("files", [])
    failures: list[str] = []
    if payload.get("passed") is not True or payload.get("full") is not True:
        failures.append("data audit is not a passed full audit")
    if payload.get("data_root") != "<external-data-root>":
        failures.append("data audit is not portable")
    if payload.get("expected_shots") != {"EAST": 470, "J-TEXT": 1733}:
        failures.append("unexpected required-shot counts")
    if payload.get("hdf5_files_opened") != 2203:
        failures.append("not all required HDF5 files were opened")
    if manifest.get("algorithm") != "sha256" or manifest.get("file_count") != 2203:
        failures.append("unexpected content-manifest algorithm or file count")
    if len(rows) != 2203:
        failures.append("content-manifest row count mismatch")
    aggregate = hashlib.sha256()
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        if (
            not relative or relative in seen or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            failures.append(f"invalid or duplicate HDF5 relative path: {relative!r}")
            continue
        seen.add(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            failures.append(f"invalid HDF5 SHA256: {relative}")
        aggregate.update(
            f"{relative}\0{row.get('bytes')}\0{row.get('sha256')}\n".encode()
        )
    if aggregate.hexdigest() != manifest.get("aggregate_sha256"):
        failures.append("HDF5 content aggregate hash mismatch")
    return not failures, failures


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = resolve_artifact(path)
    opener = gzip.open if resolved.suffix == ".gz" else open
    with opener(resolved, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _contains_forbidden_outcome_key(value: Any) -> bool:
    forbidden = {
        "is_disrupt", "down_time_s", "revealed_is_disrupt",
        "revealed_down_time_s", "outcome", "outcome_label",
    }
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_forbidden_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_outcome_key(item) for item in value)
    return False


def validate_q30_manifest(path: Path, static_predictions: Path) -> tuple[bool, list[str]]:
    """Validate the recoverable Q30 identity evidence without inventing its lost ledger."""
    if not path.is_file():
        return False, [f"missing manifest: {path}"]
    payload = json.loads(path.read_text())
    failures: list[str] = []
    positions = payload.get("positions_zero_based", [])
    shots = payload.get("shots", [])
    source = payload.get("source", {})
    if payload.get("kind") != "sealed_chronological_policy_queries":
        failures.append("unexpected Q30 manifest kind")
    if payload.get("label_oracle_used") is not False:
        failures.append("Q30 selection is not declared label-free")
    if payload.get("source_row_count") != 360:
        failures.append("Q30 source row count is not 360")
    if len(positions) != 30 or len(shots) != 30:
        failures.append("Q30 manifest does not contain exactly 30 identities")
    if positions != sorted(set(positions)) or any(
        not isinstance(value, int) or value < 0 or value >= 360
        for value in positions
    ):
        failures.append("Q30 positions are not unique, ordered, and in range")
    if len(set(shots)) != len(shots):
        failures.append("Q30 shot identities are not unique")
    for key in ("query_decisions_sha256", "static_predictions_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))):
            failures.append(f"invalid Q30 source hash: {key}")
    if not static_predictions.is_file():
        failures.append(f"missing Q30 static roster: {static_predictions}")
    else:
        rows = read_jsonl(static_predictions)
        if len(rows) != 360:
            failures.append("Q30 static roster does not contain 360 rows")
        elif any(rows[position].get("shot") != shot for position, shot in zip(positions, shots)):
            failures.append("Q30 shot identities do not match their sealed stream positions")
        if sha256_file(static_predictions) != source.get("static_predictions_sha256"):
            failures.append("Q30 static prediction hash mismatch")
    note = str(source.get("recovery_note", "")).lower()
    if not all(token in note for token in ("reboot", "tmpfs", "restored")):
        failures.append("Q30 lost-ledger recovery boundary is absent")
    return not failures, failures


def validate_primary_causal_ledgers(seed_dir: Path) -> tuple[bool, list[str]]:
    """Check the archived predict/query/reveal/finalize relations fail closed."""
    failures: list[str] = []
    required = {
        name: seed_dir / name for name in (
            "predictions.jsonl", "query_decisions.jsonl", "reveals.jsonl",
            "updates.jsonl", "static_predictions.jsonl",
        )
    }
    if any(not resolve_artifact(path).is_file() for path in required.values()):
        return False, [
            f"missing {resolve_artifact(path)}" for path in required.values()
            if not resolve_artifact(path).is_file()
        ]
    predictions = read_jsonl(required["predictions.jsonl"])
    queries = read_jsonl(required["query_decisions.jsonl"])
    reveals = read_jsonl(required["reveals.jsonl"])
    updates = read_jsonl(required["updates.jsonl"])
    static = read_jsonl(required["static_predictions.jsonl"])
    if not all(len(rows) == 360 for rows in (predictions, queries, static)):
        failures.append("primary prediction/query/static ledgers are not 360 rows")
        return False, failures
    canonical = [(row.get("stream_index"), row.get("shot")) for row in predictions]
    expected = [(index, shot) for index, (_, shot) in enumerate(canonical)]
    if canonical != expected or len({shot for _, shot in canonical}) != 360:
        failures.append("primary prediction stream is not unique and contiguous")
    for name, rows in (("query", queries), ("static", static)):
        identity = [(row.get("stream_index"), row.get("shot")) for row in rows]
        if identity != canonical:
            failures.append(f"{name} ledger identity/order differs from predictions")
    if any(_contains_forbidden_outcome_key(row) for row in predictions + queries):
        failures.append("prediction/query ledger contains an outcome field")
    selected = {
        (row.get("stream_index"), row.get("shot"))
        for row in queries if row.get("selected") is True
    }
    prediction_queries = {
        (row.get("stream_index"), row.get("shot"))
        for row in predictions if row.get("queried") is True
    }
    revealed = {(row.get("stream_index"), row.get("shot")) for row in reveals}
    if selected != prediction_queries or revealed != selected:
        failures.append("reveals, selected queries, and prediction query receipts differ")
    if len(revealed) != len(reveals):
        failures.append("reveal ledger contains duplicate shot/position rows")
    revealed_shots = {shot for _, shot in revealed}
    revealed_final = [row for row in updates if row.get("event") == "unified_revealed_finalize"]
    unqueried_final = [row for row in updates if row.get("event") == "unified_unqueried_finalize"]
    if {row.get("shot") for row in revealed_final} != revealed_shots:
        failures.append("revealed finalize receipts do not match revealed shots")
    if len(revealed_final) != len(revealed_shots):
        failures.append("revealed shots do not have exactly one finalize receipt")
    if ({row.get("shot") for row in unqueried_final} !=
            {shot for _, shot in canonical} - revealed_shots):
        failures.append("unqueried finalize receipts do not match skipped shots")
    if len(unqueried_final) != 360 - len(revealed_shots):
        failures.append("unqueried shots do not have exactly one finalize receipt")
    intra = [row for row in updates if row.get("event") == "unified_intra_update"]
    if not intra or any(row.get("strict_prediction_sealed_before_update") is not True for row in intra):
        failures.append("intra-shot update lacks a sealed-before-update receipt")
    return not failures, failures


def validate_text_artifact_portability(root: Path) -> tuple[bool, list[str]]:
    """Reject workstation-specific absolute paths from released text artifacts."""
    failures: list[str] = []
    suffixes = {
        ".json", ".jsonl", ".md", ".tex", ".csv", ".yaml", ".yml", ".log",
    }
    patterns = ("/media/", "/home/", "/dev/shm/", "C:\\Users\\")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(errors="replace")
        matched = [pattern for pattern in patterns if pattern in text]
        if matched:
            failures.append(f"workstation path in {path.relative_to(root)}: {matched}")
    return not failures, failures


def main() -> None:
    results: list[dict[str, Any]] = []
    evidence = json.loads(EVIDENCE.read_text())

    results.append(check(
        evidence["status"] == "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION",
        "evidence status preserves non-independent boundary",
        evidence["status"],
    ))
    contract = evidence["data_contract"]
    jtext_roles = contract["jtext"]["roles"]
    results.extend([
        check(
            jtext_roles["train"]["shots"] == 1353
            and jtext_roles["val"]["shots"] == 160
            and jtext_roles["test"]["shots"] == 220,
            "J-TEXT split counts", jtext_roles,
        ),
        check(
            (jtext_roles["train"]["disruptions"], jtext_roles["val"]["disruptions"],
             jtext_roles["test"]["disruptions"]) == (188, 80, 110),
            "J-TEXT HDF5-authoritative class counts", jtext_roles,
        ),
        check(not contract["jtext"]["strict_chronological_role_order"],
              "J-TEXT non-chronological split is disclosed", contract["jtext"]),
        check(not contract["jtext"]["mimics_operational_prevalence"],
              "J-TEXT balanced comparator boundary is disclosed", contract["jtext"]),
        check(contract["east_support"]["shots"] == 110
              and contract["east_support"]["disruptions"] == 15
              and contract["east_support"]["safe"] == 95,
              "EAST support is HDF5-authoritative 15D/95N",
              contract["east_support"]),
        check(contract["east_support"]["used_for_target_fitting"]
              and contract["east_support"]["used_for_target_threshold"]
              and not contract["east_support"]["independent_target_validation"],
              "support reuse and missing target validation are disclosed",
              contract["east_support"]),
        check(contract["east_balanced360"] == {"shots": 360, "disruptions": 180, "safe": 180},
              "EAST balanced360 counts"),
        check(contract["support_test_overlap"] == 0, "support/test zero overlap"),
        check(not contract["demonstrably_pristine_local_future_stream"],
              "no fabricated pristine future stream"),
    ])

    metric_contract = evidence["official_metric_contract"]
    results.extend([
        check(metric_contract["primary_metric"] == "lead-qualified operational TPR",
              "formal metric is lead-qualified operational TPR", metric_contract),
        check(metric_contract["raw_alarm_detection"] == "diagnostic_only",
              "raw alarm occurrence is diagnostic only", metric_contract),
        check(metric_contract["median_lead"]
              == "median over lead-qualified operational true positives only",
              "median lead uses operational TPs only", metric_contract),
        check(metric_contract["alarm_persistence"]
              == "first accepted causal supported crossing; no ten-point requirement",
              "alarm persistence contract", metric_contract),
    ])

    expected = {
        ("source_test220", "operational_tpr"): 0.8515151515151516,
        ("source_test220", "fpr"): 0.21212121212121213,
        ("source_test220", "auc"): 0.9219283746556474,
        ("source_test220", "operational_median_lead_ms"): 116.63208101704923,
        ("east_static360", "operational_tpr"): 0.8407407407407407,
        ("east_static360", "fpr"): 0.16666666666666666,
        ("east_static360", "operational_median_lead_ms"): 242.03333333333322,
        ("east_online360", "operational_tpr"): 0.8481481481481481,
        ("east_online360", "fpr"): 0.15185185185185185,
        ("east_online360", "operational_median_lead_ms"): 229.49999999999952,
    }
    lineage = evidence["primary_lineage"]
    official = evidence["official_results"]
    for (stage, metric), target in expected.items():
        actual = official[stage]["aggregate"][metric]["mean"]
        results.append(check(close(actual, target), f"metric {stage}.{metric}", actual))
    results.append(check(
        all(
            official["source_test220"]["seeds"][str(seed)]["lead_recomputation"]["status"]
            == "PASS"
            and official["source_test220"]["seeds"][str(seed)]["lead_recomputation"]
            ["alarm_endpoint_missing"] == 0
            for seed in (131, 133, 137)
        ),
        "J-TEXT lead is independently recomputed from alarm time and HDF5",
    ))
    evidence_text = EVIDENCE.read_text()
    results.append(check(
        '"/home/' not in evidence_text and '"/media/' not in evidence_text,
        "aggregate evidence is free of workstation absolute paths",
    ))

    bootstrap = official["bootstrap"]
    results.extend([
        check(bootstrap["iterations"] == 5000, "moving-block repeat count"),
        check(bootstrap["block_size"] == 20, "displayed moving-block size"),
        check("descriptive sensitivity only" in bootstrap["inference_boundary"],
              "post-selection inference boundary", bootstrap["inference_boundary"]),
        check(close(bootstrap["delta_fpr"]["median"], -0.013888888888888853),
              "paired block delta FPR", bootstrap["delta_fpr"]),
        check(bootstrap["delta_fpr"]["ci95"][1] < 0.0,
              "paired block delta FPR interval below zero", bootstrap["delta_fpr"]),
        check(close(bootstrap["delta_tpr"]["median"], 0.007168458781361964),
              "paired block delta operational TPR", bootstrap["delta_tpr"]),
        check(bootstrap["delta_tpr"]["ci95"][0] > 0.0,
              "paired block delta operational TPR interval above zero", bootstrap["delta_tpr"]),
        check(close(bootstrap["delta_median_lead_ms"]["median"], -11.65844899999945),
              "paired block delta operational lead", bootstrap["delta_median_lead_ms"]),
        check(set(official["moving_block_sensitivity"]) == {"10", "20", "30", "45"},
              "moving-block sensitivity sizes"),
        check(official["iid_stratified_bootstrap_secondary"]["iterations"] == 5000,
              "iid bootstrap retained as secondary diagnostic"),
    ])

    alarm_audit = lineage["alarm_state_machine"]
    results.extend([
        check(alarm_audit["adaptive_parameters"] == [
            "high_threshold", "physics_weight", "trend_weight", "early_threshold",
        ], "four adaptive alarm parameters", alarm_audit["adaptive_parameters"]),
        check(alarm_audit["fixed_causal_widths"] == {"fast": 5, "slow": 100},
              "fixed causal alarm widths", alarm_audit["fixed_causal_widths"]),
        check([
            alarm_audit["seeds"][str(seed)]["changed_update_count"]
            for seed in (131, 133, 137)
        ] == [38, 30, 40], "real alarm parameter updates", {
            seed: alarm_audit["seeds"][str(seed)]["changed_update_count"]
            for seed in (131, 133, 137)
        }),
    ])

    old_probes = [
        lineage["source_test220"]["shortcut_audit"]["seeds"][str(seed)]
        ["audit"]["paired_context_probe_macro_auc"]
        for seed in (131, 133, 137)
    ]
    padfix_probes = [
        evidence["padfix_ablation"]["validation_shortcut_audit"][str(seed)]
        ["paired_context_probe_macro_auc"]
        for seed in (131, 133, 137)
    ]
    results.extend([
        check(all(value > 0.65 for value in old_probes),
              "primary source shortcut failure retained", old_probes),
        check(all(value <= 0.65 for value in padfix_probes),
              "PADFIX context gates pass", padfix_probes),
        check(
            evidence["padfix_ablation"]["east_online360"]["aggregate"]["fpr"]["mean"]
            > evidence["padfix_ablation"]["east_static360"]["aggregate"]["fpr"]["mean"],
            "PADFIX online negative transfer retained",
        ),
        check(evidence["negative_controls"]["screen_candidate_count"] == 972,
              "development multiplicity retained"),
        check(evidence["negative_controls"]["equal_budget_random_query_available"],
              "causal equal-label-budget random-query online baseline available"),
    ])

    causal_random = json.loads(CAUSAL_RANDOM.read_text())
    random_per_seed = causal_random["per_seed"]
    results.extend([
        check(
            causal_random["schema_version"]
            == "causal-random-query-online-report-v1",
            "causal random-query report schema",
            causal_random["schema_version"],
        ),
        check(
            causal_random["seeds"] == [131, 133, 137]
            and causal_random["replicates_per_seed"] == 3,
            "causal random-query seeds and replicates",
        ),
        check(
            all(
                random_per_seed[str(seed)]["parent_and_static_hashes_match"]
                and random_per_seed[str(seed)]["label_budget_match"]
                and random_per_seed[str(seed)]["configured_optimizer_budget_match"]
                for seed in (131, 133, 137)
            ),
            "causal random-query parent and configured-budget fidelity",
        ),
        check(
            all(
                random_per_seed[str(seed)]["active_minus_random_mean_utility"] > 0
                for seed in (131, 133, 137)
            ),
            "active query utility exceeds random-query mean in every seed",
            {
                seed: random_per_seed[str(seed)]["active_minus_random_mean_utility"]
                for seed in (131, 133, 137)
            },
        ),
        check(
            not all(
                random_per_seed[str(seed)]["realized_compute_exact_match"]
                for seed in (131, 133, 137)
            )
            and causal_random["aggregate"]["confirmatory_significance_claim"] is False,
            "random-query realized-compute and significance limits retained",
            causal_random["limitations"],
        ),
    ])

    equal_compute = json.loads(EQUAL_COMPUTE.read_text())
    compute_per_seed = equal_compute["per_seed"]
    expected_compute = {
        131: (102, 1632, 204, -0.0018518518518518823),
        133: (90, 1440, 180, 0.0),
        137: (99, 1584, 198, -0.005555555555555425),
    }
    results.extend([
        check(
            equal_compute["schema_version"]
            == "equal-compute-selector-report-v2"
            and equal_compute["seeds"] == [131, 133, 137]
            and equal_compute["replicates_per_seed"] == 3,
            "equal-compute selector report schema and runs",
        ),
        check(
            all(
                compute_per_seed[str(seed)]["equal_query_budget"]
                and compute_per_seed[str(seed)]["equal_parent_and_static"]
                and compute_per_seed[str(seed)]["equal_realized_compute_shape"]
                for seed in (131, 133, 137)
            ),
            "equal-compute parent, budget and realized-shape fidelity",
        ),
        check(
            all(
                compute_per_seed[str(seed)]["active_archived_roster"]
                ["query_count"] == expected_compute[seed][0]
                and compute_per_seed[str(seed)]["active_archived_roster"]
                ["counts"]["fixed_compute_current_windows"]
                == expected_compute[seed][1]
                and compute_per_seed[str(seed)]["active_archived_roster"]
                ["counts"]["fixed_compute_replay_windows"]
                == expected_compute[seed][1]
                and compute_per_seed[str(seed)]["active_archived_roster"]
                ["counts"]["fixed_compute_shared_steps"]
                == expected_compute[seed][2]
                and compute_per_seed[str(seed)]["active_archived_roster"]
                ["counts"]["fixed_compute_private_steps"]
                == expected_compute[seed][2]
                for seed in (131, 133, 137)
            ),
            "equal-compute realized window and optimizer totals",
        ),
        check(
            all(close(
                compute_per_seed[str(seed)]["active_minus_random_mean_utility"],
                expected_compute[seed][3],
            ) for seed in (131, 133, 137)),
            "equal-compute selector deltas",
            {
                seed: compute_per_seed[str(seed)][
                    "active_minus_random_mean_utility"
                ] for seed in (131, 133, 137)
            },
        ),
        check(
            equal_compute["aggregate"]["screening_gate_passed"] is False
            and equal_compute["aggregate"]
            ["archived_roster_superiority_supported"] is False
            and equal_compute["aggregate"]
            ["adaptive_selector_policy_effect_estimated"] is False
            and equal_compute["aggregate"]["equivalence_or_null_proven"] is False,
            "equal-compute negative adjudication and claim boundary",
            equal_compute["aggregate"],
        ),
        check(
            all(
                all(
                    arm["metric_recomputation"][phase]["status"] == "PASS"
                    and arm["metric_recomputation"][phase]["truth_contract"]
                    == "HDF5 /meta/IsDisrupt and /meta/DownTime"
                    and arm["metric_recomputation"][phase]["metrics"]
                    == {
                        name: (arm["metrics"] if phase == "online" else arm["static_metrics"])[name]
                        for name in (
                            "TP", "FN", "FP", "TN", "tpr", "tpr_at_30ms",
                            "fpr", "median_lead_ms", "over_3000ms_count",
                        )
                    }
                    for phase in ("online", "static")
                )
                for seed in (131, 133, 137)
                for arm in (
                    [compute_per_seed[str(seed)]["active_archived_roster"]]
                    + compute_per_seed[str(seed)]["random_rosters"]
                )
            ),
            "equal-compute predictions recomputed from HDF5 truth",
        ),
    ])

    target_only = json.loads(EAST_TARGET_ONLY.read_text())
    target_local = target_only["target_only_arms"]["target_only_local17"]
    target_matched = target_only["target_only_arms"]["target_only_matched17"]
    results.extend([
        check(
            target_only["schema_version"] == "east-target-only-report-v1"
            and target_only["claim_boundary"] == "TEST_DRIVEN_EXPLORATION",
            "EAST target-only report schema and claim boundary",
        ),
        check(
            close(target_local["equal_seed_mean"]["operational_tpr"],
                  0.7962962962962963)
            and close(target_local["equal_seed_mean"]["fpr"],
                      0.14444444444444443)
            and close(target_local["equal_seed_mean"]["utility_tpr30_minus_fpr"],
                      0.6518518518518519),
            "strict EAST-only local-scaler aggregate",
            target_local["equal_seed_mean"],
        ),
        check(
            close(target_matched["equal_seed_mean"]["operational_tpr"],
                  0.7314814814814815)
            and close(target_matched["equal_seed_mean"]["fpr"],
                      0.10555555555555556),
            "target-only source-scaler diagnostic aggregate",
            target_matched["equal_seed_mean"],
        ),
        check(
            close(target_local["transfer_static_minus_target_only"]["utility_tpr30_minus_fpr"],
                  0.022222222222222143),
            "transfer-static utility increment over strict EAST-only",
            target_local["transfer_static_minus_target_only"],
        ),
    ])

    equal_budget = json.loads(EQUAL_BUDGET.read_text())
    direct_budget = equal_budget["direct_selected_complement"]
    direct_deltas = direct_budget["aggregate_deltas"]
    results.extend([
        check(
            equal_budget["schema_version"]
            == "equal-label-budget-common-mask-report-v2",
            "equal-label-budget report schema",
            equal_budget["schema_version"],
        ),
        check(
            equal_budget["status"]
            == "DIAGNOSTIC_ONLY_TEST_DRIVEN_EXPLORATION",
            "equal-label-budget status preserves development boundary",
            equal_budget["status"],
        ),
        check(
            equal_budget["completed_seeds"] == [131, 133, 137]
            and len(equal_budget["comparisons"]) == 9,
            "equal-label-budget seeds and random replicates",
        ),
        check(
            all(row["passed"] for row in equal_budget["parent_fidelity"].values()),
            "front-load control reproduces all archived parents exactly",
            equal_budget["parent_fidelity"],
        ),
        check(
            close(direct_deltas["selected_frontload_minus_static_tpr30"],
                  0.0641498255289189)
            and close(direct_deltas["selected_frontload_minus_static_fpr"],
                      0.16734354451922337)
            and close(direct_deltas["selected_frontload_minus_static_j"],
                      -0.1031937189903045)
            and close(direct_deltas["online_minus_selected_frontload_j"],
                      0.11628029499921815),
            "equal-label-budget aggregate deltas",
            direct_deltas,
        ),
        check(
            direct_budget["all_seed_selected_j_worse_than_static"]
            and direct_budget["all_seed_online_j_better_than_selected_frontload"],
            "equal-label-budget directions hold in all seeds",
        ),
        check(
            equal_budget["claim_boundary"].endswith(
                "not optimizer-step/FLOP matched to historical online updates"
            ),
            "equal-label-budget compute limitation disclosed",
            equal_budget["claim_boundary"],
        ),
    ])

    figures_ok, figure_failures = validate_manifest(
        FIGURE_MANIFEST, PACKAGE / "figures", "figures",
    )
    tables_ok, table_failures = validate_manifest(
        TABLE_MANIFEST, MANUSCRIPT / "tables", "tables",
    )
    results.extend([
        check(figures_ok, "figure hashes", figure_failures),
        check(tables_ok, "table hashes", table_failures),
    ])

    random_archive_ok, random_archive_failures = validate_compact_artifact_manifest(
        PACKAGE / "evidence/causal_random_query_online_v1/artifact_manifest.json"
    )
    target_archive_ok, target_archive_failures = validate_compact_artifact_manifest(
        PACKAGE / "evidence/east_target_only_v1/artifact_manifest.json"
    )
    compute_archive_ok, compute_archive_failures = validate_compact_artifact_manifest(
        PACKAGE
        / "evidence/equal_compute_selector_control_v1/artifact_manifest.json"
    )
    hdf5_manifest_ok, hdf5_manifest_failures = validate_hdf5_content_manifest(
        HDF5_CONTENT_MANIFEST
    )
    code_manifest_ok, code_manifest_failures = validate_release_code_manifest(
        EQUAL_COMPUTE_CODE_MANIFEST
    )
    results.extend([
        check(random_archive_ok, "causal random-query compact artifact hashes",
              random_archive_failures),
        check(target_archive_ok, "EAST target-only compact artifact hashes",
              target_archive_failures),
        check(compute_archive_ok, "equal-compute selector artifact hashes",
              compute_archive_failures),
        check(code_manifest_ok, "equal-compute release code/environment hashes",
              code_manifest_failures),
        check(hdf5_manifest_ok, "portable full HDF5 content manifest",
              hdf5_manifest_failures),
    ])
    portable_ok, portable_failures = validate_text_artifact_portability(PACKAGE)
    results.append(check(portable_ok, "all released text artifacts are portable",
                         portable_failures))

    q30_ok, q30_failures = validate_q30_manifest(
        Q30_MANIFEST,
        PACKAGE / "evidence/primary_online_artifacts/seed131/static_predictions.jsonl",
    )
    results.append(check(q30_ok, "Q30 frozen identity and provenance boundary",
                         q30_failures))
    prospective = json.loads(PROSPECTIVE_TEMPLATE.read_text())
    prospective_arms = {
        "static", "active", "no_label", "causal_random_query",
        "label_frontload", "east_only",
    }
    prospective_exclusion_sources = {
        source.get("path"): set(source.get("json_pointers", []))
        for source in prospective.get("excluded_shot_sources", [])
        if isinstance(source, dict)
    }
    required_exclusion_sources = {
        "config/splits/east_paper131_split_all_in.json": {
            "/train/disruption_shots", "/train/non_disruption_shots",
            "/test/disruption_shots", "/test/non_disruption_shots",
        },
        "config/splits/east_hdf5_authoritative_balanced360_v1.json": {
            "/test/disruption_shots", "/test/non_disruption_shots",
        },
    }
    results.extend([
        check(
            prospective.get("schema_version")
            == "prospective-east-confirmation-v1"
            and prospective.get("claim_status")
            == "PROSPECTIVE_PREREGISTRATION_ONLY"
            and set(prospective.get("arms", [])) == prospective_arms
            and prospective.get("retuning_prohibited") is True
            and prospective.get("complete_stream_reporting") is True
            and prospective.get("target_validation_completed") is True
            and all(
                prospective_exclusion_sources.get(path) == pointers
                for path, pointers in required_exclusion_sources.items()
            ),
            "prospective confirmation template preserves frozen claim contract",
        ),
        check(
            all(
                prospective.get(field) in prospective.get("frozen_artifacts", [])
                for field in (
                    "target_validation_artifact",
                    "random_schedule_artifact",
                    "execution_plan_artifact",
                )
            ),
            "prospective validation, random schedules and execution plan are hash-bound",
        ),
    ])
    for seed in (131, 133, 137):
        seed_dir = PACKAGE / f"evidence/primary_online_artifacts/seed{seed}"
        causal_ok, causal_failures = validate_primary_causal_ledgers(seed_dir)
        results.append(check(
            causal_ok, f"primary causal ledger relations seed{seed}", causal_failures,
        ))
        parent = (
            PACKAGE / "evidence/equal_label_budget_frontload_v2"
            / f"seed{seed}/support_only/frontloaded_target_state.pt"
        )
        parent_summary = json.loads((parent.parent / "summary.json").read_text())
        expected_parent_hash = parent_summary["hashes"]["frontloaded_target_state"]
        results.append(check(
            parent.is_file() and sha256_file(parent) == expected_parent_hash,
            f"sealed EAST target-parent artifact seed{seed}",
            {"expected_sha256": expected_parent_hash},
        ))

    required_files = [
        PACKAGE / "README.md", REPORT, MANUSCRIPT / "main.tex",
        MANUSCRIPT / "supplement.tex", MANUSCRIPT / "references.bib",
        PACKAGE / "NATURE_PORTFOLIO_READINESS.md", PACKAGE / "MODEL_CARD.md",
        PACKAGE / "FIVE_WAY_AUDIT.md",
        PACKAGE / "evidence/case_study_seed131_shot73790.npz",
        PACKAGE / "figures/fig5_adaptive_alarm_state_machine.pdf",
        PACKAGE / "figures/fig8_equal_label_budget_frontload_ablation.pdf",
        PACKAGE / "figures/fig11_equal_compute_selector_screen.pdf",
        PACKAGE / "evidence/equal_label_budget_frontload_v2/common_mask_report/ADJUDICATION_zh.md",
        PACKAGE / "evidence/equal_label_budget_frontload_v2/common_mask_report/adjudication.json",
        CAUSAL_RANDOM,
        EQUAL_COMPUTE,
        PACKAGE / "evidence/causal_random_query_online_v1/ADJUDICATION.md",
        EAST_TARGET_ONLY,
        PACKAGE / "evidence/east_target_only_v1/ADJUDICATION.md",
        HDF5_CONTENT_MANIFEST,
        Q30_MANIFEST,
        EQUAL_COMPUTE_CODE_MANIFEST,
        PROSPECTIVE_TEMPLATE,
        ROOT / "reproducibility/prospective.py",
    ]
    results.append(check(all(path.is_file() for path in required_files),
                         "required source/report files",
                         [str(path) for path in required_files if not path.is_file()]))
    compute_manifest_payload = json.loads((
        PACKAGE / "evidence/equal_compute_selector_control_v1/artifact_manifest.json"
    ).read_text())
    compute_release_files = [
        PACKAGE / "evidence/equal_compute_selector_control_v1" / row["path"]
        for row in compute_manifest_payload["artifacts"]
    ]
    tracked_ok, tracked_failures = validate_git_tracked([
        *required_files,
        *compute_release_files,
        PACKAGE / "figures/fig11_equal_compute_selector_screen.svg",
        PACKAGE / "figures/fig11_equal_compute_selector_screen.png",
        MANUSCRIPT / "tables/equal_compute_selector.tex",
        PACKAGE / "scripts/run_equal_compute_selector_control.py",
        PACKAGE / "scripts/build_equal_compute_selector_report.py",
        PACKAGE / "scripts/build_equal_compute_artifact_manifest.py",
        PACKAGE / "scripts/build_equal_compute_code_manifest.py",
        PACKAGE / "scripts/archive_equal_compute_selector_results.py",
    ])
    results.append(check(
        tracked_ok, "release-critical equal-compute files are Git tracked",
        tracked_failures,
    ))

    main_text = (MANUSCRIPT / "main.tex").read_text()
    supplement_text = (MANUSCRIPT / "supplement.tex").read_text()
    report_text = REPORT.read_text()
    readiness_text = (PACKAGE / "NATURE_PORTFOLIO_READINESS.md").read_text()
    combined = "\n".join((main_text, supplement_text, report_text, readiness_text))
    mandatory = (
        "TEST_DRIVEN_EXPLORATION", "15D/95N", "972", "count-aligned",
        "random-query", "front-load", "label-access", "not ready",
        "padding", "future", "HDF5", "predict", "seal", "query", "reveal",
    )
    results.append(check(all(token.lower() in combined.lower() for token in mandatory),
                         "mandatory caveats and contracts appear",
                         [token for token in mandatory if token.lower() not in combined.lower()]))
    forbidden_main = (
        "deployment-ready model", "independent east test confirms",
        "learned all disruption mechanisms", "solved disruption prediction",
    )
    results.append(check(not any(token in main_text.lower() for token in forbidden_main),
                         "main manuscript avoids forbidden overclaims"))
    forbidden_evidence_labels = (
        "isolated source test", "independent j-text test",
        "strict chronological j-text test",
    )
    results.append(check(
        not any(token in combined.lower() for token in forbidden_evidence_labels),
        "source comparator is not mislabeled independent/chronological",
        [token for token in forbidden_evidence_labels if token in combined.lower()],
    ))
    results.extend([
        check("\\section*{Data availability}" in main_text,
              "separate Data availability section"),
        check("\\section*{Code availability}" in main_text,
              "separate Code availability section"),
        check("AI-assisted development disclosure" in main_text,
              "generative-AI assistance disclosed"),
        check("NOT READY" in (PACKAGE / "NATURE_PORTFOLIO_READINESS.md").read_text(),
              "Nature readiness blocker is explicit"),
    ])
    publication_surfaces = "\n".join((
        main_text, supplement_text, report_text, (PACKAGE / "README.md").read_text(),
    ))
    deprecated_headline_terms = ("raw tpr", "timely tpr")
    results.append(check(
        not any(token in publication_surfaces.lower() for token in deprecated_headline_terms),
        "publication surfaces use one operational TPR contract",
        [token for token in deprecated_headline_terms if token in publication_surfaces.lower()],
    ))

    pdfs = {name: pdf_info(MANUSCRIPT / name) for name in ("main.pdf", "supplement.pdf")}
    results.extend([
        check(pdfs["main.pdf"]["exists"] and pdfs["main.pdf"]["pages"] >= 6
              and pdfs["main.pdf"]["bytes"] >= 50_000
              and pdfs["main.pdf"]["text_chars"] >= 20_000,
              "compiled main PDF", pdfs["main.pdf"]),
        check(pdfs["supplement.pdf"]["exists"] and pdfs["supplement.pdf"]["pages"] >= 8
              and pdfs["supplement.pdf"]["bytes"] >= 50_000
              and pdfs["supplement.pdf"]["text_chars"] >= 20_000,
              "compiled supplement PDF", pdfs["supplement.pdf"]),
    ])

    for seed, hashes in lineage["artifact_hashes"].items():
        bundled = PACKAGE / f"evidence/primary_online_artifacts/seed{seed}"
        run = (
            bundled if bundled.is_dir() else
            ROOT / f"experiments/transfer_methods_2026/online_episode_value_v3_runs/final_seed{seed}_episode_v3"
        )
        for name, expected_hash in hashes.items():
            candidate = resolve_artifact(run / name)
            results.append(check(candidate.is_file() and sha256_artifact(candidate) == expected_hash,
                                 f"primary artifact hash seed{seed}/{name}"))

    for seed, hashes in lineage["source_artifact_hashes"].items():
        bundled = PACKAGE / f"evidence/source_lineage/seed{seed}"
        run = (
            bundled if bundled.is_dir() else
            ROOT / f"experiments/transfer_methods_2026/source_transformer_runs/seed{seed}"
        )
        for name, expected_hash in hashes.items():
            candidate = run / name
            results.append(check(candidate.is_file() and sha256_file(candidate) == expected_hash,
                                 f"source artifact hash seed{seed}/{name}"))

    figure_source = (PACKAGE / "scripts/generate_figures.py").read_text()
    style_source = (PACKAGE / "scripts/publication_style.py").read_text()
    results.extend([
        check("linewidth=0." not in figure_source,
              "figure strokes are not sub-1-point explicit widths"),
        check('"axes.linewidth": 1.0' in style_source,
              "publication axes use one-point width"),
    ])

    latex_logs = {
        name: (MANUSCRIPT / name).read_text(errors="replace")
        for name in ("main.log", "supplement.log")
        if (MANUSCRIPT / name).is_file()
    }
    results.append(check(
        len(latex_logs) == 2 and not any(
            token in text for text in latex_logs.values()
            for token in ("Overfull \\hbox", "Undefined references", "Citation `")
        ),
        "LaTeX logs have no overfull boxes or unresolved references",
    ))

    passed = all(item["passed"] for item in results)
    payload = {
        "schema_version": "paper-package-validation-v4",
        "status": "PASSED" if passed else "FAILED",
        "validation_scope": "local package integrity and disclosure completeness",
        "nature_submission_readiness": "BLOCKED_PENDING_NEW_EVIDENCE_AND_ARCHIVAL_RELEASE",
        "passed": passed,
        "checks_passed": sum(item["passed"] for item in results),
        "checks_total": len(results),
        "checks": results,
        "claim_boundary": evidence["claim_boundary"],
        "pdfs": pdfs,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
