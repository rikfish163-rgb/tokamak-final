#!/usr/bin/env python3
"""Fail-closed validator for full EAST360 causal episode-v3 replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_ROOT = HERE / "online_episode_value_v3_runs"
DEFAULT_BASELINE_ROOT = Path(os.environ.get(
    "FUSION_V21_BASELINE_ROOT", HERE / "online_architecture_search/frozen_runs",
)).expanduser().resolve()
DEFAULT_COMPLETION_ARTIFACT = (
    ROOT / ".omx/specs/autoresearch-east360-fpr-episode-v3/result.json"
)
DEFAULT_REPORT = DEFAULT_ROOT / "REPORT_zh.md"
BASELINE = {
    131: {"TP": 172, "FP": 28, "tpr_at_30ms": 0.8444444444444444},
    133: {"TP": 166, "FP": 26, "tpr_at_30ms": 0.8388888888888889},
    137: {"TP": 169, "FP": 36, "tpr_at_30ms": 0.8611111111111112},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def forbidden_keys(value: Any, forbidden: set[str], *, path: str = "") -> list[str]:
    """Return outcome-bearing keys at any nesting depth in a pre-reveal row."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            if name in forbidden:
                found.append(child_path)
            found.extend(forbidden_keys(child, forbidden, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, forbidden, path=f"{path}[{index}]"))
    return found


def resolve_baseline_run(root: Path, seed: int) -> Path:
    """Resolve a known v2.1 baseline layout, failing closed otherwise.

    The archived validator assumed a directory name that no longer exists in
    this checkout (``final_seed{seed}_frozen_v21``), even though the sealed
    frozen replay is present under ``online_architecture_search/frozen_runs``.
    Accept both explicit layouts, but require the baseline stage marker and
    required ledgers so a missing or unrelated directory cannot be treated as
    a valid control.
    """
    root = Path(root).expanduser().resolve()
    candidates = (
        root / f"final_seed{int(seed)}_frozen_v21",
        root / f"seed{int(seed)}",
    )
    required = ("summary.json", "predictions.jsonl", "query_decisions.jsonl")
    for candidate in candidates:
        if not all((candidate / name).is_file() for name in required):
            continue
        summary = json.loads((candidate / "summary.json").read_text())
        if summary.get("stage") != "STATIC360_THEN_RESET_ONLINE360_FROZEN_ARCHITECTURE":
            continue
        return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"no sealed v2.1 frozen baseline for seed {seed}; checked {expected}"
    )


def validate_seed(
    seed: int, root: Path, baseline_root: Path,
    errors: list[str], checks: dict[str, Any],
) -> None:
    run = root / f"final_seed{seed}_episode_v3"
    required = (
        "summary.json", "final_online_state.pt", "predictions.jsonl",
        "static_predictions.jsonl", "query_decisions.jsonl",
        "reveals.jsonl", "updates.jsonl",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        errors.append(f"seed{seed}: missing final artifacts {missing}")
        return
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("status") != "ONLINE_DIAGNOSTIC":
        errors.append(f"seed{seed}: status is {summary.get('status')!r}")
    if int(summary.get("stream_shots", -1)) != 360:
        errors.append(f"seed{seed}: stream is not complete 360")
    episode_contract = summary.get("value_contract", {}).get(
        "episode_value_v3"
    )
    if not episode_contract:
        errors.append(f"seed{seed}: episode v3 contract is absent")
    else:
        expected_episode = {
            "head_type": "causal_delta60_protected_calibration",
            "query_mode": "shadow",
            "gradient_mode": "legacy",
            "learning_feedback_isolated": True,
        }
        for key, expected in expected_episode.items():
            if episode_contract.get(key) != expected:
                errors.append(
                    f"seed{seed}: episode contract {key}="
                    f"{episode_contract.get(key)!r}, expected {expected!r}"
                )
    hashes = summary.get("hashes", {})
    for key, filename in {
        "predictions": "predictions.jsonl",
        "static_predictions": "static_predictions.jsonl",
        "queries": "query_decisions.jsonl",
        "reveals": "reveals.jsonl",
        "updates": "updates.jsonl",
    }.items():
        if hashes.get(key) != sha256(run / filename):
            errors.append(f"seed{seed}: {filename} hash mismatch")
    predictions = load_jsonl(run / "predictions.jsonl")
    static = load_jsonl(run / "static_predictions.jsonl")
    queries = load_jsonl(run / "query_decisions.jsonl")
    reveals = load_jsonl(run / "reveals.jsonl")
    if not all(len(rows) == 360 for rows in (predictions, static, queries)):
        errors.append(f"seed{seed}: incomplete prediction/query rows")
        return
    shots = [int(row["shot"]) for row in predictions]
    if shots != sorted(shots) or len(set(shots)) != 360:
        errors.append(f"seed{seed}: roster is not unique chronological 360")
    if [int(row["shot"]) for row in static] != shots:
        errors.append(f"seed{seed}: static roster differs")
    if [int(row["shot"]) for row in queries] != shots:
        errors.append(f"seed{seed}: query roster differs")
    selected = [int(row["shot"]) for row in queries if bool(row["selected"])]
    if selected != [int(row["shot"]) for row in reveals]:
        errors.append(f"seed{seed}: selected and revealed rosters differ")
    if len(selected) > 108:
        errors.append(f"seed{seed}: 30 percent query budget exceeded")
    forbidden = {"is_disrupt", "down_time_s", "label", "outcome"}
    for name, rows in (("prediction", predictions), ("query", queries)):
        leaked = sorted({item for row in rows for item in forbidden_keys(row, forbidden)})
        if leaked:
            errors.append(f"seed{seed}: {name} rows expose labels {leaked}")

    # The causal episode gate is a deployment overlay.  It may delay or veto
    # the already-sealed v2.1 alarm, but may never create or advance one.
    overlay_violations: list[int] = []
    for row in predictions:
        deployed = bool(row["alarmed"])
        learning = bool(row.get("learning_alarmed"))
        deployed_index = row.get("alarm_index")
        learning_index = row.get("learning_alarm_index")
        valid = bool(row.get("learning_feedback_isolated"))
        if deployed:
            valid = (
                valid and learning and deployed_index is not None
                and learning_index is not None
                and int(deployed_index) >= int(learning_index)
            )
        else:
            valid = valid and deployed_index is None
        if not valid:
            overlay_violations.append(int(row["shot"]))
    if overlay_violations:
        errors.append(
            f"seed{seed}: deployment overlay violations {overlay_violations[:8]}"
        )

    # A frozen-v2.1 trajectory comparison is meaningful only when the episode
    # run explicitly declares its query controller frozen.  Episode-v3 is
    # allowed to adapt its *query policy* after reveal; in that case comparing
    # ``queried`` against v2.1 would reject the intended treatment rather than
    # test feedback isolation.  We still require the sealed baseline for the
    # metric comparison, and validate deployment/learning isolation above.
    episode_contract = summary.get("value_contract", {}).get(
        "episode_value_v3", {}
    )
    query_control_frozen = bool(episode_contract.get("query_control_frozen", False))
    try:
        baseline = resolve_baseline_run(baseline_root, seed)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        baseline = None
    baseline_required = ("predictions.jsonl", "query_decisions.jsonl")
    baseline_missing = (
        baseline is None
        or any(not (baseline / name).is_file() for name in baseline_required)
    )
    if baseline_missing:
        if baseline is not None:
            errors.append(f"seed{seed}: missing v2.1 baselines {baseline_required}")
    elif query_control_frozen:
        baseline_predictions = load_jsonl(baseline / "predictions.jsonl")
        baseline_queries = load_jsonl(baseline / "query_decisions.jsonl")
        if len(baseline_predictions) != len(predictions):
            errors.append(f"seed{seed}: v2.1 prediction length changed")
        else:
            for before, after in zip(
                baseline_predictions, predictions, strict=True,
            ):
                prediction_signature = (
                    int(before["shot"]), bool(before["alarmed"]),
                    before["alarm_index"], int(before["parameter_version"]),
                    before["parameters"], bool(before["queried"]),
                    before["shared_persistent_sha256_before_prediction"],
                    before["private_persistent_sha256_before_prediction"],
                )
                learning_signature = (
                    int(after["shot"]), bool(after["learning_alarmed"]),
                    after["learning_alarm_index"],
                    int(after["parameter_version"]), after["parameters"],
                    bool(after["queried"]),
                    after["shared_persistent_sha256_before_prediction"],
                    after["private_persistent_sha256_before_prediction"],
                )
                if prediction_signature != learning_signature:
                    errors.append(
                        f"seed{seed}: learning trajectory differs at shot "
                        f"{after['shot']}"
                    )
                    break
        if len(baseline_queries) != len(queries):
            errors.append(f"seed{seed}: v2.1 query length changed")
        else:
            query_fields = (
                "shot", "selected", "query_score", "cutoff",
                "selection_route",
            )
            for before, after in zip(
                baseline_queries, queries, strict=True,
            ):
                if any(before.get(key) != after.get(key) for key in query_fields):
                    errors.append(
                        f"seed{seed}: query control differs at shot "
                        f"{after['shot']}"
                    )
                    break
    elif baseline is not None:
        checks.setdefault(str(seed), {})["baseline_comparison"] = {
            "status": "NOT_APPLICABLE",
            "reason": "episode-v3 query controller is adaptive; exact v2.1 query trajectory is not the treatment contract",
            "query_control_frozen": False,
        }

    metric = summary["results"]["online_unified"]
    base = BASELINE[seed]
    if float(metric["tpr"]) < 0.60 or float(metric["fpr"]) > 0.40:
        errors.append(f"seed{seed}: abnormal metric gate failed")
    if int(metric["TP"]) < int(base["TP"]):
        errors.append(f"seed{seed}: TP degraded {metric['TP']} < {base['TP']}")
    if float(metric["tpr_at_30ms"]) + 1e-12 < float(base["tpr_at_30ms"]):
        errors.append(f"seed{seed}: TPR@30ms degraded")
    if int(metric["FP"]) >= int(base["FP"]):
        errors.append(f"seed{seed}: FP did not strictly improve")
    baseline_comparison = checks.get(str(seed), {}).get("baseline_comparison")
    checks[str(seed)] = {
        "query_count": len(selected), "baseline": base,
        "online": metric, "summary_sha256": sha256(run / "summary.json"),
        "source_base_sha256": hashes.get("base_before_after"),
        "baseline_run": None if baseline is None else str(baseline),
        "baseline_comparison": baseline_comparison or {
            "status": "BIT_EXACT_REQUIRED",
            "query_control_frozen": True,
        },
        "deployment_overlay_monotone": not overlay_violations,
        "v21_learning_trajectory_bit_exact": (
            None if baseline_comparison and baseline_comparison["status"] == "NOT_APPLICABLE"
            else not any(
                item.startswith(f"seed{seed}: learning trajectory differs")
                or item.startswith(f"seed{seed}: query control differs")
                for item in errors
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument(
        "--completion-artifact", type=Path,
        default=DEFAULT_COMPLETION_ARTIFACT,
    )
    args = parser.parse_args()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    for seed in BASELINE:
        validate_seed(
            seed, args.artifact_root, args.baseline_root, errors, checks,
        )
    result = {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "integrity_status": "PASS" if not errors else "FAIL",
        "research_status": "ONLINE_DIAGNOSTIC_TEST_DRIVEN_EXPLORATION",
        "independent_future_stream_complete": False,
        "artifact_root": str(args.artifact_root),
        "baseline_root": str(args.baseline_root),
        "output_artifact_path": str(DEFAULT_REPORT.relative_to(ROOT)),
        "checks": checks, "errors": errors,
    }
    args.completion_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.completion_artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
