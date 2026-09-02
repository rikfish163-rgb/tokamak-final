#!/usr/bin/env python3
"""Audit the retained J-TEXT -> EAST diagnostic mainline.

The audit is deliberately a read-mostly, fail-closed check over the published
receipts.  It does not retrain a model, choose a threshold, or inspect labels
from an unqueried online shot.  A successful report means that the two
lineages and their stored causal receipts are internally consistent; it does
not turn balanced360 or TEST220 into independent future evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

try:  # Package import in tests; direct import when invoked as a script.
    from .mainline_paths import assert_final_artifact_root, assert_project_workspace_path
    from .support_label_audit import audit_east_support110
except ImportError:  # pragma: no cover - exercised by the CLI entry point
    from mainline_paths import assert_final_artifact_root, assert_project_workspace_path
    from support_label_audit import audit_east_support110


REPO_ROOT = Path(__file__).resolve().parents[2]
MOUNTED_ROOT = REPO_ROOT / "fuison_ext4_runs"
SUPPORT_ROSTER_PATH = REPO_ROOT / "config/splits/east_paper131_split_all_in.json"
CAUSAL_ORDER_DETAIL = {
    "endpoint": [
        "observe the current causal signal prefix",
        "predict and seal the current endpoint score",
        "optionally apply the unlabeled intra-shot update to fast state",
        "continue to the next endpoint",
    ],
    "shot": [
        "seal the complete shot prediction trace",
        "select a query without reading the current shot outcome",
        "reveal HDF5 outcome only for a selected shot",
        "apply persistent inter-shot feedback before the next shot",
    ],
    "outcome_access_during_endpoint": "none",
}
EXPECTED_SOURCE_SHA = {
    131: "1ab6b8e3b30992400ea6df8ffe235cb9d0023730f5a0c77d4de0f087ae85912d",
    133: "c8c5945f58a938d9c7d8197650b01b9dacf2be7cf0ae317825f422d1283142da",
    137: "5183dc09239409325dd4f10fde8a6af4b6da4e28accc9ff88b11ff07a4d826cf",
}
EXPECTED_PARENT_SHA = {
    131: "3dd1366ae3dc671d6a693d52c1e8eb0ab2d3a168a6d811ad7679637d79d92bbd",
    133: "396fe442d8f35c7f2b40fb31dbe594ee396a67d226dc56a7eeda2e62c6f9c178",
    137: "f1726381a730969ca497b2a376bc4dc346759409f16865524112077c3e50cce4",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required receipt is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must contain a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return number


def _metrics(
    row: Mapping[str, Any], *, label: str, disruptive: int, safe: int,
    tpr_key: str, count_keys: Mapping[str, str] | None = None,
    raw_tpr_key: str | None = None,
) -> dict[str, Any]:
    count_source: Mapping[str, Any] = row.get("counts", row)
    count_keys = count_keys or {key: key for key in ("TP", "FP", "FN", "TN")}
    for key in count_keys.values():
        _require(key in count_source, f"{label} lacks count {key}")
    _require(tpr_key in row, f"{label} lacks metric {tpr_key}")
    _require("fpr" in row, f"{label} lacks metric fpr")
    counts = {key: int(count_source[count_keys[key]]) for key in ("TP", "FP", "FN", "TN")}
    _require(all(value >= 0 for value in counts.values()), f"{label} has negative counts")
    _require(
        counts["TP"] + counts["FN"] == disruptive,
        f"{label} disruptive denominator is not {disruptive}: {counts}",
    )
    _require(
        counts["FP"] + counts["TN"] == safe,
        f"{label} safe denominator is not {safe}: {counts}",
    )
    tpr = _finite(row[tpr_key], f"{label}.{tpr_key}")
    fpr = _finite(row["fpr"], f"{label}.fpr")
    _require(0.0 <= tpr <= 1.0, f"{label} TPR is outside [0, 1]")
    # EAST stores both raw TPR (matching TP/FN) and operational TPR@30ms.
    # The latter has no separate count field in the historical receipt, so
    # only the raw value is checked against the confusion counts.
    raw_tpr_key = raw_tpr_key or tpr_key
    if raw_tpr_key in row:
        raw_tpr = _finite(row[raw_tpr_key], f"{label}.{raw_tpr_key}")
        _require(abs(raw_tpr - counts["TP"] / disruptive) < 1e-8,
                 f"{label} raw TPR/count mismatch")
    _require(abs(fpr - counts["FP"] / safe) < 1e-8, f"{label} FPR/count mismatch")
    result = dict(counts)
    median = row.get("median_lead_ms")
    if median is None:
        median = row.get("operational_tp_lead_ms", {}).get("median_ms")
    result.update({
        "tpr": tpr,
        "tpr_key": tpr_key,
        "fpr": fpr,
        "median_lead_ms": _finite(median, f"{label}.median_lead_ms"),
        "over_3000ms_count": int(row.get("over_3000ms_count", 0)),
    })
    return result


def _assert_mounted_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    assert_final_artifact_root(resolved)
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    _require(
        os.stat(resolved).st_dev != os.stat("/").st_dev,
        f"{label} is on the local root filesystem: {resolved}",
    )
    _require(Path("/home") not in resolved.parents and resolved != Path("/home"),
             f"{label} resolves under /home: {resolved}")
    return resolved


def _audit_jtext(root: Path) -> dict[str, Any]:
    root = _assert_mounted_directory(root, "J-TEXT result root")
    manifest = _read_json(root / "mainline_manifest.json")
    summary = _read_json(root / "summary.json")
    _require(manifest.get("schema_version") == "jtext-adv25-active-mounted-v1",
             "J-TEXT schema is not the retained Adv25 mainline")
    _require(manifest.get("claim_status") == "TEST_DRIVEN_EXPLORATION",
             "J-TEXT claim boundary changed unexpectedly")
    _require(manifest.get("method") == "delayed-feedback-active-hedge-inter-only",
             "J-TEXT result is not the registered active branch")
    _require(manifest.get("representation_updated") is False,
             "J-TEXT receipt claims an unregistered representation update")
    _require(manifest.get("jtext_online_state_used_for_east") is False,
             "J-TEXT online state is marked as an EAST input")
    _require(manifest.get("east_transfer_parent") == "static_adv25_source_checkpoint_only",
             "J-TEXT/EAST parent isolation contract changed")
    _require(manifest.get("causal_order") ==
             "signal prefix -> predict -> seal -> query -> reveal -> inter update",
             "J-TEXT causal order is not the registered order")
    causal_detail = manifest.get("causal_order_detail")
    if causal_detail is not None:
        _require(
            isinstance(causal_detail, Mapping)
            and causal_detail.get("outcome_access_before_query") == "none",
            "J-TEXT detailed causal contract is invalid",
        )
    _require(manifest.get("test_outcomes_used_before_query") is False,
             "J-TEXT receipt does not explicitly prohibit pre-query outcomes")
    _require(manifest.get("test_outcomes_used_for_trace_or_query_score") is False,
             "J-TEXT outcomes entered a trace or query score")
    _require(manifest.get("hdf5_outcomes_used_only_after_selected_query") is True,
             "J-TEXT HDF5 reveal contract is missing")
    _require(int(manifest.get("budget_percent", -1)) == 20, "J-TEXT budget is not 20%")
    _require(int(manifest.get("query_count", -1)) == 44, "J-TEXT query count is not 44")
    _require(int(summary.get("shot_count", -1)) == 220, "J-TEXT did not process TEST220")
    _require(int(summary.get("query_count", -1)) == 44, "J-TEXT summary query count is not 44")
    roster = manifest.get("ordered_test220")
    _require(isinstance(roster, list) and len(roster) == 220 and len(set(roster)) == 220,
             "J-TEXT TEST220 roster is not a unique 220-shot sequence")
    _require("VAL160" in str(manifest.get("threshold_source")),
             "J-TEXT threshold source is not VAL160")
    static = _metrics(
        summary.get("static", {}), label="J-TEXT static", disruptive=110, safe=110,
        tpr_key="operational_tpr",
        count_keys={"TP": "operational_tp", "FP": "fp", "FN": "operational_fn", "TN": "tn"},
    )
    online = _metrics(
        summary.get("online", {}), label="J-TEXT online", disruptive=110, safe=110,
        tpr_key="operational_tpr",
        count_keys={"TP": "operational_tp", "FP": "fp", "FN": "operational_fn", "TN": "tn"},
    )
    delta = summary.get("delta", {}).get("candidate_minus_baseline", {})
    return {
        "path": str(root),
        "claim_status": manifest["claim_status"],
        "budget_percent": int(manifest["budget_percent"]),
        "query_count": int(manifest["query_count"]),
        "static": static,
        "online": online,
        "delta": {
            "tpr": online["tpr"] - static["tpr"],
            "fpr": online["fpr"] - static["fpr"],
            "j": _finite(delta.get("j_operational"), "J-TEXT delta.j_operational"),
            "auc": _finite(delta.get("auc"), "J-TEXT delta.auc"),
        },
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "causal_order_detail": causal_detail or {
            "shot": [
                "observe the current causal TEST220 signal prefix",
                "predict and seal the complete shot trace",
                "select queries without reading the current outcome",
                "reveal HDF5 outcome only for selected shots",
                "update the mixture before the next shot/block",
            ],
            "outcome_access_before_query": "none",
            "inferred_from_legacy_receipt": True,
        },
    }


def _audit_east(root: Path, *, seed: int, fresh: bool) -> dict[str, Any]:
    root = _assert_mounted_directory(root, f"EAST seed{seed} result root")
    manifest = _read_json(root / "mainline_manifest.json")
    summary = _read_json(root / "final_episode_v3" / "summary.json")
    _require(manifest.get("schema_version") == "east-v3c-shotfirst-mounted-v1",
             f"EAST seed{seed} schema is not the retained V3C mainline")
    _require(manifest.get("status") == "ONLINE_DIAGNOSTIC",
             f"EAST seed{seed} status changed unexpectedly")
    _require(manifest.get("claim_status") == "TEST_DRIVEN_EXPLORATION",
             f"EAST seed{seed} is not marked diagnostic")
    _require(int(manifest.get("seed", -1)) == seed, f"EAST seed mismatch: expected {seed}")
    _require(manifest.get("support_scope") == "EAST support110 only",
             f"EAST seed{seed} support scope changed")
    _require(manifest.get("test_scope") == "EAST balanced360 chronological stream",
             f"EAST seed{seed} test scope changed")
    _require(manifest.get("support_sampler") ==
             "shot-first -> endpoint (historical V3C compatibility)",
             f"EAST seed{seed} support sampler changed")
    _require(manifest.get("causal_order") == "predict -> seal -> query -> reveal -> update",
             f"EAST seed{seed} causal order changed")
    causal_detail = manifest.get("causal_order_detail")
    if causal_detail is not None:
        _require(
            isinstance(causal_detail, Mapping)
            and causal_detail.get("outcome_access_during_endpoint") == "none",
            f"EAST seed{seed} detailed causal contract is invalid",
        )
    _require(manifest.get("balanced360_outcomes_used_before_query") is False,
             f"EAST seed{seed} has a pre-query outcome access flag")
    _require(manifest.get("jtext_online_checkpoint_used_for_east") is False,
             f"EAST seed{seed} consumed J-TEXT online state")
    source = Path(str(manifest.get("source_checkpoint", ""))).expanduser().resolve()
    _require(source.is_file(), f"EAST seed{seed} source checkpoint is missing: {source}")
    source_sha = _sha256(source)
    _require(source_sha == EXPECTED_SOURCE_SHA[seed], f"EAST seed{seed} source SHA mismatch")
    _require(manifest.get("source_checkpoint_sha256") == source_sha,
             f"EAST seed{seed} manifest source SHA mismatch")
    hashes = summary.get("hashes", {})
    _require(hashes.get("source_checkpoint") == source_sha,
             f"EAST seed{seed} summary source SHA mismatch")
    _require(hashes.get("base_before_after") == EXPECTED_PARENT_SHA[seed],
             f"EAST seed{seed} support parent SHA mismatch")
    _require(int(summary.get("stream_shots", -1)) == 360,
             f"EAST seed{seed} did not process all 360 shots")
    order = summary.get("order_contract", {})
    _require(order.get("chronological_main_result") is True and order.get("is_reordered") is False,
             f"EAST seed{seed} is not chronological")
    _require(order.get("query_selection_label_oracle_used") is False,
             f"EAST seed{seed} query selection used labels")
    execution = summary.get("execution_protocol", {})
    _require(execution.get("phase1_static_complete_before_phase2") is True,
             f"EAST seed{seed} static phase was not sealed")
    _require(execution.get("phase2_reset_to_same_migrated_initial_sha") is True,
             f"EAST seed{seed} online phase did not reset to the static parent")
    _require(execution.get("selection_on_360_in_this_run") is True,
             f"EAST seed{seed} lacks its development-selection marker")
    static = _metrics(
        summary.get("results", {}).get("static_unified", {}),
        label=f"EAST seed{seed} static", disruptive=180, safe=180,
        tpr_key="tpr_at_30ms",
        raw_tpr_key="tpr",
    )
    online = _metrics(
        summary.get("results", {}).get("online_unified", {}),
        label=f"EAST seed{seed} online", disruptive=180, safe=180,
        tpr_key="tpr_at_30ms",
        raw_tpr_key="tpr",
    )
    return {
        "seed": seed,
        "path": str(root),
        "fresh_replay": fresh,
        "claim_status": manifest["claim_status"],
        "query_count": int(summary.get("query_count", -1)),
        "static": static,
        "online": online,
        "delta": {
            "tpr": online["tpr"] - static["tpr"],
            "fpr": online["fpr"] - static["fpr"],
            "j": (online["tpr"] - online["fpr"])
                 - (static["tpr"] - static["fpr"]),
        },
        "causal_order_detail": causal_detail or {
            **CAUSAL_ORDER_DETAIL,
            "inferred_from_legacy_receipt": True,
        },
        "support_contract_mode": manifest.get(
            "support_contract_mode", "legacy_role_roster_hdf5_authoritative"
        ),
    }


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    mounted = _assert_mounted_directory(MOUNTED_ROOT, "published artifact root")
    jtext = _audit_jtext(args.jtext_root)
    # The compatibility replay's support IDs come from the historical role
    # roster, but every support label is read from HDF5.  Audit that boundary
    # explicitly; do not inspect any balanced360 outcome here.
    east_manifest = _read_json(args.east_root / "seed131" / "mainline_manifest.json")
    data_root = east_manifest.get("data_root")
    _require(data_root is not None, "EAST seed131 manifest lacks data_root")
    data_root_path = assert_project_workspace_path(data_root, label="EAST audit data root")
    support_audit = audit_east_support110(data_root_path, SUPPORT_ROSTER_PATH)
    for seed in (131, 133, 137):
        manifest = _read_json(
            args.east_root / f"seed{seed}" / "mainline_manifest.json"
        )
        if "support_contract_mode" in manifest:
            _require(
                manifest["support_contract_mode"]
                == support_audit["support_contract_mode"],
                f"EAST seed{seed} support contract mode drifted",
            )
            _require(
                manifest.get("support_authoritative_counts")
                == support_audit["authoritative_counts"],
                f"EAST seed{seed} support counts drifted",
            )
            _require(
                manifest.get("support_roster_sha256")
                == support_audit["roster_sha256"],
                f"EAST seed{seed} support roster hash drifted",
            )
            _require(
                manifest.get("support_label_mismatches")
                == support_audit["mismatches"],
                f"EAST seed{seed} support label audit drifted",
            )
    east = [
        _audit_east(args.east_root / f"seed{seed}", seed=seed, fresh=False)
        for seed in (131, 133, 137)
    ]
    fresh_east = _audit_east(args.fresh_east_root, seed=131, fresh=True)
    return {
        "schema_version": "mainline-audit-v1",
        "status": "DIAGNOSTIC_ONLY",
        "claim_boundary": "TEST_DRIVEN_EXPLORATION; no independent future confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "published_root": str(mounted),
            "published_device": os.stat(mounted).st_dev,
            "local_root_device": os.stat("/").st_dev,
            "legacy_local_root_exists": Path("/home/hetaisheng/fuison_ext4_runs").exists(),
            "final_root_on_mounted_device": os.stat(mounted).st_dev != os.stat("/").st_dev,
        },
        "mainline": {
            "jtext_static_is_east_parent": False,
            "jtext_online_state_used_for_east": False,
            "east_static_and_online_share_parent": True,
            "causal_order": "predict -> seal -> query -> reveal -> update",
            "causal_order_detail": CAUSAL_ORDER_DETAIL,
        },
        "east_support_label_audit": support_audit,
        "jtext": jtext,
        "east_three_seed_diagnostic": {
            "runs": east,
            "mean_static_tpr_at_30ms": _mean([row["static"] for row in east], "tpr"),
            "mean_static_fpr": _mean([row["static"] for row in east], "fpr"),
            "mean_online_tpr_at_30ms": _mean([row["online"] for row in east], "tpr"),
            "mean_online_fpr": _mean([row["online"] for row in east], "fpr"),
            "mean_delta_j": _mean([row["delta"] for row in east], "j"),
        },
        "east_fresh_seed131": fresh_east,
        "interpretation": {
            "jtext_active_gain": "delayed queried-feedback expert-weight update; no Transformer representation update",
            "east_active_gain": "retained V3C shot-first parent with chronological intra/inter online path and alarm state machine",
            "formal_admission": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jtext-root", type=Path,
        default=MOUNTED_ROOT / "jtext_adv25_active_canonical_replay_20260831/budget20",
    )
    parser.add_argument(
        "--east-root", type=Path,
        default=MOUNTED_ROOT / "adv25_v3c_success_shotfirst_replay_20260831",
    )
    parser.add_argument(
        "--fresh-east-root", type=Path,
        default=MOUNTED_ROOT / "adv25_v3c_success_shotfirst_replay_20260831/seed131",
    )
    parser.add_argument(
        "--output", type=Path,
        # ``mainline_audit_20260831.json`` is an older class-first receipt.
        # Keep the current shot-first audit on an unambiguous filename so a
        # routine rerun cannot silently overwrite or be confused with it.
        default=MOUNTED_ROOT / "mainline_audit_20260831_v2.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args)
    output = args.output.expanduser().resolve()
    assert_final_artifact_root(output.parent)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "status": report["status"],
        "jtext_online_delta_j": report["jtext"]["delta"]["j"],
        "east_mean_delta_j": report["east_three_seed_diagnostic"]["mean_delta_j"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
