#!/usr/bin/env python3
"""Orchestrate the only selected dynamic-source -> fixed125 EAST path.

The runner is intentionally a fail-closed control plane.  It records exact
commands and hash-bound receipts, but delegates source preprocessing to the
source trainer and EAST replay to the existing specialized runners.  A source
checkpoint without the revised prefix-causal input contract is rejected even
when the source command itself completed successfully.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
DEFAULT_CONFIG = PACKAGE / "configs/primary_chain_dynamic_source_v1.json"
HDF5_CONTENT_MANIFEST = (
    PACKAGE / "evidence/required_hdf5_content_manifest.json"
)
SOURCE_TRAINER = HERE / "train_jtext_source_transformer.py"
UNIFIED_RUNNER = HERE / "run_unified_online_active.py"
JTEXT_ADVANCED_TIME = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import rebuilt_harness as rebuilt  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_two_timescale_online as two_runner  # noqa: E402
import run_unified_online_active as unified  # noqa: E402
import selected_mainline_contract as selected  # noqa: E402
import train_jtext_source_transformer as source_trainer  # noqa: E402
import jtext_online_io as jtext_io  # noqa: E402


RUN_SCHEMA = "selected-full-mainline-run-v1"
RECEIPT_SCHEMA = "selected-mainline-stage-receipt-v1"
STAGES = (
    "source-train", "jtext-static", "east-parent", "east-static", "east-online",
)


class ResumeBindingMismatch(RuntimeError):
    """A finalized selected run no longer matches one of its bound inputs."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_selected_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    checks = (
        (payload.get("schema_version"), "primary-chain-dynamic-source-v1"),
        (payload.get("selected_label_arm"), selected.SELECTED_LABEL_ARM),
        (payload.get("parent_schema"), selected.SELECTED_PARENT_SCHEMA),
        (payload.get("claim_status"), selected.CLAIM_STATUS),
        (payload.get("selection_role"), selected.SELECTION_ROLE),
        (payload.get("historical_admission"), selected.HISTORICAL_ADMISSION),
        (
            payload.get("confirmation_eligible"),
            selected.CONFIRMATION_ELIGIBLE,
        ),
        (
            payload.get("corrected_ablation_report_sha256"),
            selected.CORRECTED_ABLATION_REPORT_SHA256,
        ),
        (payload.get("selection_metric"), selected.SELECTION_METRIC),
        (
            payload.get("historical_failure_reason"),
            selected.HISTORICAL_FAILURE_REASON,
        ),
        (payload.get("source", {}).get("label_policy"), "advanced_floor"),
        (
            payload.get("source", {}).get("input_contract_schema"),
            selected.SOURCE_INPUT_SCHEMA,
        ),
        (payload.get("east", {}).get("target_label_policy"), "fixed125"),
        (payload.get("east", {}).get("target_horizon_ms"), 125.0),
        (payload.get("east", {}).get("threshold_source"), "EAST support110"),
        (payload.get("online", {}).get("query_block_size"), 20),
        (payload.get("online", {}).get("query_quota"), 4),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("selected mainline configuration contract mismatch")
    seeds = payload.get("seeds")
    if seeds != list(selected.ALLOWED_SEEDS):
        raise ValueError("selected mainline seed set mismatch")
    if payload.get("stage_sequence") != list(STAGES):
        raise ValueError("selected mainline stage sequence mismatch")
    return payload


def source_stage_command(
    *, seed: int, output: Path, config: Mapping[str, Any], python: Path,
) -> list[str]:
    source = config["source"]
    return [
        str(Path(python)), str(SOURCE_TRAINER),
        "--seed", str(int(seed)),
        "--output", str(Path(output)),
        "--device", "cuda",
        "--epochs", str(int(source["epochs"])),
        "--minimum-epochs", str(int(source["minimum_epochs"])),
        "--patience", str(int(source["patience"])),
        "--steps-per-epoch", str(int(source["steps_per_epoch"])),
        "--batch-size", str(int(source["batch_size"])),
        "--learning-rate", str(float(source["learning_rate"])),
        "--target-tpr", str(float(source["target_tpr"])),
        "--label-policy", "advanced_floor",
        "--advanced-time-path", str(JTEXT_ADVANCED_TIME),
        "--skip-test",
    ]


def _add_option(command: list[str], name: str, value: Any) -> None:
    command.extend([name, str(value)])


def unified_stage_command(
    *, stage: str, seed: int, output: Path, source_checkpoint: Path,
    parent_state: Path, run_contract: Path, roster_manifest: Path,
    config: Mapping[str, Any], python: Path,
) -> list[str]:
    if stage not in ("east-static", "east-online"):
        raise ValueError("unified selected stage must be east-static or east-online")
    online = config["online"]
    command = [
        str(Path(python)), str(UNIFIED_RUNNER),
        "--seed", str(int(seed)),
        "--out", str(Path(output)),
        "--source-checkpoint", str(Path(source_checkpoint)),
        "--sealed-parent-state", str(Path(parent_state)),
        "--selected-mainline-contract", str(Path(run_contract)),
        "--roster-manifest", str(Path(roster_manifest)),
    ]
    option_names = {
        "max_shots": "--max-shots",
        "private_support_epochs": "--private-support-epochs",
        "adaptation_events": "--adaptation-events",
        "intra_steps": "--intra-steps",
        "inter_steps": "--inter-steps",
        "intra_lr": "--intra-lr",
        "private_intra_lr": "--private-intra-lr",
        "intra_physics_weight": "--intra-physics-weight",
        "inter_lr": "--inter-lr",
        "private_inter_lr": "--private-inter-lr",
        "alarm_lr": "--alarm-lr",
        "query_block_size": "--query-block-size",
        "query_quota": "--query-quota",
        "query_history_size": "--query-history-size",
        "query_quantile": "--query-quantile",
        "query_uncertainty_weight": "--query-uncertainty-weight",
        "query_value_lr": "--query-value-lr",
        "query_initial_cutoff": "--query-initial-cutoff",
        "query_token_capacity": "--query-token-capacity",
    }
    for key, option in option_names.items():
        _add_option(command, option, online[key])
    if bool(online.get("development_search")):
        command.append("--development-search")
    if stage == "east-static":
        command.append("--static-only")
    return command


def make_selected_parent_payload(
    *, contract: selected.SelectedMainlineContract, threshold: float,
    encoder: torch.nn.Module, head: torch.nn.Module,
    shared: torch.nn.Module, private: torch.nn.Module,
    target_label_audit: Mapping[str, Any],
    source_val_threshold_receipt: Mapping[str, Any] | None = None,
    east_threshold_receipt: Mapping[str, Any] | None = None,
    target_scaler: Mapping[str, Any] | None = None,
    raw8_scaler: Mapping[str, Any] | None = None,
    physics_scales: Mapping[str, Any] | None = None,
    reconstruction_constants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if target_label_audit.get("policy") != selected.EAST_TARGET_LABEL_POLICY:
        raise ValueError("selected parent requires a fixed125 target label audit")
    states = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "shared": shared.state_dict(),
        "private": private.state_dict(),
    }
    def content(
        value: Mapping[str, Any] | None, digest: str, name: str,
    ) -> dict[str, Any]:
        resolved = (
            dict(value)
            if value is not None
            else selected.resolve_json_content(digest, name=name)
        )
        if selected.sha256_json(resolved) != digest:
            raise RuntimeError(f"selected parent {name} SHA256 mismatch")
        return resolved

    def receipt(
        value: Mapping[str, Any] | None, digest: str, name: str,
    ) -> dict[str, Any]:
        if value is not None:
            resolved = dict(value)
        else:
            body = selected.resolve_json_content(digest, name=name)
            resolved = {**body, "receipt_sha256": digest}
        if resolved.get("receipt_sha256") != digest:
            raise RuntimeError(f"selected parent {name} SHA256 mismatch")
        return resolved

    source_receipt = receipt(
        source_val_threshold_receipt,
        contract.source_val_threshold_receipt_sha256,
        "source VAL threshold receipt",
    )
    east_receipt = receipt(
        east_threshold_receipt,
        contract.east_threshold_receipt_sha256,
        "EAST support threshold receipt",
    )
    preprocessing = {
        "target_scaler": content(
            target_scaler, contract.target_scaler_sha256, "target scaler",
        ),
        "raw8_scaler": content(
            raw8_scaler, contract.raw8_scaler_sha256, "raw8 scaler",
        ),
        "physics_scales": content(
            physics_scales, contract.physics_scales_sha256, "physics scales",
        ),
        "reconstruction_constants": content(
            reconstruction_constants,
            contract.reconstruction_constants_sha256,
            "reconstruction constants",
        ),
    }
    payload = {
        "schema_version": selected.SELECTED_PARENT_SCHEMA,
        "selected_label_arm": selected.SELECTED_LABEL_ARM,
        "lineage": {"source": "source_dynamic", "target": "east_fixed"},
        **contract.to_dict(),
        "east_threshold": float(threshold),
        "threshold_source": "EAST support110",
        "target_label_audit": dict(target_label_audit),
        "source_val_threshold_receipt": source_receipt,
        "east_threshold_receipt": east_receipt,
        **preprocessing,
        **{
            f"{name}_sha256": getattr(contract, f"{name}_sha256")
            for name in preprocessing
        },
        **states,
        "target_encoder_sha256": selected.state_dict_sha256(states["encoder"]),
        "target_head_sha256": selected.state_dict_sha256(states["head"]),
        "target_base_sha256": selected.state_bundle_sha256(
            states["encoder"], states["head"],
        ),
        "shared_parent_sha256": selected.state_dict_sha256(states["shared"]),
        "private_parent_sha256": selected.state_dict_sha256(states["private"]),
    }
    payload["canonical_parent_fingerprint_sha256"] = (
        selected.canonical_parent_fingerprint(payload)
    )
    return payload


def _write_roster(path: Path, shots: Sequence[int]) -> str:
    ordered = [int(shot) for shot in shots]
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("roster must contain unique shots")
    payload = "".join(
        json.dumps({"stream_index": index, "shot": shot}, sort_keys=True) + "\n"
        for index, shot in enumerate(ordered)
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return selected.sha256_file(path)


def build_selected_east_parent(
    *, seed: int, source_checkpoint: Path,
    contract: selected.SelectedMainlineContract | None, output: Path,
    balanced360_roster: Sequence[int], private_support_epochs: int = 8,
    jtext_test_roster: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit support-only base/shared/private components and serialize a new parent."""

    test_roster = [int(shot) for shot in balanced360_roster]
    if len(test_roster) != 360:
        raise RuntimeError(f"selected balanced360 roster has {len(test_roster)} rows")
    if len(set(test_roster)) != 360:
        raise RuntimeError("selected balanced360 roster must contain unique shots")
    source_path = Path(source_checkpoint).expanduser().resolve()
    _source_encoder, _source_head, source_payload = (
        selected.load_selected_source_checkpoint(source_path, int(seed))
    )
    del _source_encoder, _source_head
    if (
        contract is not None
        and source_payload["source_checkpoint_sha256"]
        != contract.source_checkpoint_sha256
    ):
        raise RuntimeError("selected parent source checkpoint contract mismatch")
    support_parent = a1.train_selected_support_parent(
        seed=int(seed), source_checkpoint=source_path,
        target_label_policy="fixed125", target_tpr=0.80,
    )
    support_ids = sorted(int(shot["shot"]) for shot in support_parent["support"])
    support_roster_sha = selected.sha256_json(support_ids)
    if (
        contract is not None
        and support_roster_sha != contract.east_support_roster_sha256
    ):
        raise RuntimeError("selected EAST support roster SHA256 mismatch")
    shared_result = dha.fit_selected_shared_parent(
        support_parent, seed=int(seed),
    )
    minimum_positive, positive_count = positive_gate.support_positive_threshold(
        shared_result["shared"], shared_result["support_cache"], 0.10,
    )
    median8, scale8 = unified.fit_causal_raw8_scaler(support_ids)
    private, private_batches, private_contract = unified.fit_private_model(
        support_parent["encoder"], support_parent["head"],
        shared_result["shared"], support_parent["support"],
        support_parent["physics_scales"], median8, scale8,
        threshold=float(support_parent["threshold"]),
        minimum_positive=float(minimum_positive), seed=int(seed),
        epochs=int(private_support_epochs),
    )
    exact_support = isinstance(support_parent, a1.SelectedSupportParent)
    parent_options: dict[str, Any] = {}
    if exact_support:
        target_scaler = {
            "center": np.asarray(support_parent["center"]).tolist(),
            "scale": np.asarray(support_parent["scale"]).tolist(),
        }
        raw8_scaler = {
            "center": np.asarray(median8).tolist(),
            "scale": np.asarray(scale8).tolist(),
        }
        reconstruction_constants = {
            "minimum_positive_residual": float(minimum_positive),
            "positive_support_windows": int(positive_count),
            "shared_reconstruction_q80": float(
                unified.shared_reconstruction_scale(
                    shared_result["shared"], shared_result["support_cache"],
                )
            ),
            "private_reconstruction_q80": float(
                private_contract["private_reconstruction_q80"]
            ),
        }
        parent_options = {
            "source_val_threshold_receipt": source_payload[
                "validation_threshold_receipt"
            ],
            "east_threshold_receipt": support_parent["threshold_receipt"],
            "target_scaler": target_scaler,
            "raw8_scaler": raw8_scaler,
            "physics_scales": support_parent["physics_scales"],
            "reconstruction_constants": reconstruction_constants,
        }
        if contract is None:
            if jtext_test_roster is None:
                raise ValueError(
                    "selected parent construction requires explicit J-TEXT TEST220 roster"
                )
            jtext_roster = [int(shot) for shot in jtext_test_roster]
            if len(jtext_roster) != 220 or len(set(jtext_roster)) != 220:
                raise RuntimeError("selected J-TEXT TEST220 roster mismatch")
            contract = selected.contract_from_source(
                source_path,
                source_payload,
                seed=int(seed),
                east_support_roster_sha256=support_parent[
                    "support_roster_sha256"
                ],
                east_support_data_manifest_sha256=support_parent[
                    "support_data_manifest_sha256"
                ],
                east_support_outcome_manifest_sha256=support_parent[
                    "support_outcome_manifest_sha256"
                ],
                east_threshold_receipt_sha256=support_parent[
                    "threshold_receipt"
                ]["receipt_sha256"],
                target_scaler_sha256=selected.sha256_json(target_scaler),
                raw8_scaler_sha256=selected.sha256_json(raw8_scaler),
                physics_scales_sha256=selected.sha256_json(
                    support_parent["physics_scales"]
                ),
                reconstruction_constants_sha256=selected.sha256_json(
                    reconstruction_constants
                ),
                jtext_test_roster_sha256=selected.sha256_json(jtext_roster),
                east_test_roster_sha256=selected.sha256_json(test_roster),
            )
        for name, expected in {
            "support_roster_sha256": contract.east_support_roster_sha256,
            "support_data_manifest_sha256": (
                contract.east_support_data_manifest_sha256
            ),
            "support_outcome_manifest_sha256": (
                contract.east_support_outcome_manifest_sha256
            ),
        }.items():
            if support_parent[name] != expected:
                raise RuntimeError(f"selected EAST {name} mismatch")
    if contract is None:
        raise RuntimeError("selected parent requires a typed support110 result")
    payload = make_selected_parent_payload(
        contract=contract, threshold=float(support_parent["threshold"]),
        encoder=support_parent["encoder"], head=support_parent["head"],
        shared=shared_result["shared"], private=private,
        target_label_audit=support_parent["target_label_audit"],
        **parent_options,
    )
    selected.validate_selected_parent(payload, contract)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"selected EAST parent output exists: {output}")
    output.mkdir(parents=True)
    parent_path = output / "east_fixed_parent.pt"
    torch.save(payload, parent_path)
    if (
        contract.east_test_roster_sha256 is None
        or selected.sha256_json(test_roster) != contract.east_test_roster_sha256
    ):
        raise RuntimeError("selected balanced360 roster SHA256 mismatch")
    roster_path = output / "balanced360_roster.jsonl"
    roster_file_sha = _write_roster(roster_path, test_roster)
    manifest = {
        "schema_version": selected.SELECTED_PARENT_SCHEMA,
        "seed": int(seed),
        "source_checkpoint_sha256": contract.source_checkpoint_sha256,
        "source_input_contract_sha256": contract.source_input_contract_sha256,
        "source_preprocessing_sha256": contract.source_preprocessing_sha256,
        "east_target_label_policy": "fixed125",
        "east_target_horizon_ms": 125.0,
        "east_support_roster_sha256": support_roster_sha,
        "east_threshold": float(support_parent["threshold"]),
        "threshold_source": "EAST support110",
        "minimum_positive_residual": float(minimum_positive),
        "positive_support_windows": int(positive_count),
        "private_support_batches": len(private_batches),
        "private_contract": private_contract,
        "parent_sha256": selected.sha256_file(parent_path),
        "balanced360_roster_file_sha256": roster_file_sha,
        "claim_status": selected.CLAIM_STATUS,
    }
    _write_json(output / "parent_manifest.json", manifest)
    return {
        "payload": payload,
        "parent_path": parent_path,
        "parent_manifest": output / "parent_manifest.json",
        "roster_manifest": roster_path,
        "contract": contract,
    }


def seal_static_predictions_then_read_outcomes(
    *, rows: Sequence[Mapping[str, Any]], seal_path: Path,
    stage: str, expected_count: int,
    outcome_oracle_factory: Callable[[], Callable[[int], Mapping[str, Any]]],
    threshold_receipt_sha256: str, parent_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Seal every prediction and trace before constructing the outcome oracle."""

    records = [dict(row) for row in rows]
    count = int(expected_count)
    if len(records) != count:
        raise RuntimeError(
            f"static seal requires exactly {count} rows, got {len(records)}"
        )
    if [int(row.get("stream_index", -1)) for row in records] != list(range(count)):
        raise RuntimeError(
            f"static seal stream indices must be exactly 0..{count - 1}"
        )
    shots = [int(row["shot"]) for row in records]
    if len(set(shots)) != count:
        raise RuntimeError(f"static seal requires {count} unique shots")

    seal_path = Path(seal_path).expanduser().resolve()
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = seal_path.parent / "risk_traces"
    trace_dir.mkdir(exist_ok=False)
    serialized: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    for row in records:
        stream_index = int(row["stream_index"])
        shot = int(row["shot"])
        if "risk_trace" not in row:
            raise RuntimeError(f"shot {shot} is missing its static risk trace")
        trace_relative = Path("risk_traces") / f"{stream_index:04d}_{shot}.npz"
        trace_sha = selected.write_trace_file(
            seal_path.parent / trace_relative,
            shot=shot,
            risk_trace=row.pop("risk_trace"),
        )
        row["risk_trace_path"] = str(trace_relative)
        row["risk_trace_sha256"] = trace_sha
        serialized.append(row)
        trace_manifest.append({
            "shot": shot, "path": str(trace_relative), "sha256": trace_sha,
        })

    writer = a1.JsonlSealWriter(seal_path)
    try:
        for row in serialized:
            writer.append(row)
    finally:
        if not writer.handle.closed:
            writer.handle.flush()
            os.fsync(writer.handle.fileno())
        writer.close()
    prediction_sha = selected.sha256_file(seal_path)
    manifest_path = seal_path.parent / "static_seal_manifest.json"
    _manifest, manifest_sha = selected.write_static_seal_manifest(
        manifest_path,
        stage=stage,
        prediction_path=seal_path,
        roster=shots,
        risk_traces=trace_manifest,
        threshold_receipt_sha256=threshold_receipt_sha256,
        parent_fingerprint_sha256=parent_fingerprint_sha256,
    )
    selected.validate_static_seal_manifest(
        manifest_path, expected_count=count, expected_stage=stage,
    )

    # This is intentionally the first point at which the lazy outcome oracle
    # exists.  Concrete HDF5 readers normalize their own scalar metadata;
    # globally replacing h5py.Dataset.__getitem__ would be process-wide and
    # could bypass or race access guards in another thread.
    outcome_reader = outcome_oracle_factory()
    outcomes = {shot: dict(outcome_reader(shot)) for shot in shots}
    return {
        "prediction_seal_sha256": prediction_sha,
        "seal_manifest_sha256": manifest_sha,
        "seal_manifest_path": manifest_path,
        "outcomes": outcomes,
    }


def execute_static_stage_boundary(
    *, stage: str, roster: Sequence[int], output: Path,
    prediction_builder: Callable[[int, int], Mapping[str, Any]],
    outcome_oracle_factory: Callable[[], Callable[[int], Mapping[str, Any]]],
    threshold_receipt_sha256: str, parent_fingerprint_sha256: str,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Execute a real static phase with a hard prediction/outcome boundary."""

    selected_counts = {"jtext-static": 220, "east-static": 360}
    if stage not in selected_counts:
        raise ValueError("selected static stage must be jtext-static or east-static")
    count = selected_counts[stage] if expected_count is None else int(expected_count)
    ordered = [int(shot) for shot in roster]
    if len(ordered) != count or len(set(ordered)) != count:
        raise RuntimeError(f"{stage} requires {count} unique roster shots")
    rows: list[dict[str, Any]] = []
    for stream_index, shot in enumerate(ordered):
        row = dict(prediction_builder(stream_index, shot))
        if (
            int(row.get("stream_index", -1)) != stream_index
            or int(row.get("shot", -1)) != shot
        ):
            raise RuntimeError("static prediction builder changed roster identity")
        rows.append(row)

    output = Path(output).expanduser().resolve()
    result = seal_static_predictions_then_read_outcomes(
        rows=rows,
        seal_path=output / "static_predictions.jsonl",
        stage=stage,
        expected_count=count,
        outcome_oracle_factory=outcome_oracle_factory,
        threshold_receipt_sha256=threshold_receipt_sha256,
        parent_fingerprint_sha256=parent_fingerprint_sha256,
    )
    summary = {
        "schema_version": selected.STATIC_SEAL_SCHEMA,
        "stage": stage,
        "row_count": count,
        "prediction_seal_sha256": result["prediction_seal_sha256"],
        "seal_manifest_sha256": result["seal_manifest_sha256"],
        "outcomes_read_after_complete_static_seal": True,
    }
    _write_json(output / "summary.json", summary)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "stage": stage,
        "status": "completed",
        "prediction_seal_sha256": result["prediction_seal_sha256"],
        "seal_manifest_sha256": result["seal_manifest_sha256"],
    }
    with (output / "stage_receipts.jsonl").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {**result, "summary_path": output / "summary.json"}


def run_selected_jtext_static_stage(
    *, source_checkpoint: Path, seed: int, roster_path: Path,
    data_root: Path, output: Path,
) -> dict[str, Any]:
    """Seal label-free TEST220 source predictions before any outcome access."""

    encoder, head, source_payload = selected.load_selected_source_checkpoint(
        source_checkpoint, seed,
    )
    roster = selected.load_label_free_roster(roster_path, expected_count=220)
    center = source_payload["scaler_center"].detach().cpu().numpy()
    scale = source_payload["scaler_scale"].detach().cpu().numpy()
    threshold = float(source_payload["validation_threshold"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError("selected source VAL threshold is invalid")

    def shot_path(shot: int) -> Path:
        return (
            Path(data_root) / "J-TEXT/unified_hdf5"
            / str(int(shot) // 100 * 100) / f"{int(shot)}.hdf5"
        )

    def predict(stream_index: int, shot: int) -> dict[str, Any]:
        signal = jtext_io.load_jtext_signal_only(data_root, shot)
        values = np.asarray(signal.values, dtype=np.float32)
        available = np.asarray(signal.availability, dtype=bool)
        values = np.clip((values - center) / scale, -10.0, 10.0)
        values = np.where(available, values, 0.0).astype(np.float32)
        _endpoints, trace = source_trainer.score_trace(
            encoder, head, values, device="cpu",
        )
        return {
            "stream_index": stream_index,
            "shot": shot,
            "score": float(np.max(trace)),
            "alarmed": bool(np.max(trace) >= threshold),
            "risk_trace": trace,
        }

    def oracle_factory() -> Callable[[int], Mapping[str, Any]]:
        # AdvancedTime is outcome-bearing for this protocol and is loaded only
        # after execute_static_stage_boundary has hashed the complete seal.
        advanced = json.loads(selected.JTEXT_ADVANCED_TIME.read_text())

        def read(shot: int) -> Mapping[str, Any]:
            with h5py.File(shot_path(shot), "r") as handle:
                is_disrupt = bool(
                    np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0]
                )
                down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
            advanced_ms = advanced.get(str(int(shot))) if is_disrupt else None
            return {
                "is_disrupt": is_disrupt,
                "down_time_s": (
                    float(down[0]) if is_disrupt and down.size else None
                ),
                "advanced_time_ms": (
                    float(advanced_ms) if advanced_ms is not None else None
                ),
            }

        return read

    return execute_static_stage_boundary(
        stage="jtext-static",
        roster=roster,
        output=output,
        prediction_builder=predict,
        outcome_oracle_factory=oracle_factory,
        threshold_receipt_sha256=(
            source_payload["source_val_threshold_receipt_sha256"]
        ),
        parent_fingerprint_sha256=source_payload["source_checkpoint_sha256"],
    )


def append_stage_receipt(
    ledger: Path, *, stage: str, binding: Mapping[str, Any],
    outputs: Mapping[str, Path], command: Sequence[str],
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown selected stage: {stage}")
    serialized_outputs: dict[str, dict[str, str]] = {}
    for name, raw_path in outputs.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"stage output does not exist: {path}")
        serialized_outputs[name] = {
            "path": str(path), "sha256": selected.sha256_file(path),
        }
    record = {
        "schema_version": RECEIPT_SCHEMA,
        "stage": stage,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "binding": dict(binding),
        "binding_sha256": selected.sha256_json(dict(binding)),
        "command": [str(value) for value in command],
        "outputs": serialized_outputs,
    }
    ledger = Path(ledger).expanduser().resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _stage_receipts(ledger: Path, stage: str) -> list[dict[str, Any]]:
    path = Path(ledger).expanduser().resolve()
    if not path.is_file():
        return []
    return [
        row for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("stage") == stage and row.get("status") == "completed"
    ]


def require_resumable_stage(
    ledger: Path, *, stage: str, binding: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = _stage_receipts(ledger, stage)
    if not receipts:
        raise FileNotFoundError(f"no completed receipt for selected stage {stage}")
    receipt = receipts[-1]
    if (
        receipt.get("binding") != dict(binding)
        or receipt.get("binding_sha256") != selected.sha256_json(dict(binding))
    ):
        raise RuntimeError(f"{stage} resume binding mismatch")
    for name, output in receipt.get("outputs", {}).items():
        path = Path(output["path"])
        if not path.is_file() or selected.sha256_file(path) != output.get("sha256"):
            raise RuntimeError(f"{stage} output hash mismatch: {name}")
    return receipt


def selected_code_closure_paths() -> tuple[Path, ...]:
    """Files whose current bytes define selected-stage resume compatibility."""

    relative = (
        "advanced_time_labels.py",
        "common.py",
        "dual_threshold_alarm.py",
        "dynamic_hazard_adapter.py",
        "east_private23_sidecar.py",
        "east_private_continual.py",
        "evaluate_dynamic_positive_gate.py",
        "grid_search.py",
        "online_active_audit.py",
        "online_auxiliary.py",
        "online_episode_value_v3.py",
        "online_protected_residual.py",
        "online_timescale_ablation.py",
        "rebuilt_harness.py",
        "reverse_online_value.py",
        "run_dynamic_hazard_adapter.py",
        "run_east_private23_online.py",
        "run_online_active_a1.py",
        "run_online_active_a3.py",
        "run_online_protected_residual.py",
        "run_selected_full_mainline.py",
        "run_two_timescale_online.py",
        "run_unified_online_active.py",
        "selected_mainline_contract.py",
        "shot_alarm.py",
        "train_jtext_source_transformer.py",
        "two_timescale_online_adapter.py",
        "unified_online_value.py",
    )
    return tuple(HERE / name for name in relative) + (
        ROOT / "config/mainline_code_manifest.json",
    )


def _collection_sha256(paths: Sequence[Path]) -> str:
    def identity(raw: Path) -> str:
        path = Path(raw).expanduser().resolve()
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    rows = [
        {
            "index": index,
            "path": identity(Path(path)),
            "sha256": selected.sha256_file(Path(path)),
        }
        for index, path in enumerate(paths)
    ]
    return selected.sha256_json(rows)


def _resume_binding(
    *, config_path: Path, checkpoint_path: Path, parent_path: Path,
    roster_path: Path, threshold_receipt_path: Path,
    scaler_receipt_path: Path, hdf5_paths: Sequence[Path],
    code_files: Sequence[Path],
) -> dict[str, str]:
    named = {
        "config_sha256": config_path,
        "checkpoint_sha256": checkpoint_path,
        "parent_sha256": parent_path,
        "roster_sha256": roster_path,
        "threshold_receipt_sha256": threshold_receipt_path,
        "scaler_receipt_sha256": scaler_receipt_path,
    }
    for path in [*named.values(), *hdf5_paths, *code_files]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"selected resume input does not exist: {path}")
    return {
        "code_closure_sha256": _collection_sha256(code_files),
        **{
            field: selected.sha256_file(Path(path))
            for field, path in named.items()
        },
        "hdf5_contents_sha256": _collection_sha256(hdf5_paths),
    }


def _verify_resume_document(output: Path, binding: Mapping[str, str]) -> None:
    path = output / "run_contract.json"
    if not path.is_file():
        raise ResumeBindingMismatch("resume binding mismatch: run_contract")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeBindingMismatch(
            "resume binding mismatch: run_contract"
        ) from exc
    recorded = document.get("resume_binding")
    if not isinstance(recorded, Mapping):
        raise ResumeBindingMismatch("resume binding mismatch: run_contract")
    for field, actual in binding.items():
        if recorded.get(field) != actual:
            raise ResumeBindingMismatch(f"resume binding mismatch: {field}")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ResumeBindingMismatch("resume binding mismatch: outputs")
    for relative, digest in outputs.items():
        candidate = output / str(relative)
        if not candidate.is_file() or selected.sha256_file(candidate) != digest:
            raise ResumeBindingMismatch(
                f"resume binding mismatch: output:{relative}"
            )


def run_two_shot_cpu_resume_stage(
    *, output: Path, config_path: Path, checkpoint_path: Path,
    parent_path: Path, roster_path: Path, threshold_receipt_path: Path,
    scaler_receipt_path: Path, hdf5_paths: Sequence[Path],
    code_files: Sequence[Path],
) -> dict[str, Any]:
    """Small real-HDF5 stage proving no-write-before-resume-validation."""

    output = Path(output).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    parent_path = Path(parent_path).expanduser().resolve()
    roster_path = Path(roster_path).expanduser().resolve()
    threshold_receipt_path = Path(threshold_receipt_path).expanduser().resolve()
    scaler_receipt_path = Path(scaler_receipt_path).expanduser().resolve()
    hdf5_paths = tuple(Path(path).expanduser().resolve() for path in hdf5_paths)
    code_files = tuple(Path(path).expanduser().resolve() for path in code_files)
    binding = _resume_binding(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        parent_path=parent_path,
        roster_path=roster_path,
        threshold_receipt_path=threshold_receipt_path,
        scaler_receipt_path=scaler_receipt_path,
        hdf5_paths=hdf5_paths,
        code_files=code_files,
    )

    # No mkdir, effective-config write, or contract rewrite occurs before this
    # comparison when the output already exists.
    if output.exists():
        _verify_resume_document(output, binding)
        return {"status": "resumed", "resume_binding": binding}

    roster = selected.load_label_free_roster(
        roster_path, expected_count=len(hdf5_paths),
    )
    by_shot = dict(zip(roster, hdf5_paths))

    def predict(stream_index: int, shot: int) -> dict[str, Any]:
        with h5py.File(by_shot[shot], "r") as handle:
            signal = handle["data/ip"]
            selected.read_required_timebase(signal)
            values = np.asarray(signal[:], dtype=np.float32).reshape(-1)
        scale = float(np.max(np.abs(values))) if len(values) else 1.0
        trace = np.clip(values / max(scale, 1e-6), -1.0, 1.0)
        return {
            "stream_index": stream_index,
            "shot": shot,
            "score": float(np.max(trace)) if len(trace) else 0.0,
            "risk_trace": trace,
        }

    def oracle_factory() -> Callable[[int], Mapping[str, Any]]:
        def read(shot: int) -> Mapping[str, Any]:
            with h5py.File(by_shot[shot], "r") as handle:
                is_disrupt = bool(
                    np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0]
                )
                down = float(
                    np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0]
                )
            return {"is_disrupt": is_disrupt, "down_time_s": down}

        return read

    result = execute_static_stage_boundary(
        stage="east-static",
        roster=roster,
        output=output,
        prediction_builder=predict,
        outcome_oracle_factory=oracle_factory,
        threshold_receipt_sha256=binding["threshold_receipt_sha256"],
        parent_fingerprint_sha256=binding["parent_sha256"],
        expected_count=len(roster),
    )
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    _write_json(output / "effective_config.json", config_payload)
    output_hashes = {
        str(path.relative_to(output)): selected.sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "run_contract.json"
    }
    _write_json(output / "run_contract.json", {
        "schema_version": RUN_SCHEMA,
        "resume_binding": binding,
        "outputs": output_hashes,
        "prediction_seal_sha256": result["prediction_seal_sha256"],
        "seal_manifest_sha256": result["seal_manifest_sha256"],
    })
    return {"status": "completed", "resume_binding": binding}


def _code_sha256() -> str:
    files = selected_code_closure_paths()
    return selected.sha256_json({
        str(path.relative_to(ROOT)): selected.sha256_file(path) for path in files
    })


def bind_runtime_data_root(data_root: Path) -> dict[str, str]:
    """Bind every in-process selected reader to the explicit CLI data root.

    Several historical modules cache ``FUSION_DATA_ROOT`` at import time.  The
    selected orchestrator also builds the EAST parent in-process, so merely
    passing an environment variable to later subprocesses is insufficient.
    This explicit binding is the single selected-path compatibility bridge;
    all values are checked immediately after assignment.
    """

    resolved = Path(data_root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"selected data root does not exist: {resolved}")
    bindings = (
        ("common.DATA", common, "DATA"),
        ("rebuilt_harness.DATA", rebuilt, "DATA"),
        ("train_jtext_source_transformer.DATA_ROOT", source_trainer, "DATA_ROOT"),
        ("run_two_timescale_online.DATA", two_runner, "DATA"),
        (
            "run_two_timescale_online.reverse_online_value.DATA",
            two_runner.reverse_value,
            "DATA",
        ),
        ("run_unified_online_active.DATA", unified, "DATA"),
        (
            "run_unified_online_active.run_east_private23_online.DATA",
            unified.old_private,
            "DATA",
        ),
    )
    os.environ["FUSION_DATA_ROOT"] = str(resolved)
    result: dict[str, str] = {}
    for name, module, attribute in bindings:
        setattr(module, attribute, resolved)
        actual = Path(getattr(module, attribute)).expanduser().resolve()
        if actual != resolved:
            raise RuntimeError(f"selected runtime data-root binding failed: {name}")
        result[name] = str(actual)
    return result


def validate_live_hdf5_content_manifest(
    *, data_root: Path, manifest_path: Path,
    expected_manifest_sha256: str, expected_aggregate_sha256: str,
) -> dict[str, Any]:
    """Rehash every registered HDF5 byte beneath ``data_root`` fail closed."""

    root = Path(data_root).expanduser().resolve()
    path = Path(manifest_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"selected data root does not exist: {root}")
    if not path.is_file():
        raise FileNotFoundError(f"selected HDF5 manifest does not exist: {path}")
    if selected.sha256_file(path) != str(expected_manifest_sha256):
        raise RuntimeError("selected HDF5 content manifest SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("selected HDF5 content manifest is unreadable") from exc
    manifest = payload.get("content_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("selected HDF5 content manifest body is missing")
    rows = manifest.get("files")
    if (
        manifest.get("algorithm") != "sha256"
        or not isinstance(rows, list)
        or int(manifest.get("file_count", -1)) != len(rows)
        or not rows
    ):
        raise RuntimeError("selected HDF5 content manifest schema mismatch")

    aggregate = hashlib.sha256()
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("selected HDF5 content manifest row mismatch")
        relative_text = str(row.get("relative_path", ""))
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise RuntimeError(
                f"selected HDF5 manifest has unsafe/duplicate path: {relative_text!r}"
            )
        seen.add(relative_text)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"selected HDF5 path escapes data root: {relative_text}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"selected HDF5 file does not exist: {candidate}"
            )
        expected_size = int(row.get("bytes", -1))
        actual_size = int(candidate.stat().st_size)
        if actual_size != expected_size:
            raise ResumeBindingMismatch(
                f"HDF5 live content mismatch: {relative_text} (size)"
            )
        actual_sha = selected.sha256_file(candidate)
        if actual_sha != str(row.get("sha256", "")):
            raise ResumeBindingMismatch(
                f"HDF5 live content mismatch: {relative_text} (sha256)"
            )
        aggregate.update(
            f"{relative_text}\0{actual_size}\0{actual_sha}\n".encode("utf-8")
        )
        total_bytes += actual_size
    actual_aggregate = aggregate.hexdigest()
    if (
        actual_aggregate != manifest.get("aggregate_sha256")
        or actual_aggregate != str(expected_aggregate_sha256)
    ):
        raise ResumeBindingMismatch("HDF5 live content aggregate mismatch")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "aggregate_sha256": actual_aggregate,
    }


def _binding_base(
    *, config_path: Path, config: Mapping[str, Any], data_root: Path,
) -> dict[str, Any]:
    hdf5_manifest = HDF5_CONTENT_MANIFEST
    expected_manifest_sha = config["data_contract"][
        "hdf5_content_manifest_sha256"
    ]
    live_hdf5 = validate_live_hdf5_content_manifest(
        data_root=data_root,
        manifest_path=hdf5_manifest,
        expected_manifest_sha256=expected_manifest_sha,
        expected_aggregate_sha256=config["data_contract"][
            "hdf5_content_aggregate_sha256"
        ],
    )
    return {
        "code_sha256": _code_sha256(),
        "config_sha256": selected.sha256_file(config_path),
        "data_sha256": selected.sha256_json({
            "data_root": str(data_root),
            "data_contract": config["data_contract"],
            "live_hdf5_aggregate_sha256": live_hdf5["aggregate_sha256"],
        }),
        "hdf5_content_manifest_sha256": expected_manifest_sha,
        "hdf5_content_aggregate_sha256": config["data_contract"][
            "hdf5_content_aggregate_sha256"
        ],
        "hdf5_live_file_count": live_hdf5["file_count"],
        "hdf5_live_total_bytes": live_hdf5["total_bytes"],
    }


def _configured_roster(
    config: Mapping[str, Any], *, section: str, name: str,
    digest_prefix: str, count: int,
) -> tuple[Path, list[int]]:
    relative = config[section][name]
    path = (ROOT / str(relative)).resolve()
    roster = selected.load_label_free_roster(path, expected_count=count)
    expected_file = config["data_contract"][f"{digest_prefix}_file_sha256"]
    expected_roster = config["data_contract"][f"{digest_prefix}_sha256"]
    if selected.sha256_file(path) != expected_file:
        raise RuntimeError(f"selected {section} {name} file SHA256 mismatch")
    if selected.sha256_json(roster) != expected_roster:
        raise RuntimeError(f"selected {section} {name} SHA256 mismatch")
    return path, roster


def _source_binding(
    *, base: Mapping[str, Any], source_payload: Mapping[str, Any],
    jtext_roster: Sequence[int], support_roster: Sequence[int],
    east_roster: Sequence[int], parent_sha256: str | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data = config["data_contract"]
    return {
        **dict(base),
        "source_split_manifest_sha256": source_payload["split_sha256"],
        "source_label_map_sha256": source_payload["source_label_map_sha256"],
        "source_checkpoint_sha256": source_payload[
            "source_checkpoint_sha256"
        ],
        "source_input_contract_sha256": source_payload[
            "source_input_contract_sha256"
        ],
        "source_preprocessing_sha256": source_payload[
            "source_preprocessing_sha256"
        ],
        "jtext_test_roster_sha256": selected.sha256_json(jtext_roster),
        "east_support_roster_sha256": selected.sha256_json(support_roster),
        "east_test_roster_sha256": selected.sha256_json(east_roster),
        "source_label_manifest_sha256": data["jtext_advanced_time_sha256"],
        "east_support_manifest_sha256": data[
            "east_support_manifest_sha256"
        ],
        "east_test_hdf5_manifest_sha256": data[
            "east_balanced360_manifest_sha256"
        ],
        "parent_sha256": parent_sha256,
    }


def _load_run_contract(
    path: Path,
) -> tuple[selected.SelectedMainlineContract, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("selected run contract is missing or unreadable") from exc
    contract = selected.SelectedMainlineContract.from_dict(
        document.get("selected_contract", {})
    )
    return contract, document


def _run_subprocess(command: Sequence[str], *, data_root: Path) -> None:
    environment = dict(os.environ)
    environment["FUSION_DATA_ROOT"] = str(data_root)
    environment["FUSION_PROJECT_ROOT"] = str(ROOT)
    subprocess.run(
        [str(value) for value in command], check=True, cwd=ROOT, env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=selected.ALLOWED_SEEDS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=(*STAGES, "all"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"selected data root does not exist: {data_root}")
    bind_runtime_data_root(data_root)
    config_path = args.config.expanduser().resolve()
    config = load_selected_config(config_path)
    jtext_roster_path, jtext_roster = _configured_roster(
        config, section="source", name="test_roster",
        digest_prefix="jtext_test_roster", count=220,
    )
    support_roster_path, support_roster = _configured_roster(
        config, section="east", name="support_roster",
        digest_prefix="east_support_roster", count=110,
    )
    east_roster_path, east_roster = _configured_roster(
        config, section="east", name="test_roster",
        digest_prefix="east_balanced360_roster", count=360,
    )
    if support_roster_path != common.ROOT / "config/splits/east_support110_label_free_roster.jsonl":
        raise RuntimeError("selected support loader/config roster path mismatch")

    output = args.output.expanduser().resolve()
    effective_config = output / "effective_config.json"
    # Recompute current code, roster/config and every external HDF5 byte before
    # creating or modifying the output tree.  This ordering is part of the
    # selected resume contract, not an optimization.
    base_binding = _binding_base(
        config_path=config_path, config=config, data_root=data_root,
    )
    if output.exists():
        if not effective_config.is_file():
            raise ResumeBindingMismatch(
                "resume binding mismatch: effective_config"
            )
        if json.loads(effective_config.read_text()) != config:
            raise ResumeBindingMismatch(
                "resume binding mismatch: effective_config"
            )
    else:
        output.mkdir(parents=True)
        _write_json(effective_config, config)
    ledger = output / "stage_receipts.jsonl"
    run_contract_path = output / "run_contract.json"
    source_dir = output / "source-train"
    source_path = source_dir / "source_checkpoint.pt"
    parent_dir = output / "east-parent"
    parent_path = parent_dir / "east_fixed_parent.pt"
    roster_path = parent_dir / "balanced360_roster.jsonl"
    requested = STAGES if args.stage == "all" else (args.stage,)

    source_command = source_stage_command(
        seed=args.seed, output=source_dir, config=config,
        python=Path(sys.executable),
    )
    if "source-train" in requested:
        binding = {
            **base_binding,
            "source_split_manifest_sha256": config["data_contract"][
                "jtext_train_sha256"
            ],
            "source_label_map_sha256": config["data_contract"][
                "jtext_advanced_time_sha256"
            ],
            "jtext_test_roster_sha256": selected.sha256_json(jtext_roster),
            "east_support_roster_sha256": selected.sha256_json(support_roster),
            "east_test_roster_sha256": selected.sha256_json(east_roster),
            "hdf5_content_manifest_sha256": base_binding[
                "hdf5_content_manifest_sha256"
            ],
            "parent_sha256": None,
        }
        receipts = _stage_receipts(ledger, "source-train")
        if receipts:
            require_resumable_stage(
                ledger, stage="source-train", binding=binding,
            )
        else:
            if source_dir.exists() and any(source_dir.iterdir()):
                raise ResumeBindingMismatch(
                    "source-train has unreceipted pre-existing output"
                )
            _run_subprocess(source_command, data_root=data_root)
            # Revised Task 2 must add input_contract.  Current legacy dynamic
            # checkpoints stop here rather than entering EAST silently.
            selected.load_selected_source_checkpoint(source_path, args.seed)
            append_stage_receipt(
                ledger, stage="source-train", binding=binding,
                outputs={"source_checkpoint": source_path},
                command=source_command,
            )

    _source_encoder, _source_head, source_payload = (
        selected.load_selected_source_checkpoint(source_path, args.seed)
    )
    del _source_encoder, _source_head
    source_binding = _source_binding(
        base=base_binding,
        source_payload=source_payload,
        jtext_roster=jtext_roster,
        support_roster=support_roster,
        east_roster=east_roster,
        parent_sha256=None,
        config=config,
    )

    if "jtext-static" in requested:
        receipts = _stage_receipts(ledger, "jtext-static")
        if receipts:
            require_resumable_stage(
                ledger, stage="jtext-static", binding=source_binding,
            )
        else:
            result = run_selected_jtext_static_stage(
                source_checkpoint=source_path,
                seed=args.seed,
                roster_path=jtext_roster_path,
                data_root=data_root,
                output=output / "jtext-static",
            )
            append_stage_receipt(
                ledger,
                stage="jtext-static",
                binding=source_binding,
                outputs={
                    "static_predictions": output / "jtext-static/static_predictions.jsonl",
                    "static_seal_manifest": result["seal_manifest_path"],
                    "summary": result["summary_path"],
                },
                command=[str(Path(sys.executable)), str(Path(__file__)), "--stage", "jtext-static"],
            )

    if "east-parent" in requested:
        if not _stage_receipts(ledger, "jtext-static"):
            raise RuntimeError("east-parent requires completed jtext-static receipt")
        receipts = _stage_receipts(ledger, "east-parent")
        if receipts:
            require_resumable_stage(
                ledger, stage="east-parent", binding=source_binding,
            )
        else:
            result = build_selected_east_parent(
                seed=args.seed, source_checkpoint=source_path,
                contract=None, output=parent_dir,
                balanced360_roster=east_roster,
                jtext_test_roster=jtext_roster,
                private_support_epochs=int(config["east"]["private_support_epochs"]),
            )
            contract = result["contract"]
            run_document: dict[str, Any] = {
                "schema_version": RUN_SCHEMA,
                "seed": args.seed,
                "selected_label_arm": selected.SELECTED_LABEL_ARM,
                "selection_role": selected.SELECTION_ROLE,
                "historical_admission": selected.HISTORICAL_ADMISSION,
                "confirmation_eligible": selected.CONFIRMATION_ELIGIBLE,
                "claim_status": selected.CLAIM_STATUS,
                "config_sha256": selected.sha256_file(config_path),
                "code_sha256": _code_sha256(),
                "selected_contract": contract.to_dict(),
                "source_checkpoint_sha256": contract.source_checkpoint_sha256,
                "source_input_contract": source_payload["input_contract"],
                "target_parent_sha256": selected.sha256_file(parent_path),
            }
            _write_json(run_contract_path, run_document)
            append_stage_receipt(
                ledger, stage="east-parent", binding=source_binding,
                outputs={
                    "parent_state": result["parent_path"],
                    "parent_manifest": result["parent_manifest"],
                    "balanced360_roster": result["roster_manifest"],
                },
                command=[
                    str(Path(sys.executable)), str(Path(__file__)),
                    "--seed", str(args.seed), "--stage", "east-parent",
                ],
            )
    if any(stage in requested for stage in ("east-static", "east-online")):
        if not parent_path.is_file() or not roster_path.is_file():
            raise FileNotFoundError("selected EAST parent stage is incomplete")
        contract, run_document = _load_run_contract(run_contract_path)
        parent_payload = torch.load(
            parent_path, map_location="cpu", weights_only=True,
        )
        selected.validate_selected_parent(parent_payload, contract)
        parent_sha = selected.sha256_file(parent_path)
        if run_document.get("target_parent_sha256") != parent_sha:
            raise RuntimeError("run contract target parent SHA256 mismatch")
        actual_roster = selected.load_label_free_roster(
            roster_path, expected_count=360,
        )
        if actual_roster != east_roster:
            raise RuntimeError("selected parent/test roster mismatch")
        phase_binding = _source_binding(
            base=base_binding,
            source_payload=source_payload,
            jtext_roster=jtext_roster,
            support_roster=support_roster,
            east_roster=east_roster,
            parent_sha256=parent_sha,
            config=config,
        )

    if "east-static" in requested:
        static_dir = output / "east-static"
        command = unified_stage_command(
            stage="east-static", seed=args.seed, output=static_dir,
            source_checkpoint=source_path, parent_state=parent_path,
            run_contract=run_contract_path, roster_manifest=roster_path,
            config=config, python=Path(sys.executable),
        )
        receipts = _stage_receipts(ledger, "east-static")
        if receipts:
            require_resumable_stage(
                ledger, stage="east-static", binding=phase_binding,
            )
        else:
            _run_subprocess(command, data_root=data_root)
            append_stage_receipt(
                ledger, stage="east-static", binding=phase_binding,
                outputs={
                    "static_predictions": static_dir / "static_predictions.jsonl",
                    "static_seal_manifest": static_dir / "static_seal_manifest.json",
                    "summary": static_dir / "summary.json",
                },
                command=command,
            )
        command_path = output / "effective_command_east_static.json"
        if not command_path.exists():
            _write_json(command_path, command)

    if "east-online" in requested:
        static_receipts = _stage_receipts(ledger, "east-static")
        if not static_receipts:
            raise RuntimeError("online stage requires a completed static stage receipt")
        static_manifest = Path(
            static_receipts[-1]["outputs"]["static_seal_manifest"]["path"]
        )
        selected.validate_static_seal_manifest(
            static_manifest, expected_count=360, expected_stage="east-static",
        )
        static_seal_sha = static_receipts[-1]["outputs"][
            "static_seal_manifest"
        ]["sha256"]
        online_dir = output / "east-online"
        command = unified_stage_command(
            stage="east-online", seed=args.seed, output=online_dir,
            source_checkpoint=source_path, parent_state=parent_path,
            run_contract=run_contract_path, roster_manifest=roster_path,
            config=config, python=Path(sys.executable),
        )
        online_binding = {
            **phase_binding,
            "static_seal_sha256": static_seal_sha,
        }
        receipts = _stage_receipts(ledger, "east-online")
        if receipts:
            require_resumable_stage(
                ledger, stage="east-online", binding=online_binding,
            )
        else:
            _run_subprocess(command, data_root=data_root)
            append_stage_receipt(
                ledger, stage="east-online", binding=online_binding,
                outputs={
                    "static_predictions": online_dir / "static_predictions.jsonl",
                    "predictions": online_dir / "predictions.jsonl",
                    "summary": online_dir / "summary.json",
                },
                command=command,
            )
        command_path = output / "effective_command_east_online.json"
        if not command_path.exists():
            _write_json(command_path, command)

    print(json.dumps({
        "status": "SELECTED_STAGE_COMPLETE",
        "stage": args.stage,
        "run_contract": str(run_contract_path),
        "claim_status": selected.CLAIM_STATUS,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
