#!/usr/bin/env python3
"""Factorial J-TEXT online adaptation audit for the Adv25 source checkpoint.

This is an independent diagnostic runner.  It keeps one static, outcome-blind
query manifest fixed for all four arms and then runs the ordered TEST220 stream
with separate adapter state:

    static, intra_only, inter_only, intra_inter

The J-TEXT online branch is never exported to EAST.  The purpose of this
runner is attribution, not to select a new parent or to claim an independent
test result.  Labels are exposed by the oracle only for selected shots after
the prediction and query decision have been sealed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import run_jtext_rebuilt17_online as online  # noqa: E402
import train_jtext_source_transformer as SOURCE  # noqa: E402


MODES = ("static", "intra_only", "inter_only", "intra_inter")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def module_hash(module: nn.Module) -> str:
    return online.module_sha256(module)


def set_deterministic(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")


def build_static_query_manifest(
    encoder: nn.Module,
    head: nn.Module,
    test_ids: list[int],
    center: np.ndarray,
    scale: np.ndarray,
    *,
    threshold: float,
    budget_fraction: float,
    query_cutoff: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a label-free manifest once, before any mode can update."""
    selector = online.CausalBlockBudgetSelector(
        total_shots=len(test_ids), fraction=budget_fraction, cutoff=query_cutoff,
        block_size=20,
    )
    history: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    static_records: list[dict[str, Any]] = []
    for index, shot_id in enumerate(test_ids):
        shot = online.load_signal_only(shot_id, center, scale)
        endpoints, _embeddings, logits = online.causal_embeddings(
            encoder, head, shot["v"], device=device,
        )
        risk = torch.sigmoid(torch.from_numpy(logits)).numpy()
        signature = online.shot_signature(shot["v"], shot["a"])
        novelty = online.normalized_novelty(signature, history)
        components = online.query_components(
            risk, logits, threshold, novelty,
        )
        selected, reason = selector.decide(
            score=components["query_score"], stream_index=index,
        )
        alarm_index = online.first_alarm(endpoints, risk, threshold=threshold)
        static_records.append({
            "stream_index": index,
            "shot": int(shot_id),
            "score": float(np.max(risk)) if len(risk) else 0.0,
            "alarm_index": alarm_index,
            "endpoints": endpoints.tolist(),
            "start": float(shot["start"]),
            "rate": float(shot["rate"]),
            "threshold": float(threshold),
        })
        manifest.append({
            "stream_index": index,
            "block": int(index // 20),
            "position": int(index % 20),
            "shot": int(shot_id),
            "query_score": float(components["query_score"]),
            "components": components,
            "selected": bool(selected),
            "selection_reason": reason,
            "label_access_before_decision": False,
        })
        history.append(signature)
    if selector.selected != selector.total_budget:
        raise RuntimeError(
            f"J-TEXT query manifest budget failed: {selector.selected} != "
            f"{selector.total_budget}"
        )
    return manifest, static_records


def score_mode_shot(
    mode: str,
    *,
    persistent: online.LowRankRiskAdapter,
    encoder: nn.Module,
    head: nn.Module,
    shot: dict[str, Any],
    device: str,
    threshold: float,
    intra_interval: int,
    intra_learning_rate: float,
    intra_steps: int,
    intra_max_relative_change: float,
    intra_risk_forecast_weight: float,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Score one shot with an optional resettable fast adapter."""
    endpoints, embeddings, base_logits = online.causal_embeddings(
        encoder, head, shot["v"], device=device,
    )
    base_risk = torch.sigmoid(torch.from_numpy(base_logits)).numpy()
    if mode == "static":
        # The static arm has no fast state.  Recording the persistent hash is
        # explicit and unambiguous for downstream lineage checks.
        return base_risk.copy(), [], module_hash(persistent)

    fast = copy.deepcopy(persistent).to(device)
    fast.eval()
    risk = np.empty(len(endpoints), dtype=np.float32)
    receipts: list[dict[str, Any]] = []
    history_z: list[np.ndarray] = []
    history_base: list[np.ndarray] = []
    interval = max(int(intra_interval), 1)
    for point in range(len(endpoints)):
        # Seal the current prediction before using that observation to adapt
        # the state for the next endpoint.
        with torch.no_grad():
            adjusted, _residual = fast(
                torch.from_numpy(embeddings[point:point + 1]).to(device),
                torch.from_numpy(base_logits[point:point + 1]).to(device),
            )
        risk[point] = float(torch.sigmoid(adjusted[0]).cpu())
        history_z.append(embeddings[point].copy())
        history_base.append(base_logits[point:point + 1].copy())
        if (
            mode in ("intra_only", "intra_inter")
            and point >= 1
            and point + 1 < len(endpoints)
            and point % interval == 0
        ):
            receipt = online.intra_update(
                fast,
                embeddings[point - 1], embeddings[point],
                history_z[:-1], history_base[:-1],
                device=device,
                learning_rate=intra_learning_rate,
                steps=intra_steps,
                max_relative_change=intra_max_relative_change,
                current_base_logit=base_logits[point],
                risk_forecast_weight=intra_risk_forecast_weight,
            )
            receipt.update({"shot": int(shot["shot"]), "point": point})
            receipts.append(receipt)
    # Capture the resettable fast state before the shot is sealed.  This must
    # not be replaced by the persistent hash: the latter can remain unchanged
    # even when accepted intra-shot updates were applied.
    return risk, receipts, module_hash(fast)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("J-TEXT ablation requires CUDA in rl_env")
    set_deterministic(args.seed)
    device = args.device
    checkpoint_path = args.source_checkpoint.expanduser().resolve()
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(online.RH.REBUILD),
        expected_seed=args.seed, device=device,
    )
    online.validate_advanced_source_contract(
        checkpoint, SOURCE.JTEXT_ADVANCED_TIME_PATH,
    )
    center = checkpoint["scaler_center"].cpu().numpy().astype(np.float32)
    scale = checkpoint["scaler_scale"].cpu().numpy().astype(np.float32)
    roles = online.load_roles()
    test_ids = roles["test"]
    threshold = float(checkpoint["validation_threshold"])
    parent_hash = module_hash(nn.Sequential(encoder, head))

    # The validation guard is legal source-side data and is identical for all
    # arms.  TEST220 is not opened here.
    validation = SOURCE.load_jtext_shots(roles["val"])
    GS.apply_scale(validation, center, scale)
    guard = online.build_guard(encoder, head, validation, device=device)

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    manifest, static_records = build_static_query_manifest(
        encoder, head, test_ids, center, scale,
        threshold=threshold,
        budget_fraction=args.budget_fraction,
        query_cutoff=args.query_cutoff,
        device=device,
    )
    manifest_by_shot = {int(row["shot"]): row for row in manifest}
    if list(manifest_by_shot) != test_ids:
        raise RuntimeError("query manifest does not preserve TEST220 order")
    manifest_writer = online.JsonlWriter(output / "query_manifest.jsonl")
    try:
        for row in manifest:
            manifest_writer.append(row)
    finally:
        manifest_writer.close()
    static_writer = online.JsonlWriter(output / "static_predictions.jsonl")
    try:
        for row in static_records:
            static_writer.append(row)
    finally:
        static_writer.close()

    # All modes start from byte-identical adapter parameters.  Static does not
    # use the adapter, but retaining an explicit empty state makes the lineage
    # and hash checks uniform.
    torch.manual_seed(int(args.seed) + 9000)
    template = online.LowRankRiskAdapter().to(device)
    initial_state = copy.deepcopy(template.state_dict())
    persistent = {}
    for mode in MODES:
        item = online.LowRankRiskAdapter().to(device)
        item.load_state_dict(initial_state, strict=True)
        persistent[mode] = item
    initial_adapter_hashes = {mode: module_hash(item) for mode, item in persistent.items()}

    writers = {
        mode: online.JsonlWriter(output / f"predictions_{mode}.jsonl")
        for mode in MODES
    }
    update_writer = online.JsonlWriter(output / "updates.jsonl")
    reveal_writer = online.JsonlWriter(output / "reveals.jsonl")
    oracle = online.OutcomeOracle(test_ids)
    advanced = online.LazyAdvancedTime(SOURCE.JTEXT_ADVANCED_TIME_PATH)
    replays: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    update_counts: dict[str, dict[str, int]] = {
        mode: {
            "intra_attempts": 0,
            "intra_applied": 0,
            "inter_accepted": 0,
            "inter_rollbacks": 0,
            "correct_reveals": 0,
        }
        for mode in MODES
    }

    try:
        for stream_index, shot_id in enumerate(test_ids):
            shot = online.load_signal_only(shot_id, center, scale)
            shot_risks: dict[str, np.ndarray] = {}
            shot_rows: dict[str, dict[str, Any]] = {}
            for mode in MODES:
                risk, receipts, fast_hash_at_seal = score_mode_shot(
                    mode,
                    persistent=persistent[mode],
                    encoder=encoder,
                    head=head,
                    shot=shot,
                    device=device,
                    threshold=threshold,
                    intra_interval=args.intra_interval,
                    intra_learning_rate=args.intra_learning_rate,
                    intra_steps=args.intra_steps,
                    intra_max_relative_change=args.intra_max_relative_change,
                    intra_risk_forecast_weight=args.intra_risk_forecast_weight,
                )
                shot_risks[mode] = risk
                update_counts[mode]["intra_attempts"] += len(receipts)
                update_counts[mode]["intra_applied"] += sum(
                    bool(item.get("applied")) for item in receipts
                )
                for receipt in receipts:
                    update_writer.append({"mode": mode, "kind": "intra", **receipt})
                alarm_index = online.first_alarm(
                    np.arange(len(risk), dtype=np.int64), risk, threshold=threshold,
                )
                row = {
                    "stream_index": stream_index,
                    "shot": int(shot_id),
                    "score": float(np.max(risk)) if len(risk) else 0.0,
                    "alarm_index": alarm_index,
                    "endpoints": list(range(len(risk))),
                    "start": float(shot["start"]),
                    "rate": float(shot["rate"]),
                    "threshold": threshold,
                    "queried": False,
                    "persistent_adapter_sha256_before": module_hash(persistent[mode]),
                    "fast_adapter_sha256_at_seal": fast_hash_at_seal,
                    "intra_update_count": len(receipts),
                }
                # Store actual endpoint indices: the J-TEXT source trace has
                # one risk value for each endpoint, starting at zero.
                row["endpoints"] = list(range(len(risk)))
                writers[mode].append(row)
                records[mode].append(row)
                shot_rows[mode] = row

            oracle.seal(shot_id)
            query = manifest_by_shot[shot_id]
            selected = bool(query["selected"])
            for mode in MODES:
                shot_rows[mode]["queried"] = selected
            if selected:
                oracle.query(shot_id)
                outcome = oracle.reveal(shot_id)
                advanced_time = (
                    advanced.lookup_after_reveal(shot_id)
                    if bool(outcome["is_disrupt"]) else None
                )
                reveal_writer.append({
                    "stream_index": stream_index,
                    "shot": int(shot_id),
                    "is_disrupt": bool(outcome["is_disrupt"]),
                    "down_time_s": outcome["down_time_s"],
                    "advanced_time_ms": advanced_time,
                })
                for mode in MODES:
                    group, lead_ms = online.outcome_group(
                        is_disrupt=bool(outcome["is_disrupt"]),
                        down_time_s=outcome["down_time_s"],
                        start=shot["start"], rate=shot["rate"],
                        endpoints=np.arange(len(shot_risks[mode]), dtype=np.int64),
                        risk=shot_risks[mode], threshold=threshold,
                    )
                    shot_rows[mode]["revealed_group"] = group
                    shot_rows[mode]["lead_ms"] = lead_ms
                    if mode in ("inter_only", "intra_inter"):
                        endpoints = np.arange(len(shot_risks[mode]), dtype=np.int64)
                        embeddings_end, embeddings, base_logits = online.causal_embeddings(
                            encoder, head, shot["v"], device=device,
                        )
                        if not np.array_equal(embeddings_end, endpoints):
                            raise RuntimeError("J-TEXT endpoint cache changed during reveal")
                        bag = online.labelled_bag(
                            shot, outcome, advanced_time, embeddings, base_logits,
                            endpoints, max_windows=args.max_bag_windows,
                        )
                        receipt: dict[str, Any]
                        if bag is None:
                            receipt = {
                                "updated": False,
                                "reason": "no_usable_labelled_windows",
                                "group": group,
                            }
                        else:
                            receipt = online.persistent_update(
                                persistent[mode], bag, replays[mode], guard,
                                group=group,
                                device=device,
                                threshold=threshold,
                                learning_rate=args.inter_learning_rate,
                                steps=args.inter_steps,
                                max_relative_change=args.inter_max_relative_change,
                                max_guard_harms=args.max_guard_harms,
                            )
                            replays[mode].append({"bag": bag, "group": group})
                            if len(replays[mode]) > int(args.max_replay_shots):
                                replays[mode] = replays[mode][-int(args.max_replay_shots):]
                        update_counts[mode]["inter_accepted"] += int(
                            bool(receipt.get("updated"))
                        )
                        update_counts[mode]["inter_rollbacks"] += int(
                            receipt.get("reason") == "guard_or_change_rollback"
                        )
                        update_writer.append({
                            "mode": mode, "kind": "inter", "shot": int(shot_id),
                            "group": group, "lead_ms": lead_ms,
                            "receipt": receipt,
                            "persistent_adapter_sha256_after": module_hash(persistent[mode]),
                        })
                    else:
                        update_counts[mode]["correct_reveals"] += int(group in ("TP", "TN"))
                        update_writer.append({
                            "mode": mode, "kind": "inter", "shot": int(shot_id),
                            "group": group, "lead_ms": lead_ms,
                            "receipt": {"updated": False, "reason": "ablation_disabled"},
                        })
            else:
                oracle.discard(shot_id)
                for mode in MODES:
                    update_writer.append({
                        "mode": mode, "kind": "inter", "shot": int(shot_id),
                        "updated": False, "reason": "unqueried_no_label_access",
                    })
            if (stream_index + 1) % 20 == 0:
                print(
                    f"[JTEXT-ABLATION] sealed {stream_index + 1}/{len(test_ids)} "
                    f"queries={sum(int(row['selected']) for row in manifest[:stream_index + 1])}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()
        update_writer.close()
        reveal_writer.close()

    if sum(int(row["selected"]) for row in manifest) != round(
        len(test_ids) * float(args.budget_fraction)
    ):
        raise RuntimeError("fixed J-TEXT query budget changed during replay")
    oracle.open_final_evaluation()
    outcomes = online.read_all_outcomes(test_ids)
    results: dict[str, Any] = {}
    for mode in MODES:
        metrics = online.metric_from_records(
            records[mode], outcomes, threshold=threshold,
        )
        metrics["mean_abs_score_change_vs_static"] = float(np.mean(np.abs(
            np.asarray([row["score"] for row in records[mode]], dtype=np.float64)
            - np.asarray([row["score"] for row in static_records], dtype=np.float64)
        )))
        metrics["initial_adapter_sha256"] = initial_adapter_hashes[mode]
        metrics["final_adapter_sha256"] = module_hash(persistent[mode])
        metrics["updates"] = update_counts[mode]
        results[mode] = metrics

    if module_hash(nn.Sequential(encoder, head)) != parent_hash:
        raise RuntimeError("J-TEXT ablation changed the frozen source model")
    if results["static"]["final_adapter_sha256"] != results["static"]["initial_adapter_sha256"]:
        raise RuntimeError("static ablation arm changed adapter state")
    if results["intra_only"]["final_adapter_sha256"] != results["intra_only"]["initial_adapter_sha256"]:
        raise RuntimeError("intra-only arm leaked persistent state across shots")

    summary = {
        "status": "JTEXT_ONLINE_ABLATION_DIAGNOSTIC",
        "stage": "JTEXT_ADVANCED_FLOOR25_FIXED_QUERY_FACTORIAL",
        "seed": int(args.seed),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_model_sha256_before_after": parent_hash,
        "stream": {
            "role": "TEST220",
            "shots": len(test_ids),
            "order": "ascending_shot_id",
        },
        "threshold": threshold,
        "threshold_source": "J-TEXT VAL160 embedded in source checkpoint",
        "query_contract": {
            "budget_fraction": float(args.budget_fraction),
            "budget": int(sum(int(row["selected"]) for row in manifest)),
            "cutoff": float(args.query_cutoff),
            "block_size": 20,
            "per_block_quota": int(round(20 * float(args.budget_fraction))),
            "manifest_shared_by_all_arms": True,
            "selection_label_free": True,
        },
        "adapter_contract": {
            "parameters": online.adapter_parameter_count(template),
            "intra_scope": "resettable fast low-rank adapter; no persistent write",
            "inter_scope": "persistent low-rank adapter after selected reveal only",
            "intra_learning_rate": float(args.intra_learning_rate),
            "inter_learning_rate": float(args.inter_learning_rate),
        "intra_steps": int(args.intra_steps),
        "intra_risk_forecast_weight": float(args.intra_risk_forecast_weight),
        "inter_steps": int(args.inter_steps),
        },
        "results": results,
        "causal_contract": {
            "prediction_order": (
                "signal prefix -> prediction/seal -> causal fast update for the "
                "next endpoint -> fixed query -> reveal -> later-shot update"
            ),
            "test_outcomes_read_before_query": False,
            "unqueried_outcomes_revealed_during_stream": False,
            "jtext_online_state_exported_to_east": False,
            "all_test_labels_opened_only_after_stream": True,
        },
        "hashes": {
            "query_manifest": sha256_file(output / "query_manifest.jsonl"),
            "static_predictions": sha256_file(output / "static_predictions.jsonl"),
            "updates": sha256_file(output / "updates.jsonl"),
            "reveals": sha256_file(output / "reveals.jsonl"),
        },
        "claim_boundary": (
            "TEST220 was used for method-development diagnostics.  These paired "
            "results attribute intra/inter effects under a fixed manifest but are "
            "not independent future-stream confirmation."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {
        "seed": int(args.seed),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "budget_fraction": float(args.budget_fraction),
        "query_cutoff": float(args.query_cutoff),
        "intra_interval": int(args.intra_interval),
        "intra_learning_rate": float(args.intra_learning_rate),
        "intra_steps": int(args.intra_steps),
        "intra_max_relative_change": float(args.intra_max_relative_change),
        "inter_learning_rate": float(args.inter_learning_rate),
        "inter_steps": int(args.inter_steps),
        "inter_max_relative_change": float(args.inter_max_relative_change),
        "max_guard_harms": int(args.max_guard_harms),
        "max_bag_windows": int(args.max_bag_windows),
        "max_replay_shots": int(args.max_replay_shots),
    })
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--query-cutoff", type=float, default=0.50)
    parser.add_argument("--intra-interval", type=int, default=8)
    parser.add_argument("--intra-learning-rate", type=float, default=1e-4)
    parser.add_argument("--intra-steps", type=int, default=1)
    parser.add_argument("--intra-max-relative-change", type=float, default=1e-3)
    parser.add_argument("--intra-risk-forecast-weight", type=float, default=0.5)
    parser.add_argument("--inter-learning-rate", type=float, default=3e-4)
    parser.add_argument("--inter-steps", type=int, default=2)
    parser.add_argument("--inter-max-relative-change", type=float, default=5e-2)
    parser.add_argument("--max-guard-harms", type=int, default=0)
    parser.add_argument("--max-bag-windows", type=int, default=16)
    parser.add_argument("--max-replay-shots", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
