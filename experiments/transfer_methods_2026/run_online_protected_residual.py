#!/usr/bin/env python3
"""Causal EAST stream screen with protected inter-shot residual updates.

This is a test-driven diagnostic replay of balanced360.  A1 query decisions
are consumed exactly as previously sealed.  The current shot is predicted in
full before its outcome/DownTime can be read.  Unqueried outcomes remain in an
OutcomeVault until final evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dynamic_hazard_adapter as dynamic  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import online_protected_residual as protected  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_active_a3 as a3  # noqa: E402
import shot_alarm as alarm  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_a1_query_contract(
    seals: list[dict], queries: list[dict], *, expected_queries: int = 72
) -> dict[int, dict]:
    """Join pre-decision seals with selected receipts without copying outcomes."""
    merged = {int(row["shot"]): dict(row) for row in seals}
    if len(merged) != len(seals):
        raise RuntimeError("duplicate shots in A1 seals")
    selected = {int(row["shot"]): row for row in queries}
    if len(selected) != expected_queries:
        raise RuntimeError(f"A1 query budget is not exactly {expected_queries}")
    if not set(selected).issubset(merged):
        raise RuntimeError("A1 query receipt is outside the sealed roster")
    for shot, record in merged.items():
        record["queried"] = shot in selected
        if shot in selected:
            selected_score = float(selected[shot]["query_score"])
            if abs(selected_score - float(record["query_score"])) > 1e-12:
                raise RuntimeError("A1 query score changed after sealing")
    return merged


def select_revealed_windows(
    sealed_logit: np.ndarray, labels: np.ndarray, *, count: int = 16
) -> np.ndarray:
    """Select a fixed post-reveal batch without fabricating positive windows."""
    sealed_logit = np.asarray(sealed_logit, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if len(sealed_logit) != len(labels):
        raise ValueError("logits and labels must align")
    positive = np.flatnonzero(labels)
    chosen: list[int] = []
    if len(positive):
        chosen.extend(dha.evenly_limit(positive, min(count // 2, len(positive))).tolist())
    remaining = count - len(chosen)
    negative = np.flatnonzero(~labels)
    if remaining and len(negative):
        edges = np.linspace(0, len(labels), remaining + 1, dtype=np.int64)
        for left, right in zip(edges[:-1], edges[1:]):
            candidates = negative[(negative >= left) & (negative < right)]
            if len(candidates):
                chosen.append(int(candidates[np.argmax(sealed_logit[candidates])]))
    if len(chosen) < count:
        unused = np.setdiff1d(negative, np.asarray(chosen), assume_unique=False)
        hardest = unused[np.argsort(sealed_logit[unused])[-(count - len(chosen)):]]
        chosen.extend(hardest.tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def subset_batch(batch: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long, device=batch["base_logit"].device)
    return {key: value.index_select(0, index) for key, value in batch.items()}


def confusion(labels: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    return {
        "TP": int(np.sum(labels & predicted)),
        "FP": int(np.sum(~labels & predicted)),
        "FN": int(np.sum(labels & ~predicted)),
        "TN": int(np.sum(~labels & ~predicted)),
    }


def metrics(matrix: dict[str, int]) -> dict[str, float]:
    return {
        "tpr": matrix["TP"] / (matrix["TP"] + matrix["FN"]),
        "fpr": matrix["FP"] / (matrix["FP"] + matrix["TN"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--a1", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    args.a1 = args.a1 or HERE / f"online_active_a1_seed{args.seed}"
    args.adapter_dir = args.adapter_dir or HERE / f"dynamic_hazard_adapter_seed{args.seed}"
    args.gate_dir = args.gate_dir or HERE / f"dynamic_hazard_adapter_seed{args.seed}_positive_gate"
    args.out = args.out or HERE / f"online_protected_residual_seed{args.seed}"
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)

    a1_summary = json.loads((args.a1 / "summary.json").read_text())
    gate_summary = json.loads((args.gate_dir / "summary.json").read_text())
    threshold = float(a1_summary["threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    minimum_positive = float(gate_summary["minimum_positive_residual"])
    scales = json.loads((args.a1 / "config.json").read_text())["innovation_scales"]
    query_rows = merge_a1_query_contract(
        load_jsonl(args.a1 / "seals.jsonl"),
        load_jsonl(args.a1 / "queries.jsonl"),
        expected_queries=72,
    )

    print(f"[ONLINE-RESIDUAL] rebuild frozen seed={args.seed} on {alarm.DEV}", flush=True)
    enc, head, transfer, test = alarm.train(args.seed)
    base_hash = a1.model_sha256(enc, head)
    if base_hash != a1_summary["model_sha256_before_after"]:
        raise RuntimeError("frozen base differs from A1")
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    adapter = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM, physics_dim=dha.PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    checkpoint = torch.load(args.adapter_dir / "adapter.pt", map_location=alarm.DEV, weights_only=True)
    if checkpoint["base_sha256"] != base_hash:
        raise RuntimeError("adapter parent base hash mismatch")
    adapter.load_state_dict(checkpoint["adapter"])
    initial_adapter_hash = auxiliary.module_sha256(adapter)
    if initial_adapter_hash != gate_summary["adapter_sha256"]:
        raise RuntimeError("adapter differs from support-calibrated gate parent")

    print("[ONLINE-RESIDUAL] build support replay and immutable guard", flush=True)
    support_cache = dha.build_support_cache(enc, head, transfer, scales)
    guard_rng = np.random.default_rng(args.seed + 9000)
    guard = dha.to_device(dha.sample_examples(
        support_cache, guard_rng, batch_size=512, balanced=True
    ))
    updater = protected.ProtectedResidualUpdater(
        adapter,
        threshold_logit=threshold_logit,
        guard_batch=guard,
        lr=args.lr,
        max_guard_harms=0,
        minimum_positive_residual=minimum_positive,
    )
    immutable_hash = protected.module_subset_sha256(adapter, exclude_prefix="residual_head")

    static_rows = {
        int(row["shot"]): row for row in load_jsonl(args.gate_dir / "predictions.jsonl")
    }
    labels_map = {int(shot["shot"]): bool(shot["is_disrupt"]) for shot in test}
    vault = audit.OutcomeVault(labels_map)
    test_by_shot = {int(shot["shot"]): shot for shot in test}
    ordered_shots = sorted(test_by_shot)
    if ordered_shots != sorted(query_rows) or ordered_shots != sorted(static_rows):
        raise RuntimeError("A1/static/test shot rosters differ")

    prediction_writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    receipt_writer = a1.JsonlSealWriter(args.out / "updates.jsonl")
    records: list[dict] = []
    replay_rng = np.random.default_rng(args.seed + 10000)
    accepted = rejected = 0
    try:
        for stream_index, shot_id in enumerate(ordered_shots):
            shot = test_by_shot[shot_id]
            endpoints = np.arange(alarm.W - 1 + dha.LAG, len(shot["v"]), dtype=np.int64)
            if not len(endpoints):
                raise RuntimeError(f"shot {shot_id} has no legal dynamic endpoints")
            frozen, direct_logit = dha.extract_frozen_batch(enc, head, shot["v"], endpoints)
            physics, available = a3.physics_family_trace(shot_id, scales)
            full_batch = dha.to_device({
                "frozen": frozen,
                "base_logit": direct_logit,
                "physics": physics[endpoints],
                "available": available[endpoints],
            })
            sealed = updater.predict_and_seal(shot_id, full_batch)
            adjusted_logit = sealed["adjusted_logit"].numpy()
            base_risk = np.load(args.a1 / "risk_traces" / f"{shot_id}.npz")["risk"]
            prefix_max = float(base_risk[:dha.LAG].max()) if len(base_risk[:dha.LAG]) else 0.0
            online_score = max(prefix_max, float(torch.sigmoid(sealed["adjusted_logit"]).max()))
            query = query_rows[shot_id]
            selected = bool(query["queried"])
            record = {
                "shot": shot_id,
                "stream_index": stream_index,
                "static_score": float(static_rows[shot_id]["adapted_score"]),
                "online_score": online_score,
                "queried": selected,
                "query_score": float(query["query_score"]),
                "predictor_sha256_before_prediction": sealed["predictor_sha256"],
            }
            prediction_writer.append(record)
            records.append(record)

            if selected:
                revealed = vault.reveal_for_query(shot_id, selected=True)
                times = float(shot["start"]) + endpoints / float(shot["rate"])
                window_labels = dynamic.alarm_window_labels(
                    times,
                    is_disrupt=revealed,
                    down_time_s=shot["down"],
                    horizon_ms=125.0,
                )
                full_batch["label"] = torch.as_tensor(
                    window_labels, dtype=torch.float32, device=alarm.DEV
                )
                chosen = select_revealed_windows(adjusted_logit, window_labels, count=16)
                current = subset_batch(full_batch, chosen)
                replay = dha.to_device(dha.sample_examples(
                    support_cache, replay_rng, batch_size=16, balanced=True
                ))
                receipt = updater.reveal_and_update(
                    shot_id,
                    selected=True,
                    current_batch=current,
                    replay_batch=replay,
                    query_score=float(query["query_score"]),
                    steps=2,
                )
                accepted += int(receipt["accepted"])
                rejected += int(not receipt["accepted"])
            else:
                receipt = updater.discard_without_reveal(shot_id)
            receipt_writer.append(receipt)
            if (stream_index + 1) % 40 == 0:
                print(
                    f"[ONLINE-RESIDUAL] sealed {stream_index + 1}/360 "
                    f"accepted={accepted} rollback={rejected}", flush=True,
                )
    finally:
        prediction_writer.close()
        receipt_writer.close()

    if protected.module_subset_sha256(adapter, exclude_prefix="residual_head") != immutable_hash:
        raise RuntimeError("online stream changed immutable target parameters")
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("online stream changed frozen base predictor")

    vault.open_final_evaluation()
    final_labels = vault.all_labels()
    labels = np.asarray([final_labels[row["shot"]] for row in records], dtype=bool)
    static_scores = np.asarray([row["static_score"] for row in records])
    online_scores = np.asarray([row["online_score"] for row in records])
    static_prediction = static_scores >= threshold
    online_prediction = online_scores >= threshold
    static_conf = confusion(labels, static_prediction)
    online_conf = confusion(labels, online_prediction)
    transitions = {
        "FP_rescued": int(np.sum(~labels & static_prediction & ~online_prediction)),
        "FN_rescued": int(np.sum(labels & ~static_prediction & online_prediction)),
        "TP_harmed": int(np.sum(labels & static_prediction & ~online_prediction)),
        "TN_harmed": int(np.sum(~labels & ~static_prediction & online_prediction)),
    }
    static_metrics = metrics(static_conf)
    online_metrics = metrics(online_conf)
    normality = {
        "tpr_ge_0_60": online_metrics["tpr"] >= 0.60,
        "fpr_le_0_40": online_metrics["fpr"] <= 0.40,
    }
    screen = {
        "tpr_non_degraded": online_conf["TP"] >= static_conf["TP"],
        "fpr_not_above_static_plus_0_01": online_metrics["fpr"] <= static_metrics["fpr"] + 0.01,
        "tp_harmed_le_2": transitions["TP_harmed"] <= 2,
        "tn_harmed_le_2": transitions["TN_harmed"] <= 2,
        "accepted_updates_gt_0": accepted > 0,
    }
    normality["passed"] = all(normality.values())
    screen["passed"] = all(screen.values())
    summary = {
        "status": "ONLINE_DIAGNOSTIC",
        "stage": "PROTECTED_INTERSHOT_RESIDUAL_SEED_SCREEN",
        "seed": args.seed,
        "query_contract": "sealed A1 causal 20% budget; 4 of every ordered 20 shots",
        "query_count": sum(bool(row["queried"]) for row in records),
        "accepted_updates": accepted,
        "rolled_back_updates": rejected,
        "threshold": threshold,
        "minimum_positive_residual": minimum_positive,
        "initial_adapter_sha256": initial_adapter_hash,
        "final_adapter_sha256": auxiliary.module_sha256(adapter),
        "base_model_sha256_before_after": base_hash,
        "immutable_adapter_sha256_before_after": immutable_hash,
        "static_confusion": static_conf,
        "online_confusion": online_conf,
        "static_metrics": static_metrics,
        "online_metrics": online_metrics,
        "transitions": transitions,
        "normality": normality,
        "screen_gate": screen,
        "hashes": {
            "predictions": a1.sha256_file(args.out / "predictions.jsonl"),
            "updates": a1.sha256_file(args.out / "updates.jsonl"),
        },
        "claim_boundary": (
            "balanced360 test-driven prequential diagnostic; query formula reused from A1; "
            "not independent confirmation and not deployment evidence"
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    torch.save({
        "adapter": adapter.state_dict(), "base_sha256": base_hash,
        "parent_adapter_sha256": initial_adapter_hash,
    }, args.out / "final_adapter.pt")
    print(json.dumps({
        "static": static_metrics,
        "online": online_metrics,
        "transitions": transitions,
        "accepted": accepted,
        "rollback": rejected,
        "screen": screen,
    }, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
