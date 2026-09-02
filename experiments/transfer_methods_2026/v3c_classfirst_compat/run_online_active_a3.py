#!/usr/bin/env python3
"""A3 shadow run: frozen base plus causal EAST auxiliary dynamics learning."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import shot_alarm as alarm  # noqa: E402


FAMILIES = ("electrical", "position", "radiation", "mhd")
FEATURE_DIM = 64 * 3
PHYSICS_DIM = len(FAMILIES)


def selected_endpoints(length: int, max_transitions: int) -> np.ndarray:
    if length <= alarm.W:
        return np.zeros(0, dtype=np.int64)
    count = min(int(max_transitions) + 1, length - alarm.W + 1)
    return np.unique(np.linspace(alarm.W - 1, length - 1, count, dtype=np.int64))


def physics_family_trace(shot_id: int, scales: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    values, available, rate = a1.load_raw_rebuilt(shot_id)
    innovations = audit.causal_innovations(values, available, sample_rate_hz=rate)
    scores, masks = audit.score_physics_families(innovations, scales)
    state = np.stack([scores.get(name, np.zeros(len(values))) for name in FAMILIES], axis=1)
    valid = np.stack([masks.get(name, np.zeros(len(values), bool)) for name in FAMILIES], axis=1)
    return state.astype(np.float32), valid.astype(bool)


def frozen_step_feature(
    enc: torch.nn.Module,
    shot: dict[str, Any],
    endpoint: int,
    previous_last: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = np.ascontiguousarray(shot["v"][endpoint - alarm.W + 1:endpoint + 1])
    x = torch.from_numpy(window[None]).to(alarm.DEV)
    cls, last = auxiliary.frozen_token_features(enc, x)
    cls, last = cls[0], last[0]
    delta = torch.zeros_like(last) if previous_last is None else last - previous_last
    return torch.cat([cls, last, delta]).detach(), last.detach()


def run_shot_pair(
    *,
    enc: torch.nn.Module,
    shot: dict[str, Any],
    scales: dict[str, float],
    static: auxiliary.OnlineAuxiliaryLearner,
    online: auxiliary.OnlineAuxiliaryLearner,
    max_transitions: int,
    patch_writer=None,
) -> dict[str, float]:
    endpoints = selected_endpoints(len(shot["v"]), max_transitions)
    static.reset_shot(); online.reset_shot()
    if len(endpoints) < 2:
        return {"static_error": 0.0, "online_error": 0.0, "transitions": 0}
    physics, valid = physics_family_trace(int(shot["shot"]), scales)
    previous_last = None
    static_errors, online_errors = [], []
    for step, endpoint in enumerate(endpoints):
        frozen, previous_last = frozen_step_feature(enc, shot, int(endpoint), previous_last)
        state = torch.from_numpy(physics[endpoint]).to(alarm.DEV)
        available = torch.from_numpy(valid[endpoint]).to(alarm.DEV)
        if step > 0:
            static_receipt = static.observe_and_update(
                step, frozen, state, available, update=False
            )
            online_receipt = online.observe_and_update(
                step, frozen, state, available, update=True
            )
            static_errors.append(float(static_receipt["prediction_error_mean"]))
            online_errors.append(float(online_receipt["prediction_error_mean"]))
            if patch_writer is not None:
                patch_writer.write(json.dumps({
                    "shot": int(shot["shot"]), "endpoint": int(endpoint),
                    "stream_event": "observe_update", **online_receipt,
                }, sort_keys=True) + "\n")
        static_seal = static.predict_and_seal(step, frozen, state, available)
        online_seal = online.predict_and_seal(step, frozen, state, available)
        if patch_writer is not None:
            patch_writer.write(json.dumps({
                "shot": int(shot["shot"]), "endpoint": int(endpoint),
                "stream_event": "predict_seal", "step_id": step,
                "predicted_next": online_seal["predicted_next"],
            }, sort_keys=True) + "\n")
    static.reset_shot(); online.reset_shot()
    if patch_writer is not None:
        patch_writer.flush(); os.fsync(patch_writer.fileno())
    return {
        "static_error": float(np.mean(static_errors)) if static_errors else 0.0,
        "online_error": float(np.mean(online_errors)) if online_errors else 0.0,
        "transitions": len(static_errors),
    }


def clone_learner(source, *, lr: float):
    clone = auxiliary.OnlineAuxiliaryLearner(
        feature_dim=FEATURE_DIM, physics_dim=PHYSICS_DIM, hidden=32,
        lr=lr, device=alarm.DEV,
    )
    clone.sidecar.load_state_dict(copy.deepcopy(source.sidecar.state_dict()))
    clone.teacher.load_state_dict(copy.deepcopy(source.teacher.state_dict()))
    return clone


def grouped_errors(records: list[dict[str, Any]], labels: dict[int, bool]) -> dict[str, Any]:
    result = {}
    for name in ("TP", "FP", "FN", "TN"):
        subset = []
        for row in records:
            actual = labels[int(row["shot"])]
            predicted = bool(row["base_alarm"])
            group = "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
            if group == name:
                subset.append(row)
        result[name] = {
            "n": len(subset),
            "static_error_mean": float(np.mean([r["static_error"] for r in subset])),
            "online_error_mean": float(np.mean([r["online_error"] for r in subset])),
            "online_novelty_mean": float(np.mean([r["aux_novelty"] for r in subset])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--a1", type=Path, default=HERE / "online_active_a1_seed131")
    parser.add_argument("--out", type=Path, default=HERE / "online_active_a3_seed131")
    parser.add_argument("--support-transitions", type=int, default=24)
    parser.add_argument("--stream-transitions", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reset-online-each-shot", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)

    a1_summary = json.loads((args.a1 / "summary.json").read_text())
    a1_records = [json.loads(line) for line in (args.a1 / "seals.jsonl").read_text().splitlines()]
    a1_by_shot = {int(row["shot"]): row for row in a1_records}
    a1_cutoff = float(a1_summary["query_cutoff"])
    scales = json.loads((args.a1 / "config.json").read_text())["innovation_scales"]

    print(f"[A3] deterministic base training seed={args.seed} device={alarm.DEV}", flush=True)
    enc, head, transfer, test = alarm.train(args.seed)
    if alarm.DEV != "cuda":
        raise RuntimeError("A3 registered run requires CUDA")
    base_hash = a1.model_sha256(enc, head)
    if base_hash != a1_summary["model_sha256_before_after"]:
        raise RuntimeError("A3 base model differs from sealed A1")
    for parameter in enc.parameters(): parameter.requires_grad_(False)
    for parameter in head.parameters(): parameter.requires_grad_(False)

    torch.manual_seed(args.seed + 3000)
    trainer = auxiliary.OnlineAuxiliaryLearner(
        feature_dim=FEATURE_DIM, physics_dim=PHYSICS_DIM, hidden=32,
        lr=args.lr, device=alarm.DEV,
    )
    print("[A3] label-free sidecar pretraining on fixed 110 support", flush=True)
    for number, shot in enumerate(sorted(transfer, key=lambda row: int(row["shot"])), 1):
        # Identical learners are not needed during pretraining; use a zero-LR
        # static clone only to satisfy the paired causal execution function.
        static_dummy = clone_learner(trainer, lr=0.0)
        run_shot_pair(
            enc=enc, shot=shot, scales=scales, static=static_dummy, online=trainer,
            max_transitions=args.support_transitions,
        )
        trainer.events.clear()
        if number % 25 == 0:
            print(f"[A3] support pretrained {number}/{len(transfer)}", flush=True)

    pretrained_hash = auxiliary.module_sha256(trainer.sidecar, trainer.teacher)
    static = clone_learner(trainer, lr=0.0)
    online = clone_learner(trainer, lr=args.lr)

    print("[A3] fitting support-only auxiliary novelty cutoff", flush=True)
    novelty_reference = []
    support_static = clone_learner(trainer, lr=0.0)
    support_online = clone_learner(trainer, lr=0.0)
    for shot in sorted(transfer, key=lambda row: int(row["shot"])):
        result = run_shot_pair(
            enc=enc, shot=shot, scales=scales, static=support_static, online=support_online,
            max_transitions=args.stream_transitions,
        )
        novelty_reference.append(result["online_error"])
        support_static.events.clear(); support_online.events.clear()
    novelty_contract = auxiliary.support_novelty_contract(novelty_reference)
    novelty_cutoff = novelty_contract["reference_raw"]

    labels = {int(shot["shot"]): bool(shot["is_disrupt"]) for shot in test}
    vault = audit.OutcomeVault(labels)
    selector = audit.BlockBudgetSelector(cutoff=1.0, block_size=20, quota=4)
    seal_writer = a1.JsonlSealWriter(args.out / "shot_seals.jsonl")
    query_writer = a1.JsonlSealWriter(args.out / "queries.jsonl")
    patch_path = args.out / "patch_events.jsonl"
    patch_writer = patch_path.open("x", encoding="utf-8")
    records = []
    try:
        for index, shot in enumerate(sorted(test, key=lambda row: int(row["shot"]))):
            if args.reset_online_each_shot:
                online = clone_learner(trainer, lr=args.lr)
            shot_id = int(shot["shot"])
            result = run_shot_pair(
                enc=enc, shot=shot, scales=scales, static=static, online=online,
                max_transitions=args.stream_transitions, patch_writer=patch_writer,
            )
            base = a1_by_shot[shot_id]
            base_component = float(base["query_score"]) / max(a1_cutoff, 1e-8)
            aux_component = float(result["online_error"]) / novelty_cutoff
            query_score = max(base_component, aux_component)
            record = {
                "stream_index": index, "shot": shot_id,
                "base_max_risk": float(base["base_max_risk"]),
                "base_alarm": bool(base["base_alarm"]),
                **result,
                "aux_novelty": aux_component,
                "base_query_normalized": base_component,
                "query_score": query_score,
                "queried": False,
                "base_model_sha256": base_hash,
            }
            seal_writer.append(record)
            selected = selector.decide(score=query_score, stream_index=index)
            record["queried"] = selected
            if selected:
                revealed = vault.reveal_for_query(shot_id, selected=True)
                query_writer.append({
                    "stream_index": index, "shot": shot_id,
                    "query_score": query_score, "revealed_is_disrupt": revealed,
                })
            records.append(record)
            static.events.clear(); online.events.clear()
            if (index + 1) % 40 == 0:
                print(f"[A3] sealed {index + 1}/{len(test)}", flush=True)
    finally:
        seal_writer.close(); query_writer.close(); patch_writer.close()

    if selector.total_selected != 72:
        raise RuntimeError(f"A3 query budget failed: {selector.total_selected}")
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("A3 modified the frozen base predictor")
    vault.open_final_evaluation()
    final_labels = vault.all_labels()
    capture = audit.confusion_and_query_capture(records, final_labels, threshold=float(a1_summary["threshold"]))
    total_static = float(np.average(
        [r["static_error"] for r in records], weights=[max(r["transitions"], 1) for r in records]
    ))
    total_online = float(np.average(
        [r["online_error"] for r in records], weights=[max(r["transitions"], 1) for r in records]
    ))
    summary = {
        "status": "ONLINE_DIAGNOSTIC",
        "stage": "A3_INTRA_SHOT_AUXILIARY_SSL_SHADOW",
        "seed": args.seed,
        "base_predictor_updated": False,
        "base_model_sha256_before_after": base_hash,
        "pretrained_sidecar_sha256": pretrained_hash,
        "final_online_sidecar_sha256": auxiliary.module_sha256(online.sidecar, online.teacher),
        "online_state_policy": (
            "reset_to_support_pretrained_each_shot"
            if args.reset_online_each_shot else "carry_across_shots"
        ),
        "support_novelty_q80": novelty_cutoff,
        "support_novelty_contract": novelty_contract,
        "query_budget": {"total": selector.total_selected, "block_size": 20, "per_block": 4},
        "query_capture": capture,
        "a1_error_enrichment_reference": a1_summary["query_capture"]["error_enrichment_vs_stream"],
        "auxiliary_prediction": {
            "static_error": total_static,
            "online_error": total_online,
            "online_over_static": total_online / total_static if total_static else None,
            "by_base_group": grouped_errors(records, final_labels),
        },
        "hashes": {
            "shot_seals": a1.sha256_file(args.out / "shot_seals.jsonl"),
            "queries": a1.sha256_file(args.out / "queries.jsonl"),
            "patch_events": a1.sha256_file(patch_path),
        },
        "claim_boundary": "A3 changes query features only; base risk and alarm metrics remain exactly A0.",
    }
    summary["screen_decision"] = {
        "next_state_improved": bool(total_online < total_static),
        "query_enrichment_improved": bool(
            capture["error_enrichment_vs_stream"]
            > a1_summary["query_capture"]["error_enrichment_vs_stream"]
        ),
    }
    summary["screen_decision"]["passed"] = all(summary["screen_decision"].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
