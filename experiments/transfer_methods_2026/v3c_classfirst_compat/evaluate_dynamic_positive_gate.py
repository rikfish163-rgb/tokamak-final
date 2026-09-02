#!/usr/bin/env python3
"""Support-calibrated positive residual gate for a trained hazard adapter."""

from __future__ import annotations

import argparse
import json
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
import run_dynamic_hazard_adapter as runner  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import shot_alarm as alarm  # noqa: E402


def support_positive_threshold(adapter, cache, quantile: float) -> tuple[float, int]:
    values = []
    adapter.eval()
    with torch.no_grad():
        for row in cache:
            selected = row["label"] > 0.5
            if not np.any(selected):
                continue
            _, delta, _, _ = adapter(
                torch.from_numpy(row["frozen"][selected]).to(alarm.DEV),
                torch.from_numpy(row["physics"][selected]).to(alarm.DEV),
                torch.from_numpy(row["available"][selected]).to(alarm.DEV),
                torch.from_numpy(row["base_logit"][selected]).to(alarm.DEV),
            )
            positive = delta[delta > 0].cpu().numpy()
            if len(positive):
                values.append(positive)
    joined = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    if not len(joined):
        raise RuntimeError("support positive windows produced no positive residual")
    return float(np.quantile(joined, quantile)), int(len(joined))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--a1", type=Path, default=HERE / "online_active_a1_seed131")
    parser.add_argument("--parent", type=Path, default=HERE / "dynamic_hazard_adapter_seed131")
    parser.add_argument("--out", type=Path, default=HERE / "dynamic_hazard_adapter_seed131_positive_gate")
    parser.add_argument("--support-positive-quantile", type=float, default=0.10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    parent_summary = json.loads((args.parent / "summary.json").read_text())
    a1_summary = json.loads((args.a1 / "summary.json").read_text())
    threshold = float(a1_summary["threshold"])
    scales = json.loads((args.a1 / "config.json").read_text())["innovation_scales"]

    print(f"[DHA-GATE] reproduce frozen base seed={args.seed}", flush=True)
    enc, head, transfer, test = alarm.train(args.seed)
    base_hash = a1.model_sha256(enc, head)
    if base_hash != parent_summary["base_model_sha256_before_after"]:
        raise RuntimeError("parent/base hash mismatch")
    for parameter in enc.parameters(): parameter.requires_grad_(False)
    for parameter in head.parameters(): parameter.requires_grad_(False)
    adapter = dynamic.DynamicHazardAdapter(
        feature_dim=runner.FEATURE_DIM, physics_dim=runner.PHYSICS_DIM,
        hidden=16, max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    checkpoint = torch.load(args.parent / "adapter.pt", map_location=alarm.DEV, weights_only=True)
    adapter.load_state_dict(checkpoint["adapter"]); adapter.eval()
    if auxiliary.module_sha256(adapter) != parent_summary["adapter_sha256"]:
        raise RuntimeError("adapter checkpoint hash mismatch")

    print("[DHA-GATE] calibrate positive evidence on 110 support only", flush=True)
    cache = runner.build_support_cache(enc, head, transfer, scales)
    minimum_positive, positive_count = support_positive_threshold(
        adapter, cache, args.support_positive_quantile
    )
    labels_map = {int(shot["shot"]): bool(shot["is_disrupt"]) for shot in test}
    vault = audit.OutcomeVault(labels_map)
    writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    records = []
    print(f"[DHA-GATE] seal 360 with minimum positive residual={minimum_positive:.6f}", flush=True)
    try:
        for number, shot in enumerate(sorted(test, key=lambda row: int(row["shot"])), 1):
            shot_id = int(shot["shot"])
            base_risk = np.load(args.a1 / "risk_traces" / f"{shot_id}.npz")["risk"]
            adapted, delta = runner.score_shot(
                adapter, enc, head, shot, scales, base_risk,
                minimum_positive_residual=minimum_positive,
            )
            record = {
                "shot": shot_id, "base_score": float(base_risk.max()),
                "adapted_score": float(adapted.max()),
                "max_abs_effective_residual": float(np.max(np.abs(delta))) if len(delta) else 0.0,
                "minimum_positive_residual": minimum_positive,
            }
            writer.append(record); records.append(record)
            if number % 40 == 0:
                print(f"[DHA-GATE] sealed {number}/{len(test)}", flush=True)
    finally:
        writer.close()

    vault.open_final_evaluation(); labels = vault.all_labels()
    y = np.asarray([labels[int(row["shot"])] for row in records], bool)
    base_scores = np.asarray([row["base_score"] for row in records])
    adapted_scores = np.asarray([row["adapted_score"] for row in records])
    bp, ap = base_scores >= threshold, adapted_scores >= threshold
    base_conf = runner.confusion(y, base_scores, threshold)
    adapted_conf = runner.confusion(y, adapted_scores, threshold)
    transitions = {
        "FP_rescued": int(np.sum((~y) & bp & (~ap))),
        "FN_rescued": int(np.sum(y & (~bp) & ap)),
        "TP_harmed": int(np.sum(y & bp & (~ap))),
        "TN_harmed": int(np.sum((~y) & (~bp) & ap)),
    }
    summary = {
        "status": "ONLINE_DIAGNOSTIC",
        "stage": "SUPPORT_CALIBRATED_POSITIVE_RESIDUAL_GATE",
        "seed": args.seed,
        "threshold": threshold,
        "support_positive_quantile": args.support_positive_quantile,
        "support_positive_count": positive_count,
        "minimum_positive_residual": minimum_positive,
        "base_confusion": base_conf, "adapted_confusion": adapted_conf,
        "base_metrics": {
            "tpr": base_conf["TP"] / (base_conf["TP"] + base_conf["FN"]),
            "fpr": base_conf["FP"] / (base_conf["FP"] + base_conf["TN"]),
        },
        "adapted_metrics": {
            "tpr": adapted_conf["TP"] / (adapted_conf["TP"] + adapted_conf["FN"]),
            "fpr": adapted_conf["FP"] / (adapted_conf["FP"] + adapted_conf["TN"]),
        },
        "transitions": transitions,
        "base_model_sha256_before_after": base_hash,
        "adapter_sha256": auxiliary.module_sha256(adapter),
        "screen_gate": {
            "tp_harmed_le_2": transitions["TP_harmed"] <= 2,
            "tn_harmed_le_2": transitions["TN_harmed"] <= 2,
            "fn_rescued_ge_6": transitions["FN_rescued"] >= 6,
            "tpr_improved": adapted_conf["TP"] > base_conf["TP"],
            "fpr_not_above_0_10": adapted_conf["FP"] / (adapted_conf["FP"] + adapted_conf["TN"]) <= 0.10,
        },
        "hashes": {"predictions": a1.sha256_file(args.out / "predictions.jsonl")},
        "claim_boundary": "Support-calibrated gate, evaluated on already test-driven balanced360.",
    }
    summary["screen_gate"]["passed"] = all(summary["screen_gate"].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "minimum_positive_residual": minimum_positive,
        "base": summary["base_metrics"], "adapted": summary["adapted_metrics"],
        "transitions": transitions, "screen_gate": summary["screen_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
