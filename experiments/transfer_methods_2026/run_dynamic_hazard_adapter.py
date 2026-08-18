#!/usr/bin/env python3
"""Train an EAST-private dynamics-constrained hazard residual on fixed support."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dynamic_hazard_adapter as dynamic  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_active_a3 as a3  # noqa: E402
import shot_alarm as alarm  # noqa: E402


FEATURE_DIM = 64 * 3
PHYSICS_DIM = len(a3.FAMILIES)
LAG = 8


def evenly_limit(indices: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def extract_frozen_batch(
    enc: torch.nn.Module,
    head: torch.nn.Module,
    values: np.ndarray,
    endpoints: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray(endpoints, dtype=np.int64)
    features, logits = [], []
    enc.eval(); head.eval()
    for start in range(0, len(endpoints), batch_size):
        batch = endpoints[start:start + batch_size]
        current = np.stack([values[e - alarm.W + 1:e + 1] for e in batch])
        previous = np.stack([values[e - LAG - alarm.W + 1:e - LAG + 1] for e in batch])
        x = torch.from_numpy(np.concatenate([current, previous])).to(alarm.DEV)
        cls, last = auxiliary.frozen_token_features(enc, x)
        count = len(batch)
        current_cls, current_last = cls[:count], last[:count]
        previous_last = last[count:]
        frozen = torch.cat([current_cls, current_last, current_last - previous_last], dim=1)
        with torch.no_grad():
            base_logit = head(current_cls)
        features.append(frozen.cpu().numpy().astype(np.float32))
        logits.append(base_logit.cpu().numpy().astype(np.float32))
    return np.concatenate(features), np.concatenate(logits)


def build_support_cache(enc, head, shots, scales, *, max_per_class: int = 32):
    cache = []
    for shot in sorted(shots, key=lambda row: int(row["shot"])):
        n = len(shot["v"])
        candidates = np.arange(alarm.W - 1 + LAG, n - LAG, dtype=np.int64)
        if not len(candidates):
            continue
        times = float(shot["start"]) + candidates / float(shot["rate"])
        labels = dynamic.alarm_window_labels(
            times,
            is_disrupt=bool(shot["is_disrupt"]),
            down_time_s=shot["down"],
            horizon_ms=125.0,
        )
        positive = evenly_limit(candidates[labels], max_per_class)
        negative = evenly_limit(candidates[~labels], max_per_class)
        endpoints = np.sort(np.concatenate([positive, negative]))
        selected_labels = np.isin(endpoints, positive).astype(np.float32)
        frozen, base_logit = extract_frozen_batch(enc, head, shot["v"], endpoints)
        physics, available = a3.physics_family_trace(int(shot["shot"]), scales)
        cache.append({
            "shot": int(shot["shot"]),
            "frozen": frozen,
            "base_logit": base_logit,
            "physics": physics[endpoints],
            "available": available[endpoints],
            "next_physics": physics[endpoints + LAG],
            "next_available": available[endpoints + LAG],
            "label": selected_labels,
        })
    return cache


def sample_examples(cache, rng, *, batch_size: int, balanced: bool):
    rows = []
    positive_shots = [row for row in cache if np.any(row["label"] > 0.5)]
    for index in range(batch_size):
        want_positive = balanced and index < batch_size // 2
        pool = positive_shots if want_positive else cache
        shot = pool[int(rng.integers(len(pool)))]
        candidates = np.flatnonzero(shot["label"] > 0.5 if want_positive else shot["label"] <= 0.5)
        if not len(candidates):
            candidates = np.arange(len(shot["label"]))
        rows.append((shot, int(candidates[int(rng.integers(len(candidates)))])))
    keys = ("frozen", "base_logit", "physics", "available", "next_physics", "next_available", "label")
    result = {}
    for key in keys:
        result[key] = np.stack([row[key][i] for row, i in rows])
    return result


def to_device(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        dtype = torch.bool if key in ("available", "next_available") else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=alarm.DEV)
    return result


def masked_next_loss(prediction, target, available):
    valid = available & torch.isfinite(target)
    if bool(valid.any()):
        return F.smooth_l1_loss(prediction[valid], target[valid])
    return prediction.sum() * 0.0


def train_adapter(adapter, cache, *, seed: int, threshold: float):
    rng = np.random.default_rng(seed + 5000)
    history = []
    optimizer = torch.optim.AdamW(
        list(adapter.projector.parameters()) + list(adapter.next_head.parameters()),
        lr=5e-4, weight_decay=1e-4,
    )
    adapter.train()
    for epoch in range(4):
        losses = []
        for _ in range(40):
            batch = to_device(sample_examples(cache, rng, batch_size=128, balanced=False))
            latent = adapter.encode(batch["frozen"], batch["physics"], batch["available"])
            loss = masked_next_loss(
                adapter.predict_next(latent), batch["next_physics"], batch["next_available"]
            )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"phase": "aux_pretrain", "epoch": epoch, "loss": float(np.mean(losses))})

    threshold_logit = math.log(threshold / (1.0 - threshold))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(8):
        totals, hazards, auxiliaries, anchors = [], [], [], []
        for _ in range(48):
            batch = to_device(sample_examples(cache, rng, batch_size=128, balanced=True))
            adjusted, delta, _, latent = adapter(
                batch["frozen"], batch["physics"], batch["available"], batch["base_logit"]
            )
            hazard = F.binary_cross_entropy_with_logits(adjusted, batch["label"])
            aux = masked_next_loss(
                adapter.predict_next(latent), batch["next_physics"], batch["next_available"]
            )
            sign = batch["label"] * 2.0 - 1.0
            base_margin = sign * (batch["base_logit"] - threshold_logit)
            adjusted_margin = sign * (adjusted - threshold_logit)
            protected = base_margin >= 0
            anchor = (
                F.relu(base_margin[protected] - adjusted_margin[protected]).mean()
                if bool(protected.any()) else adjusted.sum() * 0.0
            )
            loss = hazard + 0.20 * aux + 1.0 * anchor + 0.05 * delta.square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
            totals.append(float(loss.detach().cpu())); hazards.append(float(hazard.detach().cpu()))
            auxiliaries.append(float(aux.detach().cpu())); anchors.append(float(anchor.detach().cpu()))
        history.append({
            "phase": "joint_hazard", "epoch": epoch,
            "loss": float(np.mean(totals)), "hazard": float(np.mean(hazards)),
            "aux": float(np.mean(auxiliaries)), "anchor": float(np.mean(anchors)),
        })
    adapter.eval()
    return history


def score_shot(
    adapter, enc, head, shot, scales, base_risk, *, minimum_positive_residual: float = 0.0
):
    base_risk = np.asarray(base_risk, dtype=np.float32)
    adjusted = base_risk.copy()
    endpoints = np.arange(alarm.W - 1 + LAG, len(shot["v"]), dtype=np.int64)
    if not len(endpoints):
        return adjusted, np.zeros_like(adjusted)
    physics, available = a3.physics_family_trace(int(shot["shot"]), scales)
    all_delta = np.zeros(len(endpoints), dtype=np.float32)
    for start in range(0, len(endpoints), 512):
        batch_endpoints = endpoints[start:start + 512]
        frozen, direct_logit = extract_frozen_batch(enc, head, shot["v"], batch_endpoints)
        risk_indices = batch_endpoints - (alarm.W - 1)
        sealed_logit = np.log(np.clip(base_risk[risk_indices], 1e-6, 1 - 1e-6) /
                              np.clip(1 - base_risk[risk_indices], 1e-6, 1 - 1e-6))
        if np.max(np.abs(torch.sigmoid(torch.from_numpy(direct_logit)).numpy() - base_risk[risk_indices])) > 1e-5:
            raise RuntimeError(f"shot {shot['shot']}: direct/sealed base risk mismatch")
        with torch.no_grad():
            _adapted_logit, delta, _, _ = adapter(
                torch.from_numpy(frozen).to(alarm.DEV),
                torch.from_numpy(physics[batch_endpoints]).to(alarm.DEV),
                torch.from_numpy(available[batch_endpoints]).to(alarm.DEV),
                torch.from_numpy(sealed_logit.astype(np.float32)).to(alarm.DEV),
            )
            effective_delta = dynamic.apply_positive_residual_gate(
                delta, minimum_positive=minimum_positive_residual
            )
            probability = torch.sigmoid(
                torch.from_numpy(sealed_logit.astype(np.float32)).to(alarm.DEV) + effective_delta
            ).cpu().numpy()
        adjusted[risk_indices] = probability
        all_delta[start:start + len(batch_endpoints)] = effective_delta.cpu().numpy()
    return adjusted, all_delta


def confusion(labels, scores, threshold):
    labels = np.asarray(labels, bool); predicted = np.asarray(scores) >= threshold
    return {
        "TP": int(np.sum(labels & predicted)), "FP": int(np.sum(~labels & predicted)),
        "FN": int(np.sum(labels & ~predicted)), "TN": int(np.sum(~labels & ~predicted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--a1", type=Path, default=HERE / "online_active_a1_seed131")
    parser.add_argument("--out", type=Path, default=HERE / "dynamic_hazard_adapter_seed131")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    a1_summary = json.loads((args.a1 / "summary.json").read_text())
    threshold = float(a1_summary["threshold"])
    scales = json.loads((args.a1 / "config.json").read_text())["innovation_scales"]

    print(f"[DHA] train frozen base seed={args.seed} device={alarm.DEV}", flush=True)
    enc, head, transfer, test = alarm.train(args.seed)
    base_hash = a1.model_sha256(enc, head)
    if base_hash != a1_summary["model_sha256_before_after"]:
        raise RuntimeError("base model differs from A1 seal")
    for parameter in enc.parameters(): parameter.requires_grad_(False)
    for parameter in head.parameters(): parameter.requires_grad_(False)

    print("[DHA] cache HDF5-authoritative 110 support features", flush=True)
    cache = build_support_cache(enc, head, transfer, scales)
    counts = {
        "shots": len(cache),
        "positive_windows": int(sum(row["label"].sum() for row in cache)),
        "negative_windows": int(sum((row["label"] <= 0.5).sum() for row in cache)),
    }
    torch.manual_seed(args.seed + 5000)
    adapter = dynamic.DynamicHazardAdapter(
        feature_dim=FEATURE_DIM, physics_dim=PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
    if parameter_count >= 5000:
        raise RuntimeError(f"adapter parameter budget exceeded: {parameter_count}")
    print(f"[DHA] train adapter parameters={parameter_count} support={counts}", flush=True)
    history = train_adapter(adapter, cache, seed=args.seed, threshold=threshold)
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("adapter training changed frozen base")

    labels_map = {int(shot["shot"]): bool(shot["is_disrupt"]) for shot in test}
    vault = audit.OutcomeVault(labels_map)
    seal_writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    records = []
    print("[DHA] seal 360 fixed-threshold predictions", flush=True)
    try:
        for number, shot in enumerate(sorted(test, key=lambda row: int(row["shot"])), 1):
            shot_id = int(shot["shot"])
            base_risk = np.load(args.a1 / "risk_traces" / f"{shot_id}.npz")["risk"]
            adapted_risk, delta = score_shot(adapter, enc, head, shot, scales, base_risk)
            record = {
                "shot": shot_id,
                "base_score": float(base_risk.max()),
                "adapted_score": float(adapted_risk.max()),
                "mean_abs_logit_residual": float(np.mean(np.abs(delta))),
                "max_abs_logit_residual": float(np.max(np.abs(delta))) if len(delta) else 0.0,
                "threshold": threshold,
                "base_model_sha256": base_hash,
                "adapter_sha256": auxiliary.module_sha256(adapter),
            }
            seal_writer.append(record); records.append(record)
            if number % 40 == 0:
                print(f"[DHA] sealed {number}/{len(test)}", flush=True)
    finally:
        seal_writer.close()

    vault.open_final_evaluation()
    final_labels = vault.all_labels()
    ordered_labels = np.asarray([final_labels[int(row["shot"])] for row in records])
    base_scores = np.asarray([row["base_score"] for row in records])
    adapted_scores = np.asarray([row["adapted_score"] for row in records])
    base_conf = confusion(ordered_labels, base_scores, threshold)
    adapted_conf = confusion(ordered_labels, adapted_scores, threshold)
    base_predicted = base_scores >= threshold
    adapted_predicted = adapted_scores >= threshold
    transitions = {
        "FP_rescued": int(np.sum((~ordered_labels) & base_predicted & (~adapted_predicted))),
        "FN_rescued": int(np.sum(ordered_labels & (~base_predicted) & adapted_predicted)),
        "TP_harmed": int(np.sum(ordered_labels & base_predicted & (~adapted_predicted))),
        "TN_harmed": int(np.sum((~ordered_labels) & (~base_predicted) & adapted_predicted)),
    }
    summary = {
        "status": "ONLINE_DIAGNOSTIC",
        "stage": "EAST_PRIVATE_DYNAMICS_HAZARD_RESIDUAL",
        "seed": args.seed,
        "threshold": threshold,
        "threshold_source": "fixed 110-shot support base threshold; not reselected",
        "support_counts": counts,
        "adapter_parameters": parameter_count,
        "base_model_sha256_before_after": base_hash,
        "adapter_sha256": auxiliary.module_sha256(adapter),
        "base_confusion": base_conf,
        "adapted_confusion": adapted_conf,
        "base_metrics": {
            "tpr": base_conf["TP"] / (base_conf["TP"] + base_conf["FN"]),
            "fpr": base_conf["FP"] / (base_conf["FP"] + base_conf["TN"]),
        },
        "adapted_metrics": {
            "tpr": adapted_conf["TP"] / (adapted_conf["TP"] + adapted_conf["FN"]),
            "fpr": adapted_conf["FP"] / (adapted_conf["FP"] + adapted_conf["TN"]),
        },
        "transitions": transitions,
        "history": history,
        "normality": {
            "tpr_ge_0_60": adapted_conf["TP"] / (adapted_conf["TP"] + adapted_conf["FN"]) >= 0.60,
            "fpr_le_0_40": adapted_conf["FP"] / (adapted_conf["FP"] + adapted_conf["TN"]) <= 0.40,
        },
        "screen_gate": {
            "tp_harmed_le_2": transitions["TP_harmed"] <= 2,
            "tn_harmed_le_2": transitions["TN_harmed"] <= 2,
            "fp_rescued_ge_1": transitions["FP_rescued"] >= 1,
            "tpr_non_degraded": adapted_conf["TP"] >= base_conf["TP"],
            "fpr_strictly_lower": adapted_conf["FP"] < base_conf["FP"],
        },
        "hashes": {"predictions": a1.sha256_file(args.out / "predictions.jsonl")},
        "claim_boundary": "110-support-trained, balanced360 test-driven diagnostic; no online updates during test.",
    }
    summary["normality"]["passed"] = all(summary["normality"].values())
    summary["screen_gate"]["passed"] = all(summary["screen_gate"].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    torch.save({"adapter": adapter.state_dict(), "base_sha256": base_hash}, args.out / "adapter.pt")
    print(json.dumps({
        "base": summary["base_metrics"], "adapted": summary["adapted_metrics"],
        "transitions": transitions, "screen_gate": summary["screen_gate"],
    }, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
