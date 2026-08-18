#!/usr/bin/env python3
"""Run immutable A0 plus causal A1 query auditing on ordered EAST balanced360.

The migrated predictor and its support-selected threshold never update here.
Labels are held by OutcomeVault until a query has been sealed, and full labels
are opened only after all 360 predictions and query decisions are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
sys.path.insert(0, str(HERE))

import common as common  # noqa: E402
import online_active_audit as audit  # noqa: E402
import rebuilt_harness as rebuilt  # noqa: E402
import shot_alarm as alarm  # noqa: E402


MANIFEST = ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json"
REFERENCE_DIR = HERE / "final_delivery_hdf5_deterministic_v2" / "seeds"
METHOD = "rebuilt17+ea125+adapt_high+per-shot-max"


def selection_score(components: dict[str, float], threshold: float, mode: str) -> float:
    if mode == "balanced":
        return float(components["query_score"])
    if mode == "fp_first":
        if float(components["max_risk"]) >= float(threshold):
            return float(components["fp_contradiction"])
        return 0.25 * float(components["fn_contradiction"])
    raise ValueError(mode)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_sha256(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class JsonlSealWriter:
    """Durably append one JSON record at a time; existing ledgers are refused."""

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"refusing to append to an existing seal ledger: {path}")
        self.path = path
        self.handle = path.open("x", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def load_raw_rebuilt(shot: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = rebuilt.DATA / "EAST" / "unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    arrays, rate, _, _, _ = rebuilt.load_raw(path, "EAST")
    values, available = rebuilt.rebuild_features(arrays, "EAST")
    return values, available, rate


def physics_trace(
    shot: int,
    scales: dict[str, float],
    *,
    risk_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values, available, rate = load_raw_rebuilt(shot)
    innovations = audit.causal_innovations(values, available, sample_rate_hz=rate)
    family_scores, family_available = audit.score_physics_families(innovations, scales)
    support, known, count = audit.combine_family_scores(family_scores, family_available)
    support = support[alarm.W - 1:]
    known = known[alarm.W - 1:]
    count = count[alarm.W - 1:]
    if len(support) != risk_length:
        raise RuntimeError(f"shot {shot}: risk/physics length mismatch {risk_length} != {len(support)}")
    meta = {
        "physics_known_fraction": float(np.mean(known)),
        "physics_family_count_max": int(np.max(count)) if len(count) else 0,
    }
    return support, known, meta


def fit_support_physics_scales(shots: list[dict[str, Any]]) -> dict[str, float]:
    traces = []
    for shot in shots:
        values, available, rate = load_raw_rebuilt(int(shot["shot"]))
        traces.append(audit.causal_innovations(values, available, sample_rate_hz=rate))
    return audit.fit_innovation_scales(traces)


def reference_contract(seed: int) -> tuple[float, dict[int, float], Path]:
    path = REFERENCE_DIR / f"seed{seed}.json"
    data = json.loads(path.read_text())
    if data["method"] != METHOD:
        raise RuntimeError(f"reference method mismatch: {data['method']}")
    threshold = float(data["support_selected_threshold"]["threshold"])
    scores = {int(row["shot"]): float(row["score"]) for row in data["predictions"]}
    return threshold, scores, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path, default=HERE / "online_active_a1_seed131")
    parser.add_argument("--query-quantile", type=float, default=0.80)
    parser.add_argument("--query-mode", choices=("balanced", "fp_first"), default="balanced")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output already exists; use a new path: {args.out}")
    args.out.mkdir(parents=True)
    risk_dir = args.out / "risk_traces"
    risk_dir.mkdir()

    print(f"[A0] deterministic training seed={args.seed} device={alarm.DEV}", flush=True)
    enc, head, transfer, test = alarm.train(args.seed)
    if alarm.DEV != "cuda":
        raise RuntimeError("this registered run requires CUDA in rl_env")
    model_hash_before = model_sha256(enc, head)
    threshold, reference_scores, reference_path = reference_contract(args.seed)
    if set(reference_scores) != {int(s["shot"]) for s in test}:
        raise RuntimeError("current balanced360 shots differ from the frozen reference")

    print("[A1] fitting label-free innovation scales on fixed 110 support", flush=True)
    scales = fit_support_physics_scales(transfer)
    support_query_scores = []
    for number, shot in enumerate(sorted(transfer, key=lambda row: int(row["shot"])), 1):
        risk = alarm.full_risk(enc, head, shot)
        physics, known, _ = physics_trace(int(shot["shot"]), scales, risk_length=len(risk))
        support_components = audit.query_components(
            risk=risk, physics_support=physics, physics_known=known, threshold=threshold
        )
        support_query_scores.append(selection_score(support_components, threshold, args.query_mode))
        if number % 25 == 0:
            print(f"[A1] support scored {number}/{len(transfer)}", flush=True)
    query_cutoff = float(np.quantile(support_query_scores, args.query_quantile))

    labels = {int(shot["shot"]): bool(shot["is_disrupt"]) for shot in test}
    stream = [{"shot": int(shot["shot"])} for shot in test]
    by_shot = {int(shot["shot"]): shot for shot in test}
    risk_cache: dict[int, np.ndarray] = {}
    physics_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    seal_writer = JsonlSealWriter(args.out / "seals.jsonl")
    query_writer = JsonlSealWriter(args.out / "queries.jsonl")
    selector = audit.BlockBudgetSelector(cutoff=query_cutoff, block_size=20, quota=4)
    vault = audit.OutcomeVault(labels)

    def predict(item: dict[str, Any]) -> np.ndarray:
        shot_id = int(item["shot"])
        risk = alarm.full_risk(enc, head, by_shot[shot_id])
        risk_cache[shot_id] = risk
        physics_cache[shot_id] = physics_trace(shot_id, scales, risk_length=len(risk))
        np.savez_compressed(risk_dir / f"{shot_id}.npz", risk=risk)
        return risk

    def summarize(item: dict[str, Any], risk: np.ndarray) -> dict[str, float]:
        physics, known, meta = physics_cache[int(item["shot"])]
        components = audit.query_components(
            risk=risk, physics_support=physics, physics_known=known, threshold=threshold
        )
        components["unmodified_query_score"] = components["query_score"]
        components["query_score"] = selection_score(components, threshold, args.query_mode)
        components.update(meta)
        return components

    def on_seal(record: dict[str, Any]) -> None:
        shot_id = int(record["shot"])
        risk = risk_cache[shot_id]
        record.update(
            model_sha256=model_hash_before,
            risk_sha256=sha256_array(risk),
            base_alarm=bool(float(record["base_max_risk"]) >= threshold),
            threshold=threshold,
        )
        seal_writer.append(record)
        if (int(record["stream_index"]) + 1) % 40 == 0:
            print(f"[A1] sealed {int(record['stream_index']) + 1}/{len(stream)}", flush=True)

    def on_reveal(record: dict[str, Any]) -> None:
        query_writer.append({
            "stream_index": int(record["stream_index"]),
            "shot": int(record["shot"]),
            "query_score": float(record["query_score"]),
            "revealed_is_disrupt": bool(record["revealed_is_disrupt"]),
        })

    try:
        records = audit.run_prequential_audit(
            shots=stream,
            predict=predict,
            summarize=summarize,
            selector=selector,
            outcomes=vault,
            on_seal=on_seal,
            on_reveal=on_reveal,
        )
    finally:
        seal_writer.close()
        query_writer.close()

    if len(records) != 360 or selector.total_selected != 72:
        raise RuntimeError(f"budget contract failed: shots={len(records)} queries={selector.total_selected}")
    block_counts = [
        sum(bool(row["queried"]) for row in records[start:start + 20])
        for start in range(0, len(records), 20)
    ]
    if block_counts != [4] * 18:
        raise RuntimeError(f"block budget contract failed: {block_counts}")

    reproduced = {int(row["shot"]): float(row["base_max_risk"]) for row in records}
    max_abs_reference_delta = max(abs(reproduced[s] - reference_scores[s]) for s in reference_scores)
    if max_abs_reference_delta > 1e-6:
        raise RuntimeError(f"static reference reproduction failed: max delta={max_abs_reference_delta}")
    if model_sha256(enc, head) != model_hash_before:
        raise RuntimeError("A1 changed the immutable migrated predictor")

    vault.open_final_evaluation()
    final_labels = vault.all_labels()
    capture = audit.confusion_and_query_capture(records, final_labels, threshold=threshold)
    confusion = capture["confusion"]
    metrics = {
        "tpr": confusion["TP"] / (confusion["TP"] + confusion["FN"]),
        "fpr": confusion["FP"] / (confusion["FP"] + confusion["TN"]),
    }
    summary = {
        "status": "LOCKED_PREQUENTIAL_DIAGNOSTIC_REPLAY",
        "stage": "A0_STATIC_PLUS_A1_QUERY_AUDIT",
        "seed": args.seed,
        "method": METHOD,
        "base_predictor_updated": False,
        "test_labels_used_for_query_scoring": False,
        "threshold": threshold,
        "threshold_source": "fixed 110-shot EAST support",
        "query_cutoff": query_cutoff,
        "query_cutoff_source": f"label-free support query-score q{args.query_quantile:.2f}",
        "query_mode": args.query_mode,
        "query_budget": {"block_size": 20, "per_block": 4, "total": 72},
        "block_query_counts": block_counts,
        "base_metrics": metrics,
        "query_capture": capture,
        "model_sha256_before_after": model_hash_before,
        "max_abs_reference_score_delta": max_abs_reference_delta,
        "hashes": {
            "test_manifest": sha256_file(MANIFEST),
            "reference_seed_json": sha256_file(reference_path),
            "seals_jsonl": sha256_file(args.out / "seals.jsonl"),
            "queries_jsonl": sha256_file(args.out / "queries.jsonl"),
        },
        "claim_boundary": "A1 audits query enrichment only; it does not improve predictions and is not an independent locked-test result.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "query_quantile": args.query_quantile,
        "query_mode": args.query_mode,
        "physics_scale_quantile": 0.90,
        "physics_innovation_lag_samples": 8,
        "threshold": threshold,
        "model_sha256": model_hash_before,
        "innovation_scales": scales,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"base_metrics": metrics, "query_capture": capture}, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
