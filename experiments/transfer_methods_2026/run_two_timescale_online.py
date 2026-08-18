#!/usr/bin/env python3
"""Strict sequential EAST stream with intra-shot SSL and inter-shot supervision."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import h5py


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dynamic_hazard_adapter as dynamic  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_active_a3 as a3  # noqa: E402
import run_online_protected_residual as old_online  # noqa: E402
import rebuilt_harness as rebuilt  # noqa: E402
import reverse_online_value as reverse_value  # noqa: E402
import shot_alarm as alarm  # noqa: E402
import two_timescale_online_adapter as two_scale  # noqa: E402


MODES = ("static", "intra_only", "inter_only", "intra_inter")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_east_signal_only(shot_id: int, med: np.ndarray, scale: np.ndarray) -> dict:
    """Load data/* and timebase attrs without touching any HDF5 meta dataset."""
    path = DATA / "EAST/unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    with h5py.File(path, "r") as handle:
        arrays = {
            name: np.asarray(handle[f"data/{name}"][:], dtype=np.float32)
            for name in rebuilt.RAW23
        }
        rate = float(np.asarray(
            handle["data/ip"].attrs.get("SampleRate", 1000.0)
        ).reshape(-1)[0])
        start = float(np.asarray(
            handle["data/ip"].attrs.get("StartTime", 0.0)
        ).reshape(-1)[0])
    values, available = rebuilt.rebuild_features(arrays, "EAST")
    values = np.clip((values - med) / scale, -10, 10)
    values = np.where(available, values, 0.0).astype(np.float32)
    return {
        "shot": int(shot_id), "v": values, "a": available,
        "start": start, "rate": rate,
    }


def read_hdf5_outcome(shot_id: int) -> dict:
    path = DATA / "EAST/unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    with h5py.File(path, "r") as handle:
        is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    return {
        "is_disrupt": is_disrupt,
        "down_time_s": float(down[0]) if is_disrupt and down.size else None,
    }


def fit_support_query_cutoff(
    enc, head, support_shots, scales, *, threshold: float, quantile: float
) -> float:
    scores = []
    for shot in sorted(support_shots, key=lambda row: int(row["shot"])):
        risk = alarm.full_risk(enc, head, shot)
        physics, known, _meta = a1.physics_trace(
            int(shot["shot"]), scales, risk_length=len(risk)
        )
        components = audit.query_components(
            risk=risk, physics_support=physics,
            physics_known=known, threshold=threshold,
        )
        scores.append(float(components["query_score"]))
    return float(np.quantile(scores, quantile))


def make_batch(
    frozen: np.ndarray,
    base_logit: np.ndarray,
    physics: np.ndarray,
    available: np.ndarray,
    indices: np.ndarray,
    *,
    next_physics: np.ndarray | None = None,
    next_available: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    indices = np.asarray(indices, dtype=np.int64)
    batch = {
        "frozen": torch.from_numpy(frozen[indices]),
        "base_logit": torch.from_numpy(base_logit[indices]),
        "physics": torch.from_numpy(physics[indices]),
        "available": torch.from_numpy(available[indices]),
    }
    if next_physics is not None:
        batch["next_physics"] = torch.from_numpy(next_physics[indices])
    if next_available is not None:
        batch["next_available"] = torch.from_numpy(next_available[indices])
    if labels is not None:
        batch["label"] = torch.from_numpy(labels[indices].astype(np.float32))
    return batch


def concatenate_batches(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = tuple(batches[0])
    if any(tuple(batch) != keys for batch in batches[1:]):
        raise ValueError("replay batches have different fields")
    return {
        name: torch.cat([batches[0][name], *(batch[name] for batch in batches[1:])])
        for name in keys
    }


def fit_fast_physics_support_threshold(
    support_shots: list[dict], scales: dict[str, float], *, preserve_fraction: float
) -> dict[str, float]:
    if not 0 < preserve_fraction <= 1:
        raise ValueError("fast physics preserve fraction must be in (0, 1]")
    positive_support = []
    for shot in support_shots:
        risk_length = len(shot["v"]) - (alarm.W - 1)
        support, known, _count = a1.physics_trace(
            int(shot["shot"]), scales, risk_length=risk_length
        )
        times = float(shot["start"]) + np.arange(alarm.W - 1, len(shot["v"])) / float(shot["rate"])
        labels = dynamic.alarm_window_labels(
            times, is_disrupt=bool(shot["is_disrupt"]),
            down_time_s=shot["down"], horizon_ms=125.0,
        )
        positive_support.extend(support[labels & known].tolist())
    if not positive_support:
        raise RuntimeError("support set has no physics-observable positive windows")
    quantile = 1.0 - float(preserve_fraction)
    return {
        "preserve_fraction": float(preserve_fraction),
        "lower_quantile": quantile,
        "minimum_support": float(np.quantile(positive_support, quantile)),
        "positive_window_count": len(positive_support),
    }


def causal_score_shot(
    controller: two_scale.TwoTimescaleAdapterController,
    *,
    shot_id: int,
    base_risk: np.ndarray,
    endpoints: np.ndarray,
    frozen: np.ndarray,
    base_logit: np.ndarray,
    physics_at_endpoint: np.ndarray,
    available_at_endpoint: np.ndarray,
    full_physics: np.ndarray,
    full_available: np.ndarray,
    physics_support_at_endpoint: np.ndarray,
    physics_known_at_endpoint: np.ndarray,
    minimum_fast_physics_support: float | None,
    use_intra: bool,
    adaptation_events: int,
    intra_steps: int,
) -> tuple[np.ndarray, list[dict]]:
    controller.begin_shot(shot_id)
    adjusted = np.asarray(base_risk, dtype=np.float32).copy()
    risk_indices = endpoints - (alarm.W - 1)
    receipts: list[dict] = []

    def score_positions(positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        if not len(positions):
            return
        batch = make_batch(
            frozen, base_logit, physics_at_endpoint, available_at_endpoint, positions,
        )
        fast_probability = torch.sigmoid(controller.predict(batch))
        if minimum_fast_physics_support is not None:
            persistent_probability = torch.sigmoid(controller.predict_persistent(batch))
            fast_probability = two_scale.protect_positive_fast_increment(
                fast_probability, persistent_probability,
                physics_support=torch.from_numpy(physics_support_at_endpoint[positions]),
                physics_known=torch.from_numpy(physics_known_at_endpoint[positions]),
                minimum_support=minimum_fast_physics_support,
            )
        adjusted[risk_indices[positions]] = fast_probability.numpy()

    legal_sources = np.flatnonzero(endpoints + dha.LAG < len(full_physics))
    safe_next = np.minimum(endpoints + dha.LAG, len(full_physics) - 1)
    if use_intra and len(legal_sources):
        source_positions = dha.evenly_limit(legal_sources, adaptation_events)
        next_unscored = 0
        for source_position in source_positions:
            observed_endpoint = int(endpoints[source_position] + dha.LAG)
            visible_end = int(np.searchsorted(endpoints, observed_endpoint, side="right"))
            current_position = visible_end - 1
            if current_position > next_unscored:
                score_positions(np.arange(next_unscored, current_position))
            # x_t and its physics proxies are now observable.  Adapt the fast
            # copy before predicting t, without revising any earlier endpoint.
            current_batch = make_batch(
                frozen, base_logit, physics_at_endpoint, available_at_endpoint,
                np.asarray([current_position]),
            )
            receipts.append(controller.intra_physics_update(
                batch=current_batch, steps=intra_steps,
            ))
            score_positions(np.asarray([current_position]))
            next_unscored = visible_end
            source_batch = make_batch(
                frozen, base_logit, physics_at_endpoint, available_at_endpoint,
                np.asarray([source_position]),
                next_physics=full_physics[safe_next],
                next_available=full_available[safe_next],
            )
            receipts.append(controller.intra_update(
                source_endpoint=int(endpoints[source_position]),
                observed_endpoint=observed_endpoint,
                batch=source_batch,
                steps=intra_steps,
            ))
        if next_unscored < len(endpoints):
            score_positions(np.arange(next_unscored, len(endpoints)))
    else:
        batch = make_batch(
            frozen, base_logit, physics_at_endpoint, available_at_endpoint,
            np.arange(len(endpoints)),
        )
        adjusted[risk_indices] = torch.sigmoid(controller.predict(batch)).numpy()
    return adjusted, receipts


def confusion(labels: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.sum(labels & predicted)), "FP": int(np.sum(~labels & predicted)),
        "FN": int(np.sum(labels & ~predicted)), "TN": int(np.sum(~labels & ~predicted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--intra-lr", type=float, default=1e-4)
    parser.add_argument("--inter-lr", type=float, default=5e-4)
    parser.add_argument("--adaptation-events", type=int, default=24)
    parser.add_argument("--intra-steps", type=int, default=1)
    parser.add_argument("--inter-steps", type=int, default=2)
    parser.add_argument("--max-guard-harms", type=int, default=0)
    parser.add_argument("--physics-constraint-weight", type=float, default=0.10)
    parser.add_argument("--intra-physics-weight", type=float, default=0.10)
    parser.add_argument("--physics-increment-gate", action="store_true")
    parser.add_argument("--negative-max-physics-support", type=float, default=0.10)
    parser.add_argument("--positive-min-physics-support", type=float, default=0.65)
    parser.add_argument("--value-config", type=Path, default=None)
    parser.add_argument(
        "--fast-physics-preserve", type=float, default=None,
        help="support-positive fraction used to fit the positive fast-increment physics gate",
    )
    args = parser.parse_args()
    args.out = args.out or HERE / f"two_timescale_online_seed{args.seed}"
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)

    a1_dir = HERE / f"online_active_a1_seed{args.seed}"
    adapter_dir = HERE / f"dynamic_hazard_adapter_seed{args.seed}"
    gate_dir = HERE / f"dynamic_hazard_adapter_seed{args.seed}_positive_gate"
    a1_config = json.loads((a1_dir / "config.json").read_text())
    threshold = float(a1_config["threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    scales = a1_config["innovation_scales"]
    unlabeled_roster = [int(row["shot"]) for row in load_jsonl(a1_dir / "seals.jsonl")]
    value_config = None
    auxiliary_by_shot = None
    if args.value_config is not None:
        value_config = json.loads(args.value_config.read_text())
        if value_config.get("claim_boundary") != "RETROSPECTIVE_DESIGNED_PREQUENTIAL_360":
            raise ValueError("value config has the wrong retrospective design boundary")
        auxiliary_rows = load_jsonl(
            HERE / f"online_active_a3_seed{args.seed}_intrareset" / "shot_seals.jsonl"
        )
        forbidden = {"is_disrupt", "label", "outcome", "revealed_is_disrupt"}
        if any(forbidden.intersection(row) for row in auxiliary_rows):
            raise ValueError("value auxiliary ledger contains outcome columns")
        if [int(row["shot"]) for row in auxiliary_rows] != unlabeled_roster:
            raise RuntimeError("value auxiliary ledger does not match the ordered stream")
        auxiliary_by_shot = {int(row["shot"]): row for row in auxiliary_rows}

    print(f"[TWO-SCALE] rebuild seed={args.seed} device={alarm.DEV}", flush=True)
    enc, head, transfer, med, scale = alarm.train_support_only(args.seed)
    if alarm.DEV != "cuda":
        raise RuntimeError("registered two-timescale run requires rl_env CUDA")
    base_hash = a1.model_sha256(enc, head)
    if base_hash != a1_config["model_sha256"]:
        raise RuntimeError("frozen base differs from A1")
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    stream = [load_east_signal_only(shot_id, med, scale) for shot_id in unlabeled_roster]
    vault = audit.LazyShotOutcomeOracle(unlabeled_roster, read_hdf5_outcome)

    print("[TWO-SCALE] build 110-shot support replay/guard", flush=True)
    support_cache = dha.build_support_cache(enc, head, transfer, scales)
    fast_physics_contract = (
        fit_fast_physics_support_threshold(
            transfer, scales, preserve_fraction=args.fast_physics_preserve
        )
        if args.fast_physics_preserve is not None else None
    )
    guard_rng = np.random.default_rng(args.seed + 12000)
    guard = dha.to_device(dha.sample_examples(
        support_cache, guard_rng, batch_size=512, balanced=True
    ))
    checkpoint = torch.load(adapter_dir / "adapter.pt", map_location=alarm.DEV, weights_only=True)
    if checkpoint["base_sha256"] != base_hash:
        raise RuntimeError("adapter parent base mismatch")
    parent = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM, physics_dim=dha.PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(alarm.DEV)
    parent.load_state_dict(checkpoint["adapter"])
    parent_hash = auxiliary.module_sha256(parent)
    minimum_positive, support_positive_count = positive_gate.support_positive_threshold(
        parent, support_cache, 0.10,
    )
    query_cutoff = fit_support_query_cutoff(
        enc, head, transfer, scales,
        threshold=threshold, quantile=float(a1_config["query_quantile"]),
    )
    effective_query_cutoff = (
        float(value_config["cutoff"]) if value_config is not None else query_cutoff
    )
    query_selector = audit.BlockBudgetSelector(
        cutoff=effective_query_cutoff, block_size=20, quota=4
    )

    controllers = {
        mode: two_scale.TwoTimescaleAdapterController(
            copy.deepcopy(parent), guard_batch=guard,
            intra_lr=args.intra_lr, inter_lr=args.inter_lr,
            risk_anchor_weight=0.25, guard_anchor_weight=0.25,
            max_guard_harms=args.max_guard_harms,
            physics_constraint_weight=args.physics_constraint_weight,
            intra_physics_weight=args.intra_physics_weight,
            physics_increment_gate=args.physics_increment_gate,
            negative_max_physics_support=args.negative_max_physics_support,
            positive_min_physics_support=args.positive_min_physics_support,
            minimum_positive_residual=minimum_positive,
            threshold_logit=threshold_logit,
        )
        for mode in MODES
    }
    initial_hashes = {mode: controller.persistent_hash() for mode, controller in controllers.items()}
    writers = {
        mode: a1.JsonlSealWriter(args.out / f"predictions_{mode}.jsonl") for mode in MODES
    }
    update_writer = a1.JsonlSealWriter(args.out / "updates.jsonl")
    query_writer = a1.JsonlSealWriter(args.out / "query_decisions.jsonl")
    reveal_writer = a1.JsonlSealWriter(args.out / "reveals.jsonl")
    records = {mode: [] for mode in MODES}
    update_counts = {
        mode: {"intra": 0, "inter_accepted": 0, "inter_rollback": 0, "protection_only": 0}
        for mode in MODES
    }
    replay_rng = np.random.default_rng(args.seed + 13000)
    replay_buffers = {
        mode: two_scale.CausalReplayBuffer(capacity_shots=64)
        for mode in ("inter_only", "intra_inter")
    }

    try:
        for stream_index, shot in enumerate(stream):
            shot_id = int(shot["shot"])
            endpoints = np.arange(alarm.W - 1 + dha.LAG, len(shot["v"]), dtype=np.int64)
            frozen, direct_logit = dha.extract_frozen_batch(enc, head, shot["v"], endpoints)
            base_risk = np.load(a1_dir / "risk_traces" / f"{shot_id}.npz")["risk"]
            risk_indices = endpoints - (alarm.W - 1)
            if np.max(np.abs(torch.sigmoid(torch.from_numpy(direct_logit)).numpy() - base_risk[risk_indices])) > 1e-5:
                raise RuntimeError(f"shot {shot_id}: base risk trace mismatch")
            sealed_logit = np.log(
                np.clip(base_risk[risk_indices], 1e-6, 1 - 1e-6)
                / np.clip(1 - base_risk[risk_indices], 1e-6, 1 - 1e-6)
            ).astype(np.float32)
            physics, available = a3.physics_family_trace(shot_id, scales)
            physics_at_endpoint = physics[endpoints]
            available_at_endpoint = available[endpoints]
            family_scores = {
                family: physics[:, index] for index, family in enumerate(a3.FAMILIES)
            }
            family_available = {
                family: available[:, index] for index, family in enumerate(a3.FAMILIES)
            }
            physics_support, physics_known, _family_count = audit.combine_family_scores(
                family_scores, family_available
            )
            adjusted_by_mode = {}
            shot_rows = {}
            for mode, controller in controllers.items():
                adjusted, intra_receipts = causal_score_shot(
                    controller,
                    shot_id=shot_id, base_risk=base_risk, endpoints=endpoints,
                    frozen=frozen, base_logit=sealed_logit,
                    physics_at_endpoint=physics_at_endpoint,
                    available_at_endpoint=available_at_endpoint,
                    full_physics=physics, full_available=available,
                    physics_support_at_endpoint=physics_support[endpoints],
                    physics_known_at_endpoint=physics_known[endpoints],
                    minimum_fast_physics_support=(
                        None if fast_physics_contract is None
                        else fast_physics_contract["minimum_support"]
                    ),
                    use_intra=mode in ("intra_only", "intra_inter"),
                    adaptation_events=args.adaptation_events,
                    intra_steps=args.intra_steps,
                )
                adjusted_by_mode[mode] = adjusted
                seal = controller.seal_shot(shot_id)
                update_counts[mode]["intra"] += len(intra_receipts)
                for receipt in intra_receipts:
                    update_writer.append({"mode": mode, **receipt})
                score = float(np.max(adjusted))
                row = {
                    "stream_index": stream_index, "shot": shot_id,
                    "score": score, "threshold": threshold,
                    "queried": False,
                    "persistent_sha256_before_prediction": seal["persistent_sha256"],
                    "fast_sha256_at_seal": seal["fast_sha256"],
                }
                writers[mode].append(row)
                records[mode].append(row)
                shot_rows[mode] = row

            vault.seal_current(shot_id)
            query_components = audit.query_components(
                risk=base_risk,
                physics_support=physics_support[alarm.W - 1:],
                physics_known=physics_known[alarm.W - 1:],
                threshold=threshold,
            )
            value_features = None
            transformed_value = None
            if value_config is not None:
                value_features = reverse_value.runtime_value_features(
                    query_components=query_components,
                    auxiliary_row=auxiliary_by_shot[shot_id],
                    risk=base_risk,
                )
                query_score, transformed_value = reverse_value.score_value_row(
                    value_features, value_config
                )
            else:
                query_score = float(query_components["query_score"])
            selected = query_selector.decide(score=query_score, stream_index=stream_index)
            query_writer.append({
                "stream_index": stream_index, "shot": shot_id,
                "query_score": query_score,
                "selected": bool(selected),
                "value_features": value_features,
                "transformed_value_features": transformed_value,
            })
            for mode in MODES:
                shot_rows[mode]["queried"] = bool(selected)

            if selected:
                revealed = vault.reveal_for_query(shot_id, selected=True)
                reveal_writer.append({
                    "stream_index": stream_index, "shot": shot_id,
                    "revealed_is_disrupt": revealed["is_disrupt"],
                    "revealed_down_time_s": revealed["down_time_s"],
                })
                legal = np.flatnonzero(endpoints + dha.LAG < len(physics))
                safe_next = np.minimum(endpoints + dha.LAG, len(physics) - 1)
                times = float(shot["start"]) + endpoints / float(shot["rate"])
                labels = dynamic.alarm_window_labels(
                    times, is_disrupt=revealed["is_disrupt"],
                    down_time_s=revealed["down_time_s"], horizon_ms=125.0,
                )
                for mode, controller in controllers.items():
                    if mode in ("inter_only", "intra_inter"):
                        logits = np.log(
                            np.clip(adjusted_by_mode[mode][risk_indices], 1e-6, 1 - 1e-6)
                            / np.clip(1 - adjusted_by_mode[mode][risk_indices], 1e-6, 1 - 1e-6)
                        )
                        chosen_local = old_online.select_revealed_windows(
                            logits[legal], labels[legal], count=16
                        )
                        chosen = legal[chosen_local]
                        current = make_batch(
                            frozen, sealed_logit, physics_at_endpoint, available_at_endpoint,
                            chosen,
                            next_physics=physics[safe_next],
                            next_available=available[safe_next],
                            labels=labels,
                        )
                        predicted_disrupt = bool(shot_rows[mode]["score"] >= threshold)
                        actual_disrupt = bool(revealed["is_disrupt"])
                        group = (
                            "TP" if actual_disrupt and predicted_disrupt
                            else "FN" if actual_disrupt
                            else "FP" if predicted_disrupt else "TN"
                        )
                        if group in ("FP", "FN"):
                            online_replay = replay_buffers[mode].sample_opposite_bag(
                                current_is_disruptive=actual_disrupt,
                                count=8, seed=args.seed * 100000 + stream_index
                            )
                            support_count = 16 if online_replay is None else 8
                            replay_np = dha.sample_examples(
                                support_cache, replay_rng,
                                batch_size=support_count, balanced=True,
                            )
                            support_replay = {
                                key: torch.from_numpy(value) for key, value in replay_np.items()
                            }
                            receipt = controller.reveal_and_inter_update(
                                shot_id, selected=True, current_batch=current,
                                replay_batch=support_replay,
                                opposite_batch=online_replay, steps=args.inter_steps,
                            )
                            update_counts[mode][
                                "inter_accepted" if receipt["accepted"] else "inter_rollback"
                            ] += 1
                            receipt["error_group"] = group
                        else:
                            receipt = controller.finalize_revealed_protection(
                                shot_id, group=group
                            )
                            update_counts[mode]["protection_only"] += 1
                        receipt["online_replay_shots_before_update"] = len(
                            replay_buffers[mode].shots
                        )
                        replay_buffers[mode].add(shot_id, current)
                    else:
                        receipt = controller.discard_without_reveal(shot_id)
                    update_writer.append({"mode": mode, **receipt})
            else:
                for mode, controller in controllers.items():
                    update_writer.append({"mode": mode, **controller.discard_without_reveal(shot_id)})
                vault.discard_unqueried(shot_id)

            if (stream_index + 1) % 20 == 0:
                print(
                    f"[TWO-SCALE] sealed {stream_index + 1}/360 "
                    f"queries={query_selector.total_selected}", flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()
        update_writer.close(); query_writer.close(); reveal_writer.close()

    if query_selector.total_selected != 72:
        raise RuntimeError("causal query budget failed")
    if a1.model_sha256(enc, head) != base_hash:
        raise RuntimeError("online adaptation changed frozen base")
    vault.open_final_evaluation()
    outcomes = vault.all_outcomes()
    labels = np.asarray([outcomes[row["shot"]]["is_disrupt"] for row in records["static"]], bool)
    results = {}
    for mode in MODES:
        scores = np.asarray([row["score"] for row in records[mode]])
        matrix = confusion(labels, scores >= threshold)
        results[mode] = {
            "confusion": matrix,
            "tpr": matrix["TP"] / 180.0,
            "fpr": matrix["FP"] / 180.0,
            "mean_abs_score_change_vs_static": float(np.mean(np.abs(
                scores - np.asarray([row["score"] for row in records["static"]])
            ))),
            "initial_persistent_sha256": initial_hashes[mode],
            "final_persistent_sha256": controllers[mode].persistent_hash(),
            "updates": update_counts[mode],
            "predictions_sha256": a1.sha256_file(args.out / f"predictions_{mode}.jsonl"),
        }
    static_reference = {
        int(row["shot"]): float(row["adapted_score"])
        for row in load_jsonl(gate_dir / "predictions.jsonl")
    }
    static_delta = max(
        abs(float(row["score"]) - static_reference[int(row["shot"])])
        for row in records["static"]
    )
    if static_delta > 1e-5:
        raise RuntimeError(f"static positive-gate reproduction failed: {static_delta}")
    if results["static"]["initial_persistent_sha256"] != results["static"]["final_persistent_sha256"]:
        raise RuntimeError("static arm changed persistent parameters")
    if results["intra_only"]["initial_persistent_sha256"] != results["intra_only"]["final_persistent_sha256"]:
        raise RuntimeError("intra-only arm leaked fast updates across shots")

    summary = {
        "status": "ONLINE_DIAGNOSTIC",
        "stage": "TWO_TIMESCALE_INTRA_INTER_PREQUENTIAL_SCREEN",
        "seed": args.seed,
        "stream_shots": 360,
        "query_count": query_selector.total_selected,
        "query_cutoff": effective_query_cutoff,
        "query_cutoff_source": (
            "retrospective 360 temporal-block value fit"
            if value_config is not None else "label-free q80 on fixed 110-shot support"
        ),
        "threshold": threshold,
        "threshold_source": "fixed 110-shot support",
        "hyperparameters": {
            "intra_lr": args.intra_lr, "inter_lr": args.inter_lr,
            "adaptation_events": args.adaptation_events,
            "intra_steps": args.intra_steps, "inter_steps": args.inter_steps,
            "max_guard_harms": args.max_guard_harms,
            "physics_constraint_weight": args.physics_constraint_weight,
            "intra_physics_weight": args.intra_physics_weight,
            "physics_increment_gate": args.physics_increment_gate,
            "negative_max_physics_support": args.negative_max_physics_support,
            "positive_min_physics_support": args.positive_min_physics_support,
        },
        "adaptation_contract": {
            "fast_scope": "projector.3 + next_head + residual_head; reset from persistent each shot",
            "slow_scope": "residual_head only",
            "slow_objective": "current shot-level MIL + prior opposite-outcome shot-level MIL + sealed-logit support distillation",
            "residual_parameterization": "signed online increment around immutable gated parent",
            "prediction_order": "observe x_t -> unlabeled fast update -> predict t -> seal shot -> query -> reveal -> slow update next shots only",
        },
        "fast_physics_contract": fast_physics_contract,
        "retrospective_value_config": (
            None if args.value_config is None else {
                "path": str(args.value_config),
                "sha256": a1.sha256_file(args.value_config),
                "reference_weights": value_config["reference_weights"],
                "weight_ranges": value_config["weight_ranges"],
            }
        ),
        "results": results,
        "static_reference_max_abs_delta": static_delta,
        "base_model_sha256_before_after": base_hash,
        "parent_adapter_sha256": parent_hash,
        "support_positive_gate": {
            "quantile": 0.10,
            "minimum_positive_residual": minimum_positive,
            "positive_window_count": support_positive_count,
        },
        "online_replay": {
            "capacity_shots": 64,
            "support_windows_per_update_after_warmup": 8,
            "past_opposite_outcome_east_windows_per_update_after_warmup": 8,
            "final_buffer_shots": {
                mode: len(buffer.shots) for mode, buffer in replay_buffers.items()
            },
        },
        "hashes": {
            "unlabeled_query_ledger": a1.sha256_file(a1_dir / "seals.jsonl"),
            "query_decisions": a1.sha256_file(args.out / "query_decisions.jsonl"),
            "reveals": a1.sha256_file(args.out / "reveals.jsonl"),
            "updates": a1.sha256_file(args.out / "updates.jsonl"),
        },
        "leakage_contract": (
            "stream objects exclude outcome and DownTime; all four arms seal the current shot "
            "before the causal query selector consumes its current unlabeled score; outcome and "
            "DownTime are revealed only for selected shots; updates affect only later times/shots"
        ),
        "claim_boundary": (
            "balanced360 is a method-development online stream. Metrics are test-driven diagnostics; "
            "generalization requires a new untouched chronological EAST stream."
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
