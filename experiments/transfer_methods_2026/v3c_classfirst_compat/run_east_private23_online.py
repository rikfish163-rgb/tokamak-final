#!/usr/bin/env python3
"""Shared rebuilt17 transfer plus an EAST-private raw8 residual online stream."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))

import dynamic_hazard_adapter as dynamic  # noqa: E402
import east_private23_sidecar as private  # noqa: E402
import evaluate_dynamic_positive_gate as positive_gate  # noqa: E402
import online_active_audit as audit  # noqa: E402
import online_auxiliary as auxiliary  # noqa: E402
import run_dynamic_hazard_adapter as dha  # noqa: E402
import run_online_active_a1 as a1  # noqa: E402
import run_online_protected_residual as old_online  # noqa: E402
import run_two_timescale_online as two_runner  # noqa: E402
import shot_alarm as alarm  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw8_signal_only(shot_id: int) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / "EAST/unified_hdf5" / str(shot_id // 100 * 100) / f"{shot_id}.hdf5"
    columns = []
    with h5py.File(path, "r") as handle:
        for name in private.DROPPED_RAW8:
            columns.append(np.asarray(handle[f"data/{name}"][:], dtype=np.float32))
    values = np.stack(columns, axis=-1)
    available = np.isfinite(values)
    ciii = private.DROPPED_RAW8.index("CIII")
    available[:, ciii] &= ~np.isclose(values[:, ciii], -10.0, atol=1e-2)
    for name in ("MNM", "Mir_VV", "Mir_avg_amp"):
        index = private.DROPPED_RAW8.index(name)
        observed = values[available[:, index], index]
        if len(observed) and np.allclose(observed, 0.0):
            available[:, index] = False
    return values, available


def fit_raw8_scaler(shot_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    per_channel: list[list[np.ndarray]] = [[] for _ in private.DROPPED_RAW8]
    for shot_id in shot_ids:
        values, available = raw8_signal_only(shot_id)
        positions = np.linspace(0, len(values) - 1, min(len(values), 4096), dtype=np.int64)
        for channel in range(values.shape[1]):
            keep = available[positions, channel]
            if keep.any():
                per_channel[channel].append(values[positions[keep], channel])
    median = np.zeros(len(per_channel), dtype=np.float32)
    scale = np.ones(len(per_channel), dtype=np.float32)
    for channel, chunks in enumerate(per_channel):
        if not chunks:
            continue
        joined = np.concatenate(chunks)
        median[channel] = np.median(joined)
        q25, q75 = np.percentile(joined, [25, 75])
        scale[channel] = max(float(q75 - q25), 1e-6)
    return median, scale


def scale_raw8(
    values: np.ndarray, available: np.ndarray,
    median: np.ndarray, scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.clip((values - median) / scale, -10, 10)
    scaled = np.where(available, scaled, 0.0).astype(np.float32)
    return scaled, available.astype(bool)


def window_batch(
    values: np.ndarray, available: np.ndarray, indices: np.ndarray,
    base_logit: np.ndarray, *, target: np.ndarray | None = None,
    sealed_logit: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    indices = np.asarray(indices, dtype=np.int64)
    endpoints = alarm.W - 1 + indices
    raw = np.stack([values[end - alarm.W + 1:end + 1] for end in endpoints])
    mask = np.stack([available[end - alarm.W + 1:end + 1] for end in endpoints])
    batch = {
        "raw": torch.from_numpy(raw), "available": torch.from_numpy(mask),
        "base_logit": torch.from_numpy(base_logit[indices].astype(np.float32)),
    }
    if target is not None:
        batch["target"] = torch.from_numpy(target[indices].astype(np.float32))
    if sealed_logit is not None:
        batch["sealed_logit"] = torch.from_numpy(
            sealed_logit[indices].astype(np.float32)
        )
    return batch


def temporal_targets(shot: dict, risk_length: int) -> tuple[np.ndarray, np.ndarray]:
    horizon_ms = float(shot.get("positive_horizon_ms", 125.0))
    if not math.isfinite(horizon_ms) or horizon_ms <= 0.0:
        raise ValueError(f"shot {shot.get('shot')}: invalid positive horizon")
    times = float(shot["start"]) + np.arange(alarm.W - 1, alarm.W - 1 + risk_length) / float(shot["rate"])
    binary = dynamic.alarm_window_labels(
        times, is_disrupt=bool(shot["is_disrupt"]),
        down_time_s=shot["down"], horizon_ms=horizon_ms,
    )
    soft = np.zeros(risk_length, dtype=np.float32)
    if shot["is_disrupt"] and shot["down"] is not None:
        ttd = (float(shot["down"]) - times) * 1000.0
        legal = (ttd >= 0.0) & (ttd <= horizon_ms)
        soft[legal] = np.clip(
            (horizon_ms - ttd[legal]) / horizon_ms, 0.0, 1.0,
        )
    return binary, soft


def concatenate(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = tuple(batches[0])
    if any(tuple(batch) != keys for batch in batches[1:]):
        raise ValueError("private batches do not align")
    return {name: torch.cat([batch[name] for batch in batches]) for name in keys}


def select_error_coreset(
    sealed_probability: np.ndarray, labels: np.ndarray, *,
    threshold: float, per_error_type: int = 16,
) -> np.ndarray:
    """Select only sealed temporal FP/FN windows, balanced by error direction."""
    probability = np.asarray(sealed_probability, dtype=np.float32)
    target = np.asarray(labels, dtype=bool)
    predicted = probability >= float(threshold)
    false_negative = np.flatnonzero(target & ~predicted)
    false_positive = np.flatnonzero(~target & predicted)
    chosen = []
    for pool in (false_negative, false_positive):
        if len(pool):
            positions = np.linspace(
                0, len(pool) - 1, min(len(pool), int(per_error_type)), dtype=np.int64
            )
            chosen.extend(pool[positions].tolist())
    if not chosen:
        raise RuntimeError("error coreset requested for a shot without temporal errors")
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def predict_trace(
    model: private.EastPrivateResidual, values: np.ndarray,
    available: np.ndarray, base_risk: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.clip(base_risk, 1e-6, 1 - 1e-6)
    base_logit = np.log(base / (1 - base)).astype(np.float32)
    adjusted = np.empty_like(base_risk, dtype=np.float32)
    residual = np.empty_like(base_risk, dtype=np.float32)
    model.eval()
    for start in range(0, len(base_risk), 512):
        indices = np.arange(start, min(start + 512, len(base_risk)))
        batch = window_batch(values, available, indices, base_logit)
        with torch.no_grad():
            logit, delta, _features = model(
                batch["raw"].to(device), batch["available"].to(device),
                batch["base_logit"].to(device),
            )
        adjusted[indices] = torch.sigmoid(logit).cpu().numpy()
        residual[indices] = delta.cpu().numpy()
    return adjusted, residual


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.sum(labels & predictions)), "FP": int(np.sum(~labels & predictions)),
        "FN": int(np.sum(labels & ~predictions)), "TN": int(np.sum(~labels & ~predictions)),
    }


def result(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    matrix = confusion(labels, scores >= threshold)
    return {
        "confusion": matrix, "tpr": matrix["TP"] / int(labels.sum()),
        "fpr": matrix["FP"] / int((~labels).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path, default=HERE / "east_private23_online_seed131_v1")
    parser.add_argument("--support-epochs", type=int, default=12)
    parser.add_argument("--online-lr", type=float, default=2e-4)
    parser.add_argument("--online-scope", choices=("head", "all"), default="all")
    parser.add_argument(
        "--window-selection", choices=("balanced16", "error_coreset"),
        default="balanced16",
    )
    parser.add_argument("--query-ledger", type=Path, default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    device = torch.device(alarm.DEV)
    if device.type != "cuda":
        raise RuntimeError("EAST-private experiment requires rl_env CUDA")

    a1_dir = HERE / f"online_active_a1_seed{args.seed}"
    adapter_dir = HERE / f"dynamic_hazard_adapter_seed{args.seed}"
    config = json.loads((a1_dir / "config.json").read_text())
    threshold = float(config["threshold"])
    scales = config["innovation_scales"]
    roster = [int(row["shot"]) for row in load_jsonl(a1_dir / "seals.jsonl")]
    query_ledger = (
        args.query_ledger
        if args.query_ledger is not None
        else a1_dir / "queries.jsonl"
    )
    query_rows = load_jsonl(query_ledger)
    selected_shots = {
        int(row["shot"]) for row in query_rows if bool(row.get("selected", True))
    }
    if len(roster) != 360 or len(selected_shots) != 72:
        raise RuntimeError("private23 requires the fixed balanced360/72-query contract")

    print(f"[PRIVATE23] rebuild shared17 seed={args.seed} device={device}", flush=True)
    enc, head, transfer, med17, scale17 = alarm.train_support_only(args.seed)
    base_hash = a1.model_sha256(enc, head)
    if base_hash != config["model_sha256"]:
        raise RuntimeError("private23 shared parent differs from A1")
    for module in (enc, head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    adapter = dynamic.DynamicHazardAdapter(
        feature_dim=dha.FEATURE_DIM, physics_dim=dha.PHYSICS_DIM, hidden=16,
        max_logit_residual=1.0, min_physics_families=2,
    ).to(device)
    checkpoint = torch.load(adapter_dir / "adapter.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(checkpoint["adapter"]); adapter.eval()
    adapter_hash = auxiliary.module_sha256(adapter)
    support_cache = dha.build_support_cache(enc, head, transfer, scales)
    minimum_positive, _positive_count = positive_gate.support_positive_threshold(
        adapter, support_cache, 0.10
    )

    support_ids = [int(shot["shot"]) for shot in transfer]
    median8, scale8 = fit_raw8_scaler(support_ids)
    support_batches = []
    support_guard = []
    print("[PRIVATE23] build HDF5-authoritative 110-shot private support", flush=True)
    for shot in transfer:
        shot_id = int(shot["shot"])
        base_risk = alarm.full_risk(enc, head, shot)
        shared_risk, _delta = dha.score_shot(
            adapter, enc, head, shot, scales, base_risk,
            minimum_positive_residual=minimum_positive,
        )
        binary, soft = temporal_targets(shot, len(shared_risk))
        base_logit = np.log(np.clip(shared_risk, 1e-6, 1 - 1e-6) /
                            np.clip(1 - shared_risk, 1e-6, 1 - 1e-6))
        chosen = old_online.select_revealed_windows(base_logit, binary, count=16)
        raw, available = raw8_signal_only(shot_id)
        raw, available = scale_raw8(raw, available, median8, scale8)
        batch = window_batch(
            raw, available, chosen, base_logit,
            target=soft, sealed_logit=base_logit,
        )
        batch["shot_label"] = torch.full((len(chosen),), float(shot["is_disrupt"]))
        support_batches.append(batch)
    support = concatenate(*support_batches)

    torch.manual_seed(args.seed + 23000)
    pretrained = private.EastPrivateResidual(channel_count=8, hidden=16).to(device)
    if sum(parameter.numel() for parameter in pretrained.parameters()) >= 5000:
        raise RuntimeError("EAST-private parameter budget exceeded")
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = torch.nonzero(support["target"] > 0, as_tuple=False).flatten()
    negative = torch.nonzero(support["target"] <= 0, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(args.seed + 23001)
    print(f"[PRIVATE23] train private raw8 support epochs={args.support_epochs}", flush=True)
    for epoch in range(args.support_epochs):
        pretrained.train()
        losses = []
        for _step in range(max(1, len(support["target"]) // 64)):
            pos_index = positive[torch.randint(len(positive), (32,), generator=generator)]
            neg_index = negative[torch.randint(len(negative), (32,), generator=generator)]
            index = torch.cat([pos_index, neg_index])
            current, _residual, _features = pretrained(
                support["raw"].index_select(0, index).to(device),
                support["available"].index_select(0, index).to(device),
                support["base_logit"].index_select(0, index).to(device),
            )
            loss = private.sealed_temporal_residual_loss(
                current, support["sealed_logit"].index_select(0, index).to(device),
                support["target"].index_select(0, index).to(device),
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrained.parameters(), 0.5)
            optimizer.step(); losses.append(float(loss.detach()))
        print(f"[PRIVATE23] epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)
    pretrained.eval()
    pretrained_hash = auxiliary.module_sha256(pretrained)
    fixed = copy.deepcopy(pretrained).eval()
    online = copy.deepcopy(pretrained).train()
    if args.online_scope == "head":
        for parameter in online.parameters():
            parameter.requires_grad_(False)
        for parameter in online.network[-1].parameters():
            parameter.requires_grad_(True)
    online_parameters = [parameter for parameter in online.parameters() if parameter.requires_grad]
    online_optimizer = torch.optim.AdamW(
        online_parameters, lr=args.online_lr, weight_decay=1e-4
    )
    replay = private.SameErrorReplayBuffer(capacity_shots=64)

    # Correct support predictions are immutable guard bags for online updates.
    for shot, batch in zip(transfer, support_batches):
        with torch.no_grad():
            logits, _delta, _features = online(
                batch["raw"].to(device), batch["available"].to(device),
                batch["base_logit"].to(device),
            )
        predicted = bool((torch.sigmoid(logits).max() >= threshold).item())
        if predicted == bool(shot["is_disrupt"]):
            support_guard.append({
                "batch": batch, "predicted": predicted, "shot": int(shot["shot"]),
            })

    vault = audit.LazyShotOutcomeOracle(roster, two_runner.read_hdf5_outcome)
    prediction_writer = a1.JsonlSealWriter(args.out / "predictions.jsonl")
    update_writer = a1.JsonlSealWriter(args.out / "updates.jsonl")
    rows = []
    online_guards = []
    try:
        for stream_index, shot_id in enumerate(roster):
            shot = two_runner.load_east_signal_only(shot_id, med17, scale17)
            base_risk = np.load(a1_dir / "risk_traces" / f"{shot_id}.npz")["risk"]
            shared_risk, _shared_delta = dha.score_shot(
                adapter, enc, head, shot, scales, base_risk,
                minimum_positive_residual=minimum_positive,
            )
            raw, available = raw8_signal_only(shot_id)
            raw, available = scale_raw8(raw, available, median8, scale8)
            fixed_risk, _fixed_delta = predict_trace(fixed, raw, available, shared_risk, device)
            online_risk, online_delta = predict_trace(online, raw, available, shared_risk, device)
            row = {
                "stream_index": stream_index, "shot": shot_id,
                "shared_score": float(shared_risk.max()),
                "private110_score": float(fixed_risk.max()),
                "private_online_score": float(online_risk.max()),
                "queried": shot_id in selected_shots,
                "online_model_sha256_before_prediction": auxiliary.module_sha256(online),
                "max_abs_private_residual": float(np.max(np.abs(online_delta))),
            }
            prediction_writer.append(row); rows.append(row)
            vault.seal_current(shot_id)
            if shot_id not in selected_shots:
                vault.discard_unqueried(shot_id)
            else:
                outcome = vault.reveal_for_query(shot_id, selected=True)
                revealed = {
                    **shot, "is_disrupt": bool(outcome["is_disrupt"]),
                    "down": outcome["down_time_s"],
                }
                binary, soft = temporal_targets(revealed, len(shared_risk))
                sealed_logit = np.log(np.clip(online_risk, 1e-6, 1 - 1e-6) /
                                      np.clip(1 - online_risk, 1e-6, 1 - 1e-6))
                base_logit = np.log(np.clip(shared_risk, 1e-6, 1 - 1e-6) /
                                    np.clip(1 - shared_risk, 1e-6, 1 - 1e-6))
                predicted = row["private_online_score"] >= threshold
                actual = bool(outcome["is_disrupt"])
                group = "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
                chosen = (
                    select_error_coreset(
                        online_risk, binary, threshold=threshold, per_error_type=16
                    )
                    if args.window_selection == "error_coreset" and group in ("FP", "FN")
                    else old_online.select_revealed_windows(
                        sealed_logit, binary, count=16
                    )
                )
                current_batch = window_batch(
                    raw, available, chosen, base_logit,
                    target=soft, sealed_logit=sealed_logit,
                )
                receipt = {"stream_index": stream_index, "shot": shot_id, "group": group}
                if group in ("FP", "FN"):
                    state_before = copy.deepcopy(online.state_dict())
                    optimizer_before = copy.deepcopy(online_optimizer.state_dict())
                    same_error = replay.sample(
                        error_group=group, count=16,
                        seed=args.seed * 100000 + stream_index,
                    )
                    train_batch = (
                        current_batch if same_error is None
                        else concatenate(current_batch, same_error)
                    )
                    losses = []
                    online.train()
                    for _step in range(2):
                        current, _residual, _features = online(
                            train_batch["raw"].to(device),
                            train_batch["available"].to(device),
                            train_batch["base_logit"].to(device),
                        )
                        loss = private.sealed_temporal_residual_loss(
                            current, train_batch["sealed_logit"].to(device),
                            train_batch["target"].to(device),
                        )
                        online_optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(online_parameters, 0.5)
                        online_optimizer.step(); losses.append(float(loss.detach()))
                    harms = 0
                    online.eval()
                    with torch.no_grad():
                        for guard in [*support_guard, *online_guards]:
                            batch = guard["batch"]
                            logits, _delta, _features = online(
                                batch["raw"].to(device), batch["available"].to(device),
                                batch["base_logit"].to(device),
                            )
                            guard_prediction = bool((torch.sigmoid(logits).max() >= threshold).item())
                            harms += int(guard_prediction != guard["predicted"])
                    accepted = harms == 0
                    if not accepted:
                        online.load_state_dict(state_before)
                        online_optimizer.load_state_dict(optimizer_before)
                    replay.add(shot_id, error_group=group, batch=current_batch)
                    receipt.update({
                        "updated": bool(accepted), "accepted": bool(accepted),
                        "guard_harms": harms, "same_error_replay": same_error is not None,
                        "loss": float(np.mean(losses)),
                    })
                else:
                    online_guards.append({
                        "batch": current_batch, "predicted": bool(predicted), "shot": shot_id,
                    })
                    receipt.update({"updated": False, "reason": "protection_only"})
                receipt["online_model_sha256_after"] = auxiliary.module_sha256(online)
                update_writer.append(receipt)
            if (stream_index + 1) % 20 == 0:
                print(f"[PRIVATE23] sealed {stream_index + 1}/360", flush=True)
    finally:
        prediction_writer.close(); update_writer.close()

    vault.open_final_evaluation(); outcomes = vault.all_outcomes()
    labels = np.asarray([outcomes[int(row["shot"])]["is_disrupt"] for row in rows], bool)
    shared_scores = np.asarray([row["shared_score"] for row in rows])
    fixed_scores = np.asarray([row["private110_score"] for row in rows])
    online_scores = np.asarray([row["private_online_score"] for row in rows])
    results = {
        "shared17": result(labels, shared_scores, threshold),
        "shared17_plus_private8_110": result(labels, fixed_scores, threshold),
        "shared17_plus_private8_online": result(labels, online_scores, threshold),
    }
    summary = {
        "status": "ONLINE_DIAGNOSTIC", "stage": "EAST_PRIVATE_RAW8_ONLINE",
        "seed": args.seed, "threshold": threshold, "query_count": 72,
        "query_ledger": str(query_ledger),
        "query_ledger_sha256": a1.sha256_file(query_ledger),
        "private_channels": list(private.DROPPED_RAW8),
        "raw23_coverage": "rebuilt17 shared branch plus raw8 EAST-private branch",
        "results": results,
        "private_parameter_count": sum(p.numel() for p in online.parameters()),
        "online_scope": args.online_scope,
        "window_selection": args.window_selection,
        "online_trainable_parameter_count": sum(p.numel() for p in online_parameters),
        "support_hdf5_outcomes": {
            "disruptive": int(sum(bool(shot["is_disrupt"]) for shot in transfer)),
            "safe": int(sum(not bool(shot["is_disrupt"]) for shot in transfer)),
        },
        "pretrained_private_sha256": pretrained_hash,
        "final_online_private_sha256": auxiliary.module_sha256(online),
        "shared_base_sha256": base_hash, "shared_adapter_sha256": adapter_hash,
        "scaler_source": "raw8 median/IQR from fixed 110 EAST support only, equal per-shot cap",
        "supervision": "ea125 continuous temporal state from HDF5 after seal; FP/FN same-error replay",
        "claim_boundary": "RETROSPECTIVE_DESIGNED_PREQUENTIAL_360; not an untouched test",
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
