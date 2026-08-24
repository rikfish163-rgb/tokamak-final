#!/usr/bin/env python3
"""Train the rebuilt17 Transformer on J-TEXT before any EAST adaptation.

The checkpoint produced here is the only permitted parent for the current
``adapt_high`` and online-active chain.  Model selection reads TRAIN1353 and
VAL160 only.  TEST220 is loaded only after the checkpoint and threshold are
frozen, and HDF5 ``/meta`` is the label authority for every shot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import h5py


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
DATA_ROOT = Path(os.environ.get("FUSION_DATA_ROOT", ROOT / "data")).expanduser().resolve()
sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import advanced_time_labels as ATL  # noqa: E402
import jtext_online_io as JIO  # noqa: E402
import selected_mainline_contract as SELECTED  # noqa: E402


CANONICAL_INDEX = Path("artifacts/canonical/jtext_source_bundle_p131_v1/index.json")
EXPECTED_COUNTS = {"train": 1353, "val": 160, "test": 220}
SOURCE_HORIZON_MS = 125.0
MINIMUM_LEAD_MS = 10.0
NEGATIVE_MAX_MS = 60_000.0
JTEXT_ADVANCED_TIME = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_artifact_path(path: Path, *, anchor: Path) -> str:
    """Serialize generated artifact paths without binding them to one workstation."""
    resolved = path.expanduser().resolve()
    for base in (anchor.expanduser().resolve(), ROOT):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            pass
    return str(resolved)


def split_sha256(roles: dict[str, list[int]]) -> str:
    payload = json.dumps(roles, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_canonical_jtext_roles(root: Path = ROOT) -> dict[str, list[int]]:
    return JIO.load_canonical_jtext_roles(root)


def load_jtext_shots(
    shot_ids: list[int], *, root: Path = ROOT, data_root: Path | None = None,
) -> list[dict]:
    shots: list[dict] = []
    missing: list[int] = []
    resolved_data_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else Path(os.environ.get("FUSION_DATA_ROOT", root / "data")).expanduser().resolve()
    )
    for shot_id in shot_ids:
        path = JIO.jtext_shot_path(resolved_data_root, shot_id)
        try:
            signal = JIO.load_jtext_signal_only(resolved_data_root, shot_id)
        except Exception:  # noqa: BLE001 - fail closed with the complete list
            missing.append(int(shot_id))
            continue
        # TRAIN1353 and VAL160 are supervised roles.  Outcome access remains a
        # separate explicit HDF5 read; the signal loader above cannot carry it.
        with h5py.File(path, "r") as handle:
            raw_outcome = np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)
            if raw_outcome.size != 1:
                raise ValueError(f"J-TEXT shot {shot_id} has non-scalar IsDisrupt")
            is_disrupt = bool(raw_outcome[0])
            raw_down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
        down = float(raw_down[0]) if is_disrupt and raw_down.size else None
        if is_disrupt and (down is None or not math.isfinite(float(down))):
            raise ValueError(f"disruptive J-TEXT shot {shot_id} has no finite DownTime")
        shots.append({
            "shot": int(shot_id),
            "v": np.asarray(signal.values, dtype=np.float32).copy(),
            "a": np.asarray(signal.availability, dtype=bool).copy(),
            "is_disrupt": bool(is_disrupt),
            "down": float(down) if is_disrupt else None,
            "start": float(signal.start_time_s),
            "rate": float(signal.sample_rate_hz),
        })
    if missing:
        raise FileNotFoundError(f"missing J-TEXT HDF5 shots: {missing}")
    return shots


def causal_window(
    values: np.ndarray, *, endpoint: int, window: int = GS.W,
) -> tuple[np.ndarray, int]:
    """Return history ending at ``endpoint`` with causal left padding."""

    return JIO.causal_prefix_window(values, endpoint, window)


class BalancedSourceSampler:
    """Class-balanced, shot-balanced J-TEXT causal window sampler."""

    def __init__(
        self, shots: list[dict], *, seed: int,
        horizon_ms: float = SOURCE_HORIZON_MS,
    ) -> None:
        self.shots = shots
        self.rng = np.random.default_rng(int(seed))
        self.positive: dict[int, np.ndarray] = {}
        self.negative: dict[int, np.ndarray] = {}
        for index, shot in enumerate(shots):
            shot_horizon_ms = float(shot.get("positive_horizon_ms", horizon_ms))
            if not math.isfinite(shot_horizon_ms) or shot_horizon_ms <= 0.0:
                raise ValueError(f"shot {shot.get('shot')}: invalid positive horizon")
            endpoints = np.arange(len(shot["v"]), dtype=np.int64)
            if shot["is_disrupt"]:
                times = shot["start"] + endpoints / shot["rate"]
                ttd_ms = (float(shot["down"]) - times) * 1000.0
                positive = endpoints[
                    (ttd_ms >= 0.0) & (ttd_ms <= shot_horizon_ms)
                ]
                negative = endpoints[
                    (ttd_ms > shot_horizon_ms)
                    & (ttd_ms <= NEGATIVE_MAX_MS)
                ]
            else:
                positive = np.empty(0, dtype=np.int64)
                negative = endpoints
            if len(positive):
                self.positive[index] = positive
            if len(negative):
                self.negative[index] = negative
        if not self.positive or not self.negative:
            raise ValueError("source sampler requires both positive and negative windows")
        self.positive_shots = np.asarray(sorted(self.positive), dtype=np.int64)
        self.negative_shots = np.asarray(sorted(self.negative), dtype=np.int64)

    def _draw(self, positive: bool) -> tuple[np.ndarray, int, np.float32]:
        lookup = self.positive if positive else self.negative
        candidates = self.positive_shots if positive else self.negative_shots
        shot_index = int(candidates[int(self.rng.integers(len(candidates)))])
        endpoints = lookup[shot_index]
        endpoint = int(endpoints[int(self.rng.integers(len(endpoints)))])
        window, valid = causal_window(self.shots[shot_index]["v"], endpoint=endpoint)
        return window, valid, np.float32(1.0 if positive else 0.0)

    def sample_batch(
        self, batch_size: int, *, positive_fraction: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        positive_count = int(round(batch_size * float(positive_fraction)))
        positive_count = min(max(positive_count, 1), batch_size - 1)
        rows = [self._draw(True) for _ in range(positive_count)]
        rows.extend(self._draw(False) for _ in range(batch_size - positive_count))
        order = self.rng.permutation(batch_size)
        windows = np.stack([rows[index][0] for index in order])
        lengths = np.asarray([rows[index][1] for index in order], dtype=np.int64)
        labels = np.asarray([rows[index][2] for index in order], dtype=np.float32)
        return windows, lengths, labels


def score_trace(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    values: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.arange(len(values), dtype=np.int64)
    scores = np.empty(len(endpoints), dtype=np.float32)
    encoder.eval()
    head.eval()
    for begin in range(0, len(endpoints), int(batch_size)):
        batch_endpoints = endpoints[begin : begin + int(batch_size)]
        rows = [causal_window(values, endpoint=int(endpoint)) for endpoint in batch_endpoints]
        windows = torch.from_numpy(np.stack([row[0] for row in rows])).to(device)
        lengths = torch.as_tensor([row[1] for row in rows], device=device)
        with torch.no_grad():
            scores[begin : begin + len(rows)] = torch.sigmoid(
                head(encoder(windows, valid_lengths=lengths))
            ).cpu().numpy()
    return endpoints, scores


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def score_shots(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    shots: list[dict],
    *,
    device: str,
) -> list[dict]:
    rows: list[dict] = []
    for shot in shots:
        endpoints, scores = score_trace(encoder, head, shot["v"], device=device)
        if shot["is_disrupt"]:
            times = shot["start"] + endpoints / shot["rate"]
            ttd_ms = (float(shot["down"]) - times) * 1000.0
            timely = ttd_ms > MINIMUM_LEAD_MS
            timely_score = float(scores[timely].max()) if timely.any() else float("-inf")
        else:
            ttd_ms = np.full(len(endpoints), np.nan, dtype=np.float64)
            timely_score = float("nan")
        rows.append({
            "shot": int(shot["shot"]),
            "is_disrupt": bool(shot["is_disrupt"]),
            "score": float(scores.max()),
            "timely_score": timely_score,
            "endpoints": endpoints,
            "scores": scores,
            "ttd_ms": ttd_ms,
            "start_time_s": float(shot["start"]),
            "sample_rate_hz": float(shot["rate"]),
        })
    return rows


def select_validation_threshold(
    rows: list[dict], *, target_tpr: float = 0.90,
) -> dict[str, float]:
    disruptive = [row for row in rows if row["is_disrupt"]]
    safe = [row for row in rows if not row["is_disrupt"]]
    candidates = sorted(
        {float(row["timely_score"]) for row in disruptive if math.isfinite(row["timely_score"])}
        | {float(row["score"]) for row in safe},
        reverse=True,
    )
    best = None
    for threshold in candidates:
        tpr = float(np.mean([row["timely_score"] >= threshold for row in disruptive]))
        fpr = float(np.mean([row["score"] >= threshold for row in safe]))
        if tpr < float(target_tpr):
            continue
        candidate = (fpr, -threshold, -tpr)
        if best is None or candidate < best[0]:
            best = (candidate, threshold, tpr, fpr)
    if best is None:
        raise RuntimeError(f"VAL160 cannot reach timely TPR target {target_tpr}")
    return {"threshold": float(best[1]), "timely_tpr": best[2], "fpr": best[3]}


def metrics_at_threshold(rows: list[dict], *, threshold: float) -> tuple[dict, list[dict]]:
    labels = np.asarray([row["is_disrupt"] for row in rows], dtype=bool)
    shot_scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    predictions: list[dict] = []
    timely_hits: list[bool] = []
    raw_hits: list[bool] = []
    false_alarms: list[bool] = []
    leads: list[float] = []
    early_over_3000: list[int] = []
    for row in rows:
        passing = np.flatnonzero(row["scores"] >= float(threshold))
        fired = bool(len(passing))
        alarm_index = int(passing[0]) if fired else None
        alarm_endpoint = (
            int(row["endpoints"][alarm_index]) if alarm_index is not None else None
        )
        alarm_time_s = (
            float(row["start_time_s"])
            + float(alarm_endpoint) / float(row["sample_rate_hz"])
            if alarm_endpoint is not None else None
        )
        lead_ms = None
        timely = False
        if row["is_disrupt"]:
            raw_hits.append(fired)
            if fired:
                lead_ms = float(row["ttd_ms"][alarm_index])
                timely = lead_ms > MINIMUM_LEAD_MS
                if timely:
                    leads.append(lead_ms)
                if lead_ms > 3000.0:
                    early_over_3000.append(int(row["shot"]))
            timely_hits.append(timely)
        else:
            false_alarms.append(fired)
        predictions.append({
            "shot": int(row["shot"]),
            "is_disrupt": bool(row["is_disrupt"]),
            "score": float(row["score"]),
            "timely_score": (
                float(row["timely_score"])
                if math.isfinite(float(row["timely_score"])) else None
            ),
            "alarm": fired,
            "alarm_trace_index": alarm_index,
            "alarm_endpoint": alarm_endpoint,
            "alarm_time_s": alarm_time_s,
            "timely": timely if row["is_disrupt"] else None,
            "lead_ms": lead_ms,
        })
    metrics = {
        "n_shots": len(rows),
        "n_disruptions": int(labels.sum()),
        "n_non_disruptions": int((~labels).sum()),
        "threshold": float(threshold),
        "threshold_origin": "J-TEXT VAL160",
        "minimum_run": 1,
        "minimum_lead_ms": MINIMUM_LEAD_MS,
        "raw_tpr": float(np.mean(raw_hits)),
        "timely_tpr": float(np.mean(timely_hits)),
        "fpr": float(np.mean(false_alarms)),
        "auc": _auc(labels, shot_scores),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "early_over_3000_ms_count": len(early_over_3000),
        "early_over_3000_ms_shots": early_over_3000,
    }
    return metrics, predictions


def source_training_label_contract(
    *, policy: str, audit: dict, advanced_time_path: Path | None,
) -> dict:
    """Serialize the source-window rule without treating it as an outcome source."""

    if policy == "fixed125":
        positive: list[float] | dict = [0.0, SOURCE_HORIZON_MS]
        negative: list[float] | dict = [SOURCE_HORIZON_MS, NEGATIVE_MAX_MS]
        formula = "upper_ms=125"
    elif policy == "advanced_floor":
        positive = {
            "lower_ms_inclusive": 0.0,
            "upper_ms_per_disruptive_shot": "max(25, AdvancedTime_ms)",
            "missing_advanced_time_fallback_ms": 25.0,
        }
        negative = {
            "lower_ms_per_disruptive_shot": "max(25, AdvancedTime_ms)",
            "upper_ms_inclusive": NEGATIVE_MAX_MS,
        }
        formula = "upper_ms=max(25, AdvancedTime_ms); missing->25"
    else:
        raise ValueError(f"unsupported source label policy: {policy}")
    return {
        "policy": policy,
        "outcome_authority": "HDF5 meta/IsDisrupt",
        "down_time_authority": "HDF5 meta/DownTime",
        "advanced_time_role": "positive-window upper boundary only",
        "advanced_time_unit": "ms",
        "advanced_time_used": bool(advanced_time_path is not None),
        "advanced_time_path": (
            str(advanced_time_path.relative_to(ROOT))
            if advanced_time_path is not None
            and advanced_time_path.is_relative_to(ROOT)
            else str(advanced_time_path) if advanced_time_path is not None else None
        ),
        "advanced_time_sha256": (
            ATL.sha256_file(advanced_time_path) if advanced_time_path is not None else None
        ),
        "formula": formula,
        "positive_ttd_ms": positive,
        "negative_ttd_ms": negative,
        "audit": dict(audit),
    }


def source_data_manifest_sha256(shots: list[dict]) -> str:
    """Bind the exact scaled causal inputs used by one supervised role."""

    rows = []
    for shot in shots:
        rows.append({
            "shot": int(shot["shot"]),
            "values_sha256": SELECTED.array_sha256(shot["v"]),
            "availability_sha256": SELECTED.array_sha256(shot["a"]),
            "sample_rate_hz": float(shot["rate"]),
            "start_time_s": float(shot["start"]),
        })
    return SELECTED.sha256_json(rows)


def source_outcome_manifest_sha256(shots: list[dict]) -> str:
    """Bind HDF5-authoritative outcomes separately from signal inputs."""

    rows = [
        {
            "shot": int(shot["shot"]),
            "is_disrupt": bool(shot["is_disrupt"]),
            "down_time_s": (
                float(shot["down"]) if shot["is_disrupt"] else None
            ),
        }
        for shot in shots
    ]
    return SELECTED.sha256_json(rows)


def threshold_receipt_predictions(rows: list[dict]) -> list[dict]:
    """Use timely scores for disruptions and whole-trace scores for safe shots."""

    predictions: list[dict] = []
    for row in rows:
        score = row["timely_score"] if row["is_disrupt"] else row["score"]
        if not math.isfinite(float(score)):
            raise RuntimeError(
                f"VAL160 shot {int(row['shot'])} has no finite threshold score"
            )
        predictions.append({
            "shot": int(row["shot"]),
            "score": float(score),
            "outcome": bool(row["is_disrupt"]),
        })
    return predictions


def _checkpoint_payload(
    *,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    center: np.ndarray,
    scale: np.ndarray,
    roles: dict[str, list[int]],
    seed: int,
    epoch: int,
    validation: dict,
    training_label: dict | None = None,
    input_contract: dict | None = None,
    validation_threshold_receipt: dict | None = None,
    train_data_manifest_sha256: str | None = None,
    train_outcome_manifest_sha256: str | None = None,
    validation_data_manifest_sha256: str | None = None,
    validation_outcome_manifest_sha256: str | None = None,
) -> dict:
    if input_contract is None:
        input_contract = JIO.build_source_input_contract(ROOT)
    if validation_threshold_receipt is None:
        raise ValueError("source checkpoint requires a VAL160 threshold receipt")
    return {
        "schema_version": GS.SOURCE_CHECKPOINT_SCHEMA,
        "architecture_version": GS.SOURCE_ARCHITECTURE_VERSION,
        "feature_names": list(RH.REBUILD),
        "model_config": GS.source_model_config(len(RH.REBUILD)),
        "seed": int(seed),
        "epoch": int(epoch),
        "encoder": copy.deepcopy(encoder.state_dict()),
        "head": copy.deepcopy(head.state_dict()),
        "scaler_center": torch.as_tensor(center, dtype=torch.float32).cpu(),
        "scaler_scale": torch.as_tensor(scale, dtype=torch.float32).cpu(),
        "split_counts": dict(EXPECTED_COUNTS),
        "split_sha256": split_sha256(roles),
        "input_contract": input_contract,
        "label_authority": "data/J-TEXT/unified_hdf5/*/meta/{IsDisrupt,DownTime}",
        "training_label": training_label or {
            "policy": "fixed125",
            "positive_ttd_ms": [0.0, SOURCE_HORIZON_MS],
            "negative_ttd_ms": [SOURCE_HORIZON_MS, NEGATIVE_MAX_MS],
            "advanced_time_used": False,
        },
        "validation_threshold": float(validation["threshold"]),
        "validation_threshold_receipt": validation_threshold_receipt,
        "train_data_manifest_sha256": train_data_manifest_sha256,
        "train_outcome_manifest_sha256": train_outcome_manifest_sha256,
        "validation_data_manifest_sha256": validation_data_manifest_sha256,
        "validation_outcome_manifest_sha256": validation_outcome_manifest_sha256,
        "validation_metrics": validation,
        "scaler_sha256": SELECTED.scaler_sha256(
            torch.as_tensor(center, dtype=torch.float32),
            torch.as_tensor(scale, dtype=torch.float32),
        ),
        "label_map_sha256": (
            training_label.get("advanced_time_sha256")
            if training_label is not None else None
        ),
        "test_shards_read": 0,
    }


def train_source(args: argparse.Namespace) -> tuple[Path, dict]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    roles = load_canonical_jtext_roles(ROOT)
    train = load_jtext_shots(roles["train"])
    validation = load_jtext_shots(roles["val"])
    advanced_path = Path(args.advanced_time_path).expanduser().resolve()
    advanced = (
        ATL.load_advanced_time_ms(advanced_path)
        if args.label_policy == "advanced_floor" else None
    )
    train, label_audit = ATL.attach_positive_horizons(
        train, device="J-TEXT", policy=args.label_policy,
        advanced_time_ms=advanced,
    )
    training_label = source_training_label_contract(
        policy=args.label_policy, audit=label_audit,
        advanced_time_path=advanced_path if advanced is not None else None,
    )
    center, scale = GS.fit_scale(train)
    GS.apply_scale(train, center, scale)
    GS.apply_scale(validation, center, scale)
    input_contract = JIO.build_source_input_contract(ROOT)
    train_data_sha = source_data_manifest_sha256(train)
    train_outcome_sha = source_outcome_manifest_sha256(train)
    validation_data_sha = source_data_manifest_sha256(validation)
    validation_outcome_sha = source_outcome_manifest_sha256(validation)
    sampler = BalancedSourceSampler(train, seed=args.seed)

    encoder = GS.Enc(len(RH.REBUILD)).to(device)
    head = GS.Head().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "source_checkpoint.pt"
    history: list[dict] = []
    best_rank = None
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        head.train()
        losses: list[float] = []
        for _ in range(int(args.steps_per_epoch)):
            windows, lengths, labels = sampler.sample_batch(args.batch_size)
            x = torch.from_numpy(windows).to(device)
            n = torch.from_numpy(lengths).to(device)
            y = torch.from_numpy(labels).to(device)
            logits = head(encoder(x, valid_lengths=n))
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_rows = score_shots(encoder, head, validation, device=device)
        operating_point = select_validation_threshold(
            val_rows, target_tpr=args.target_tpr
        )
        validation_receipt = SELECTED.build_threshold_receipt(
            scope="J-TEXT VAL160",
            predictions=threshold_receipt_predictions(val_rows),
            data_manifest_sha256=validation_data_sha,
            outcome_manifest_sha256=validation_outcome_sha,
            target_tpr=args.target_tpr,
        )
        if not math.isclose(
            float(validation_receipt["chosen_threshold"]),
            float(operating_point["threshold"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("VAL160 threshold selector and receipt disagree")
        val_metrics, _ = metrics_at_threshold(
            val_rows, threshold=operating_point["threshold"]
        )
        val_metrics["loss"] = float(np.mean(losses))
        val_metrics["epoch"] = epoch
        history.append(val_metrics)
        rank = (
            -float(val_metrics["fpr"]),
            float(val_metrics["timely_tpr"]),
            float(val_metrics["auc"]),
        )
        print(
            f"epoch={epoch:02d} loss={val_metrics['loss']:.5f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_tpr10={val_metrics['timely_tpr']:.4f} "
            f"val_fpr={val_metrics['fpr']:.4f} "
            f"threshold={val_metrics['threshold']:.6f}",
            flush=True,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            payload = _checkpoint_payload(
                encoder=encoder,
                head=head,
                center=center,
                scale=scale,
                roles=roles,
                seed=args.seed,
                epoch=epoch,
                validation=val_metrics,
                training_label=training_label,
                input_contract=input_contract,
                validation_threshold_receipt=validation_receipt,
                train_data_manifest_sha256=train_data_sha,
                train_outcome_manifest_sha256=train_outcome_sha,
                validation_data_manifest_sha256=validation_data_sha,
                validation_outcome_manifest_sha256=validation_outcome_sha,
            )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        if epoch >= int(args.minimum_epochs) and stale >= int(args.patience):
            break

    summary = {
        "status": "SOURCE_TRAINED_TEST_UNREAD",
        "schema_version": GS.SOURCE_CHECKPOINT_SCHEMA,
        "seed": int(args.seed),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "split_counts": dict(EXPECTED_COUNTS),
        "split_sha256": split_sha256(roles),
        "label_authority": "HDF5 meta only",
        "training_label": training_label,
        "training_label_audit": label_audit,
        "input_contract": input_contract,
        "train_data_manifest_sha256": train_data_sha,
        "train_outcome_manifest_sha256": train_outcome_sha,
        "validation_data_manifest_sha256": validation_data_sha,
        "validation_outcome_manifest_sha256": validation_outcome_sha,
        "feature_names": list(RH.REBUILD),
        "model_config": GS.source_model_config(len(RH.REBUILD)),
        "trainable_parameters": sum(
            parameter.numel() for parameter in list(encoder.parameters()) + list(head.parameters())
        ),
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": portable_artifact_path(checkpoint_path, anchor=output),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "encoder_state_sha256": state_sha256(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)["encoder"]
        ),
        "test_shards_read": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return checkpoint_path, summary


def evaluate_test(
    checkpoint_path: Path, *, output: Path, seed: int, device: str,
) -> dict:
    run_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    roles = load_canonical_jtext_roles(ROOT)
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path,
        channels=len(RH.REBUILD),
        expected_seed=seed,
        device=run_device,
    )
    if checkpoint["split_sha256"] != split_sha256(roles):
        raise ValueError("source checkpoint is bound to a different J-TEXT split")
    test = load_jtext_shots(roles["test"])
    center = checkpoint["scaler_center"].cpu().numpy()
    scale = checkpoint["scaler_scale"].cpu().numpy()
    GS.apply_scale(test, center, scale)
    rows = score_shots(encoder, head, test, device=run_device)
    metrics, predictions = metrics_at_threshold(
        rows, threshold=float(checkpoint["validation_threshold"])
    )
    prediction_path = output / "test220_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    result = {
        "status": "SOURCE_TEST220_EVALUATED",
        "checkpoint": portable_artifact_path(checkpoint_path, anchor=output),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold_selection_split": "VAL160",
        "test_used_for_training_or_selection": False,
        "split_counts": dict(EXPECTED_COUNTS),
        "split_sha256": split_sha256(roles),
        "label_authority": "HDF5 meta only",
        "metrics": metrics,
        "predictions": portable_artifact_path(prediction_path, anchor=output),
        "predictions_sha256": sha256_file(prediction_path),
    }
    (output / "test220_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-tpr", type=float, default=0.90)
    parser.add_argument(
        "--label-policy", choices=ATL.POLICIES, default="fixed125",
    )
    parser.add_argument(
        "--advanced-time-path", type=Path, default=JTEXT_ADVANCED_TIME,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = HERE / "source_transformer_runs" / f"seed{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    checkpoint, summary = train_source(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.skip_test:
        result = evaluate_test(
            checkpoint,
            output=Path(args.output).expanduser().resolve(),
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
