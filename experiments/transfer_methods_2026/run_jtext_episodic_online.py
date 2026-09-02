#!/usr/bin/env python3
"""Run a causal local-neural episodic experiment on J-TEXT TEST220.

This is an independent diagnostic branch.  It starts from a frozen Adv25
source checkpoint and a pre-registered, label-free query manifest.  A queried
FP/FN/TARDY can fit a small local adapter after reveal; the adapter is routed
to later endpoints only through a causal descriptor and a correct-shot
blocker gate.  The source encoder/head and the threshold never change.

The runner intentionally does not reuse the J-TEXT online mixture weights and
does not export its state to EAST.  TEST220 has been used for development, so
the result is always ``DIAGNOSTIC_ONLY``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .mainline_paths import assert_final_artifact_root, assert_project_workspace_path
except ImportError:  # pragma: no cover - direct CLI execution
    from mainline_paths import assert_final_artifact_root, assert_project_workspace_path

import grid_search as GS  # noqa: E402
import jtext_label_contract as LABELS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import run_jtext_rebuilt17_online as ONLINE  # noqa: E402
import train_jtext_source_transformer as SOURCE  # noqa: E402
from jtext_episodic_adapter import (  # noqa: E402
    ERROR_GROUPS,
    EpisodicRiskAdapter,
    OrderedEpisodicBank,
    PrototypeConditionedRiskAdapter,
    adapter_parameter_count,
    endpoint_context,
    endpoint_descriptor,
    fit_episode_adapter,
    hidden_prototype,
    module_sha256,
)


TEST_COUNT = 220
DEFAULT_MAX_POINTS = 16
DEFAULT_QUERY_MANIFEST = (
    ROOT / "fuison_ext4_runs" /
    "jtext_adv25_active_canonical_replay_20260831" / "budget20" /
    "query_ledger.jsonl"
)
FORBIDDEN_LABEL_KEYS = frozenset({
    "is_disrupt", "down_time_s", "advanced_time_ms", "outcome", "group",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite ledger: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(
            row, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


class OutcomeGate:
    """Make HDF5 outcomes readable only after a selected query."""

    def __init__(self, data_root: Path, shot_ids: Sequence[int]) -> None:
        roster = [int(value) for value in shot_ids]
        if roster != sorted(roster) or len(roster) != len(set(roster)):
            raise ValueError("J-TEXT outcome roster must be ordered and unique")
        self.data_root = Path(data_root).expanduser().resolve()
        self.shot_ids = roster
        self.cursor = 0
        self.pending: int | None = None
        self.queried = False
        self.evaluation_open = False
        self.revealed: list[int] = []

    def _path(self, shot_id: int) -> Path:
        return (
            self.data_root / "J-TEXT" / "unified_hdf5" /
            str(int(shot_id) // 100 * 100) / f"{int(shot_id)}.hdf5"
        )

    def seal(self, shot_id: int) -> None:
        if self.pending is not None:
            raise RuntimeError("previous J-TEXT shot is still pending")
        if self.cursor >= len(self.shot_ids) or int(shot_id) != self.shot_ids[self.cursor]:
            raise RuntimeError("shot does not match the causal roster cursor")
        self.pending = int(shot_id)
        self.queried = False

    def query(self, shot_id: int) -> None:
        if self.pending != int(shot_id):
            raise PermissionError("query requires a sealed current shot")
        if self.queried:
            raise RuntimeError("current shot was queried twice")
        self.queried = True

    def reveal(self, shot_id: int) -> dict[str, Any]:
        if self.pending != int(shot_id) or not self.queried:
            raise PermissionError("reveal requires a selected sealed shot")
        path = self._path(shot_id)
        with h5py.File(path, "r") as handle:
            is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
            down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
        if is_disrupt and (not len(down) or not np.isfinite(float(down[0]))):
            raise ValueError(f"disruptive shot {shot_id} has no finite DownTime")
        self.pending = None
        self.queried = False
        self.cursor += 1
        self.revealed.append(int(shot_id))
        return {
            "is_disrupt": is_disrupt,
            "down_time_s": float(down[0]) if is_disrupt and len(down) else None,
        }

    def discard(self, shot_id: int) -> None:
        if self.pending != int(shot_id) or self.queried:
            raise RuntimeError("invalid discard transition")
        self.pending = None
        self.cursor += 1

    def open_final_evaluation(self) -> None:
        if self.pending is not None or self.queried or self.cursor != len(self.shot_ids):
            raise RuntimeError("stream is not complete")
        self.evaluation_open = True

    def read_for_final_evaluation(self, shot_id: int) -> dict[str, Any]:
        if not self.evaluation_open:
            raise PermissionError("final outcomes are not open")
        path = self._path(shot_id)
        with h5py.File(path, "r") as handle:
            is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
            down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
        return {
            "is_disrupt": is_disrupt,
            "down_time_s": float(down[0]) if is_disrupt and len(down) else None,
        }


def load_signal_only(data_root: Path, shot_id: int, center: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    """Read only current-shot signal and time attributes; never ``meta``."""

    path = (
        Path(data_root).expanduser().resolve() / "J-TEXT" / "unified_hdf5" /
        str(int(shot_id) // 100 * 100) / f"{int(shot_id)}.hdf5"
    )
    with h5py.File(path, "r") as handle:
        arrays = {
            name: np.asarray(handle[f"data/{name}"][:], dtype=np.float32)
            for name in RH.RAW23
        }
        signal = handle["data/ip"]
        rate = float(np.asarray(signal.attrs.get("SampleRate", 1000.0)).reshape(-1)[0])
        start = float(np.asarray(signal.attrs.get("StartTime", 0.0)).reshape(-1)[0])
    values, available = RH.rebuild_features(arrays, "J-TEXT")
    values = np.clip((values - center) / scale, -10.0, 10.0)
    values = np.where(available, values, 0.0).astype(np.float32)
    return {
        "shot": int(shot_id), "v": values, "a": available.astype(bool),
        "start": start, "rate": rate,
    }


def load_manifest(path: Path, roster: Sequence[int]) -> tuple[list[dict[str, Any]], str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"query manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("query manifest rows must be objects")
        if FORBIDDEN_LABEL_KEYS.intersection(row):
            raise ValueError("query manifest contains forbidden outcome fields")
        rows.append(row)
    if len(rows) != len(roster):
        raise ValueError("query manifest length does not match TEST220")
    for index, (row, shot) in enumerate(zip(rows, roster)):
        if int(row.get("stream_index", -1)) != index or int(row.get("shot", -1)) != int(shot):
            raise ValueError("query manifest is not the declared chronological roster")
        if not isinstance(row.get("selected"), (bool, np.bool_)):
            raise ValueError("query manifest selected field must be boolean")
    return rows, sha256_file(path)


def trace_contexts(trace: dict[str, Any]) -> np.ndarray:
    logits = np.asarray(trace["base_logits"], dtype=np.float32)
    endpoints = np.asarray(trace["endpoints"], dtype=np.int64)
    rate = float(trace["rate"])
    return np.stack([
        endpoint_context(
            float(logits[index]),
            0.0 if index == 0 else float(logits[index] - logits[index - 1]),
            float(endpoints[index]) / rate,
        )
        for index in range(len(logits))
    ]).astype(np.float32)


def make_trace(
    encoder: nn.Module,
    head: nn.Module,
    shot: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    endpoints, embeddings, logits = ONLINE.causal_embeddings(
        encoder, head, shot["v"], device=device, batch_size=1024,
    )
    risk = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)
    descriptors = np.stack([
        endpoint_descriptor(
            embeddings[index],
            source_logit=float(logits[index]),
            trend=(0.0 if index == 0 else float(logits[index] - logits[index - 1])),
            elapsed_s=float(endpoints[index]) / float(shot["rate"]),
        )
        for index in range(len(endpoints))
    ]).astype(np.float32)
    return {
        "shot": int(shot["shot"]), "values": shot["v"], "available": shot["a"],
        "start": float(shot["start"]), "rate": float(shot["rate"]),
        "endpoints": endpoints, "embeddings": embeddings,
        "base_logits": logits, "base_risk": risk,
        "contexts": trace_contexts({
            "base_logits": logits, "endpoints": endpoints, "rate": shot["rate"],
        }),
        "descriptors": descriptors,
    }


def outcome_group(
    trace: dict[str, Any], risk: np.ndarray, outcome: dict[str, Any], threshold: float,
) -> tuple[str, float | None]:
    alarm = np.flatnonzero(np.asarray(risk) >= float(threshold))
    if not bool(outcome["is_disrupt"]):
        return ("FP" if len(alarm) else "TN"), None
    if not len(alarm):
        return "FN", None
    down = outcome.get("down_time_s")
    if down is None:
        raise ValueError("disruptive outcome has no DownTime")
    alarm_time = float(trace["start"]) + float(trace["endpoints"][alarm[0]]) / float(trace["rate"])
    lead_ms = (float(down) - alarm_time) * 1000.0
    return ("TP" if lead_ms > 10.0 else "TARDY"), float(lead_ms)


def target_labels(
    trace: dict[str, Any], outcome: dict[str, Any], advanced_ms: float | None,
) -> np.ndarray:
    labels = np.zeros(len(trace["endpoints"]), dtype=np.float32)
    if not bool(outcome["is_disrupt"]):
        return labels
    down = outcome.get("down_time_s")
    if down is None:
        raise ValueError("disruptive outcome has no DownTime")
    horizon = max(
        float(LABELS.DEFAULT_FLOOR_MS),
        float(LABELS.DEFAULT_FLOOR_MS if advanced_ms is None else advanced_ms),
    )
    times = float(trace["start"]) + np.asarray(trace["endpoints"], dtype=np.float64) / float(trace["rate"])
    ttd = (float(down) - times) * 1000.0
    labels[(ttd >= 0.0) & (ttd <= horizon)] = 1.0
    return labels


def error_indices(trace: dict[str, Any], labels: np.ndarray, group: str, max_points: int) -> np.ndarray:
    logits = np.asarray(trace["base_logits"], dtype=np.float64)
    if group == "FP":
        candidates = np.flatnonzero(labels < 0.5)
        order = candidates[np.argsort(logits[candidates])[::-1]] if len(candidates) else candidates
    else:
        candidates = np.flatnonzero(labels >= 0.5)
        order = candidates[np.argsort(logits[candidates])[::-1]] if len(candidates) else candidates
    if not len(order):
        return np.empty(0, dtype=np.int64)
    quantiles = np.linspace(0, len(logits) - 1, min(int(max_points), len(logits)), dtype=np.int64)
    return np.unique(np.concatenate((order[:int(max_points)], quantiles))).astype(np.int64)


def build_validation_guard(
    encoder: nn.Module,
    head: nn.Module,
    validation: list[dict[str, Any]],
    *,
    threshold: float,
    device: str,
    max_shots: int = 16,
) -> list[dict[str, Any]]:
    """Build a label-bearing guard from VAL160 only."""

    guard: list[dict[str, Any]] = []
    for shot in validation:
        if len(guard) >= int(max_shots):
            break
        endpoints, embeddings, logits = ONLINE.causal_embeddings(
            encoder, head, shot["v"], device=device, batch_size=1024,
        )
        risk = torch.sigmoid(torch.from_numpy(logits)).numpy()
        outcome = {"is_disrupt": bool(shot["is_disrupt"]), "down_time_s": shot.get("down")}
        group, _lead = outcome_group(
            {"start": shot["start"], "rate": shot["rate"], "endpoints": endpoints},
            risk, outcome, threshold,
        )
        if group not in {"TP", "TN"}:
            continue
        descriptors = np.stack([
            endpoint_descriptor(
                embeddings[index], source_logit=float(logits[index]),
                trend=(0.0 if index == 0 else float(logits[index] - logits[index - 1])),
                elapsed_s=float(endpoints[index]) / float(shot["rate"]),
            )
            for index in range(len(endpoints))
        ]).astype(np.float32)
        contexts = trace_contexts({"base_logits": logits, "endpoints": endpoints, "rate": shot["rate"]})
        guard.append({
            "shot": int(shot["shot"]), "group": group,
            "base_logits": logits, "embeddings": embeddings,
            "descriptors": descriptors, "contexts": contexts,
            "endpoints": endpoints, "start": float(shot["start"]),
            "rate": float(shot["rate"]), "outcome": outcome,
        })
    if not guard:
        raise RuntimeError("no correct VAL160 guard shots available")
    return guard


def routed_guard_harms(
    episode: Any,
    guard: list[dict[str, Any]],
    *,
    threshold: float,
    minimum_similarity: float,
    minimum_margin: float,
    risk_margin: float,
    device: str,
) -> int:
    """Replay a candidate only under its own causal route and count harm."""

    harms = 0
    adapter = episode.adapter.eval()
    with torch.no_grad():
        for item in guard:
            adjusted = np.asarray(item["base_logits"], dtype=np.float32).copy()
            blocker_similarity = None
            for index, descriptor in enumerate(item["descriptors"]):
                similarity = OrderedEpisodicBank._max_similarity(
                    descriptor, episode.prototypes,
                )
                if similarity < float(minimum_similarity):
                    continue
                probability = float(1.0 / (1.0 + math.exp(-max(min(float(item["base_logits"][index]), 40.0), -40.0))))
                if episode.group == "FP" and probability < float(threshold) - float(risk_margin):
                    continue
                if episode.group != "FP" and probability > float(threshold) + float(risk_margin):
                    continue
                # Validation guards are not blockers for a candidate source;
                # only the candidate's own similarity gate is replayed here.
                z = torch.from_numpy(item["embeddings"][index:index + 1]).to(device)
                base = torch.from_numpy(item["base_logits"][index:index + 1]).to(device)
                context = torch.from_numpy(item["contexts"][index:index + 1]).to(device)
                value = float(adapter(z, base, context)[0][0].cpu())
                adjusted[index] = value
            base_group, _ = outcome_group(item, item["base_logits"], item["outcome"], threshold)
            candidate_group, _ = outcome_group(item, adjusted, item["outcome"], threshold)
            if candidate_group != base_group:
                harms += 1
    return int(harms)


def metrics(records: list[dict[str, Any]], outcomes: dict[int, dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = np.asarray([bool(outcomes[int(row["shot"])] ["is_disrupt"]) for row in records])
    scores = np.asarray([float(row["score"]) for row in records], dtype=np.float64)
    predicted = scores >= float(threshold)
    tp = int(np.sum(labels & predicted)); fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted)); tn = int(np.sum(~labels & ~predicted))
    timely: list[bool] = []; leads: list[float] = []; early: list[int] = []
    for row in records:
        outcome = outcomes[int(row["shot"])]
        if not bool(outcome["is_disrupt"]):
            continue
        alarm = row.get("alarm_index")
        lead = None
        if alarm is not None:
            alarm_time = float(row["start"]) + float(row["endpoints"][int(alarm)]) / float(row["rate"])
            lead = (float(outcome["down_time_s"]) - alarm_time) * 1000.0
        ok = bool(lead is not None and lead > 10.0)
        timely.append(ok)
        if ok:
            leads.append(float(lead))
        if lead is not None and lead > 3000.0:
            early.append(int(row["shot"]))
    pos = scores[labels]; neg = scores[~labels]
    auc = float("nan")
    if len(pos) and len(neg):
        delta = pos[:, None] - neg[None, :]
        auc = float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))
    timely_tpr = float(np.mean(timely)) if timely else 0.0
    return {
        "n_shots": len(records), "n_disruptions": int(labels.sum()),
        "n_non_disruptions": int((~labels).sum()), "TP": tp, "FP": fp,
        "FN": fn, "TN": tn, "tpr": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)), "auc": auc,
        "timely_tpr": timely_tpr,
        "operational_tpr": timely_tpr,
        "j_operational": timely_tpr - float(fp / max(fp + tn, 1)),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "early_over_3000_ms_count": len(early),
        "early_over_3000_ms_shots": early,
        "threshold": float(threshold), "minimum_lead_ms": 10.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name != "rl_env":
        raise RuntimeError(f"episodic runner requires rl_env, got {executable}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("episodic runner requires CUDA in rl_env")
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    if args.device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = args.device
    data_root = assert_project_workspace_path(args.data_root, label="J-TEXT data root")
    checkpoint_path = assert_project_workspace_path(args.source_checkpoint, label="source checkpoint")
    output = assert_final_artifact_root(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(RH.REBUILD), expected_seed=args.seed, device=device,
    )
    training_label = ONLINE.validate_advanced_source_contract(
        checkpoint, SOURCE.JTEXT_ADVANCED_TIME_PATH,
    )
    center = checkpoint["scaler_center"].detach().cpu().numpy().astype(np.float32)
    scale = checkpoint["scaler_scale"].detach().cpu().numpy().astype(np.float32)
    roles = ONLINE.load_roles(); roster = roles["test"]
    manifest, manifest_sha = load_manifest(
        args.query_manifest if args.query_manifest is not None else DEFAULT_QUERY_MANIFEST,
        roster,
    )
    selected_count = int(sum(bool(row["selected"]) for row in manifest))
    if selected_count <= 0 or selected_count > TEST_COUNT:
        raise ValueError("query manifest has an invalid selected count")
    threshold = float(checkpoint["validation_threshold"])
    source_hash_before = module_sha256(nn.Sequential(encoder, head))

    # VAL160 is used only for a fixed safety guard.  It is never refreshed by
    # TEST220 labels and is recorded separately from the online stream.
    validation = SOURCE.load_jtext_shots(roles["val"])
    GS.apply_scale(validation, center, scale)
    guard = build_validation_guard(
        encoder, head, validation, threshold=threshold, device=device,
        max_shots=int(args.guard_shots),
    )

    output.mkdir(parents=True)
    prediction_writer = JsonlWriter(output / "predictions.jsonl")
    query_writer = JsonlWriter(output / "queries.jsonl")
    reveal_writer = JsonlWriter(output / "reveals.jsonl")
    update_writer = JsonlWriter(output / "updates.jsonl")
    route_writer = JsonlWriter(output / "routes.jsonl")
    bank = OrderedEpisodicBank(
        threshold=threshold,
        minimum_similarity=float(args.minimum_similarity),
        minimum_margin=float(args.minimum_margin),
        risk_margin=float(args.risk_margin),
        minimum_episode_support=int(args.minimum_episode_support),
        maximum_episodes=int(args.maximum_episodes),
    )
    oracle = OutcomeGate(data_root, roster)
    advanced = LABELS.load_advanced_time_ms(SOURCE.JTEXT_ADVANCED_TIME_PATH)
    static_records: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = []
    update_count = 0; rollback_count = 0; route_count = 0
    error_episode_count = 0; blocker_count = 0
    try:
        for stream_index, shot_id in enumerate(roster):
            shot = load_signal_only(data_root, shot_id, center, scale)
            trace = make_trace(encoder, head, shot, device=device)
            static_alarm = ONLINE.first_alarm(
                trace["endpoints"], trace["base_risk"], threshold=threshold,
            )
            static_row = {
                "stream_index": int(stream_index), "shot": int(shot_id),
                "score": float(np.max(trace["base_risk"])) if len(trace["base_risk"]) else 0.0,
                "alarm_index": static_alarm,
                "endpoints": trace["endpoints"].tolist(),
                "start": trace["start"], "rate": trace["rate"],
                "threshold": threshold,
            }
            static_records.append(static_row)

            online_risk = np.empty(len(trace["endpoints"]), dtype=np.float32)
            route_ids: set[str] = set(); route_reasons: dict[str, int] = {}
            for point in range(len(trace["endpoints"])):
                descriptor = trace["descriptors"][point]
                probability = float(trace["base_risk"][point])
                episode, route = bank.route(
                    stream_index=stream_index, descriptor=descriptor,
                    source_probability=probability,
                )
                value = float(trace["base_logits"][point])
                if episode is not None:
                    with torch.no_grad():
                        z = torch.from_numpy(trace["embeddings"][point:point + 1]).to(device)
                        base = torch.from_numpy(trace["base_logits"][point:point + 1]).to(device)
                        context = torch.from_numpy(trace["contexts"][point:point + 1]).to(device)
                        value = float(episode.adapter(z, base, context)[0][0].cpu())
                    route_count += 1
                    route_ids.add(episode.episode_id)
                    route_reasons[route.reason] = route_reasons.get(route.reason, 0) + 1
                    route_writer.append({
                        "stream_index": int(stream_index), "shot": int(shot_id),
                        "endpoint": int(point), "episode_id": episode.episode_id,
                        **route.as_dict(),
                    })
                online_risk[point] = float(1.0 / (1.0 + math.exp(-max(min(value, 40.0), -40.0))))
            online_alarm = ONLINE.first_alarm(
                trace["endpoints"], online_risk, threshold=threshold,
            )
            online_row = {
                "stream_index": int(stream_index), "shot": int(shot_id),
                "score": float(np.max(online_risk)) if len(online_risk) else 0.0,
                "alarm_index": online_alarm,
                "endpoints": trace["endpoints"].tolist(),
                "start": trace["start"], "rate": trace["rate"],
                "threshold": threshold, "query_selected": bool(manifest[stream_index]["selected"]),
                "routed_endpoint_count": int(sum(route_reasons.values())),
                "routed_episode_ids": sorted(route_ids),
                "route_reasons": route_reasons,
            }
            online_records.append(online_row)
            prediction_writer.append({"arm": "static", **static_row})
            prediction_writer.append({"arm": "episodic_inter", **online_row})

            # The reveal gate is opened only after both arms have sealed the
            # current prediction.  The manifest is immutable and label-free.
            oracle.seal(shot_id)
            selected = bool(manifest[stream_index]["selected"])
            query_writer.append({
                "stream_index": int(stream_index), "shot": int(shot_id),
                "selected": selected, "manifest_sha256": manifest_sha,
                "label_access_before_decision": False,
            })
            if not selected:
                oracle.discard(shot_id)
                continue
            oracle.query(shot_id)
            outcome = oracle.reveal(shot_id)
            group, lead_ms = outcome_group(trace, online_risk, outcome, threshold)
            advanced_ms = float(advanced[shot_id]) if bool(outcome["is_disrupt"]) and shot_id in advanced else None
            labels = target_labels(trace, outcome, advanced_ms)
            reveal_writer.append({
                "stream_index": int(stream_index), "shot": int(shot_id),
                "is_disrupt": bool(outcome["is_disrupt"]),
                "down_time_s": outcome["down_time_s"],
                "advanced_time_ms": advanced_ms, "group": group, "lead_ms": lead_ms,
            })
            if group in ERROR_GROUPS:
                indices = error_indices(trace, labels, group, int(args.max_points))
                if len(indices):
                    if args.adapter_kind == "prototype":
                        adapter = PrototypeConditionedRiskAdapter(
                            hidden_prototype(trace["embeddings"][indices]),
                            descriptor_prototype=trace["descriptors"][indices],
                            direction=-1.0 if group == "FP" else 1.0,
                            rank=int(args.adapter_rank),
                            max_logit_delta=float(args.max_logit_delta),
                        ).to(device)
                    else:
                        adapter = EpisodicRiskAdapter(
                            dimension=int(trace["embeddings"].shape[1]),
                            rank=int(args.adapter_rank),
                            max_logit_delta=float(args.max_logit_delta),
                        ).to(device)
                    fit = fit_episode_adapter(
                        adapter,
                        embeddings=trace["embeddings"][indices],
                        base_logits=trace["base_logits"][indices],
                        contexts=trace["contexts"][indices],
                        labels=labels[indices], device=device,
                        learning_rate=float(args.learning_rate), steps=int(args.fit_steps),
                        max_relative_change=float(args.max_relative_change),
                        hard_example_weight=float(args.hard_example_weight),
                    )
                    candidate = type("Candidate", (), {
                        "prototypes": trace["descriptors"][indices],
                        "adapter": adapter, "group": group,
                    })()
                    guard_harms = routed_guard_harms(
                        candidate, guard, threshold=threshold,
                        minimum_similarity=float(args.minimum_similarity),
                        minimum_margin=float(args.minimum_margin),
                        risk_margin=float(args.risk_margin), device=device,
                    ) if bool(args.use_guard) and fit["accepted"] else 0
                    accepted = bool(fit["accepted"] and guard_harms <= int(args.max_guard_harms))
                    if not accepted:
                        rollback_count += 1
                    else:
                        commit = bank.add_error_episode(
                            shot_id=shot_id, stream_index=stream_index,
                            group=group, descriptors=trace["descriptors"][indices],
                            adapter=adapter,
                            source_risk_min=float(np.min(trace["base_risk"][indices])),
                            source_risk_max=float(np.max(trace["base_risk"][indices])),
                        )
                        update_count += 1; error_episode_count += 1
                    update_writer.append({
                        "stream_index": int(stream_index), "shot": int(shot_id),
                        "group": group, "indices": indices.tolist(),
                        "fit": fit, "guard_harms": int(guard_harms),
                        "accepted": accepted,
                        "commit": commit if accepted else None,
                        "post_reveal_only": True,
                    })
            else:
                # A correct queried shot is a blocker, not a training target.
                blocker = bank.add_blocker(
                    shot_id=shot_id, stream_index=stream_index,
                    descriptors=trace["descriptors"],
                )
                blocker_count += 1
                update_writer.append({
                    "stream_index": int(stream_index), "shot": int(shot_id),
                    "group": group, "accepted": False,
                    "reason": "correct_shot_blocker", "blocker": blocker,
                    "post_reveal_only": True,
                })
            if (stream_index + 1) % 20 == 0:
                print(
                    f"[JTEXT-EPISODIC] sealed {stream_index + 1}/{len(roster)} "
                    f"queries={len(oracle.revealed)} episodes={bank.episode_count}",
                    flush=True,
                )
    finally:
        prediction_writer.close(); query_writer.close(); reveal_writer.close()
        update_writer.close(); route_writer.close()
    oracle.open_final_evaluation()
    outcomes = {
        int(shot): oracle.read_for_final_evaluation(shot) for shot in roster
    }
    static_metrics = metrics(static_records, outcomes, threshold)
    online_metrics = metrics(online_records, outcomes, threshold)
    static_scores = np.asarray([row["score"] for row in static_records], dtype=np.float64)
    online_scores = np.asarray([row["score"] for row in online_records], dtype=np.float64)
    source_hash_after = module_sha256(nn.Sequential(encoder, head))
    if source_hash_after != source_hash_before:
        raise RuntimeError("frozen J-TEXT source changed during episodic run")
    summary = {
        "status": "JTEXT_EPISODIC_DIAGNOSTIC_ONLY",
        "stage": (
            "JTEXT_ADVANCED_FLOOR25_ORDERED_PROTOTYPE_EPISODES"
            if args.adapter_kind == "prototype"
            else "JTEXT_ADVANCED_FLOOR25_ORDERED_LOCAL_NEURAL_EPISODES"
        ),
        "seed": int(args.seed), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_training_label": training_label,
        "stream": {"role": "TEST220", "shots": TEST_COUNT, "order": "ascending_shot_id"},
        "query_contract": {
            "manifest": str(args.query_manifest if args.query_manifest is not None else DEFAULT_QUERY_MANIFEST),
            "manifest_sha256": manifest_sha, "selected": selected_count,
            "selection_label_free": True, "test_labels_used_for_selection": False,
        },
        "threshold": threshold, "threshold_source": "source VAL160 checkpoint",
        "adapter_contract": {
            "architecture": (
                "prototype-conditioned nonlinear local residual with latent representation path"
                if args.adapter_kind == "prototype"
                else "endpoint-conditioned nonlinear local residual with latent representation path"
            ),
            "kind": args.adapter_kind,
            "parameters": adapter_parameter_count(
                PrototypeConditionedRiskAdapter(
                    np.ones(int(RH.D_MODEL), dtype=np.float32),
                    rank=int(args.adapter_rank),
                    max_logit_delta=float(args.max_logit_delta),
                ) if args.adapter_kind == "prototype" else EpisodicRiskAdapter(
                    dimension=int(RH.D_MODEL), rank=int(args.adapter_rank),
                    max_logit_delta=float(args.max_logit_delta),
                )
            ),
            "rank": int(args.adapter_rank), "max_logit_delta": float(args.max_logit_delta),
            "learning_rate": float(args.learning_rate), "fit_steps": int(args.fit_steps),
            "max_relative_change": float(args.max_relative_change),
            "hard_example_weight": float(args.hard_example_weight),
            "locality": {
                "center": 0.80,
                "temperature": 12.0,
                "direction": "-1 for FP, +1 for FN/TARDY",
            } if args.adapter_kind == "prototype" else None,
            "episodes_committed": error_episode_count,
            "episodes_rolled_back": rollback_count,
            "routes_applied_endpoints": route_count,
            "representation_path_trainable": True,
        },
        "routing_contract": {
            "minimum_similarity": float(args.minimum_similarity),
            "minimum_margin": float(args.minimum_margin), "risk_margin": float(args.risk_margin),
            "minimum_episode_support": int(args.minimum_episode_support),
            "maximum_episodes": int(args.maximum_episodes),
            "correct_shot_blockers": blocker_count,
            "guard_enabled": bool(args.use_guard), "max_guard_harms": int(args.max_guard_harms),
        },
        "static_metrics": static_metrics, "online_metrics": online_metrics,
        "delta": {
            "tpr": online_metrics["tpr"] - static_metrics["tpr"],
            "operational_tpr": online_metrics["operational_tpr"] - static_metrics["operational_tpr"],
            "fpr": online_metrics["fpr"] - static_metrics["fpr"],
            "j_operational": online_metrics["j_operational"] - static_metrics["j_operational"],
            "mean_abs_score_change": float(np.mean(np.abs(online_scores - static_scores))),
            "alarm_to_no_alarm": int(np.sum((static_scores >= threshold) & (online_scores < threshold))),
            "no_alarm_to_alarm": int(np.sum((static_scores < threshold) & (online_scores >= threshold))),
        },
        "frozen_source_integrity": {
            "before": source_hash_before, "after": source_hash_after,
            "source_parameters_updated": False,
        },
        "causal_contract": {
            "prediction_order": "signal-only prefix -> static/episodic predict and seal -> fixed manifest query -> selected reveal -> fit/commit -> next shot",
            "outcomes_read_before_query": False,
            "unselected_outcomes_revealed_during_stream": False,
            "advanced_time_read_after_reveal_only": True,
            "current_episode_applied_to_current_shot": False,
            "jtext_online_exported_to_east": False,
        },
        "hashes": {
            "query_manifest": manifest_sha,
            "source_checkpoint": sha256_file(checkpoint_path),
            "predictions": sha256_file(output / "predictions.jsonl"),
            "queries": sha256_file(output / "queries.jsonl"),
            "reveals": sha256_file(output / "reveals.jsonl"),
            "updates": sha256_file(output / "updates.jsonl"),
            "routes": sha256_file(output / "routes.jsonl"),
        },
        "claim_boundary": "TEST220 was used for method development; this is a causal diagnostic, not independent future confirmation. A positive delta cannot by itself prove new physical-law discovery.",
    }
    atomic_json(output / "config.json", {
        "seed": int(args.seed), "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "query_manifest": str(args.query_manifest if args.query_manifest is not None else DEFAULT_QUERY_MANIFEST),
        "query_manifest_sha256": manifest_sha, "device": device,
        "minimum_similarity": float(args.minimum_similarity),
        "minimum_margin": float(args.minimum_margin), "risk_margin": float(args.risk_margin),
        "minimum_episode_support": int(args.minimum_episode_support),
        "adapter_rank": int(args.adapter_rank), "max_logit_delta": float(args.max_logit_delta),
        "adapter_kind": args.adapter_kind,
        "learning_rate": float(args.learning_rate), "fit_steps": int(args.fit_steps),
        "max_relative_change": float(args.max_relative_change),
        "hard_example_weight": float(args.hard_example_weight),
        "use_guard": bool(args.use_guard), "max_guard_harms": int(args.max_guard_harms),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "output": str(output), "static": static_metrics,
        "online": online_metrics, "delta": summary["delta"],
        "episodes": error_episode_count, "rollbacks": rollback_count,
    }, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--minimum-similarity", type=float, default=0.90)
    parser.add_argument("--minimum-margin", type=float, default=0.05)
    parser.add_argument("--risk-margin", type=float, default=0.18)
    parser.add_argument(
        "--minimum-episode-support", type=int, default=2,
        help="number of distinct strict-past same-group episodes required before routing",
    )
    parser.add_argument("--maximum-episodes", type=int, default=32)
    parser.add_argument("--adapter-rank", type=int, default=4)
    parser.add_argument(
        "--adapter-kind", choices=("prototype", "generic"), default="prototype",
        help="local route adapter; prototype binds each update to its revealed error trajectory",
    )
    parser.add_argument("--max-logit-delta", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--fit-steps", type=int, default=8)
    parser.add_argument("--max-relative-change", type=float, default=0.20)
    parser.add_argument("--hard-example-weight", type=float, default=4.0)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--guard-shots", type=int, default=16)
    parser.add_argument("--max-guard-harms", type=int, default=0)
    parser.add_argument("--use-guard", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
