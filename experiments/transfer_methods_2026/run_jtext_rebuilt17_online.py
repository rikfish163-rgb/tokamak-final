#!/usr/bin/env python3
"""Run the J-TEXT online-active branch from a rebuilt17 static checkpoint.

This runner is intentionally separate from the EAST V3C runner.  It uses the
same causal source ``Enc/Head`` architecture as the historical rebuilt17
transfer line, but it never exports the online-updated J-TEXT adapter to EAST.

Contract
--------
* the parent checkpoint must have been trained with
  ``H_i=max(25 ms, AdvancedTime_i)``;
* TEST220 is ordered by shot number and its metadata is hidden by an oracle
  until ``predict -> seal -> query`` has completed;
* a per-shot fast adapter is updated with current/past signal only and reset
  before the next shot;
* a persistent low-rank adapter is updated only after a queried HDF5 outcome is
  revealed, using the current labelled bag and earlier revealed replay;
* threshold and all test metrics are fixed after the source VAL160 selection.

The resulting metrics are development evidence because TEST220 is the online
stream used by this project for method development.  The ledger nevertheless
enforces prospective access ordering so it can be replayed on a future stream.
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
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
HERE = ROOT / "experiments" / "transfer_methods_2026"
sys.path.insert(0, str(HERE))

import grid_search as GS  # noqa: E402
import jtext_label_contract as LABELS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import train_jtext_source_transformer as SOURCE  # noqa: E402


WINDOW = int(GS.W)
LAG = 8
EXPECTED_TEST_SHOTS = 220
DEFAULT_BUDGET_FRACTION = 0.10
DEFAULT_MAX_BAG_WINDOWS = 16
DEFAULT_MAX_REPLAY_SHOTS = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class JsonlWriter:
    """Append-only ledger writer; an existing ledger is never reused."""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite ledger: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("x", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def shot_path(shot_id: int) -> Path:
    return (
        ROOT / "data" / "J-TEXT" / "unified_hdf5"
        / str(int(shot_id) // 100 * 100) / f"{int(shot_id)}.hdf5"
    )


def load_signal_only(shot_id: int, center: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    """Load data and time-base attributes without opening HDF5 ``meta``."""

    path = shot_path(shot_id)
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
        "shot": int(shot_id),
        "v": values,
        "a": available.astype(bool),
        "start": start,
        "rate": rate,
    }


def read_outcome(shot_id: int) -> dict[str, Any]:
    """Read authoritative outcome metadata after the oracle permits it."""

    with h5py.File(shot_path(shot_id), "r") as handle:
        is_disrupt = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down_values = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    if is_disrupt and (not len(down_values) or not np.isfinite(float(down_values[0]))):
        raise ValueError(f"J-TEXT disruptive shot {shot_id} has no finite DownTime")
    return {
        "is_disrupt": is_disrupt,
        "down_time_s": float(down_values[0]) if is_disrupt and len(down_values) else None,
    }


def read_all_outcomes(shot_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    return {int(shot): read_outcome(int(shot)) for shot in shot_ids}


class OutcomeOracle:
    """A cursor that makes a current outcome inaccessible before seal/query."""

    def __init__(self, shot_ids: list[int]) -> None:
        if shot_ids != sorted(shot_ids) or len(shot_ids) != len(set(shot_ids)):
            raise ValueError("J-TEXT online roster must be unique and ordered")
        self.shot_ids = list(shot_ids)
        self.cursor = 0
        self.pending: int | None = None
        self.queried = False
        self.evaluation_open = False
        self.revealed: list[int] = []

    def seal(self, shot_id: int) -> None:
        if self.pending is not None:
            raise RuntimeError("a previous J-TEXT shot is still pending")
        if self.cursor >= len(self.shot_ids) or int(shot_id) != self.shot_ids[self.cursor]:
            raise RuntimeError("J-TEXT shot does not match the causal oracle cursor")
        self.pending = int(shot_id)
        self.queried = False

    def query(self, shot_id: int) -> None:
        """Record a label query only after the current prediction is sealed."""

        if self.pending != int(shot_id):
            raise PermissionError("query requires a sealed current shot")
        if self.queried:
            raise RuntimeError("the current J-TEXT shot was queried twice")
        self.queried = True

    def reveal(self, shot_id: int) -> dict[str, Any]:
        if self.pending != int(shot_id) or not self.queried:
            raise PermissionError("outcome reveal requires a sealed and queried current shot")
        result = read_outcome(int(shot_id))
        self.pending = None
        self.queried = False
        self.cursor += 1
        self.revealed.append(int(shot_id))
        return result

    def discard(self, shot_id: int) -> None:
        if self.pending != int(shot_id):
            raise RuntimeError("discard does not match the sealed current shot")
        if self.queried:
            raise RuntimeError("a queried J-TEXT shot must be revealed")
        self.pending = None
        self.cursor += 1

    def open_final_evaluation(self) -> None:
        if self.pending is not None or self.queried or self.cursor != len(self.shot_ids):
            raise RuntimeError("J-TEXT online stream is not complete")
        self.evaluation_open = True


class LazyAdvancedTime:
    """Resolve only queried shot IDs after reveal; never used by prediction."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._payload: dict[int, float] | None = None
        self.lookups: list[int] = []

    def lookup_after_reveal(self, shot_id: int) -> float | None:
        if self._payload is None:
            self._payload = LABELS.load_advanced_time_ms(self.path)
        self.lookups.append(int(shot_id))
        return self._payload.get(int(shot_id))


class CausalBudgetSelector:
    """Immediate label-free selection with an exact total budget."""

    def __init__(self, *, total_shots: int, fraction: float, cutoff: float) -> None:
        if total_shots <= 0 or not 0.0 < float(fraction) <= 1.0:
            raise ValueError("invalid J-TEXT query budget")
        total = int(round(int(total_shots) * float(fraction)))
        if total <= 0 or total > total_shots:
            raise ValueError("query fraction yields an invalid exact budget")
        self.total_shots = int(total_shots)
        self.fraction = float(fraction)
        self.total_budget = total
        if not math.isfinite(float(cutoff)) or not 0.0 <= float(cutoff) <= 1.0:
            raise ValueError("query cutoff must be a finite probability in [0, 1]")
        self.cutoff = float(cutoff)
        self.selected = 0

    def decide(self, *, score: float, stream_index: int) -> tuple[bool, str]:
        index = int(stream_index)
        if index < 0 or index >= self.total_shots:
            raise ValueError("stream index is outside query selector roster")
        if not np.isfinite(float(score)):
            raise ValueError("query score must be finite")
        if self.selected >= self.total_budget:
            return False, "budget_exhausted"
        remaining_after = self.total_shots - index - 1
        slots_after = self.total_budget - self.selected - 1
        if float(score) >= self.cutoff:
            self.selected += 1
            return True, "score"
        if slots_after >= 0 and slots_after >= remaining_after:
            self.selected += 1
            return True, "causal_force_fill"
        return False, "below_cutoff"


class CausalBlockBudgetSelector:
    """Spend an exact query quota independently in each chronological block.

    The earlier global selector can legally force-fill the suffix of a stream:
    once high-valued shots have consumed the global budget, every remaining
    slot is spent at the end.  That is an exact budget but it is not the
    project's intended online contract (and gives the learner no opportunity
    to use an early reveal).  This selector fixes the allocation *per block*.
    Each decision reads only the current score, the current position, and the
    remaining slots in the current block; it never ranks or inspects a future
    shot.
    """

    def __init__(
        self,
        *,
        total_shots: int,
        fraction: float,
        cutoff: float,
        block_size: int = 20,
    ) -> None:
        shots = int(total_shots)
        block = int(block_size)
        if shots <= 0 or block <= 0 or shots % block:
            raise ValueError(
                "causal block budget requires a positive complete roster"
            )
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("invalid causal block budget fraction")
        quota_float = block * float(fraction)
        quota = int(round(quota_float))
        if quota <= 0 or quota > block or not math.isclose(quota, quota_float):
            raise ValueError(
                "causal block budget fraction must yield an integer block quota"
            )
        if not math.isfinite(float(cutoff)) or not 0.0 <= float(cutoff) <= 1.0:
            raise ValueError("causal block query cutoff must be finite in [0, 1]")
        self.total_shots = shots
        self.block_size = block
        self.fraction = float(fraction)
        self.quota = quota
        self.cutoff = float(cutoff)
        self.selected = 0
        self._next_index = 0
        self._block = -1
        self._used = 0

    @property
    def total_budget(self) -> int:
        return (self.total_shots // self.block_size) * self.quota

    def decide(self, *, score: float, stream_index: int) -> tuple[bool, str]:
        index = int(stream_index)
        if index != self._next_index:
            raise RuntimeError("causal block decisions must be contiguous")
        if index < 0 or index >= self.total_shots:
            raise ValueError("stream index is outside causal block roster")
        if not math.isfinite(float(score)):
            raise ValueError("causal block query score must be finite")
        block = index // self.block_size
        position = index % self.block_size
        if block != self._block:
            self._block = block
            self._used = 0
        slots = self.quota - self._used
        remaining = self.block_size - position
        selected = bool(slots > 0 and float(score) >= self.cutoff)
        route = "score" if selected else "below_cutoff"
        if not selected and slots > 0 and remaining <= slots:
            selected = True
            route = "causal_block_force_fill"
        if selected:
            self._used += 1
            self.selected += 1
        self._next_index += 1
        if position == self.block_size - 1 and self._used != self.quota:
            raise RuntimeError(
                "causal block selector failed to fill its exact block quota"
            )
        return selected, route


class LowRankRiskAdapter(nn.Module):
    """Small causal adapter with a representation path and a risk residual."""

    def __init__(self, dimension: int = 64, rank: int = 8, max_logit_delta: float = 1.0) -> None:
        super().__init__()
        if dimension <= 0 or rank <= 0 or max_logit_delta <= 0:
            raise ValueError("invalid low-rank adapter dimensions")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.max_logit_delta = float(max_logit_delta)
        self.latent_down = nn.Linear(dimension, rank, bias=False)
        self.latent_up = nn.Linear(rank, dimension, bias=False)
        self.risk_down = nn.Linear(dimension, rank, bias=False)
        self.risk_up = nn.Linear(rank, 1, bias=True)
        self.next_down = nn.Linear(dimension, rank, bias=False)
        self.next_up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.latent_down.weight, std=0.02)
        nn.init.normal_(self.risk_down.weight, std=0.02)
        nn.init.normal_(self.next_down.weight, std=0.02)
        nn.init.zeros_(self.latent_up.weight)
        nn.init.zeros_(self.risk_up.weight)
        nn.init.zeros_(self.risk_up.bias)
        nn.init.zeros_(self.next_up.weight)

    def transform(self, frozen: Tensor) -> Tensor:
        correction = self.latent_up(torch.tanh(self.latent_down(frozen)))
        return frozen + 0.10 * correction

    def forward(self, frozen: Tensor, base_logit: Tensor) -> tuple[Tensor, Tensor]:
        transformed = self.transform(frozen)
        residual = self.max_logit_delta * torch.tanh(
            self.risk_up(torch.tanh(self.risk_down(transformed))).squeeze(-1)
        )
        return base_logit + residual, residual

    def next_state(self, frozen: Tensor) -> Tensor:
        transformed = self.transform(frozen)
        return transformed + self.next_up(torch.tanh(self.next_down(transformed)))


def adapter_parameter_count(adapter: LowRankRiskAdapter) -> int:
    return sum(parameter.numel() for parameter in adapter.parameters())


def causal_embeddings(
    encoder: nn.Module,
    head: nn.Module,
    values: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache the source evaluator's complete causal trace.

    The source trainer evaluates *every* observed endpoint, including the
    left-padded prefix of a short shot, and passes the corresponding number
    of real observations as ``valid_lengths`` to the encoder.  The first
    version of this online runner kept only complete windows and omitted
    ``valid_lengths``.  That silently changed the model/evaluator pair and
    made its static arm incomparable with ``test220_metrics.json``.  Keep the
    exact source endpoint policy here so that any online delta is attributable
    to the adapter, not to a different prefix definition.
    """

    endpoints = np.arange(len(values), dtype=np.int64)
    embedding_dim = int(RH.D_MODEL)
    if not len(endpoints):
        return endpoints, np.empty((0, embedding_dim), np.float32), np.empty(0, np.float32)
    embeddings = np.empty((len(endpoints), embedding_dim), dtype=np.float32)
    logits = np.empty(len(endpoints), dtype=np.float32)
    encoder.eval(); head.eval()
    for start in range(0, len(endpoints), int(batch_size)):
        current = endpoints[start:start + int(batch_size)]
        rows = [SOURCE.causal_window(values, endpoint=int(endpoint)) for endpoint in current]
        windows = np.stack([row[0] for row in rows]).astype(np.float32, copy=False)
        valid_lengths = torch.as_tensor(
            [row[1] for row in rows], device=device, dtype=torch.long
        )
        with torch.no_grad():
            tensor = torch.from_numpy(windows).to(device)
            latent = encoder(tensor, valid_lengths=valid_lengths)
            value = head(latent)
        embeddings[start:start + len(current)] = latent.detach().cpu().numpy()
        logits[start:start + len(current)] = value.detach().cpu().numpy()
    return endpoints, embeddings, logits


def base_trace(
    encoder: nn.Module,
    head: nn.Module,
    values: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints, embeddings, logits = causal_embeddings(
        encoder, head, values, device=device
    )
    return endpoints, embeddings, torch.sigmoid(torch.from_numpy(logits)).numpy()


def shot_signature(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Outcome-blind, bounded signature used only for novelty selection."""

    tail = values[-min(128, len(values)):]
    mask = available[-len(tail):]
    features: list[float] = []
    for column in range(tail.shape[1]):
        observed = tail[mask[:, column], column]
        if len(observed):
            features.extend([
                float(np.median(observed)),
                float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25)),
            ])
        else:
            features.extend([0.0, 0.0])
    features.extend(float(value) for value in mask.mean(axis=0))
    return np.asarray(features, dtype=np.float32)


def normalized_novelty(signature: np.ndarray, history: list[np.ndarray]) -> float:
    if not history:
        return 1.0
    current = np.asarray(signature, dtype=np.float64)
    distances = []
    for previous in history[-128:]:
        value = np.asarray(previous, dtype=np.float64)
        scale = max(float(np.linalg.norm(value)) / math.sqrt(max(len(value), 1)), 1.0)
        distances.append(float(np.linalg.norm(current - value) / math.sqrt(len(current)) / scale))
    return float(np.clip(np.min(distances), 0.0, 1.0))


def query_components(
    risk: np.ndarray,
    base_logits: np.ndarray,
    threshold: float,
    novelty: float,
) -> dict[str, float]:
    risk = np.asarray(risk, dtype=np.float64)
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if not len(risk):
        # A very short shot has no complete causal source window.  It cannot
        # generate an alarm or a risk-based uncertainty term, but it remains
        # a valid stream item and may consume a query slot through its
        # outcome-blind novelty score.  Keeping this case explicit avoids an
        # invalid argmax and makes the no-window decision auditable.
        score = 0.30 * float(np.clip(novelty, 0.0, 1.0))
        return {
            "uncertainty": 0.0,
            "novelty": float(np.clip(novelty, 0.0, 1.0)),
            "instability": 0.0,
            "query_score": float(np.clip(score, 0.0, 1.0)),
            "max_risk": 0.0,
        }
    threshold_logit = math.log(float(threshold) / (1.0 - float(threshold)))
    peak_index = int(np.argmax(risk))
    peak_logit = float(base_logits[peak_index])
    uncertainty = math.exp(-abs(peak_logit - threshold_logit))
    instability = float(np.clip(np.max(np.abs(np.diff(risk))) * 8.0, 0.0, 1.0)) if len(risk) > 1 else 0.0
    score = 0.50 * uncertainty + 0.30 * float(novelty) + 0.20 * instability
    return {
        "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
        "novelty": float(np.clip(novelty, 0.0, 1.0)),
        "instability": instability,
        "query_score": float(np.clip(score, 0.0, 1.0)),
        "max_risk": float(np.max(risk)) if len(risk) else 0.0,
    }


def first_alarm(endpoints: np.ndarray, risk: np.ndarray, *, threshold: float) -> int | None:
    passing = np.flatnonzero(np.asarray(risk) >= float(threshold))
    return int(passing[0]) if len(passing) else None


def outcome_group(
    *,
    is_disrupt: bool,
    down_time_s: float | None,
    start: float,
    rate: float,
    endpoints: np.ndarray,
    risk: np.ndarray,
    threshold: float,
) -> tuple[str, float | None]:
    index = first_alarm(endpoints, risk, threshold=threshold)
    if not is_disrupt:
        return ("FP" if index is not None else "TN"), None
    if index is None:
        return "FN", None
    alarm_time = float(start) + float(endpoints[index]) / float(rate)
    lead_ms = (float(down_time_s) - alarm_time) * 1000.0
    if lead_ms > 10.0:
        return "TP", lead_ms
    return "TARDY", lead_ms


def choose_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= int(limit):
        return indices
    return indices[np.linspace(0, len(indices) - 1, int(limit), dtype=np.int64)]


def choose_error_anchored_indices(
    indices: np.ndarray,
    limit: int,
    *,
    scores: np.ndarray,
    anchors: Iterable[int] = (),
) -> np.ndarray:
    """Select labelled endpoints without dropping the error that triggered a query.

    Uniform temporal subsampling is appropriate for a representative replay
    bag, but it is unsafe for an error-correction update: a short high-risk
    excursion can lie between two uniformly selected endpoints.  The
    selection is deterministic and outcome-aware only *after* reveal.  It
    always retains registered temporal anchors (for example the first alarm
    and the disruption boundary), then the highest source-risk endpoints,
    and finally fills the remaining slots by temporal quantiles.
    """
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or len(values) != len(np.unique(values)):
        raise ValueError("labelled endpoint indices must be a unique 1-D array")
    if int(limit) <= 0:
        raise ValueError("labelled endpoint limit must be positive")
    if len(values) <= int(limit):
        return np.sort(values)
    risk = np.asarray(scores, dtype=np.float64)
    if risk.ndim != 1:
        raise ValueError("labelled endpoint scores must be one-dimensional")
    if not np.isfinite(risk).all():
        raise ValueError("labelled endpoint scores must be finite")
    selected: set[int] = {
        int(value) for value in anchors if int(value) in set(values.tolist())
    }
    # The highest-risk endpoints are the hard examples for both FP and FN
    # correction.  Stable mergesort makes ties reproducible across platforms.
    order = values[np.argsort(-risk[values], kind="mergesort")]
    selected.update(int(value) for value in order[: max(1, int(limit) // 2)])
    if len(selected) < int(limit):
        remaining = np.asarray(
            [value for value in values if int(value) not in selected],
            dtype=np.int64,
        )
        need = int(limit) - len(selected)
        selected.update(
            int(value) for value in remaining[
                np.linspace(0, len(remaining) - 1, need, dtype=np.int64)
            ]
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def labelled_bag(
    shot: dict[str, Any],
    outcome: dict[str, Any],
    advanced_time: float | None,
    embeddings: np.ndarray,
    base_logits: np.ndarray,
    endpoints: np.ndarray,
    *,
    max_windows: int,
) -> dict[str, Any] | None:
    if not len(endpoints):
        return None
    times = float(shot["start"]) + endpoints / float(shot["rate"])
    if outcome["is_disrupt"]:
        if outcome["down_time_s"] is None:
            raise ValueError("disruptive queried shot lacks DownTime")
        ttd_ms = (float(outcome["down_time_s"]) - times) * 1000.0
        horizon, _raw, _fallback = LABELS.effective_horizon_ms(
            int(shot["shot"]),
            {} if advanced_time is None else {int(shot["shot"]): float(advanced_time)},
            floor_ms=LABELS.DEFAULT_FLOOR_MS,
        )
        positive = endpoints[(ttd_ms >= 0.0) & (ttd_ms <= horizon)]
        negative = endpoints[(ttd_ms > horizon) & (ttd_ms <= 60000.0)]
    else:
        horizon = None
        positive = np.empty(0, dtype=np.int64)
        negative = endpoints
    # Keep the first alarm and the boundary closest to the revealed terminal
    # state.  These anchors are created only after the current shot has been
    # revealed; they are never available to prediction or query selection.
    positive_anchors: list[int] = []
    negative_anchors: list[int] = []
    if len(base_logits):
        peak_endpoint = int(endpoints[int(np.argmax(base_logits))])
        if peak_endpoint in set(positive.tolist()):
            positive_anchors.append(peak_endpoint)
        if peak_endpoint in set(negative.tolist()):
            negative_anchors.append(peak_endpoint)
    if outcome["is_disrupt"] and outcome["down_time_s"] is not None:
        boundary_targets = [0.0, float(horizon)]
        for target in boundary_targets:
            boundary_endpoint = int(
                endpoints[int(np.argmin(np.abs(ttd_ms - float(target))))]
            )
            if boundary_endpoint in set(positive.tolist()):
                positive_anchors.append(boundary_endpoint)
            if boundary_endpoint in set(negative.tolist()):
                negative_anchors.append(boundary_endpoint)
    selected_positive = choose_error_anchored_indices(
        positive, max_windows, scores=base_logits,
        anchors=positive_anchors,
    )
    selected_negative = choose_error_anchored_indices(
        negative, max_windows, scores=base_logits,
        anchors=negative_anchors,
    )
    selected = np.sort(np.concatenate([selected_positive, selected_negative]))
    if not len(selected):
        return None
    position = np.searchsorted(endpoints, selected)
    y = np.isin(selected, selected_positive).astype(np.float32)
    return {
        "shot": int(shot["shot"]),
        "outcome": int(bool(outcome["is_disrupt"])),
        "horizon_ms": None if horizon is None else float(horizon),
        "endpoints": selected,
        "z": np.asarray(embeddings[position], dtype=np.float32),
        "base_logit": np.asarray(base_logits[position], dtype=np.float32),
        "label": y,
        "sampling_policy": "error_anchored_toprisk_v1",
        "positive_anchor_endpoints": sorted(set(positive_anchors)),
        "negative_anchor_endpoints": sorted(set(negative_anchors)),
    }


def tensor_batch(bag: dict[str, Any], device: str) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.from_numpy(bag["z"]).to(device),
        torch.from_numpy(bag["base_logit"]).to(device),
        torch.from_numpy(bag["label"]).to(device),
    )


def guard_harms(
    adapter: LowRankRiskAdapter,
    guard: list[dict[str, Any]],
    *,
    threshold: float,
    device: str,
) -> int:
    harmed = 0
    adapter.eval()
    with torch.no_grad():
        for item in guard:
            z = torch.from_numpy(item["z"]).to(device)
            base = torch.from_numpy(item["base_logit"]).to(device)
            adjusted, _ = adapter(z, base)
            base_alarm = bool(torch.sigmoid(base).max().item() >= threshold)
            adjusted_alarm = bool(torch.sigmoid(adjusted).max().item() >= threshold)
            if base_alarm == bool(item["label"]) and adjusted_alarm != bool(item["label"]):
                harmed += 1
    return harmed


def relative_change(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in before:
        left = before[name].detach().float()
        right = after[name].detach().float()
        numerator += float((right - left).square().sum())
        denominator += float(left.square().sum())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1.0)


def project_state_delta(
    before: dict[str, Tensor],
    after: dict[str, Tensor],
    *,
    maximum: float,
) -> tuple[float, float, bool]:
    """Project a finite update into its registered norm ball.

    The adapter is intentionally identity-initialized, so a relative norm
    defined against its initial state has an absolute denominator of one.
    Rejecting every Adam step under that convention turns an online learner
    into a no-op.  We therefore keep the same safety radius, but project an
    otherwise finite step onto the boundary.  This is deterministic, bounded,
    and auditable: the receipt contains both the raw and realized changes.
    """
    limit = float(maximum)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("J-TEXT update radius must be finite and positive")
    raw = relative_change(before, after)
    if not math.isfinite(raw):
        return raw, raw, False
    if raw <= limit:
        return raw, raw, False
    scale = limit / raw
    with torch.no_grad():
        for name, value in after.items():
            value.copy_(before[name] + scale * (value - before[name]))
    realized = relative_change(before, after)
    return raw, realized, True


def intra_update(
    adapter: LowRankRiskAdapter,
    previous_z: np.ndarray,
    current_z: np.ndarray,
    history_z: list[np.ndarray],
    history_base: list[np.ndarray],
    *,
    device: str,
    learning_rate: float,
    steps: int,
    max_relative_change: float,
    current_base_logit: np.ndarray | float | None = None,
    risk_forecast_weight: float = 0.5,
) -> dict[str, Any]:
    if not math.isfinite(float(risk_forecast_weight)) or float(risk_forecast_weight) < 0.0:
        raise ValueError("J-TEXT risk forecast weight must be finite and non-negative")
    before = copy.deepcopy(adapter.state_dict())
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    previous = torch.from_numpy(np.asarray(previous_z, np.float32)[None]).to(device)
    current = torch.from_numpy(np.asarray(current_z, np.float32)[None]).to(device)
    current_base = None
    if current_base_logit is not None:
        current_base = torch.from_numpy(
            np.asarray(current_base_logit, np.float32).reshape(1)
        ).to(device)
    losses: list[float] = []
    adapter.train()
    for _ in range(int(steps)):
        prediction = adapter.next_state(previous)
        dynamics = F.smooth_l1_loss(prediction, current.detach())
        if history_z:
            old_z = torch.from_numpy(np.stack(history_z[-8:]).astype(np.float32)).to(device)
            old_base = torch.from_numpy(np.concatenate(history_base[-8:]).astype(np.float32)).to(device)
            old_adjusted, _ = adapter(old_z, old_base)
            anchor = F.smooth_l1_loss(old_adjusted, old_base.detach())
        else:
            anchor = dynamics * 0.0
        regularizer = torch.stack([
            parameter.square().mean() for parameter in adapter.parameters()
        ]).mean()
        # The current observation is available only after its prediction has
        # been sealed.  It is therefore a legal one-step, label-free teacher
        # for the *next* endpoint: forecast the current source risk from the
        # previous observed state.  Unlike an identity teacher on the current
        # endpoint, this term has a non-zero gradient at the zero residual
        # initialization and can actually adapt the risk path.
        risk_forecast = dynamics * 0.0
        if (
            float(risk_forecast_weight) > 0.0
            and current_base is not None
            and history_z
            and history_base
        ):
            previous_base = torch.from_numpy(
                np.asarray(history_base[-1], np.float32).reshape(1)
            ).to(device)
            previous_adjusted, _ = adapter(previous, previous_base)
            risk_forecast = F.smooth_l1_loss(
                previous_adjusted, current_base.detach()
            )
        loss = (
            dynamics
            + float(risk_forecast_weight) * risk_forecast
            + 0.05 * anchor
            + 0.001 * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    after = adapter.state_dict()
    finite = all(torch.isfinite(value).all() for value in after.values())
    raw_change = float("nan")
    change = float("nan")
    projected = False
    if finite:
        raw_change, change, projected = project_state_delta(
            before, after, maximum=float(max_relative_change)
        )
        finite = bool(math.isfinite(change))
    accepted = bool(finite and change > 0.0)
    if not accepted:
        adapter.load_state_dict(before, strict=True)
    return {
        "applied": accepted,
        "reason": "causal_next_state_update" if accepted else "no_op_or_nonfinite",
        "loss": float(np.mean(losses)) if losses else 0.0,
        "raw_relative_parameter_change": float(raw_change) if math.isfinite(raw_change) else None,
        "relative_parameter_change": float(change if accepted else 0.0),
        "projected_to_radius": bool(projected and accepted),
        "risk_forecast_weight": float(risk_forecast_weight),
        "parameter_sha256": module_sha256(adapter),
    }


def persistent_update(
    adapter: LowRankRiskAdapter,
    current: dict[str, Any],
    replay: list[dict[str, Any]],
    guard: list[dict[str, Any]],
    *,
    group: str,
    device: str,
    threshold: float,
    learning_rate: float,
    steps: int,
    max_relative_change: float,
    max_guard_harms: int,
    hard_example_weight: float = 4.0,
) -> dict[str, Any]:
    if group not in {"FP", "FN", "TARDY"}:
        return {"updated": False, "reason": "correct_prediction", "group": group}
    if not math.isfinite(float(hard_example_weight)) or float(hard_example_weight) < 0.0:
        raise ValueError("J-TEXT hard-example weight must be finite and non-negative")
    batches = [current]
    same_group = [item for item in replay if item.get("group") == group]
    opposite = [item for item in replay if item.get("group") != group]
    if same_group:
        batches.append(same_group[-1]["bag"])
    if opposite:
        batches.append(opposite[-1]["bag"])
    before = copy.deepcopy(adapter.state_dict())
    optimizer_before = None
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    losses: list[float] = []
    adapter.train()
    for _ in range(int(steps)):
        loss_terms = []
        for bag in batches:
            z, base, labels = tensor_batch(bag, device)
            adjusted, residual = adapter(z, base)
            # Weight the endpoint that caused the revealed error.  For a
            # negative label this is proportional to the pre-update risk;
            # for a positive label it is proportional to the pre-update
            # miss probability.  The weight is computed from detached source
            # logits, so it cannot create a label-dependent inference path.
            source_risk = torch.sigmoid(base.detach())
            hardness = torch.where(labels > 0.5, 1.0 - source_risk, source_risk)
            weights = 1.0 + float(hard_example_weight) * hardness
            point_loss = F.binary_cross_entropy_with_logits(
                adjusted, labels, reduction="none"
            )
            loss_terms.append((weights * point_loss).sum() / weights.sum().clamp_min(1e-6))
            loss_terms.append(0.01 * residual.square().mean())
        if guard:
            guard_terms = []
            for item in guard:
                z = torch.from_numpy(item["z"]).to(device)
                base = torch.from_numpy(item["base_logit"]).to(device)
                adjusted, _ = adapter(z, base)
                guard_terms.append(F.smooth_l1_loss(adjusted, base.detach()))
            loss_terms.append(0.05 * torch.stack(guard_terms).mean())
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    after = adapter.state_dict()
    change = relative_change(before, after)
    harms = guard_harms(adapter, guard, threshold=threshold, device=device)
    finite = all(torch.isfinite(value).all() for value in after.values())
    accepted = bool(
        finite and change <= float(max_relative_change) and harms <= int(max_guard_harms)
    )
    if not accepted:
        adapter.load_state_dict(before, strict=True)
        if optimizer_before is not None:
            optimizer.load_state_dict(optimizer_before)
    return {
        "updated": accepted,
        "reason": "revealed_error_supervised_update" if accepted else "guard_or_change_rollback",
        "group": group,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "relative_parameter_change": float(change if accepted else 0.0),
        "guard_harms": int(harms),
        "replay_bags": len(batches) - 1,
        "hard_example_weight": float(hard_example_weight),
        "parameter_sha256": module_sha256(adapter),
    }


def metric_from_records(
    records: list[dict[str, Any]],
    outcomes: dict[int, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([bool(outcomes[int(row["shot"])]["is_disrupt"]) for row in records])
    scores = np.asarray([float(row["score"]) for row in records], dtype=np.float64)
    predicted = scores >= float(threshold)
    tp = int(np.sum(labels & predicted)); fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted)); tn = int(np.sum(~labels & ~predicted))
    leads = []
    early = []
    timely = []
    for row in records:
        if not bool(outcomes[int(row["shot"])]["is_disrupt"]):
            continue
        down = outcomes[int(row["shot"])]["down_time_s"]
        alarm_index = row.get("alarm_index")
        lead = None
        if alarm_index is not None:
            alarm_time = float(row["start"]) + float(row["endpoints"][int(alarm_index)]) / float(row["rate"])
            lead = (float(down) - alarm_time) * 1000.0
        is_timely = bool(lead is not None and lead > 10.0)
        timely.append(is_timely)
        if is_timely:
            leads.append(float(lead))
        if lead is not None and lead > 3000.0:
            early.append(int(row["shot"]))
    pos = scores[labels]; neg = scores[~labels]
    auc_value = float("nan")
    if len(pos) and len(neg):
        comparison = pos[:, None] - neg[None, :]
        auc_value = float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))
    return {
        "n_shots": len(records),
        "n_disruptions": int(labels.sum()),
        "n_non_disruptions": int((~labels).sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "tpr": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "auc": auc_value,
        "timely_tpr": float(np.mean(timely)) if timely else 0.0,
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "early_over_3000_ms_count": len(early),
        "early_over_3000_ms_shots": early,
        "threshold": float(threshold),
        "minimum_lead_ms": 10.0,
    }


def build_guard(
    encoder: nn.Module,
    head: nn.Module,
    validation: list[dict[str, Any]],
    *,
    device: str,
    max_shots_per_class: int = 4,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for label in (True, False):
        chosen.extend([
            row for row in validation if bool(row["is_disrupt"]) is label
        ][: int(max_shots_per_class)])
    guard: list[dict[str, Any]] = []
    for shot in chosen:
        endpoints, embeddings, logits = causal_embeddings(
            encoder, head, shot["v"], device=device, batch_size=512
        )
        if len(endpoints):
            selected = np.linspace(0, len(endpoints) - 1, min(24, len(endpoints)), dtype=np.int64)
            guard.append({
                "shot": int(shot["shot"]),
                "label": int(bool(shot["is_disrupt"])),
                "z": embeddings[selected],
                "base_logit": logits[selected],
            })
    if not guard:
        raise RuntimeError("J-TEXT online guard has no validation windows")
    return guard


def load_roles() -> dict[str, list[int]]:
    roles = SOURCE.load_canonical_jtext_roles(ROOT)
    test = sorted(int(value) for value in roles["test"])
    if len(test) != EXPECTED_TEST_SHOTS:
        raise ValueError(f"J-TEXT TEST220 count changed: {len(test)}")
    return {name: sorted(int(value) for value in ids) for name, ids in roles.items()}


def validate_advanced_source_contract(
    checkpoint: dict[str, Any], advanced_path: Path
) -> dict[str, Any]:
    """Fail closed if the parent was not trained with the requested label map."""

    training_label = checkpoint.get("training_label")
    if not isinstance(training_label, dict):
        raise ValueError("source checkpoint has no training-label provenance")
    if training_label.get("policy") != SOURCE.JTEXT_LABEL_POLICY_ADVANCED_FLOOR25:
        raise ValueError("J-TEXT online requires the max(25 ms, AdvancedTime) source checkpoint")
    expected_path = Path(advanced_path).expanduser().resolve()
    recorded_path = training_label.get("advanced_time_source")
    if recorded_path is not None and Path(str(recorded_path)).expanduser().resolve() != expected_path:
        raise ValueError("source checkpoint AdvancedTime path differs from runtime contract")
    recorded_sha = training_label.get("advanced_time_source_sha256")
    actual_sha = sha256_file(expected_path)
    if recorded_sha != actual_sha:
        raise ValueError("source checkpoint AdvancedTime SHA differs from runtime file")
    if float(training_label.get("advanced_time_floor_ms", float("nan"))) != LABELS.DEFAULT_FLOOR_MS:
        raise ValueError("source checkpoint AdvancedTime floor differs from 25 ms contract")
    return training_label


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("J-TEXT online run requires CUDA in rl_env")
    device = args.device
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    checkpoint_path = Path(args.source_checkpoint).expanduser().resolve()
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(RH.REBUILD), expected_seed=args.seed, device=device
    )
    training_label = validate_advanced_source_contract(
        checkpoint, SOURCE.JTEXT_ADVANCED_TIME_PATH
    )
    center = checkpoint["scaler_center"].cpu().numpy().astype(np.float32)
    scale = checkpoint["scaler_scale"].cpu().numpy().astype(np.float32)
    roles = load_roles()
    test_ids = roles["test"]
    threshold = float(checkpoint["validation_threshold"])
    parent_hash = module_sha256(nn.Sequential(encoder, head))

    validation = SOURCE.load_jtext_shots(roles["val"])
    GS.apply_scale(validation, center, scale)
    guard = build_guard(encoder, head, validation, device=device)

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"J-TEXT online output exists: {output}")
    output.mkdir(parents=True)
    (output / "risk_traces").mkdir()
    static_writer = JsonlWriter(output / "static_predictions.jsonl")
    online_writer = JsonlWriter(output / "online_predictions.jsonl")
    query_writer = JsonlWriter(output / "queries.jsonl")
    reveal_writer = JsonlWriter(output / "reveals.jsonl")
    update_writer = JsonlWriter(output / "updates.jsonl")

    # Static phase is completed before online state is created.  It is outcome
    # blind and its only purpose is the paired frozen baseline.
    static_records: list[dict[str, Any]] = []
    for index, shot_id in enumerate(test_ids):
        shot = load_signal_only(shot_id, center, scale)
        endpoints, _z, risk = base_trace(encoder, head, shot["v"], device=device)
        alarm_index = first_alarm(endpoints, risk, threshold=threshold)
        row = {
            "stream_index": index, "shot": shot_id,
            "score": float(np.max(risk)) if len(risk) else 0.0,
            "alarm_index": alarm_index,
            "endpoints": endpoints.tolist(),
            "start": float(shot["start"]), "rate": float(shot["rate"]),
            "threshold": threshold,
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
        }
        static_writer.append(row)
        static_records.append(row)
        np.savez_compressed(output / "risk_traces" / f"{shot_id}.npz", static=risk)

    persistent = LowRankRiskAdapter().to(device)
    parameter_count = adapter_parameter_count(persistent)
    if parameter_count >= 5000:
        raise RuntimeError(f"J-TEXT online adapter exceeds parameter budget: {parameter_count}")
    initial_adapter_hash = module_sha256(persistent)
    oracle = OutcomeOracle(test_ids)
    selector = CausalBlockBudgetSelector(
        total_shots=len(test_ids), fraction=args.budget_fraction,
        cutoff=args.query_cutoff, block_size=20,
    )
    lazy_advanced = LazyAdvancedTime(SOURCE.JTEXT_ADVANCED_TIME_PATH)
    signatures: list[np.ndarray] = []
    replay: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = []
    update_count = 0
    rollback_count = 0
    try:
        for index, shot_id in enumerate(test_ids):
            # Reload only the current shot after the paired static pass.  No
            # future-shot signal is held by the online state or query policy.
            shot = load_signal_only(shot_id, center, scale)
            endpoints, embeddings, base_logits = causal_embeddings(
                encoder, head, shot["v"], device=device
            )
            base_risk = torch.sigmoid(torch.from_numpy(base_logits)).numpy()
            fast = copy.deepcopy(persistent).to(device)
            fast.eval()
            online_risk = np.empty(len(endpoints), dtype=np.float32)
            intra_receipts: list[dict[str, Any]] = []
            history_z: list[np.ndarray] = []
            history_base: list[np.ndarray] = []
            for point in range(len(endpoints)):
                with torch.no_grad():
                    adjusted, _ = fast(
                        torch.from_numpy(embeddings[point:point + 1]).to(device),
                        torch.from_numpy(base_logits[point:point + 1]).to(device),
                    )
                online_risk[point] = float(torch.sigmoid(adjusted[0]).cpu())
                history_z.append(embeddings[point].copy())
                history_base.append(base_logits[point:point + 1].copy())
                # The current endpoint is scored before it can influence the
                # fast state.  Its observed value may adapt the state used by
                # the next endpoint, which is the strict causal online order.
                if (
                    point >= 1
                    and point + 1 < len(endpoints)
                    and point % max(int(args.intra_interval), 1) == 0
                ):
                    receipt = intra_update(
                        fast,
                        embeddings[point - 1], embeddings[point],
                        history_z[:-1], history_base[:-1],
                        device=device, learning_rate=args.intra_learning_rate,
                        steps=args.intra_steps,
                        max_relative_change=args.intra_max_relative_change,
                        current_base_logit=base_logits[point],
                        risk_forecast_weight=args.intra_risk_forecast_weight,
                    )
                    receipt.update({"shot": shot_id, "point": point})
                    intra_receipts.append(receipt)
                    update_writer.append({"kind": "intra", **receipt})
            signature = shot_signature(shot["v"], shot["a"])
            novelty = normalized_novelty(signature, signatures)
            components = query_components(online_risk, base_logits, threshold, novelty)
            selected, selection_reason = selector.decide(
                score=components["query_score"], stream_index=index
            )
            alarm_index = first_alarm(endpoints, online_risk, threshold=threshold)
            row = {
                "stream_index": index, "shot": shot_id,
                "score": float(np.max(online_risk)) if len(online_risk) else 0.0,
                "alarm_index": alarm_index,
                "endpoints": endpoints.tolist(),
                "start": float(shot["start"]), "rate": float(shot["rate"]),
                "threshold": threshold, "queried": bool(selected),
                "query_score": components["query_score"],
                "query_components": components,
                "query_reason": selection_reason,
                "persistent_adapter_sha256_before": module_sha256(persistent),
                "fast_adapter_sha256_at_seal": module_sha256(fast),
                "intra_update_count": len(intra_receipts),
            }
            online_writer.append(row)
            online_records.append(row)
            np.savez_compressed(
                output / "risk_traces" / f"{shot_id}.npz",
                static=base_risk, online=online_risk, endpoints=endpoints,
            )
            oracle.seal(shot_id)
            if selected:
                oracle.query(shot_id)
            query_writer.append({
                "stream_index": index, "shot": shot_id,
                "selected": bool(selected), "selection_reason": selection_reason,
                "query_score": components["query_score"],
                "components": components,
                "label_access_before_decision": False,
            })
            if selected:
                outcome = oracle.reveal(shot_id)
                advanced = (
                    lazy_advanced.lookup_after_reveal(shot_id)
                    if outcome["is_disrupt"] else None
                )
                group, lead_ms = outcome_group(
                    is_disrupt=bool(outcome["is_disrupt"]),
                    down_time_s=outcome["down_time_s"], start=shot["start"],
                    rate=shot["rate"], endpoints=endpoints, risk=online_risk,
                    threshold=threshold,
                )
                bag = labelled_bag(
                    shot, outcome, advanced, embeddings, base_logits, endpoints,
                    max_windows=args.max_bag_windows,
                )
                receipt = {"updated": False, "reason": "no_usable_labelled_windows", "group": group}
                if bag is not None:
                    receipt = persistent_update(
                        persistent, bag, replay, guard,
                        group=group, device=device, threshold=threshold,
                        learning_rate=args.inter_learning_rate,
                        steps=args.inter_steps,
                        max_relative_change=args.inter_max_relative_change,
                        max_guard_harms=args.max_guard_harms,
                        hard_example_weight=args.hard_example_weight,
                    )
                    bag_record = {"bag": bag, "group": group}
                    replay.append(bag_record)
                    if len(replay) > int(args.max_replay_shots):
                        replay = replay[-int(args.max_replay_shots):]
                update_count += int(bool(receipt.get("updated")))
                rollback_count += int(receipt.get("reason") == "guard_or_change_rollback")
                reveal_writer.append({
                    "stream_index": index, "shot": shot_id,
                    "is_disrupt": bool(outcome["is_disrupt"]),
                    "down_time_s": outcome["down_time_s"],
                    "advanced_time_ms": advanced,
                    "discrepancy_group": group, "lead_ms": lead_ms,
                })
                update_writer.append({
                    "kind": "inter", "shot": shot_id, "group": group,
                    "receipt": receipt,
                    "persistent_adapter_sha256_after": module_sha256(persistent),
                })
            else:
                oracle.discard(shot_id)
            signatures.append(signature)
            if (index + 1) % 20 == 0:
                print(
                    f"[JTEXT-ONLINE] sealed {index + 1}/{len(test_ids)} "
                    f"queries={selector.selected}", flush=True
                )
    finally:
        static_writer.close(); online_writer.close(); query_writer.close()
        reveal_writer.close(); update_writer.close()

    if selector.selected != selector.total_budget:
        raise RuntimeError(
            f"J-TEXT exact query budget failed: {selector.selected} != {selector.total_budget}"
        )
    oracle.open_final_evaluation()
    outcomes = read_all_outcomes(test_ids)
    static_metrics = metric_from_records(static_records, outcomes, threshold=threshold)
    online_metrics = metric_from_records(online_records, outcomes, threshold=threshold)
    static_scores = np.asarray([row["score"] for row in static_records])
    online_scores = np.asarray([row["score"] for row in online_records])
    transitions = {
        "static_alarm_to_online_no_alarm": int(sum(
            float(left["score"]) >= threshold and float(right["score"]) < threshold
            for left, right in zip(static_records, online_records)
        )),
        "static_no_alarm_to_online_alarm": int(sum(
            float(left["score"]) < threshold and float(right["score"]) >= threshold
            for left, right in zip(static_records, online_records)
        )),
    }
    summary = {
        "status": "JTEXT_ONLINE_DIAGNOSTIC",
        "stage": "JTEXT_ADVANCED_FLOOR25_STATIC_PLUS_PREQUENTIAL_ONLINE",
        "seed": int(args.seed),
        "stream": {"role": "TEST220", "shots": len(test_ids), "order": "ascending_shot_id"},
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_training_label": training_label,
        "threshold": threshold,
        "threshold_source": "J-TEXT VAL160 embedded in source checkpoint",
        "budget": {
            "fraction": float(args.budget_fraction),
            "total": selector.total_budget,
            "selected": selector.selected,
            "cutoff": float(args.query_cutoff),
            "selection_is_label_free": True,
        },
        "adapter": {
            "architecture": "low-rank latent residual + risk residual + next-state head",
            "parameters": parameter_count,
            "initial_sha256": initial_adapter_hash,
            "final_sha256": module_sha256(persistent),
            "intra_scope": "fast adapter; reset before every shot",
            "inter_scope": "persistent adapter; only after queried HDF5 reveal",
            "intra_learning_rate": float(args.intra_learning_rate),
            "intra_risk_forecast_weight": float(args.intra_risk_forecast_weight),
            "inter_learning_rate": float(args.inter_learning_rate),
            "intra_steps": int(args.intra_steps),
            "inter_steps": int(args.inter_steps),
            "hard_example_weight": float(args.hard_example_weight),
            "accepted_inter_updates": update_count,
            "rollback_inter_updates": rollback_count,
        },
        "static_metrics": static_metrics,
        "online_metrics": online_metrics,
        "delta": {
            "tpr": online_metrics["tpr"] - static_metrics["tpr"],
            "fpr": online_metrics["fpr"] - static_metrics["fpr"],
            "j": (online_metrics["tpr"] - online_metrics["fpr"])
            - (static_metrics["tpr"] - static_metrics["fpr"]),
            "mean_abs_score_change": float(np.mean(np.abs(online_scores - static_scores))),
            "transitions": transitions,
        },
        "advanced_time_access": {
            "path": str(lazy_advanced.path),
            "sha256": sha256_file(lazy_advanced.path),
            "looked_up_after_reveal": lazy_advanced.lookups,
            "lookup_count": len(lazy_advanced.lookups),
        },
        "frozen_source_integrity": {
            "source_module_sha256_before": parent_hash,
            "source_module_sha256_after": module_sha256(nn.Sequential(encoder, head)),
            "source_parameters_updated": False,
        },
        "causal_contract": {
            "prediction_order": (
                "signal prefix -> predict/seal -> causal fast update for the next "
                "endpoint -> query -> reveal -> inter update"
            ),
            "test_outcomes_read_before_query": False,
            "jtext_online_checkpoint_used_for_east": False,
            "test_labels_used_for_threshold_or_query_cutoff": False,
            "online_stream_used_for_method_development": True,
        },
        "hashes": {
            "split": SOURCE.split_sha256(roles),
            "static_predictions": sha256_file(output / "static_predictions.jsonl"),
            "online_predictions": sha256_file(output / "online_predictions.jsonl"),
            "queries": sha256_file(output / "queries.jsonl"),
            "reveals": sha256_file(output / "reveals.jsonl"),
            "updates": sha256_file(output / "updates.jsonl"),
        },
        "claim_boundary": (
            "TEST220 is a development online stream; results are diagnostic until an "
            "untouched chronological J-TEXT stream confirms the frozen configuration."
        ),
    }
    atomic_json(output / "config.json", {
        "seed": int(args.seed), "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "budget_fraction": float(args.budget_fraction),
        "query_cutoff": float(args.query_cutoff),
        "intra_learning_rate": float(args.intra_learning_rate),
        "intra_risk_forecast_weight": float(args.intra_risk_forecast_weight),
        "inter_learning_rate": float(args.inter_learning_rate),
        "intra_steps": int(args.intra_steps), "inter_steps": int(args.inter_steps),
        "max_bag_windows": int(args.max_bag_windows),
        "max_replay_shots": int(args.max_replay_shots),
        "hard_example_weight": float(args.hard_example_weight),
    })
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"static": static_metrics, "online": online_metrics, "delta": summary["delta"]}, indent=2), flush=True)
    print(f"WROTE {output}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J-TEXT rebuilt17 causal online-active replay")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--budget-fraction", type=float, default=DEFAULT_BUDGET_FRACTION)
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
    parser.add_argument("--max-bag-windows", type=int, default=DEFAULT_MAX_BAG_WINDOWS)
    parser.add_argument("--max-replay-shots", type=int, default=DEFAULT_MAX_REPLAY_SHOTS)
    parser.add_argument(
        "--hard-example-weight", type=float, default=4.0,
        help="post-reveal source-risk hardness multiplier for labelled endpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
