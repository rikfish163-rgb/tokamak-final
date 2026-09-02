#!/usr/bin/env python3
"""Run a causal typed-memory ablation on the independent J-TEXT stream.

This runner is intentionally not part of the EAST V3C path.  It starts from
one frozen Adv25 J-TEXT source checkpoint and compares, under one shared
label-free query manifest:

``static`` | ``intra_only`` | ``inter_neural`` | ``memory_only`` |
``inter_neural_memory`` | ``intra_inter_neural_memory``.

The memory and the persistent residual can only consume a selected shot after
the prediction and query have been sealed.  The fast adapter is reset for
every shot.  TEST220 is a development stream, so all resulting numbers are
diagnostic rather than independent confirmation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

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
import jtext_causal_memory as MEMORY  # noqa: E402
import jtext_label_contract as LABELS  # noqa: E402
import jtext_query_policy as QUERY  # noqa: E402
import run_jtext_rebuilt17_online as ONLINE  # noqa: E402
import train_jtext_source_transformer as SOURCE  # noqa: E402


MODES = (
    "static",
    "intra_only",
    "inter_neural",
    "memory_only",
    "inter_neural_memory",
    "intra_inter_neural_memory",
)
INTRA_MODES = frozenset({"intra_only", "intra_inter_neural_memory"})
NEURAL_MODES = frozenset({
    "inter_neural", "inter_neural_memory", "intra_inter_neural_memory",
})
MEMORY_MODES = frozenset({
    "memory_only", "inter_neural_memory", "intra_inter_neural_memory",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output ledger: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("x", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def module_hash(module: nn.Module) -> str:
    return ONLINE.module_sha256(module)


def set_deterministic(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")


@dataclass
class ShotTrace:
    shot: int
    values: np.ndarray
    available: np.ndarray
    start: float
    rate: float
    endpoints: np.ndarray
    embeddings: np.ndarray
    base_logits: np.ndarray
    base_risk: np.ndarray
    signature: np.ndarray


def build_trace(
    encoder: nn.Module,
    head: nn.Module,
    shot: dict[str, Any],
    *,
    device: str,
) -> ShotTrace:
    endpoints, embeddings, base_logits = ONLINE.causal_embeddings(
        encoder, head, shot["v"], device=device,
    )
    return ShotTrace(
        shot=int(shot["shot"]),
        values=np.asarray(shot["v"], dtype=np.float32),
        available=np.asarray(shot["a"], dtype=bool),
        start=float(shot["start"]),
        rate=float(shot["rate"]),
        endpoints=endpoints,
        embeddings=embeddings,
        base_logits=base_logits,
        base_risk=torch.sigmoid(torch.from_numpy(base_logits)).numpy(),
        signature=ONLINE.shot_signature(shot["v"], shot["a"]),
    )


def build_query_manifest(
    traces: list[ShotTrace],
    *,
    threshold: float,
    budget_fraction: float,
    cutoff: float,
    error_prior: QUERY.ValidationErrorPrior | None = None,
    error_prior_weight: float = 0.75,
) -> list[dict[str, Any]]:
    """Build one outcome-blind, chronological query manifest for all arms."""

    if not 0.0 <= float(error_prior_weight) <= 1.0:
        raise ValueError("J-TEXT error-prior blend weight must lie in [0, 1]")

    selector = ONLINE.CausalBlockBudgetSelector(
        total_shots=len(traces), fraction=budget_fraction, cutoff=cutoff,
        block_size=20,
    )
    history: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        novelty = ONLINE.normalized_novelty(trace.signature, history)
        components = MEMORY.causal_query_components(
            trace.base_risk, trace.base_logits,
            threshold=threshold, novelty=novelty,
        )
        prior_score = None
        if error_prior is not None:
            prior_features = QUERY.trace_features(
                trace.base_risk, trace.base_logits, trace.values, trace.available,
                threshold=threshold,
            )
            prior_score = float(error_prior.score(prior_features)[0])
            components = {
                **components,
                "validation_error_prior": prior_score,
                "base_query_score": float(components["query_score"]),
                "query_score": float(
                    float(error_prior_weight) * prior_score
                    + (1.0 - float(error_prior_weight))
                    * float(components["query_score"])
                ),
            }
        selected, reason = selector.decide(
            score=components["query_score"], stream_index=index,
        )
        manifest.append({
            "stream_index": int(index),
            "block": int(index // 20),
            "position": int(index % 20),
            "shot": int(trace.shot),
            "selected": bool(selected),
            "selection_reason": reason,
            "query_score": float(components["query_score"]),
            "components": components,
            "query_policy": (
                "validation_error_prior_blend"
                if error_prior is not None else "causal_components"
            ),
            "validation_error_prior_score": prior_score,
            "history_size": len(history),
            "label_access_before_decision": False,
        })
        history.append(trace.signature)
    if selector.selected != selector.total_budget:
        raise RuntimeError(
            f"J-TEXT causal memory budget mismatch: {selector.selected} != "
            f"{selector.total_budget}"
        )
    return manifest


def fit_validation_error_prior(
    encoder: nn.Module,
    head: nn.Module,
    validation: list[dict[str, Any]],
    *,
    threshold: float,
    device: str,
    l2: float,
) -> tuple[QUERY.ValidationErrorPrior, dict[str, int]]:
    """Fit the query-value prior on VAL160 only.

    The target is the *shot-level source error group* (FP/FN/TARDY), not the
    TEST220 outcome.  This is legal calibration data for a query policy and is
    deliberately kept separate from the risk model and from the online
    reveal path.
    """

    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "TARDY": 0}
    for shot in validation:
        trace = build_trace(encoder, head, shot, device=device)
        outcome = {
            "is_disrupt": bool(shot["is_disrupt"]),
            "down_time_s": shot.get("down"),
        }
        group, _lead = outcome_group_for_trace(
            trace, trace.base_risk, outcome, threshold=threshold,
        )
        groups[group] = groups.get(group, 0) + 1
        rows.append(QUERY.trace_features(
            trace.base_risk, trace.base_logits, trace.values, trace.available,
            threshold=threshold,
        ))
        labels.append(int(group in {"FP", "FN", "TARDY"}))
    prior = QUERY.ValidationErrorPrior.fit(
        np.stack(rows), np.asarray(labels, dtype=np.int64), l2=float(l2),
    )
    return prior, groups


def build_timing_protections(
    encoder: nn.Module,
    head: nn.Module,
    shots: list[dict[str, Any]],
    *,
    threshold: float,
    device: str,
    max_shots_per_class: int = 4,
) -> list[dict[str, Any]]:
    """Build a source-side timing guard from correct VAL decisions.

    A binary alarm guard cannot see a TP->TARDY degradation.  These records
    retain the endpoint times and the authoritative validation DownTime so a
    candidate adapter is rejected if it changes a previously correct TP/TN
    group.  They are never updated with TEST labels.
    """

    chosen: list[dict[str, Any]] = []
    for wanted in (True, False):
        candidates = [
            shot for shot in shots if bool(shot["is_disrupt"]) is wanted
        ]
        chosen.extend(candidates[: int(max_shots_per_class)])
    protections: list[dict[str, Any]] = []
    for shot in chosen:
        trace = build_trace(encoder, head, shot, device=device)
        outcome = {
            "is_disrupt": bool(shot["is_disrupt"]),
            "down_time_s": shot.get("down"),
        }
        group, _lead = outcome_group_for_trace(
            trace, trace.base_risk, outcome, threshold=threshold,
        )
        if group not in {"TP", "TN"}:
            continue
        protections.append({
            "z": trace.embeddings,
            "base": trace.base_logits,
            "label": bool(group == "TP"),
            "shot": int(trace.shot),
            "endpoints": trace.endpoints,
            "start": float(trace.start),
            "rate": float(trace.rate),
            "outcome": outcome,
            "base_group": group,
            "source": "VAL160",
        })
    if not protections:
        raise RuntimeError("J-TEXT timing guard has no correct VAL decisions")
    return protections


def target_array(
    trace: ShotTrace,
    outcome: dict[str, Any],
    advanced_time_ms: float | None,
) -> tuple[np.ndarray, float | None]:
    """Construct endpoint labels only after the selected reveal."""

    target = np.zeros(len(trace.endpoints), dtype=np.float32)
    if not bool(outcome["is_disrupt"]):
        return target, None
    down = outcome.get("down_time_s")
    if down is None:
        raise ValueError("disruptive J-TEXT reveal has no DownTime")
    horizon, _raw, _fallback = LABELS.effective_horizon_ms(
        trace.shot,
        {} if advanced_time_ms is None else {trace.shot: float(advanced_time_ms)},
        floor_ms=LABELS.DEFAULT_FLOOR_MS,
    )
    times = trace.start + trace.endpoints.astype(np.float64) / trace.rate
    ttd_ms = (float(down) - times) * 1000.0
    target[(ttd_ms >= 0.0) & (ttd_ms <= horizon)] = 1.0
    return target, float(horizon)


def _unique_ordered(indices: list[int], length: int) -> np.ndarray:
    seen: set[int] = set()
    result: list[int] = []
    for raw in indices:
        index = int(raw)
        if 0 <= index < int(length) and index not in seen:
            seen.add(index)
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def select_anchor_indices(
    trace: ShotTrace,
    target: np.ndarray,
    *,
    threshold: float,
    per_type: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select hard/error and protection endpoints after outcome reveal."""

    labels = np.asarray(target, dtype=np.float32).reshape(-1)
    logits = np.asarray(trace.base_logits, dtype=np.float64).reshape(-1)
    if len(labels) != len(logits) or not len(labels):
        raise ValueError("J-TEXT anchor arrays are not aligned")
    negative = np.flatnonzero(labels < 0.5)
    positive = np.flatnonzero(labels >= 0.5)
    chosen: list[int] = []
    roles: list[str] = []
    # Hard negatives are the endpoints most likely to trigger a false alarm.
    for index in negative[np.argsort(logits[negative])[::-1][: int(per_type)]]:
        chosen.append(int(index)); roles.append("negative_hard")
    # Positive anchors include the lowest-risk positive endpoints, which are
    # the useful support for a future low-risk false-negative trajectory.
    if len(positive):
        positive_order = positive[np.argsort(logits[positive])]
        for index in positive_order[: int(per_type)]:
            chosen.append(int(index)); roles.append("positive_low_risk")
        threshold_distance = np.abs(
            torch.sigmoid(torch.from_numpy(logits[positive])).numpy()
            - float(threshold)
        )
        for index in positive[np.argsort(threshold_distance)[: int(per_type)]]:
            chosen.append(int(index)); roles.append("positive_boundary")
    # Always retain the first base alarm, if any, as an explicitly typed
    # protection/error endpoint.  This is still a post-reveal memory write.
    alarm = ONLINE.first_alarm(trace.endpoints, trace.base_risk, threshold=threshold)
    if alarm is not None:
        chosen.append(int(alarm))
        roles.append("first_alarm")
    indices = _unique_ordered(chosen, len(labels))
    if not len(indices):
        indices = np.asarray([int(np.argmax(np.abs(logits)))], dtype=np.int64)
        roles = ["fallback_endpoint"]
    # Deduplicate role labels in the same order as the retained indices.
    role_by_index: dict[int, str] = {}
    for index, role in zip(chosen, roles):
        role_by_index.setdefault(int(index), role)
    roles = [role_by_index[int(index)] for index in indices]
    return indices, labels[indices], roles


def select_error_memory_indices(
    trace: ShotTrace,
    target: np.ndarray,
    *,
    group: str,
    threshold: float,
    horizon_ms: float | None,
    outcome: dict[str, Any],
    per_type: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select only causal supports that explain a revealed error.

    Generic endpoint replay was too broad: every queried TP/TN contributed
    correction prototypes and positive supports were allowed to move very
    low-risk points.  This function keeps the neural update bag broad, but
    gives the typed memory only the endpoints that could have caused the
    observed FP/FN/TARDY decision.
    """

    labels = np.asarray(target, dtype=np.float32).reshape(-1)
    logits = np.asarray(trace.base_logits, dtype=np.float64).reshape(-1)
    if group not in {"FP", "FN", "TARDY"}:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), []
    if len(labels) != len(logits) or not len(labels):
        raise ValueError("J-TEXT error-memory arrays are not aligned")
    if group == "FP":
        candidates = np.flatnonzero(labels < 0.5)
        if len(candidates):
            risk = torch.sigmoid(torch.from_numpy(logits)).numpy()
            near = candidates[risk[candidates] >= float(threshold) - 0.18]
            if len(near):
                candidates = near
            chosen = candidates[np.argsort(logits[candidates])[::-1][: int(per_type)]]
        else:
            chosen = np.empty(0, dtype=np.int64)
        roles = ["error_negative_fp"] * len(chosen)
    else:
        candidates = np.flatnonzero(labels >= 0.5)
        if group == "TARDY" and len(candidates) and horizon_ms is not None:
            down = outcome.get("down_time_s")
            if down is not None:
                times = trace.start + trace.endpoints.astype(np.float64) / trace.rate
                ttd = (float(down) - times) * 1000.0
                timely = candidates[(ttd[candidates] > 10.0) & (ttd[candidates] <= float(horizon_ms))]
                if len(timely):
                    candidates = timely
        # For FN/TARDY the useful support is the most plausible positive
        # endpoint, not the easiest low-risk positive.
        chosen = candidates[np.argsort(logits[candidates])[::-1][: int(per_type)]] if len(candidates) else np.empty(0, dtype=np.int64)
        roles = [f"error_positive_{group.lower()}"] * len(chosen)
    return chosen.astype(np.int64), labels[chosen], roles


def make_samples(
    trace: ShotTrace,
    indices: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "z": np.asarray(trace.embeddings[indices], dtype=np.float32),
        "base": np.asarray(trace.base_logits[indices], dtype=np.float32),
        "label": np.asarray(labels, dtype=np.float32),
    }


def concatenate_samples(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    if not items:
        return None
    return {
        name: np.concatenate([item[name] for item in items], axis=0)
        for name in ("z", "base", "label")
    }


def sample_replay(
    replay: list[dict[str, Any]], *,
    count: int,
) -> dict[str, np.ndarray] | None:
    if not replay:
        return None
    all_items = [row["samples"] for row in replay]
    joined = concatenate_samples(all_items)
    if joined is None or not len(joined["label"]):
        return None
    labels = joined["label"] >= 0.5
    errors = np.abs(
        torch.sigmoid(torch.from_numpy(joined["base"])).numpy()
        - joined["label"]
    )
    chosen: list[int] = []
    quota = max((int(count) + 1) // 2, 1)
    for mask in (labels, ~labels):
        candidates = np.flatnonzero(mask)
        if len(candidates):
            order = candidates[np.argsort(errors[candidates])[::-1]]
            chosen.extend(int(index) for index in order[:quota])
    if not chosen:
        chosen = list(range(len(joined["label"])))
    while len(chosen) < int(count):
        chosen.extend(chosen[: int(count) - len(chosen)])
    selected = np.asarray(chosen[: int(count)], dtype=np.int64)
    return {name: value[selected] for name, value in joined.items()}


def guard_harms_extended(
    adapter: ONLINE.LowRankRiskAdapter,
    guard: list[dict[str, Any]],
    protections: list[dict[str, Any]],
    *,
    threshold: float,
    device: str,
) -> int:
    """Count flips away from already correct validation/query decisions.

    The legacy guard only compared a binary alarm bit.  That allowed a
    previously timely TP to become a late alarm (TARDY) while still looking
    like an ``alarm=True`` decision.  The extended records therefore replay
    the full endpoint trace and protect the operational group, not just the
    existence of an alarm.
    """

    harmed = ONLINE.guard_harms(adapter, guard, threshold=threshold, device=device)
    adapter.eval()
    with torch.no_grad():
        for item in protections:
            z = torch.from_numpy(item["z"]).to(device)
            base = torch.from_numpy(item["base"]).to(device)
            adjusted, _ = adapter(z, base)
            if not {"endpoints", "start", "rate", "outcome", "base_group"}.issubset(item):
                # Backward-compatible protection rows from old ledgers retain
                # the binary check, but new rows always take the timing path.
                base_alarm = bool(torch.sigmoid(base).max().item() >= threshold)
                adjusted_alarm = bool(torch.sigmoid(adjusted).max().item() >= threshold)
                if base_alarm == bool(item["label"]) and adjusted_alarm != bool(item["label"]):
                    harmed += 1
                continue
            adjusted_group, _lead = ONLINE.outcome_group(
                is_disrupt=bool(item["outcome"]["is_disrupt"]),
                down_time_s=item["outcome"]["down_time_s"],
                start=float(item["start"]), rate=float(item["rate"]),
                endpoints=np.asarray(item["endpoints"], dtype=np.int64),
                risk=torch.sigmoid(adjusted).detach().cpu().numpy(),
                threshold=threshold,
            )
            if adjusted_group != str(item["base_group"]):
                harmed += 1
    return int(harmed)


def project_adapter(
    before: dict[str, Tensor],
    after: dict[str, Tensor],
    *,
    maximum: float,
) -> tuple[float, float, bool]:
    raw = ONLINE.relative_change(before, after)
    if not math.isfinite(raw):
        return raw, raw, False
    if raw <= float(maximum):
        return raw, raw, False
    scale = float(maximum) / raw
    with torch.no_grad():
        for name, value in after.items():
            value.copy_(before[name] + scale * (value - before[name]))
    return raw, ONLINE.relative_change(before, after), True


def focused_neural_update(
    adapter: ONLINE.LowRankRiskAdapter,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray] | None,
    guard: list[dict[str, Any]],
    protections: list[dict[str, Any]],
    *,
    group: str,
    threshold: float,
    device: str,
    learning_rate: float,
    steps: int,
    max_relative_change: float,
    max_guard_harms: int,
) -> dict[str, Any]:
    """Train only on a revealed error bag, then transactionally guard it."""

    if group not in {"FP", "FN", "TARDY"}:
        return {
            "updated": False,
            "reason": "correct_prediction_no_gradient",
            "group": group,
            "guard_harms": 0,
        }
    before = copy.deepcopy(adapter.state_dict())
    batches = [current] + ([] if replay is None else [replay])
    z = torch.from_numpy(np.concatenate([item["z"] for item in batches])).to(device)
    base = torch.from_numpy(np.concatenate([item["base"] for item in batches])).to(device)
    labels = torch.from_numpy(np.concatenate([item["label"] for item in batches])).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    losses: list[float] = []
    adapter.train()
    for _ in range(int(steps)):
        adjusted, residual = adapter(z, base)
        probabilities = torch.sigmoid(adjusted)
        point_loss = F.binary_cross_entropy_with_logits(adjusted, labels, reduction="none")
        # Give the newly revealed hard/error endpoints more influence, while
        # retaining every replay class to prevent one-shot sign reversal.
        difficulty = torch.abs(probabilities.detach() - labels)
        weights = 1.0 + 2.0 * difficulty
        supervised = (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        anchor = residual.square().mean()
        loss = supervised + 0.01 * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    after = adapter.state_dict()
    finite = all(bool(torch.isfinite(value).all()) for value in after.values())
    raw_change = float("nan")
    realized = float("nan")
    projected = False
    if finite:
        raw_change, realized, projected = project_adapter(
            before, after, maximum=max_relative_change,
        )
        finite = math.isfinite(realized)
    harms = (
        guard_harms_extended(
            adapter, guard, protections, threshold=threshold, device=device,
        ) if finite else 0
    )
    accepted = bool(finite and harms <= int(max_guard_harms) and realized > 0.0)
    if not accepted:
        adapter.load_state_dict(before, strict=True)
    return {
        "updated": accepted,
        "reason": "revealed_error_focused_update" if accepted else "guard_or_change_rollback",
        "group": group,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "raw_relative_parameter_change": float(raw_change) if math.isfinite(raw_change) else None,
        "relative_parameter_change": float(realized) if accepted else 0.0,
        "projected_to_radius": bool(projected and accepted),
        "guard_harms": int(harms),
        "batch_points": int(len(labels)),
        "replay_points": int(0 if replay is None else len(replay["label"])),
        "parameter_sha256": module_hash(adapter),
    }


def score_trace(
    mode: str,
    trace: ShotTrace,
    persistent: ONLINE.LowRankRiskAdapter,
    memory: MEMORY.CausalTypedTrajectoryMemory | None,
    *,
    threshold: float,
    device: str,
    intra_interval: int,
    intra_learning_rate: float,
    intra_steps: int,
    intra_max_relative_change: float,
    memory_routes: str = "both",
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Score one shot endpoint-by-endpoint under one mode."""

    if memory_routes not in {"both", "negative_only", "positive_only"}:
        raise ValueError("memory_routes must be both, negative_only, or positive_only")

    if mode == "static":
        return trace.base_risk.copy(), [], [], module_hash(persistent)
    fast = copy.deepcopy(persistent).to(device) if mode in INTRA_MODES else persistent
    if mode in INTRA_MODES:
        fast.eval()
    risk = np.empty(len(trace.endpoints), dtype=np.float32)
    intra_receipts: list[dict[str, Any]] = []
    memory_receipts: list[dict[str, Any]] = []
    history_z: list[np.ndarray] = []
    history_base: list[np.ndarray] = []
    interval = max(int(intra_interval), 1)
    for point in range(len(trace.endpoints)):
        with torch.no_grad():
            adjusted, _ = fast(
                torch.from_numpy(trace.embeddings[point:point + 1]).to(device),
                torch.from_numpy(trace.base_logits[point:point + 1]).to(device),
            )
        adjusted_value = float(adjusted[0].cpu())
        if memory is not None:
            trend = 0.0 if point == 0 else float(
                trace.base_logits[point] - trace.base_logits[point - 1]
            )
            feature = MEMORY.endpoint_feature(
                trace.embeddings[point],
                source_logit=float(trace.base_logits[point]),
                trend=trend,
                elapsed_s=float(trace.endpoints[point]) / trace.rate,
            )
            adjusted_value, receipt = memory.correct(
                feature, adjusted_value, threshold=threshold, error_only=True,
                allow_negative=memory_routes in {"both", "negative_only"},
                allow_positive=memory_routes in {"both", "positive_only"},
            )
            receipt.update({"shot": trace.shot, "endpoint": point})
            memory_receipts.append(receipt)
        risk[point] = float(torch.sigmoid(torch.tensor(adjusted_value)).item())
        history_z.append(trace.embeddings[point].copy())
        history_base.append(trace.base_logits[point:point + 1].copy())
        if (
            mode in INTRA_MODES
            and point >= 1
            and point + 1 < len(trace.endpoints)
            and point % interval == 0
        ):
            receipt = ONLINE.intra_update(
                fast,
                trace.embeddings[point - 1], trace.embeddings[point],
                history_z[:-1], history_base[:-1], device=device,
                learning_rate=intra_learning_rate, steps=intra_steps,
                max_relative_change=intra_max_relative_change,
                current_base_logit=trace.base_logits[point],
                risk_forecast_weight=0.5,
            )
            receipt.update({"shot": trace.shot, "point": point})
            intra_receipts.append(receipt)
    return risk, intra_receipts, memory_receipts, module_hash(fast)


def outcome_group_for_trace(
    trace: ShotTrace,
    risk: np.ndarray,
    outcome: dict[str, Any],
    *,
    threshold: float,
) -> tuple[str, float | None]:
    return ONLINE.outcome_group(
        is_disrupt=bool(outcome["is_disrupt"]),
        down_time_s=outcome["down_time_s"], start=trace.start, rate=trace.rate,
        endpoints=trace.endpoints, risk=risk, threshold=threshold,
    )


def metric(records: list[dict[str, Any]], outcomes: dict[int, dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = ONLINE.metric_from_records(records, outcomes, threshold=threshold)
    result["operational_tpr"] = float(result["timely_tpr"])
    result["j_operational"] = float(result["operational_tpr"] - result["fpr"])
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("J-TEXT causal memory experiment requires CUDA in rl_env")
    set_deterministic(args.seed)
    device = args.device
    checkpoint_path = args.source_checkpoint.expanduser().resolve()
    encoder, head, checkpoint = GS.load_source_checkpoint(
        checkpoint_path, channels=len(ONLINE.RH.REBUILD),
        expected_seed=args.seed, device=device,
    )
    ONLINE.validate_advanced_source_contract(
        checkpoint, SOURCE.JTEXT_ADVANCED_TIME_PATH,
    )
    center = checkpoint["scaler_center"].cpu().numpy().astype(np.float32)
    scale = checkpoint["scaler_scale"].cpu().numpy().astype(np.float32)
    roles = ONLINE.load_roles()
    test_ids = roles["test"]
    threshold = float(checkpoint["validation_threshold"])
    source_hash_before = module_hash(nn.Sequential(encoder, head))

    # The guard is built from VAL160 and is never refreshed from TEST220.
    validation = SOURCE.load_jtext_shots(roles["val"])
    GS.apply_scale(validation, center, scale)
    guard = ONLINE.build_guard(encoder, head, validation, device=device)
    validation_protections = build_timing_protections(
        encoder, head, validation, threshold=threshold, device=device,
    )

    error_prior: QUERY.ValidationErrorPrior | None = None
    validation_error_groups: dict[str, int] | None = None
    if args.query_policy == "validation_error_prior":
        error_prior, validation_error_groups = fit_validation_error_prior(
            encoder, head, validation, threshold=threshold, device=device,
            l2=args.query_error_l2,
        )

    # Signal-only trace prepass.  It never opens HDF5 meta or AdvancedTime.
    traces: list[ShotTrace] = []
    for index, shot_id in enumerate(test_ids):
        shot = ONLINE.load_signal_only(shot_id, center, scale)
        traces.append(build_trace(encoder, head, shot, device=device))
        if (index + 1) % 20 == 0:
            print(f"[JTEXT-MEMORY] signal-only traces {index + 1}/{len(test_ids)}", flush=True)
    manifest = build_query_manifest(
        traces, threshold=threshold,
        budget_fraction=args.budget_fraction, cutoff=args.query_cutoff,
        error_prior=error_prior,
        error_prior_weight=args.query_error_prior_weight,
    )
    manifest_by_shot = {int(row["shot"]): row for row in manifest}
    if list(manifest_by_shot) != test_ids:
        raise RuntimeError("causal memory manifest changed TEST220 order")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    if error_prior is not None:
        atomic_json(output / "query_error_prior.json", {
            "schema_version": QUERY.SCHEMA_VERSION,
            "fit_source": "J-TEXT VAL160 only",
            "test_labels_used": False,
            "l2": float(args.query_error_l2),
            "blend_weight": float(args.query_error_prior_weight),
            "validation_error_groups": validation_error_groups,
            "prior_sha256": error_prior.sha256(),
            "state": error_prior.state_dict(),
        })
    manifest_writer = JsonlWriter(output / "query_manifest.jsonl")
    for row in manifest:
        manifest_writer.append(row)
    manifest_writer.close()

    # All arms receive byte-identical adapter parents and an independent
    # memory.  Static retains an explicit adapter only for hash accounting.
    torch.manual_seed(int(args.seed) + 9000)
    template = ONLINE.LowRankRiskAdapter(
        dimension=int(ONLINE.RH.D_MODEL), rank=int(args.adapter_rank),
        max_logit_delta=float(args.max_logit_delta),
    ).to(device)
    initial_state = copy.deepcopy(template.state_dict())
    persistent: dict[str, ONLINE.LowRankRiskAdapter] = {}
    memories: dict[str, MEMORY.CausalTypedTrajectoryMemory | None] = {}
    for mode in MODES:
        adapter = ONLINE.LowRankRiskAdapter(
            dimension=int(ONLINE.RH.D_MODEL), rank=int(args.adapter_rank),
            max_logit_delta=float(args.max_logit_delta),
        ).to(device)
        adapter.load_state_dict(initial_state, strict=True)
        persistent[mode] = adapter
        memories[mode] = (
            MEMORY.CausalTypedTrajectoryMemory(
                capacity=args.memory_capacity,
                neighbors=args.memory_neighbors,
                minimum_similarity=args.memory_min_similarity,
                minimum_support_shots=args.memory_min_support_shots,
                agreement_margin=args.memory_agreement_margin,
                maximum_logit_shift=args.memory_max_logit_shift,
                candidate_margin=args.memory_candidate_margin,
                positive_floor=args.memory_positive_floor,
            ) if mode in MEMORY_MODES else None
        )
    initial_hashes = {mode: module_hash(value) for mode, value in persistent.items()}

    prediction_writers = {mode: JsonlWriter(output / f"predictions_{mode}.jsonl") for mode in MODES}
    update_writer = JsonlWriter(output / "updates.jsonl")
    memory_writer = JsonlWriter(output / "memory_receipts.jsonl")
    reveal_writer = JsonlWriter(output / "reveals.jsonl")
    records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    replay: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    protections: dict[str, list[dict[str, Any]]] = {
        mode: list(validation_protections) if mode in NEURAL_MODES else []
        for mode in MODES
    }
    counts: dict[str, dict[str, int]] = {
        mode: {
            "intra_attempts": 0,
            "intra_applied": 0,
            "neural_update_attempts": 0,
            "neural_update_accepted": 0,
            "neural_update_rollbacks": 0,
            "memory_commits": 0,
            "memory_corrections": 0,
        } for mode in MODES
    }
    oracle = ONLINE.OutcomeOracle(test_ids)
    advanced = ONLINE.LazyAdvancedTime(SOURCE.JTEXT_ADVANCED_TIME_PATH)
    static_records: list[dict[str, Any]] = []

    try:
        for stream_index, trace in enumerate(traces):
            shot_risks: dict[str, np.ndarray] = {}
            shot_rows: dict[str, dict[str, Any]] = {}
            for mode in MODES:
                risk, intra_receipts, memory_receipts, fast_hash = score_trace(
                    mode, trace, persistent[mode], memories[mode],
                    threshold=threshold, device=device,
                    intra_interval=args.intra_interval,
                    intra_learning_rate=args.intra_learning_rate,
                    intra_steps=args.intra_steps,
                    intra_max_relative_change=args.intra_max_relative_change,
                    memory_routes=args.memory_routes,
                )
                shot_risks[mode] = risk
                counts[mode]["intra_attempts"] += len(intra_receipts)
                counts[mode]["intra_applied"] += sum(
                    bool(item.get("applied")) for item in intra_receipts
                )
                counts[mode]["memory_corrections"] += sum(
                    bool(item.get("applied")) for item in memory_receipts
                )
                for receipt in intra_receipts:
                    update_writer.append({"mode": mode, "kind": "intra", **receipt})
                for receipt in memory_receipts:
                    memory_writer.append({"mode": mode, **receipt})
                alarm_index = ONLINE.first_alarm(trace.endpoints, risk, threshold=threshold)
                row = {
                    "stream_index": int(stream_index),
                    "shot": int(trace.shot),
                    "score": float(np.max(risk)) if len(risk) else 0.0,
                    "alarm_index": alarm_index,
                    "endpoints": trace.endpoints.tolist(),
                    "start": trace.start,
                    "rate": trace.rate,
                    "threshold": threshold,
                    "queried": False,
                    "fast_adapter_sha256_at_seal": fast_hash,
                    "persistent_adapter_sha256_before": module_hash(persistent[mode]),
                    "intra_update_count": len(intra_receipts),
                    "memory_correction_count": sum(
                        bool(item.get("applied")) for item in memory_receipts
                    ),
                }
                prediction_writers[mode].append(row)
                records[mode].append(row)
                shot_rows[mode] = row
            static_records.append({
                "stream_index": int(stream_index),
                "shot": int(trace.shot),
                "score": float(np.max(trace.base_risk)) if len(trace.base_risk) else 0.0,
                "alarm_index": ONLINE.first_alarm(trace.endpoints, trace.base_risk, threshold=threshold),
                "endpoints": trace.endpoints.tolist(),
                "start": trace.start,
                "rate": trace.rate,
                "threshold": threshold,
            })

            # The query decision is made only after every mode has sealed this
            # current prediction.  One shared oracle reveal feeds all arms.
            oracle.seal(trace.shot)
            query_row = manifest_by_shot[trace.shot]
            selected = bool(query_row["selected"])
            for mode in MODES:
                shot_rows[mode]["queried"] = selected
            if selected:
                oracle.query(trace.shot)
                outcome = oracle.reveal(trace.shot)
                advanced_time = (
                    advanced.lookup_after_reveal(trace.shot)
                    if bool(outcome["is_disrupt"]) else None
                )
                reveal_writer.append({
                    "stream_index": int(stream_index),
                    "shot": int(trace.shot),
                    "is_disrupt": bool(outcome["is_disrupt"]),
                    "down_time_s": outcome["down_time_s"],
                    "advanced_time_ms": advanced_time,
                })
                for mode in MODES:
                    group, lead_ms = outcome_group_for_trace(
                        trace, shot_risks[mode], outcome, threshold=threshold,
                    )
                    shot_rows[mode]["revealed_group"] = group
                    shot_rows[mode]["lead_ms"] = lead_ms
                    target, horizon = target_array(trace, outcome, advanced_time)
                    indices, labels, roles_for_points = select_anchor_indices(
                        trace, target, threshold=threshold, per_type=args.anchor_points,
                    )
                    samples = make_samples(trace, indices, labels)
                    memory_indices, memory_labels, memory_roles = select_error_memory_indices(
                        trace, target, group=group, threshold=threshold,
                        horizon_ms=horizon, outcome=outcome,
                        per_type=args.memory_anchor_points,
                    )
                    if (
                        mode in MEMORY_MODES
                        and memories[mode] is not None
                        and group in {"FP", "FN", "TARDY"}
                        and len(memory_indices)
                    ):
                        features: list[np.ndarray] = []
                        for index in memory_indices:
                            trend = 0.0 if index == 0 else float(
                                trace.base_logits[index] - trace.base_logits[index - 1]
                            )
                            features.append(MEMORY.endpoint_feature(
                                trace.embeddings[index],
                                source_logit=float(trace.base_logits[index]),
                                trend=trend,
                                elapsed_s=float(trace.endpoints[index]) / trace.rate,
                            ))
                        commit = memories[mode].add_after_reveal(
                            trace.shot, features, memory_labels.astype(np.int64),
                            roles=memory_roles,
                        )
                        counts[mode]["memory_commits"] += 1
                        memory_writer.append({
                            "mode": mode,
                            "kind": "post_reveal_commit",
                            "group": group,
                            "horizon_ms": horizon,
                            "anchor_points": int(len(memory_indices)),
                            **commit,
                        })
                    # Correctly classified queried shots become a small
                    # decision guard.  They can block later harmful flips but
                    # are never used before their reveal.
                    if group in {"TP", "TN"} and mode in NEURAL_MODES:
                        protections[mode].append({
                            "z": trace.embeddings,
                            "base": trace.base_logits,
                            "label": bool(group == "TP"),
                            "shot": int(trace.shot),
                            "endpoints": trace.endpoints,
                            "start": float(trace.start),
                            "rate": float(trace.rate),
                            "outcome": outcome,
                            "base_group": group,
                            "source": "queried_TEST220",
                        })
                        # Retain every source-side validation guard plus a
                        # bounded rolling set of revealed TEST protections.
                        protections[mode] = (
                            list(validation_protections)
                            + protections[mode][len(validation_protections):][
                                -args.protection_shots:
                            ]
                        )
                    if mode in NEURAL_MODES:
                        counts[mode]["neural_update_attempts"] += int(
                            group in {"FP", "FN", "TARDY"}
                        )
                        if group in {"FP", "FN", "TARDY"}:
                            replay_samples = sample_replay(
                                replay[mode], count=args.replay_points,
                            )
                            receipt = focused_neural_update(
                                persistent[mode], samples, replay_samples,
                                guard, protections[mode], group=group,
                                threshold=threshold, device=device,
                                learning_rate=args.inter_learning_rate,
                                steps=args.inter_steps,
                                max_relative_change=args.inter_max_relative_change,
                                max_guard_harms=args.max_guard_harms,
                            )
                            counts[mode]["neural_update_accepted"] += int(receipt["updated"])
                            counts[mode]["neural_update_rollbacks"] += int(
                                receipt["reason"] == "guard_or_change_rollback"
                            )
                            update_writer.append({
                                "mode": mode,
                                "kind": "inter_neural",
                                "shot": int(trace.shot),
                                "group": group,
                                "lead_ms": lead_ms,
                                "receipt": receipt,
                                "persistent_adapter_sha256_after": module_hash(persistent[mode]),
                            })
                    replay[mode].append({
                        "shot": int(trace.shot),
                        "group": group,
                        "samples": samples,
                    })
                    replay[mode] = replay[mode][-args.replay_shots:]
            else:
                oracle.discard(trace.shot)
                for mode in MODES:
                    update_writer.append({
                        "mode": mode,
                        "kind": "inter",
                        "shot": int(trace.shot),
                        "updated": False,
                        "reason": "unqueried_no_label_access",
                    })
            if (stream_index + 1) % 20 == 0:
                print(
                    f"[JTEXT-MEMORY] sealed {stream_index + 1}/{len(traces)} "
                    f"queries={sum(int(row['selected']) for row in manifest[:stream_index + 1])}",
                    flush=True,
                )
    finally:
        for writer in prediction_writers.values():
            writer.close()
        update_writer.close(); memory_writer.close(); reveal_writer.close()

    expected_budget = round(len(test_ids) * float(args.budget_fraction))
    if sum(int(row["selected"]) for row in manifest) != expected_budget:
        raise RuntimeError("J-TEXT causal memory exact query budget changed")
    oracle.open_final_evaluation()
    outcomes = ONLINE.read_all_outcomes(test_ids)
    results: dict[str, Any] = {}
    for mode in MODES:
        results[mode] = metric(records[mode], outcomes, threshold)
        static_scores = np.asarray([row["score"] for row in static_records], dtype=np.float64)
        mode_scores = np.asarray([row["score"] for row in records[mode]], dtype=np.float64)
        results[mode]["mean_abs_score_change_vs_static"] = float(
            np.mean(np.abs(mode_scores - static_scores))
        )
        results[mode]["transitions_vs_static"] = {
            "alarm_to_no_alarm": int(np.sum((static_scores >= threshold) & (mode_scores < threshold))),
            "no_alarm_to_alarm": int(np.sum((static_scores < threshold) & (mode_scores >= threshold))),
        }
        results[mode]["initial_adapter_sha256"] = initial_hashes[mode]
        results[mode]["final_adapter_sha256"] = module_hash(persistent[mode])
        results[mode]["updates"] = counts[mode]
        results[mode]["memory_final_size"] = (
            0 if memories[mode] is None else memories[mode].size
        )
    if module_hash(nn.Sequential(encoder, head)) != source_hash_before:
        raise RuntimeError("J-TEXT causal memory experiment changed frozen source")

    summary = {
        "status": "JTEXT_CAUSAL_MEMORY_ABLATION_DIAGNOSTIC",
        "stage": "JTEXT_ADVANCED_FLOOR25_SHARED_MANIFEST_TYPED_MEMORY",
        "seed": int(args.seed),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_model_sha256_before_after": source_hash_before,
        "stream": {
            "role": "TEST220",
            "shots": len(test_ids),
            "order": "ascending_shot_id",
            "claim_status": "TEST_DRIVEN_EXPLORATION",
        },
        "threshold": threshold,
        "threshold_source": "J-TEXT VAL160 embedded in source checkpoint",
        "query_contract": {
            "budget_fraction": float(args.budget_fraction),
            "budget": int(sum(int(row["selected"]) for row in manifest)),
            "cutoff": float(args.query_cutoff),
            "block_size": 20,
            "per_block_quota": int(round(20 * float(args.budget_fraction))),
            "manifest_shared_by_all_arms": True,
            "selection_label_free": True,
            "policy": (
                "validation_error_prior_blend"
                if error_prior is not None
                else "uncertainty + trajectory_conflict + novelty + low_risk_novelty"
            ),
            "validation_error_prior_fit_source": (
                "J-TEXT VAL160" if error_prior is not None else None
            ),
            "validation_error_prior_sha256": (
                None if error_prior is None else error_prior.sha256()
            ),
            "validation_error_prior_weight": (
                None if error_prior is None else float(args.query_error_prior_weight)
            ),
            "test_labels_used_for_policy_fit": False,
        },
        "adapter_contract": {
            "architecture": "frozen rebuilt17 source + low-rank latent/risk/next residual",
            "parameters": int(sum(value.numel() for value in template.parameters())),
            "rank": int(args.adapter_rank),
            "max_logit_delta": float(args.max_logit_delta),
            "intra_modes": sorted(INTRA_MODES),
            "neural_modes": sorted(NEURAL_MODES),
            "intra_scope": "resettable fast adapter; current observation updates next endpoint",
            "inter_scope": "persistent adapter after selected reveal and error group",
        },
        "memory_contract": {
            "schema_version": MEMORY.SCHEMA_VERSION,
            "capacity": int(args.memory_capacity),
            "neighbors": int(args.memory_neighbors),
            "minimum_similarity": float(args.memory_min_similarity),
            "minimum_support_shots": int(args.memory_min_support_shots),
            "only_post_reveal_commits": True,
            "error_support_only": True,
            "feature": "causal embedding direction + source logit + trend + elapsed time",
            "routes": args.memory_routes,
        },
        "results": results,
        "causal_contract": {
            "prediction_order": (
                "current signal prefix -> all mode predictions/seal -> query -> "
                "selected reveal -> memory commit/neural update -> next shot"
            ),
            "test_outcomes_read_before_query": False,
            "unqueried_outcomes_revealed_during_stream": False,
            "advanced_time_read_only_after_reveal": True,
            "jtext_online_state_exported_to_east": False,
        "test_labels_used_for_threshold_or_query_cutoff": False,
        "validation_labels_used_for_query_prior": bool(error_prior is not None),
    },
        "hashes": {
            "query_manifest": sha256_file(output / "query_manifest.jsonl"),
            "updates": sha256_file(output / "updates.jsonl"),
            "memory_receipts": sha256_file(output / "memory_receipts.jsonl"),
            "reveals": sha256_file(output / "reveals.jsonl"),
        },
        "claim_boundary": (
            "TEST220 was used for method-development diagnostics.  The shared "
            "manifest and post-reveal ledgers establish causal attribution within "
            "this replay, but not independent future-stream confirmation."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {
        "seed": int(args.seed),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "budget_fraction": float(args.budget_fraction),
        "query_cutoff": float(args.query_cutoff),
        "query_policy": args.query_policy,
        "query_error_l2": float(args.query_error_l2),
        "query_error_prior_weight": float(args.query_error_prior_weight),
        "adapter_rank": int(args.adapter_rank),
        "max_logit_delta": float(args.max_logit_delta),
        "intra_interval": int(args.intra_interval),
        "intra_learning_rate": float(args.intra_learning_rate),
        "intra_steps": int(args.intra_steps),
        "intra_max_relative_change": float(args.intra_max_relative_change),
        "inter_learning_rate": float(args.inter_learning_rate),
        "inter_steps": int(args.inter_steps),
        "inter_max_relative_change": float(args.inter_max_relative_change),
        "max_guard_harms": int(args.max_guard_harms),
        "anchor_points": int(args.anchor_points),
        "replay_points": int(args.replay_points),
        "replay_shots": int(args.replay_shots),
        "protection_shots": int(args.protection_shots),
        "memory_capacity": int(args.memory_capacity),
        "memory_neighbors": int(args.memory_neighbors),
        "memory_min_similarity": float(args.memory_min_similarity),
        "memory_min_support_shots": int(args.memory_min_support_shots),
        "memory_agreement_margin": float(args.memory_agreement_margin),
        "memory_max_logit_shift": float(args.memory_max_logit_shift),
        "memory_candidate_margin": float(args.memory_candidate_margin),
        "memory_positive_floor": float(args.memory_positive_floor),
        "memory_anchor_points": int(args.memory_anchor_points),
        "memory_routes": args.memory_routes,
    })
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    parser.add_argument("--query-cutoff", type=float, default=0.50)
    parser.add_argument(
        "--query-policy",
        choices=("causal_components", "validation_error_prior"),
        default="causal_components",
    )
    parser.add_argument("--query-error-l2", type=float, default=0.10)
    parser.add_argument("--query-error-prior-weight", type=float, default=0.75)
    parser.add_argument("--adapter-rank", type=int, default=4)
    parser.add_argument("--max-logit-delta", type=float, default=4.0)
    parser.add_argument("--intra-interval", type=int, default=8)
    parser.add_argument("--intra-learning-rate", type=float, default=1e-4)
    parser.add_argument("--intra-steps", type=int, default=1)
    parser.add_argument("--intra-max-relative-change", type=float, default=1e-3)
    parser.add_argument("--inter-learning-rate", type=float, default=3e-3)
    parser.add_argument("--inter-steps", type=int, default=4)
    parser.add_argument("--inter-max-relative-change", type=float, default=0.10)
    parser.add_argument("--max-guard-harms", type=int, default=0)
    parser.add_argument("--anchor-points", type=int, default=8)
    parser.add_argument("--replay-points", type=int, default=16)
    parser.add_argument("--replay-shots", type=int, default=64)
    parser.add_argument("--protection-shots", type=int, default=16)
    parser.add_argument("--memory-capacity", type=int, default=512)
    parser.add_argument("--memory-neighbors", type=int, default=3)
    parser.add_argument("--memory-min-similarity", type=float, default=0.90)
    parser.add_argument("--memory-min-support-shots", type=int, default=2)
    parser.add_argument("--memory-agreement-margin", type=float, default=0.10)
    parser.add_argument("--memory-max-logit-shift", type=float, default=4.5)
    parser.add_argument("--memory-candidate-margin", type=float, default=0.18)
    parser.add_argument("--memory-positive-floor", type=float, default=0.01)
    parser.add_argument("--memory-anchor-points", type=int, default=8)
    parser.add_argument(
        "--memory-routes", choices=("both", "negative_only", "positive_only"),
        default="both",
        help="which revealed error-memory label route may change risk at inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
