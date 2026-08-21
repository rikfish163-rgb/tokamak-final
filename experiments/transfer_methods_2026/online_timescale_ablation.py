"""Pure contracts for causal online timescale ablations and order diagnostics.

This module deliberately contains no HDF5 access.  It separates protocol
decisions that must be testable without touching labels from the heavy EAST
runner.  In particular, a reordered stream is always marked diagnostic and an
oracle curriculum cannot be constructed without an explicit permission flag.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TimescaleContract:
    name: str
    enable_intra: bool
    enable_query: bool
    enable_reveal: bool
    enable_inter: bool


_TIMESCALE_CONTRACTS = {
    "intra_only": TimescaleContract(
        name="intra_only", enable_intra=True, enable_query=False,
        enable_reveal=False, enable_inter=False,
    ),
    "inter_only": TimescaleContract(
        name="inter_only", enable_intra=False, enable_query=True,
        enable_reveal=True, enable_inter=True,
    ),
    "intra_inter": TimescaleContract(
        name="intra_inter", enable_intra=True, enable_query=True,
        enable_reveal=True, enable_inter=True,
    ),
}


def timescale_contract(name: str) -> TimescaleContract:
    try:
        return _TIMESCALE_CONTRACTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown timescale mode: {name!r}") from exc


def sealed_query_score(
    mode: str, *, features: Mapping[str, float], base_score: float,
) -> float:
    """Compute a query value from values available before outcome reveal."""
    base = float(base_score)
    if not np.isfinite(base):
        raise ValueError("base query score must be finite")
    if mode == "current":
        value = base
    elif mode == "gradient_blend":
        required = ("expected_gradient", "decision_changeability")
        if any(name not in features for name in required):
            raise ValueError("gradient blend requires sealed gradient/changeability")
        gradient = float(features["expected_gradient"])
        changeability = float(features["decision_changeability"])
        if not np.isfinite([gradient, changeability]).all():
            raise ValueError("gradient blend inputs must be finite")
        value = 0.65 * gradient + 0.20 * changeability + 0.15 * base
    else:
        raise ValueError(f"unknown query score mode: {mode!r}")
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class RosterResolution:
    roster: list[int]
    is_reordered: bool
    claim_status: str
    order_source: str


def _read_shot_manifest(path: Path) -> list[int]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
        return [
            int(row["shot"] if isinstance(row, Mapping) else row)
            for row in rows
        ]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            payload = payload.get("roster", payload.get("shots"))
        if not isinstance(payload, list):
            raise ValueError("shot manifest JSON must contain a list")
        return [
            int(row["shot"] if isinstance(row, Mapping) else row)
            for row in payload
        ]
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def resolve_online_roster(
    base_roster: Sequence[int], *, order_manifest: Path | None,
    allow_diagnostic_reorder: bool,
) -> RosterResolution:
    base = [int(shot) for shot in base_roster]
    if len(base) != len(set(base)):
        raise ValueError("base roster must be unique")
    if order_manifest is None:
        return RosterResolution(
            roster=base, is_reordered=False,
            claim_status="CHRONOLOGICAL_ONLINE_DIAGNOSTIC",
            order_source="sealed_parent_roster",
        )
    requested = _read_shot_manifest(Path(order_manifest))
    if len(requested) != len(base) or Counter(requested) != Counter(base):
        raise ValueError("order manifest must be an exact permutation of the sealed roster")
    reordered = requested != base
    if reordered and not bool(allow_diagnostic_reorder):
        raise ValueError(
            "non-chronological order requires explicit diagnostic reorder permission"
        )
    return RosterResolution(
        roster=requested,
        is_reordered=reordered,
        claim_status=(
            "ORDER_ORACLE_DIAGNOSTIC" if reordered
            else "CHRONOLOGICAL_ONLINE_DIAGNOSTIC"
        ),
        order_source=str(Path(order_manifest).expanduser().resolve()),
    )


def read_query_manifest(path: Path, roster: Sequence[int]) -> list[int]:
    queries = _read_shot_manifest(Path(path))
    if len(queries) != len(set(queries)):
        raise ValueError("fixed query manifest must be unique")
    unknown = sorted(set(queries).difference(int(shot) for shot in roster))
    if unknown:
        raise ValueError(f"fixed query manifest contains shots outside roster: {unknown[:5]}")
    return queries


def query_manifest_uses_label_oracle(path: Path) -> bool:
    """Return only an explicit oracle declaration; absence is non-oracle."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        return False
    payload = json.loads(resolved.read_text())
    return bool(
        payload.get("label_oracle_used", False)
        if isinstance(payload, Mapping) else False
    )


_CURRENT_OUTCOME_FIELDS = {
    "revealed_is_disrupt", "revealed_down_time_s", "is_disrupt",
    "down_time_s", "static_group", "deployed_group", "learning_group",
    "group",
}


def fixed_query_manifest_from_decisions(
    rows: Sequence[Mapping[str, Any]], *, roster: Sequence[int],
) -> dict[str, Any]:
    """Reduce a sealed label-free decision ledger to an immutable query list."""
    ordered = [int(shot) for shot in roster]
    if len(ordered) != len(set(ordered)):
        raise ValueError("query manifest roster must be unique")
    records = [dict(row) for row in rows]
    if len(records) != len(ordered):
        raise ValueError("query decisions must cover the complete roster")
    selected_shots: list[int] = []
    selected_positions: list[int] = []
    for index, (shot, row) in enumerate(zip(ordered, records)):
        forbidden = sorted(_CURRENT_OUTCOME_FIELDS.intersection(row))
        if forbidden:
            raise ValueError(
                f"query decision contains current outcome field: {forbidden[0]}"
            )
        if int(row.get("stream_index", -1)) != index or int(
            row.get("shot", -1)
        ) != shot:
            raise ValueError("query decisions do not follow sealed stream order")
        if not isinstance(row.get("selected"), bool):
            raise ValueError("query decision selected flag must be boolean")
        if bool(row["selected"]):
            selected_shots.append(shot)
            selected_positions.append(index)
    return {
        "kind": "sealed_chronological_policy_queries",
        "label_oracle_used": False,
        "shots": selected_shots,
        "positions_zero_based": selected_positions,
        "source_row_count": len(records),
    }


def build_causal_random_query_manifest(
    *, roster: Sequence[int], reference_manifest: Mapping[str, Any],
    block_size: int, query_rate: float, token_capacity: float,
    seed: int, replicate: int,
) -> dict[str, Any]:
    """Draw a label-free random schedule with the reference per-block budget.

    Query identities are sampled from stream positions, not outcomes.  Rejection
    sampling is used only to enforce the same rolling token constraint consumed
    by :class:`CausalFixedQuerySelector`; it never inspects labels or scores.
    """
    ordered = [int(shot) for shot in roster]
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("random-query roster must be non-empty and unique")
    if int(block_size) <= 0:
        raise ValueError("random-query block size must be positive")
    if not 0.0 < float(query_rate) <= 1.0 or float(token_capacity) < 1.0:
        raise ValueError("invalid random-query rolling token budget")
    if int(replicate) < 0:
        raise ValueError("random-query replicate must be non-negative")
    if bool(reference_manifest.get("label_oracle_used", False)):
        raise PermissionError("random-query reference must be label-free")

    position_by_shot = {shot: index for index, shot in enumerate(ordered)}
    reference_shots = [int(value) for value in reference_manifest.get("shots", [])]
    if len(reference_shots) != len(set(reference_shots)) or any(
        shot not in position_by_shot for shot in reference_shots
    ):
        raise ValueError("reference queries must be a unique roster subset")
    declared_positions = reference_manifest.get("positions_zero_based")
    positions = [position_by_shot[shot] for shot in reference_shots]
    if declared_positions is not None and [int(value) for value in declared_positions] != positions:
        raise ValueError("reference query shots and positions disagree")

    counts: list[int] = []
    blocks: list[list[int]] = []
    reference_set = set(positions)
    for start in range(0, len(ordered), int(block_size)):
        block = list(range(start, min(start + int(block_size), len(ordered))))
        blocks.append(block)
        counts.append(sum(index in reference_set for index in block))
    if any(count > len(block) for count, block in zip(counts, blocks)):
        raise RuntimeError("reference block query count is impossible")

    generator = np.random.default_rng(
        int(seed) * 1_000_003 + int(replicate) * 97_409 + 7_919
    )
    chosen_positions: list[int] | None = None
    for _ in range(100_000):
        candidate: list[int] = []
        for block, count in zip(blocks, counts):
            if count:
                candidate.extend(
                    int(value) for value in generator.choice(
                        block, size=count, replace=False,
                    )
                )
        candidate.sort()
        candidate_set = set(candidate)
        tokens = 0.0
        feasible = True
        for index in range(len(ordered)):
            tokens = min(float(token_capacity), tokens + float(query_rate))
            if index in candidate_set:
                if tokens + 1e-12 < 1.0:
                    feasible = False
                    break
                tokens -= 1.0
        if feasible:
            chosen_positions = candidate
            break
    if chosen_positions is None:
        raise RuntimeError("could not sample a causal block-matched random schedule")

    return {
        "kind": "causal_block_matched_random_queries",
        "label_oracle_used": False,
        "shots": [ordered[index] for index in chosen_positions],
        "positions_zero_based": chosen_positions,
        "source_row_count": len(ordered),
        "reference_query_count": len(reference_shots),
        "reference_block_counts": counts,
        "block_size": int(block_size),
        "query_rate": float(query_rate),
        "token_capacity": float(token_capacity),
        "random_seed": int(seed),
        "replicate": int(replicate),
    }


class CausalFixedQuerySelector:
    """Replay a pre-sealed query schedule while enforcing causal token cost."""

    def __init__(
        self, *, roster: Sequence[int], query_shots: Sequence[int],
        query_rate: float, capacity: float, block_size: int, quota: int,
        query_cost: float = 0.0, enforce_block_quota: bool = True,
    ) -> None:
        self.roster = [int(shot) for shot in roster]
        if len(self.roster) != len(set(self.roster)):
            raise ValueError("fixed-query roster must be unique")
        self.query_shots = [int(shot) for shot in query_shots]
        if len(self.query_shots) != len(set(self.query_shots)):
            raise ValueError("fixed query shots must be unique")
        if not set(self.query_shots).issubset(self.roster):
            raise ValueError("fixed query shots must be a roster subset")
        if not 0.0 < float(query_rate) <= 1.0 or float(capacity) < 1.0:
            raise ValueError("invalid causal fixed-query token budget")
        if int(block_size) <= 0 or not 0 < int(quota) <= int(block_size):
            raise ValueError("invalid causal fixed-query block budget")
        if not 0.0 <= float(query_cost) <= 1.0:
            raise ValueError("query cost must lie in [0, 1]")
        self.query_rate = float(query_rate)
        self.capacity = float(capacity)
        self.block_size = int(block_size)
        self.quota = int(quota)
        self.query_cost = float(query_cost)
        self.enforce_block_quota = bool(enforce_block_quota)
        self._query_set = set(self.query_shots)
        self._tokens = 0.0
        self._next_index = 0
        self._block = -1
        self._used = 0
        self.total_selected = 0

    def decide(
        self, *, score: float, stream_index: int, shot_id: int,
    ) -> dict[str, Any]:
        index = int(stream_index)
        shot = int(shot_id)
        if index != self._next_index or self.roster[index] != shot:
            raise RuntimeError("fixed query decisions must follow the sealed roster")
        if not np.isfinite(float(score)):
            raise ValueError("query score must be finite")
        block = index // self.block_size
        if block != self._block:
            self._block = block
            self._used = 0
        self._tokens = min(self.capacity, self._tokens + self.query_rate)
        selected = shot in self._query_set
        if selected and self._tokens + 1e-12 < 1.0:
            raise RuntimeError(
                f"shot {shot}: insufficient causal query tokens for fixed manifest"
            )
        if selected and self.enforce_block_quota and self._used >= self.quota:
            raise RuntimeError(f"shot {shot}: fixed query manifest exceeds block quota")
        if selected:
            self._tokens -= 1.0
            self._used += 1
            self.total_selected += 1
        self._next_index += 1
        value = float(np.clip(float(score), 0.0, 1.0))
        return {
            "selected": bool(selected),
            "route": "fixed_manifest" if selected else None,
            "score": value,
            "cutoff": None,
            "query_cost": self.query_cost,
            "enforce_block_quota": self.enforce_block_quota,
            "budget_semantics": (
                "rolling_token_plus_hard_block"
                if self.enforce_block_quota else "rolling_token_only"
            ),
            "net_value": value - self.query_cost,
            "block": block,
            "position": index % self.block_size,
            "history_count": 0,
            "exploration_position": None,
            "tokens_after": float(self._tokens),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "causal_fixed_manifest",
            "roster": list(self.roster),
            "query_shots": list(self.query_shots),
            "query_rate": self.query_rate,
            "capacity": self.capacity,
            "block_size": self.block_size,
            "quota": self.quota,
            "query_cost": self.query_cost,
            "enforce_block_quota": self.enforce_block_quota,
            "budget_semantics": (
                "rolling_token_plus_hard_block"
                if self.enforce_block_quota else "rolling_token_only"
            ),
            "tokens": self._tokens,
            "next_index": self._next_index,
            "block": self._block,
            "used": self._used,
            "total_selected": self.total_selected,
        }


@dataclass(frozen=True)
class OracleCurriculum:
    roster: list[int]
    query_shots: list[int]
    query_positions: list[int]
    label_oracle_used: bool
    method: str = "error_prototype_first"


@dataclass(frozen=True)
class FixedQueryReorder:
    """A reordered stream with an unchanged query set and query slots."""

    roster: list[int]
    query_shots: list[int]
    query_positions: list[int]
    label_oracle_used: bool
    method: str


def _normalized_phenotypes(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vectors = [np.asarray(row["phenotype"], dtype=np.float64).reshape(-1) for row in records]
    if not vectors or any(vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("all curriculum records need equal-length phenotype vectors")
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("curriculum phenotypes must be finite")
    median = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (matrix - median) / scale


def _select_error_prototypes(
    records: Sequence[Mapping[str, Any]], matrix: np.ndarray, count: int,
) -> list[int]:
    error_indices = [
        index for index, row in enumerate(records)
        if str(row["static_group"]).upper() in {"FP", "FN", "TARDY"}
    ]
    if not error_indices or count <= 0:
        return []
    severity = {"FP": 1.0, "FN": 1.0, "TARDY": 0.85}
    selected: list[int] = []
    remaining = set(error_indices)
    while remaining and len(selected) < count:
        def objective(index: int) -> tuple[float, float, int]:
            group_score = severity[str(records[index]["static_group"]).upper()]
            diversity = (
                min(float(np.linalg.norm(matrix[index] - matrix[other])) for other in selected)
                if selected else float(np.linalg.norm(matrix[index]))
            )
            return group_score, diversity, -int(records[index]["shot"])

        chosen = max(remaining, key=objective)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def build_oracle_error_curriculum(
    records: Sequence[Mapping[str, Any]], *, query_rate: float,
    allow_label_oracle: bool,
) -> OracleCurriculum:
    """Place representative hindsight errors at causally affordable query slots.

    The result is an explicit upper-bound diagnostic.  Static error groups are
    labels, so constructing this order is forbidden unless the caller opts in.
    Predictions in a downstream replay must still be sealed before each reveal.
    """
    if not bool(allow_label_oracle):
        raise PermissionError("oracle curriculum requires explicit label oracle permission")
    if not 0.0 < float(query_rate) <= 1.0:
        raise ValueError("query rate must lie in (0, 1]")
    rows = [dict(row) for row in records]
    shots = [int(row["shot"]) for row in rows]
    if len(shots) != len(set(shots)):
        raise ValueError("curriculum shots must be unique")
    matrix = _normalized_phenotypes(rows)
    budget = int(math.floor(len(rows) * float(query_rate) + 1e-12))
    prototypes = _select_error_prototypes(rows, matrix, budget)
    if not prototypes:
        return OracleCurriculum(
            roster=shots, query_shots=[], query_positions=[],
            label_oracle_used=True,
        )
    query_positions = [
        int(math.ceil((rank + 1) / float(query_rate)) - 1)
        for rank in range(len(prototypes))
    ]
    query_positions = [min(position, len(rows) - 1) for position in query_positions]
    if len(query_positions) != len(set(query_positions)):
        raise RuntimeError("query rate produced duplicate causal token slots")

    order: list[int | None] = [None] * len(rows)
    for position, record_index in zip(query_positions, prototypes):
        order[position] = record_index
    remaining = set(range(len(rows))).difference(prototypes)
    last_prototype: int | None = None
    prototype_by_position = dict(zip(query_positions, prototypes))
    for position in range(len(order)):
        if position in prototype_by_position:
            last_prototype = prototype_by_position[position]
            continue
        if not remaining:
            break
        if last_prototype is None:
            # Before the first affordable query, spend filler slots on samples
            # least similar to all planned error prototypes.
            chosen = max(
                remaining,
                key=lambda index: (
                    min(float(np.linalg.norm(matrix[index] - matrix[p])) for p in prototypes),
                    -shots[index],
                ),
            )
        else:
            # After a prototype is revealed, place its nearest phenotypes next
            # so the replay measures the largest plausible same-error benefit.
            chosen = min(
                remaining,
                key=lambda index: (
                    float(np.linalg.norm(matrix[index] - matrix[last_prototype])),
                    shots[index],
                ),
            )
        order[position] = chosen
        remaining.remove(chosen)
    if any(index is None for index in order):
        raise RuntimeError("oracle curriculum did not place every shot")
    roster = [shots[int(index)] for index in order]
    query_shots = [shots[index] for index in prototypes]
    return OracleCurriculum(
        roster=roster, query_shots=query_shots,
        query_positions=query_positions, label_oracle_used=True,
    )


def build_chronological_error_queries(
    records: Sequence[Mapping[str, Any]], *, query_rate: float,
    allow_label_oracle: bool, token_capacity: float = 12.0,
) -> OracleCurriculum:
    """Select hindsight errors without changing the chronological shot order.

    This isolates a query-selection ceiling from an order ceiling.  A label is
    selected only when the same rolling token bucket used online can afford it;
    errors that arrived before a token existed cannot be queried retroactively.
    """
    if not bool(allow_label_oracle):
        raise PermissionError("chronological error queries require label oracle permission")
    if not 0.0 < float(query_rate) <= 1.0 or float(token_capacity) < 1.0:
        raise ValueError("invalid chronological oracle rolling budget")
    rows = [dict(row) for row in records]
    shots = [int(row["shot"]) for row in rows]
    if not rows or len(shots) != len(set(shots)):
        raise ValueError("chronological oracle records must contain unique shots")
    budget = int(math.floor(len(rows) * float(query_rate) + 1e-12))
    tokens = 0.0
    query_shots: list[int] = []
    query_positions: list[int] = []
    for position, row in enumerate(rows):
        tokens = min(float(token_capacity), tokens + float(query_rate))
        is_error = str(row["static_group"]).upper() in {"FP", "FN", "TARDY"}
        if is_error and tokens + 1e-12 >= 1.0 and len(query_shots) < budget:
            query_shots.append(shots[position])
            query_positions.append(position)
            tokens -= 1.0
    return OracleCurriculum(
        roster=shots,
        query_shots=query_shots,
        query_positions=query_positions,
        label_oracle_used=True,
        method="chronological_earliest_affordable_static_errors",
    )


def _error_family(group: str) -> str:
    normalized = str(group).upper()
    if normalized == "FP":
        return "FP"
    if normalized in {"FN", "TARDY"}:
        return "MISS"
    return "CORRECT"


def build_fixed_query_reorder(
    records: Sequence[Mapping[str, Any]], *, query_shots: Sequence[int],
    query_positions: Sequence[int], strategy: str, random_seed: int,
    allow_label_oracle: bool,
) -> FixedQueryReorder:
    """Reorder a stream without changing which labels arrive or when.

    ``random_nonquery`` is a label-free order-sensitivity control: queried
    shots remain in their original slots and only non-query shots are shuffled.
    ``hindsight_error_neighbors`` is an explicit diagnostic oracle.  It keeps
    the same query shots and slots, but places future static errors with a
    similar sealed phenotype after a matching queried error.  Thus any gain is
    an upper bound on *when* an inter-shot update could help, not a prospective
    online score.
    """
    rows = [dict(row) for row in records]
    shots = [int(row["shot"]) for row in rows]
    if not rows or len(shots) != len(set(shots)):
        raise ValueError("fixed-query reorder records must contain unique shots")
    queries = [int(shot) for shot in query_shots]
    positions = [int(position) for position in query_positions]
    if len(queries) != len(set(queries)) or len(queries) != len(positions):
        raise ValueError("query shots and positions must be unique paired values")
    if len(positions) != len(set(positions)) or any(
        position < 0 or position >= len(rows) for position in positions
    ):
        raise ValueError("query positions must be unique valid stream indices")
    if not set(queries).issubset(shots):
        raise ValueError("query shots must be a subset of curriculum records")
    if positions != sorted(positions):
        raise ValueError("query positions must be in stream order")
    if strategy not in {"random_nonquery", "hindsight_error_neighbors"}:
        raise ValueError(f"unknown fixed-query reorder strategy: {strategy!r}")
    if strategy == "hindsight_error_neighbors" and not bool(allow_label_oracle):
        raise PermissionError("hindsight fixed-query order requires label oracle permission")

    index_by_shot = {shot: index for index, shot in enumerate(shots)}
    query_indices = [index_by_shot[shot] for shot in queries]
    order: list[int | None] = [None] * len(rows)
    for position, record_index in zip(positions, query_indices):
        order[position] = record_index
    remaining = set(range(len(rows))).difference(query_indices)

    if strategy == "random_nonquery":
        rng = np.random.default_rng(int(random_seed))
        shuffled = rng.permutation(np.asarray(sorted(remaining), dtype=np.int64)).tolist()
        cursor = 0
        for position in range(len(order)):
            if order[position] is None:
                order[position] = int(shuffled[cursor])
                cursor += 1
    else:
        matrix = _normalized_phenotypes(rows)
        query_at_position = dict(zip(positions, query_indices))
        last_query: int | None = None
        for position in range(len(order)):
            if position in query_at_position:
                last_query = query_at_position[position]
                continue
            if not remaining:
                break
            if last_query is None:
                # Preserve potentially rescuable errors until after the first
                # reveal; among equally correct fillers, use the least similar.
                def prefix_objective(index: int) -> tuple[int, float, int]:
                    correct = _error_family(rows[index]["static_group"]) == "CORRECT"
                    nearest = min(
                        float(np.linalg.norm(matrix[index] - matrix[query]))
                        for query in query_indices
                    ) if query_indices else 0.0
                    return int(correct), nearest, -shots[index]

                chosen = max(remaining, key=prefix_objective)
            else:
                revealed_family = _error_family(rows[last_query]["static_group"])

                def future_objective(index: int) -> tuple[int, int, float, int]:
                    candidate_family = _error_family(rows[index]["static_group"])
                    family_mismatch = int(candidate_family != revealed_family)
                    reserve_other_error = int(
                        revealed_family != "CORRECT"
                        and candidate_family not in {revealed_family, "CORRECT"}
                    )
                    distance = float(np.linalg.norm(matrix[index] - matrix[last_query]))
                    return family_mismatch, reserve_other_error, distance, shots[index]

                chosen = min(remaining, key=future_objective)
            order[position] = chosen
            remaining.remove(chosen)
    if any(index is None for index in order):
        raise RuntimeError("fixed-query reorder did not place every shot")
    roster = [shots[int(index)] for index in order]
    if [roster[position] for position in positions] != queries:
        raise RuntimeError("fixed-query reorder changed a sealed query slot")
    return FixedQueryReorder(
        roster=roster,
        query_shots=queries,
        query_positions=positions,
        label_oracle_used=(strategy == "hindsight_error_neighbors"),
        method=strategy,
    )


def operational_group(
    prediction: Mapping[str, Any], outcome: Mapping[str, Any], *,
    minimum_lead_ms: float = 30.0,
) -> str:
    disruptive = bool(outcome["is_disrupt"])
    alarmed = bool(prediction.get("alarmed", False))
    if not disruptive:
        return "FP" if alarmed else "TN"
    if not alarmed:
        return "FN"
    alarm_time = prediction.get("alarm_time_s")
    down_time = outcome.get("down_time_s")
    if alarm_time is None or down_time is None:
        raise ValueError("disruptive alarm classification requires both times")
    lead_ms = (float(down_time) - float(alarm_time)) * 1000.0
    # Alarm times originate from integer sample indices.  Permit only a tiny
    # numerical tolerance so an exact 30-sample lead is classified identically
    # before and after conversion through floating-point seconds.
    return (
        "TP"
        if lead_ms >= float(minimum_lead_ms) - 1e-9
        else "TARDY"
    )


def rescue_transition_report(
    static_groups: Mapping[int, str], online_groups: Mapping[int, str], *,
    queried_shots: Sequence[int], stream_order: Sequence[int],
) -> dict[str, Any]:
    order = [int(shot) for shot in stream_order]
    if set(order) != set(static_groups) or set(order) != set(online_groups):
        raise ValueError("transition report requires identical shot sets")
    position = {shot: index for index, shot in enumerate(order)}
    queried = [int(shot) for shot in queried_shots]
    if not set(queried).issubset(position):
        raise ValueError("queried shots must belong to stream order")

    counts = {
        "repaired_fp_to_tn": 0,
        "rescued_fn_to_tp": 0,
        "rescued_tardy_to_tp": 0,
        "harmed_tp_to_error": 0,
        "harmed_tn_to_fp": 0,
    }
    transitions: dict[str, int] = {}
    rescue_shots: list[int] = []
    harm_shots: list[int] = []
    for shot in order:
        before = str(static_groups[shot]).upper()
        after = str(online_groups[shot]).upper()
        transitions[f"{before}->{after}"] = transitions.get(f"{before}->{after}", 0) + 1
        if before == "FP" and after == "TN":
            counts["repaired_fp_to_tn"] += 1
            rescue_shots.append(shot)
        elif before == "FN" and after == "TP":
            counts["rescued_fn_to_tp"] += 1
            rescue_shots.append(shot)
        elif before == "TARDY" and after == "TP":
            counts["rescued_tardy_to_tp"] += 1
            rescue_shots.append(shot)
        elif before == "TP" and after != "TP":
            counts["harmed_tp_to_error"] += 1
            harm_shots.append(shot)
        elif before == "TN" and after == "FP":
            counts["harmed_tn_to_fp"] += 1
            harm_shots.append(shot)

    gross_rescues = len(rescue_shots)
    gross_harms = len(harm_shots)
    net_rescues = gross_rescues - gross_harms
    query_count = len(queried)
    first_prior_query = np.full(len(order), False, dtype=bool)
    seen_query = False
    query_set = set(queried)
    for index, shot in enumerate(order):
        first_prior_query[index] = seen_query
        seen_query = seen_query or shot in query_set
    future_rescues = sum(first_prior_query[position[shot]] for shot in rescue_shots)
    future_harms = sum(first_prior_query[position[shot]] for shot in harm_shots)

    static_error = np.asarray([
        str(static_groups[shot]).upper() not in {"TP", "TN"} for shot in order
    ], dtype=np.float64)
    online_error = np.asarray([
        str(online_groups[shot]).upper() not in {"TP", "TN"} for shot in order
    ], dtype=np.float64)
    static_cumulative_area = float(np.cumsum(static_error).sum())
    online_cumulative_area = float(np.cumsum(online_error).sum())
    return {
        **counts,
        "gross_rescues": gross_rescues,
        "gross_harms": gross_harms,
        "net_rescues": net_rescues,
        "query_count": query_count,
        "query_rate": query_count / max(len(order), 1),
        "net_rescues_per_query": net_rescues / query_count if query_count else 0.0,
        "future_gross_rescues": int(future_rescues),
        "future_gross_harms": int(future_harms),
        "future_net_rescues": int(future_rescues - future_harms),
        "future_net_rescues_per_query": (
            (future_rescues - future_harms) / query_count if query_count else 0.0
        ),
        "static_cumulative_error_area": static_cumulative_area,
        "online_cumulative_error_area": online_cumulative_area,
        "cumulative_error_area_reduction": (
            static_cumulative_area - online_cumulative_area
        ),
        "rescue_shots": rescue_shots,
        "harm_shots": harm_shots,
        "transitions": dict(sorted(transitions.items())),
    }


__all__ = [
    "CausalFixedQuerySelector", "FixedQueryReorder", "OracleCurriculum",
    "RosterResolution", "TimescaleContract", "build_chronological_error_queries",
    "build_causal_random_query_manifest",
    "build_fixed_query_reorder", "build_oracle_error_curriculum",
    "operational_group", "query_manifest_uses_label_oracle",
    "read_query_manifest", "resolve_online_roster", "rescue_transition_report",
    "sealed_query_score", "timescale_contract",
]
