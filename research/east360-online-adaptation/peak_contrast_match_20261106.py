"""Deterministic offline episode matching using only supplied descriptors.

The caller is responsible for selecting the episodes, including restricting
safe episodes to those with a complete next block. No outcome, waveform, score,
or future-block data is read here. Indices in the result refer to the original
input sequences. Shot uniqueness is enforced independently on the two sides.

An episode is eligible when at least two of the channel families [0:3], [3:7]
and [7:11] have a valid channel and its anchor is at least ``high_floor`` when
one is supplied. An edge requires identical family support, Hamming distance
at most 3, and inclusive age, anchor and trend calipers of 64, 0.25 and 0.35.
Its cost is the sum of squared differences divided by their respective
calipers, including (Hamming / 3)**2. For each shot pair the least-cost episode
pair is retained; ties use danger maturity, safe maturity, then the remaining
danger and safe descriptor values.
Shot IDs, maturities and remaining descriptors are sorted before assignment, so
input order cannot change the selected descriptor pairs. Perfectly identical
duplicate rows are interchangeable; their original indices may differ after a
permutation. Matching first maximizes cardinality and then minimizes total cost.

Diagnostics count eligible, caliper-connected and paired episodes and shots on
each side. Distance and danger-minus-safe signed delta summaries describe only
the final pairs; empty summaries contain None rather than non-finite numbers.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import NamedTuple
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


_AGE_CALIPER = 64.0
_ANCHOR_CALIPER = 0.25
_TREND_CALIPER = 0.35
_HAMMING_CALIPER = 3
_BIT_COUNTS = np.array(
    [number.bit_count() for number in range(1 << 11)], dtype=np.uint8
)


class _Episode(NamedTuple):
    index: int
    shot: int
    maturity: int
    age_ms: float
    anchor: float
    trend: float
    mask: int
    families: int


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _episodes(rows, high_floor):
    episodes = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("each episode must be a mapping")
        try:
            shot = _integer(row["shot"], "shot")
            maturity = _integer(row["maturity"], "maturity")
            age_ms = _finite_number(row["age_ms"], "age_ms")
            anchor = _finite_number(row["anchor"], "anchor")
            trend = _finite_number(row["trend"], "trend")
            valid = row["valid"]
        except KeyError as error:
            raise ValueError(f"missing episode field: {error.args[0]}") from None
        if (
            not isinstance(valid, np.ndarray)
            or valid.shape != (11,)
            or valid.dtype != np.bool_
        ):
            raise ValueError("valid must be a bool ndarray with shape (11,)")
        mask = sum(1 << bit for bit in range(11) if valid[bit])
        families = int(bool(mask & 0b00000000111))
        families |= int(bool(mask & 0b00001111000)) << 1
        families |= int(bool(mask & 0b11110000000)) << 2
        if families.bit_count() < 2 or (high_floor is not None and anchor < high_floor):
            continue
        episodes.append(
            _Episode(index, shot, maturity, age_ms, anchor, trend, mask, families)
        )
    return sorted(
        episodes,
        key=lambda ep: (ep.shot, ep.maturity, ep.age_ms, ep.anchor, ep.trend, ep.mask),
    )


def _counts(episodes, selected=None):
    if selected is not None:
        episodes = [episode for episode, keep in zip(episodes, selected) if keep]
    return {
        "shots": len({episode.shot for episode in episodes}),
        "episodes": len(episodes),
    }


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _diagnostics(danger, safe, danger_matched, safe_matched, selected):
    distances = {"cost": [], "normalized_distance": [], "hamming": []}
    deltas = {"age_ms": [], "anchor": [], "trend": [], "valid_count": []}
    channel_deltas = [[] for _ in range(11)]
    for d, s, cost in selected:
        hamming = (d.mask ^ s.mask).bit_count()
        distances["cost"].append(cost)
        distances["normalized_distance"].append(math.sqrt(cost))
        distances["hamming"].append(hamming)
        deltas["age_ms"].append(d.age_ms - s.age_ms)
        deltas["anchor"].append(d.anchor - s.anchor)
        deltas["trend"].append(d.trend - s.trend)
        deltas["valid_count"].append(d.mask.bit_count() - s.mask.bit_count())
        for bit, values in enumerate(channel_deltas):
            values.append(((d.mask >> bit) & 1) - ((s.mask >> bit) & 1))
    signed = {name: _summary(values) for name, values in deltas.items()}
    signed["valid_by_channel"] = [_summary(values) for values in channel_deltas]
    paired_count = {"shots": len(selected), "episodes": len(selected)}
    return {
        "eligible": {"danger": _counts(danger), "safe": _counts(safe)},
        "with_match": {
            "danger": _counts(danger, danger_matched),
            "safe": _counts(safe, safe_matched),
        },
        "paired": {"danger": dict(paired_count), "safe": dict(paired_count)},
        "distance": {name: _summary(values) for name, values in distances.items()},
        "deltas": signed,
    }


def match_pairs(danger, safe, *, high_floor=None):
    """Return maximum-cardinality, minimum-cost shot pairs and diagnostics.

    Both inputs are sequences of mappings with integer ``shot`` and ``maturity``,
    finite real ``age_ms``, ``anchor`` and ``trend``, and a Boolean ndarray
    ``valid`` of shape (11,). Malformed descriptors or floor raise ValueError,
    even when the opposite side is empty. Other mapping fields are not read.
    Duplicate episodes are permitted. No input mapping or array is modified.
    """
    floor = None if high_floor is None else _finite_number(high_floor, "high_floor")
    d_episodes = _episodes(danger, floor)
    s_episodes = _episodes(safe, floor)
    d_matched = np.zeros(len(d_episodes), dtype=bool)
    s_matched = np.zeros(len(s_episodes), dtype=bool)
    selected = []

    if d_episodes and s_episodes:
        d_shots = sorted({episode.shot for episode in d_episodes})
        s_shots = sorted({episode.shot for episode in s_episodes})
        d_shot_index = {shot: index for index, shot in enumerate(d_shots)}
        s_shot_index = {shot: index for index, shot in enumerate(s_shots)}
        d_maturities = {
            value: rank
            for rank, value in enumerate(sorted({ep.maturity for ep in d_episodes}))
        }
        s_maturities = {
            value: rank
            for rank, value in enumerate(sorted({ep.maturity for ep in s_episodes}))
        }
        d_maturity_rank = np.array([d_maturities[ep.maturity] for ep in d_episodes])
        s_maturity_rank = np.array([s_maturities[ep.maturity] for ep in s_episodes])
        safe_shots = np.array([s_shot_index[ep.shot] for ep in s_episodes])
        safe_age = np.array([ep.age_ms for ep in s_episodes])
        safe_anchor = np.array([ep.anchor for ep in s_episodes])
        safe_trend = np.array([ep.trend for ep in s_episodes])
        safe_mask = np.array([ep.mask for ep in s_episodes], dtype=np.uint16)
        safe_families = np.array([ep.families for ep in s_episodes], dtype=np.uint8)
        shape = (len(d_shots), len(s_shots))
        costs = np.full(shape, np.inf)
        best_danger = np.full(shape, -1, dtype=np.int64)
        best_safe = np.full(shape, -1, dtype=np.int64)

        for d_index, episode in enumerate(d_episodes):
            # Finite input values may still overflow while subtracting. Such
            # distances are necessarily outside the finite calipers.
            with np.errstate(over="ignore"):
                age = episode.age_ms - safe_age
                anchor = episode.anchor - safe_anchor
                trend = episode.trend - safe_trend
            hamming = _BIT_COUNTS[np.bitwise_xor(episode.mask, safe_mask)]
            candidates = np.flatnonzero(
                (safe_families == episode.families)
                & (np.abs(age) <= _AGE_CALIPER)
                & (np.abs(anchor) <= _ANCHOR_CALIPER)
                & (np.abs(trend) <= _TREND_CALIPER)
                & (hamming <= _HAMMING_CALIPER)
            )
            if not candidates.size:
                continue
            d_matched[d_index] = True
            s_matched[candidates] = True
            candidate_costs = (
                (age[candidates] / _AGE_CALIPER) ** 2
                + (anchor[candidates] / _ANCHOR_CALIPER) ** 2
                + (trend[candidates] / _TREND_CALIPER) ** 2
                + (hamming[candidates] / _HAMMING_CALIPER) ** 2
            )
            # Safe episodes are already in canonical order. Pick the cheapest
            # per safe shot, and the first canonical episode on an exact tie.
            order = np.lexsort((candidates, candidate_costs, safe_shots[candidates]))
            ordered_shots = safe_shots[candidates[order]]
            first = np.r_[True, ordered_shots[1:] != ordered_shots[:-1]]
            choices = order[first]
            columns = safe_shots[candidates[choices]]
            row = d_shot_index[episode.shot]
            better = candidate_costs[choices] < costs[row, columns]
            ties = candidate_costs[choices] == costs[row, columns]
            if np.any(ties):
                old_danger = best_danger[row, columns[ties]]
                old_safe = best_safe[row, columns[ties]]
                new_safe = candidates[choices[ties]]
                better[ties] = (
                    d_maturity_rank[d_index] == d_maturity_rank[old_danger]
                ) & (s_maturity_rank[new_safe] < s_maturity_rank[old_safe])
            columns = columns[better]
            choices = choices[better]
            costs[row, columns] = candidate_costs[choices]
            best_danger[row, columns] = d_index
            best_safe[row, columns] = candidates[choices]

        # Fix the maximum possible cardinality before minimizing cost. Exactly
        # n_danger - cardinality dummy columns force that many unmatched rows;
        # forbidden real edges remain infinite, with no finite penalty to tune.
        graph = csr_matrix(np.isfinite(costs))
        matching = maximum_bipartite_matching(graph, perm_type="column")
        cardinality = int(np.count_nonzero(matching >= 0))
        if cardinality:
            dummy = np.zeros((len(d_shots), len(d_shots) - cardinality))
            assignment_costs = np.concatenate((costs, dummy), axis=1)
            rows, columns = linear_sum_assignment(assignment_costs)
            for row, column in zip(rows, columns):
                if column < len(s_shots):
                    d_index, s_index = best_danger[row, column], best_safe[row, column]
                    selected.append(
                        (
                            d_episodes[d_index],
                            s_episodes[s_index],
                            float(costs[row, column]),
                        )
                    )

    pairs = [
        {"danger": d.index, "safe": s.index, "cost": cost} for d, s, cost in selected
    ]
    return {
        "pairs": pairs,
        "diagnostics": _diagnostics(
            d_episodes, s_episodes, d_matched, s_matched, selected
        ),
    }
