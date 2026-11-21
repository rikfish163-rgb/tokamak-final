"""Synthetic regression tests for deterministic, descriptor-only matching."""

import itertools
import json
import math
import random

import numpy as np
import pytest

from peak_contrast_match_20261106 import match_pairs


def episode(shot, maturity=0, *, age_ms=0.0, anchor=0.0, trend=0.0, valid=None):
    return {
        "shot": shot,
        "maturity": maturity,
        "age_ms": age_ms,
        "anchor": anchor,
        "trend": trend,
        "valid": np.ones(11, dtype=bool) if valid is None else valid,
    }


def identities(result, danger, safe):
    return [
        (
            danger[pair["danger"]]["shot"],
            danger[pair["danger"]]["maturity"],
            safe[pair["safe"]]["shot"],
            safe[pair["safe"]]["maturity"],
            pair["cost"],
        )
        for pair in result["pairs"]
    ]


def test_maximum_cardinality_precedes_cost_even_for_greedy_counterexample():
    # A nearest-edge greedy choice takes 10 -> 30, leaving 20 unmatched.
    danger = [episode(10, anchor=0.0), episode(20, anchor=0.35)]
    safe = [episode(30, anchor=0.1), episode(40, anchor=-0.2)]
    result = match_pairs(danger, safe)
    assert [(d, s) for d, _, s, _, _ in identities(result, danger, safe)] == [
        (10, 40),
        (20, 30),
    ]
    assert [pair["cost"] for pair in result["pairs"]] == pytest.approx([0.64, 1.0])


def test_cost_still_decides_when_its_difference_is_smaller_than_dummy_penalties():
    danger = [episode(1, anchor=1e-9), episode(2)]
    safe = [episode(3)]
    result = match_pairs(danger, safe)
    assert result["pairs"] == [{"danger": 1, "safe": 0, "cost": 0.0}]


def test_minimum_total_cost_after_maximum_cardinality():
    danger = [episode(10, anchor=0.0), episode(20, anchor=0.2)]
    safe = [episode(30, anchor=0.19), episode(40, anchor=0.01)]
    result = match_pairs(danger, safe)
    assert [(d, s) for d, _, s, _, _ in identities(result, danger, safe)] == [
        (10, 40),
        (20, 30),
    ]
    assert sum(pair["cost"] for pair in result["pairs"]) == pytest.approx(0.0032)


def test_all_four_caliper_normalized_squared_terms_contribute():
    mask = np.ones(11, dtype=bool)
    mask[[0, 3, 7]] = False
    danger = [episode(1, age_ms=64.0, anchor=0.25, trend=0.35)]
    safe = [episode(2, valid=mask)]
    result = match_pairs(danger, safe)
    assert result["pairs"] == [{"danger": 0, "safe": 0, "cost": 4.0}]


@pytest.mark.parametrize(
    ("field", "threshold"),
    [("age_ms", 64.0), ("anchor", 0.25), ("trend", 0.35)],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_numeric_caliper_is_inclusive_but_not_fuzzy(field, threshold, sign):
    at_boundary = episode(1, **{field: sign * threshold})
    beyond = episode(1, **{field: sign * np.nextafter(threshold, np.inf)})
    assert len(match_pairs([at_boundary], [episode(2)])["pairs"]) == 1
    assert match_pairs([beyond], [episode(2)])["pairs"] == []


def test_hamming_boundary_and_exact_family_support():
    three = np.ones(11, dtype=bool)
    three[[0, 3, 7]] = False
    four = three.copy()
    four[1] = False
    assert len(match_pairs([episode(1)], [episode(2, valid=three)])["pairs"]) == 1
    assert match_pairs([episode(1)], [episode(2, valid=four)])["pairs"] == []

    first_second = np.zeros(11, dtype=bool)
    first_second[[0, 3]] = True
    first_third = np.zeros(11, dtype=bool)
    first_third[[0, 7]] = True
    result = match_pairs(
        [episode(1, valid=first_second)], [episode(2, valid=first_third)]
    )
    assert result["pairs"] == []  # Only two bits differ, but a family differs.


def test_fewer_than_two_supported_families_is_ineligible_on_either_side():
    no_support = np.zeros(11, dtype=bool)
    one_family = no_support.copy()
    one_family[:3] = True
    danger = [episode(1, valid=no_support), episode(2, valid=one_family), episode(3)]
    safe = [episode(4, valid=one_family), episode(5)]
    result = match_pairs(danger, safe)
    assert result["diagnostics"]["eligible"] == {
        "danger": {"shots": 1, "episodes": 1},
        "safe": {"shots": 1, "episodes": 1},
    }
    assert result["pairs"] == [{"danger": 2, "safe": 1, "cost": 0.0}]


def test_high_floor_is_inclusive_and_applies_to_both_sides_without_tolerance():
    floor = 0.5
    below = np.nextafter(floor, -np.inf)
    above = np.nextafter(floor, np.inf)
    danger = [episode(1, anchor=below), episode(2, anchor=floor)]
    safe = [episode(3, anchor=below), episode(4, anchor=above)]
    result = match_pairs(danger, safe, high_floor=floor)
    assert result["diagnostics"]["eligible"] == {
        "danger": {"shots": 1, "episodes": 1},
        "safe": {"shots": 1, "episodes": 1},
    }
    assert [(p["danger"], p["safe"]) for p in result["pairs"]] == [(1, 1)]
    assert len(match_pairs(danger, safe)["pairs"]) == 2


def test_best_episode_for_each_shot_edge_and_no_shot_replacement():
    danger = [
        episode(1, 30, anchor=0.1),
        episode(1, 10, anchor=0.0),
        episode(2, 5, anchor=0.2),
    ]
    safe = [
        episode(3, 40, anchor=0.1),
        episode(3, 20, anchor=0.0),
        episode(4, 15, anchor=0.2),
        episode(4, 25, anchor=0.21),
    ]
    result = match_pairs(danger, safe)
    assert identities(result, danger, safe) == [
        (1, 10, 3, 20, 0.0),
        (2, 5, 4, 15, 0.0),
    ]
    assert len({danger[p["danger"]]["shot"] for p in result["pairs"]}) == 2
    assert len({safe[p["safe"]]["shot"] for p in result["pairs"]}) == 2


def test_tied_shots_and_maturities_are_stable_under_input_permutations():
    danger = [episode(20, 3), episode(10, 2), episode(10, 1), episode(20, 1)]
    safe = [episode(40, 3), episode(30, 2), episode(30, 1), episode(40, 1)]
    expected = identities(match_pairs(danger, safe), danger, safe)
    assert len(expected) == 2
    assert all(
        d_maturity == 1 and s_maturity == 1
        for _, d_maturity, _, s_maturity, _ in expected
    )
    rng = random.Random(917)
    for _ in range(30):
        rng.shuffle(danger)
        rng.shuffle(safe)
        assert identities(match_pairs(danger, safe), danger, safe) == expected


def test_duplicate_keys_and_identical_rows_are_safe_and_content_tiebroken():
    danger = [episode(1, 3, anchor=0.1), episode(1, 3, anchor=-0.1)]
    safe = [episode(2, 4), episode(2, 4)]
    for d in (danger, list(reversed(danger))):
        for s in (safe, list(reversed(safe))):
            result = match_pairs(d, s)
            assert len(result["pairs"]) == 1
            assert d[result["pairs"][0]["danger"]]["anchor"] == -0.1
            assert result["pairs"][0]["cost"] == pytest.approx(0.16)
            assert result["diagnostics"]["eligible"]["safe"] == {
                "shots": 1,
                "episodes": 2,
            }


def test_safe_maturity_breaks_equal_cost_before_duplicate_danger_descriptors():
    danger = [episode(1, 3, anchor=-0.1), episode(1, 3, anchor=0.1)]
    safe = [episode(2, 10, anchor=-0.1), episode(2, 1, anchor=0.1)]
    result = match_pairs(danger, safe)
    assert identities(result, danger, safe) == [(1, 3, 2, 1, 0.0)]


def test_diagnostics_count_episodes_and_shots_before_and_after_matching():
    danger = [
        episode(1, 1, age_ms=10),
        episode(1, 2, age_ms=500),
        episode(2, age_ms=1000),
    ]
    safe = [episode(3, age_ms=9), episode(4, age_ms=10), episode(5, age_ms=10000)]
    result = match_pairs(danger, safe)
    assert result["pairs"] == [{"danger": 0, "safe": 1, "cost": 0.0}]
    diagnostics = result["diagnostics"]
    assert diagnostics["eligible"] == {
        "danger": {"shots": 2, "episodes": 3},
        "safe": {"shots": 3, "episodes": 3},
    }
    assert diagnostics["with_match"] == {
        "danger": {"shots": 1, "episodes": 1},
        "safe": {"shots": 2, "episodes": 2},
    }
    assert diagnostics["paired"] == {
        "danger": {"shots": 1, "episodes": 1},
        "safe": {"shots": 1, "episodes": 1},
    }
    assert diagnostics["distance"]["cost"]["mean"] == 0.0
    json.dumps(result, allow_nan=False)


def test_paired_deltas_are_signed_danger_minus_safe():
    mask = np.ones(11, dtype=bool)
    mask[0] = False
    danger = [episode(1, age_ms=10, anchor=-0.1, trend=0.2)]
    safe = [episode(2, age_ms=20, anchor=0.1, trend=-0.1, valid=mask)]
    diagnostics = match_pairs(danger, safe)["diagnostics"]
    assert diagnostics["deltas"]["age_ms"]["mean"] == pytest.approx(-10.0)
    assert diagnostics["deltas"]["anchor"]["mean"] == pytest.approx(-0.2)
    assert diagnostics["deltas"]["trend"]["mean"] == pytest.approx(0.3)
    assert diagnostics["distance"]["hamming"]["mean"] == 1.0


@pytest.mark.parametrize(
    "danger,safe", [([], []), ([episode(1)], []), ([], [episode(2)])]
)
def test_empty_sides_have_json_safe_empty_summaries(danger, safe):
    result = match_pairs(danger, safe)
    assert result["pairs"] == []
    assert result["diagnostics"]["distance"]["cost"] == {
        "count": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    [
        {"shot": True},
        {"shot": 1.5},
        {"maturity": False},
        {"maturity": "1"},
        {"age_ms": float("nan")},
        {"age_ms": float("inf")},
        {"anchor": "0"},
        {"anchor": None},
        {"trend": np.bool_(True)},
        {"trend": 10**1000},
        {"valid": None},
        {"valid": [True] * 11},
        {"valid": np.ones(10, dtype=bool)},
        {"valid": np.ones(11, dtype=int)},
        {"valid": np.ones((11, 1), dtype=bool)},
    ],
)
def test_invalid_fields_raise_value_error_even_if_other_side_is_empty(change):
    row = episode(1)
    row.update(change)
    with pytest.raises(ValueError):
        match_pairs([row], [])
    with pytest.raises(ValueError):
        match_pairs([], [row])


def test_missing_and_non_mapping_episodes_raise_value_error():
    row = episode(1)
    del row["anchor"]
    with pytest.raises(ValueError):
        match_pairs([row], [])
    with pytest.raises(ValueError):
        match_pairs(["not a mapping"], [])


@pytest.mark.parametrize(
    "floor", [True, np.bool_(False), "0", float("nan"), float("inf"), 10**1000]
)
def test_invalid_high_floor_is_rejected(floor):
    with pytest.raises(ValueError):
        match_pairs([], [], high_floor=floor)


def test_uses_only_permitted_descriptors_and_does_not_mutate_inputs():
    danger = [episode(1)]
    safe = [episode(2)]
    danger[0]["future_score"] = object()
    safe[0]["future_waveform"] = object()
    dmask, smask = danger[0]["valid"].copy(), safe[0]["valid"].copy()
    assert len(match_pairs(danger, safe)["pairs"]) == 1
    assert np.array_equal(danger[0]["valid"], dmask)
    assert np.array_equal(safe[0]["valid"], smask)
    assert danger[0]["shot"] == 1 and safe[0]["shot"] == 2


def test_same_shot_identifier_on_both_sides_is_role_specific():
    assert len(match_pairs([episode(1)], [episode(1)])["pairs"]) == 1


def test_extreme_finite_descriptors_do_not_overflow_into_a_match():
    danger = [episode(1, age_ms=1e308, anchor=1e308, trend=1e308)]
    safe = [episode(2, age_ms=-1e308, anchor=-1e308, trend=-1e308)]
    with np.errstate(all="raise"):
        result = match_pairs(danger, safe)
    assert result["pairs"] == []
    json.dumps(result, allow_nan=False)


def test_random_small_instances_against_independent_exhaustive_oracle():
    rng = random.Random(77)
    for _ in range(50):
        danger = [episode(i, anchor=rng.choice([0.0, 0.1, 0.2, 0.4])) for i in range(3)]
        safe = [
            episode(10 + i, anchor=rng.choice([-0.1, 0.0, 0.15, 0.35]))
            for i in range(4)
        ]
        choices = []
        for n in range(4):
            for d_subset in itertools.combinations(range(3), n):
                for s_order in itertools.permutations(range(4), n):
                    differences = [
                        abs(danger[d]["anchor"] - safe[s]["anchor"])
                        for d, s in zip(d_subset, s_order)
                    ]
                    if all(diff <= 0.25 for diff in differences):
                        choices.append(
                            (n, sum((diff / 0.25) ** 2 for diff in differences))
                        )
        best_count = max(count for count, _ in choices)
        best_cost = min(cost for count, cost in choices if count == best_count)
        pairs = match_pairs(danger, safe)["pairs"]
        assert len(pairs) == best_count
        assert math.isclose(sum(p["cost"] for p in pairs), best_cost, abs_tol=1e-12)


def test_multiple_episodes_per_shot_against_independent_exhaustive_oracle():
    rng = random.Random(8801)

    def edge_cost(d, s):
        families = [(0, 3), (3, 7), (7, 11)]
        if [d["valid"][a:b].any() for a, b in families] != [
            s["valid"][a:b].any() for a, b in families
        ]:
            return None
        hamming = int(np.count_nonzero(d["valid"] != s["valid"]))
        ds = [abs(d[key] - s[key]) for key in ("age_ms", "anchor", "trend")]
        limits = [64.0, 0.25, 0.35]
        if hamming > 3 or any(delta > limit for delta, limit in zip(ds, limits)):
            return None
        return (
            sum((delta / limit) ** 2 for delta, limit in zip(ds, limits))
            + (hamming / 3) ** 2
        )

    def make_side(shots):
        rows = []
        for shot in shots:
            for _ in range(rng.randint(1, 3)):
                mask = np.ones(11, dtype=bool)
                for bit in rng.sample(range(11), rng.randint(0, 3)):
                    mask[bit] = False
                rows.append(
                    episode(
                        shot,
                        rng.randint(0, 2),
                        age_ms=rng.choice([0.0, 30.0, 70.0]),
                        anchor=rng.choice([-0.2, 0.0, 0.2, 0.4]),
                        trend=rng.choice([-0.2, 0.0, 0.2, 0.4]),
                        valid=mask,
                    )
                )
        rng.shuffle(rows)
        return rows

    for _ in range(75):
        danger = make_side(range(3))
        safe = make_side(range(10, 14))
        edges = {}
        for d in danger:
            for s in safe:
                cost = edge_cost(d, s)
                if cost is not None:
                    key = (d["shot"], s["shot"])
                    edges[key] = min(cost, edges.get(key, math.inf))
        options = []
        for n in range(4):
            for ds in itertools.combinations(range(3), n):
                for ss in itertools.permutations(range(10, 14), n):
                    chosen = list(zip(ds, ss))
                    if all(edge in edges for edge in chosen):
                        options.append((n, sum(edges[edge] for edge in chosen)))
        best_count = max(count for count, _ in options)
        best_cost = min(cost for count, cost in options if count == best_count)
        result = match_pairs(danger, safe)
        pairs = result["pairs"]
        assert len(pairs) == best_count
        assert sum(pair["cost"] for pair in pairs) == pytest.approx(
            best_cost, abs=1e-12
        )
        assert len({danger[p["danger"]]["shot"] for p in pairs}) == len(pairs)
        assert len({safe[p["safe"]]["shot"] for p in pairs}) == len(pairs)
        for pair in pairs:
            d, s = danger[pair["danger"]], safe[pair["safe"]]
            assert pair["cost"] == pytest.approx(edges[d["shot"], s["shot"]], abs=1e-12)
            assert pair["cost"] == pytest.approx(edge_cost(d, s), abs=1e-12)
