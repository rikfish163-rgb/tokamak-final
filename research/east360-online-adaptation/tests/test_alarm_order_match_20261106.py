import itertools
import json
import math
import random

import numpy as np
import pytest

from alarm_order_match_20261106 import match_summary


def row(shot, rank, class_tag, **changes):
    result = {
        "shot": shot,
        "rank": rank,
        "index": 100,
        "class_tag": class_tag,
        "logit": 0.0,
        "coverage": 0.5,
        "motion_recent": 0.1,
        "motion_old": 0.1,
        "delta_recent": 0.0,
        "delta_old": 0.0,
        "delta_recent_inner": None,
        "delta_old_inner": None,
        "future_residual": None,
    }
    result.update(changes)
    return result


def split(rows, name="full_after20"):
    return match_summary(rows)["splits"][name]


def identities(result):
    return {(p["tp_shot"], p["fp_shot"]) for p in result["pairs"]}


def test_maximum_cardinality_beats_greedy_even_at_higher_total_distance():
    rows = [
        row(1, 21, "fp", logit=0.0),
        row(2, 22, "fp", logit=0.4),
        row(3, 23, "reasonable_tp", logit=0.2),
        row(4, 24, "reasonable_tp", logit=-0.2),
    ]
    result = split(rows)
    assert result["n_eligible"] == {"fp": 2, "reasonable_tp": 2}
    assert result["n_pairs"] == 2
    assert result["n_matched"] == {"fp": 2, "reasonable_tp": 2}
    assert result["n_unmatched"] == {"fp": 0, "reasonable_tp": 0}
    assert identities(result) == {(3, 2), (4, 1)}
    assert [p["cost"] for p in result["pairs"]] == pytest.approx([0.16, 0.16])


def test_minimum_normalized_total_distance_among_maximum_cardinality_matches():
    result = split([
        row(1, 21, "fp", logit=0.19),
        row(2, 22, "fp", logit=0.01),
        row(3, 23, "reasonable_tp", logit=0.0),
        row(4, 24, "reasonable_tp", logit=0.2),
    ])
    assert identities(result) == {(3, 2), (4, 1)}
    assert sum(p["cost"] for p in result["pairs"]) == pytest.approx(0.016)


def test_cost_combines_all_five_normalized_caliper_distances():
    result = split([
        row(1, 21, "fp", index=0, logit=0.0, coverage=0.0,
            motion_recent=0.0, motion_old=0.0),
        row(2, 22, "reasonable_tp", index=128, logit=0.25, coverage=0.05,
            motion_recent=0.02, motion_old=0.02),
    ])
    assert result["n_pairs"] == 1
    assert result["pairs"][0]["cost"] == pytest.approx(1.0)


def test_splits_are_matched_independently_with_inclusive_rank_boundaries():
    rows = [
        row(20, 20, "reasonable_tp"),
        row(21, 21, "reasonable_tp"),
        row(190, 190, "other"),
        row(191, 191, "fp"),
        row(360, 360, "other"),
        row(361, 361, "fp"),
    ]
    result = match_summary(rows)["splits"]
    assert result["full_after20"]["rank_range"] == [21, 360]
    assert result["front_after20"]["rank_range"] == [21, 190]
    assert result["back"]["rank_range"] == [191, 360]
    assert result["full_after20"]["n_pairs"] == 1
    assert identities(result["full_after20"]) == {(21, 191)}
    assert result["front_after20"]["n_pairs"] == 0
    assert result["back"]["n_pairs"] == 0
    assert result["front_after20"]["excluded"]["other_class"] == 1
    assert result["back"]["excluded"]["other_class"] == 1


@pytest.mark.parametrize(
    ("fp_changes", "tp_changes", "expected"),
    [
        ({"index": 319}, {"index": 320}, False),
        ({"index": 319}, {"index": 319}, True),
        ({"index": 320}, {"index": 320}, True),
        ({"index": 0}, {"index": 128}, True),
        ({"index": 0}, {"index": 129}, False),
        ({"logit": 0.0}, {"logit": 0.25}, True),
        ({"logit": 0.0}, {"logit": 0.2500001}, False),
        ({"coverage": 0.0}, {"coverage": 0.05}, True),
        ({"coverage": 0.0}, {"coverage": 0.0500001}, False),
        ({"motion_recent": 0.0}, {"motion_recent": 0.02}, True),
        ({"motion_recent": 0.01}, {"motion_recent": 0.05}, True),
        ({"motion_recent": 0.0}, {"motion_recent": 0.0200001}, False),
        ({"motion_old": 0.0}, {"motion_old": 0.02}, True),
        ({"motion_old": 0.01}, {"motion_old": 0.05}, True),
        ({"motion_old": 0.0}, {"motion_old": 0.0200001}, False),
    ],
)
def test_calipers_are_inclusive_and_phase_is_exact(fp_changes, tp_changes, expected):
    result = split([
        row(1, 21, "fp", **fp_changes),
        row(2, 22, "reasonable_tp", **tp_changes),
    ])
    assert result["n_pairs"] == int(expected)
    if expected:
        assert result["pairs"][0]["cost"] == pytest.approx(
            0.0 if fp_changes == tp_changes else 0.2
        )


def test_paired_effect_direction_contrast_and_missing_optional_support():
    rows = [
        row(1, 21, "fp", logit=0.0, delta_recent=1.0, delta_old=2.0,
            delta_recent_inner=4.0, delta_old_inner=5.0, future_residual=10.0),
        row(2, 22, "fp", logit=0.2, delta_recent=1.0, delta_old=1.0,
            delta_recent_inner=None, delta_old_inner=1.0, future_residual=4.0),
        row(3, 23, "reasonable_tp", logit=0.0, delta_recent=4.0, delta_old=3.0,
            delta_recent_inner=7.0, delta_old_inner=8.0, future_residual=12.0),
        row(4, 24, "reasonable_tp", logit=0.2, delta_recent=2.0, delta_old=1.0,
            delta_recent_inner=1.0, delta_old_inner=2.0, future_residual=None),
    ]
    summaries = split(rows)["paired_summary"]
    assert summaries["recent"]["mean"] == pytest.approx(2.0)
    assert summaries["old"]["mean"] == pytest.approx(0.5)
    assert summaries["contrast"]["mean"] == pytest.approx(1.5)
    assert summaries["contrast"]["median"] == pytest.approx(1.5)
    assert summaries["contrast"]["tp_higher"] == 2
    assert summaries["contrast"]["tp_lower"] == 0
    assert summaries["contrast"]["ties"] == 0
    assert summaries["inner_contrast"]["n"] == 1
    assert summaries["inner_contrast"]["mean"] == pytest.approx(0.0)
    assert summaries["inner_contrast"]["unavailable_pairs"] == {"missing": 1, "invalid": 0}
    assert summaries["future_residual"]["n"] == 1
    assert summaries["future_residual"]["mean"] == pytest.approx(2.0)
    assert summaries["future_residual"]["unavailable_pairs"] == {"missing": 1, "invalid": 0}


def test_contrast_is_computed_per_shot_before_tp_minus_fp():
    result = split([
        row(1, 21, "fp", delta_recent=1.0, delta_old=0.0),
        row(2, 22, "reasonable_tp", delta_recent=1e16, delta_old=1e16),
    ])
    assert result["paired_summary"]["contrast"]["mean"] == -1.0


def test_subnormal_finite_values_remain_nonzero_in_mean_median_and_ci():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    rows = [
        row(1, 21, "fp", logit=0.0),
        row(2, 22, "fp", logit=0.2),
        row(3, 23, "reasonable_tp", logit=0.0, delta_recent=tiny),
        row(4, 24, "reasonable_tp", logit=0.2, delta_recent=tiny),
    ]
    summary = split(rows)["paired_summary"]["contrast"]
    assert summary["mean"] == tiny
    assert summary["median"] == tiny
    assert summary["ci95_mean"] == [tiny, tiny]


def test_large_finite_opposite_values_have_finite_descriptive_statistics():
    rows = [
        row(1, 21, "fp", logit=0.0),
        row(2, 22, "fp", logit=0.2),
        row(3, 23, "reasonable_tp", logit=0.0, delta_recent=-1e308),
        row(4, 24, "reasonable_tp", logit=0.2, delta_recent=1e308),
    ]
    summary = split(rows)["paired_summary"]["contrast"]
    assert summary["mean"] == 0.0
    assert summary["median"] == 0.0
    assert summary["ci95_mean"] == [-1e308, 1e308]
    json.dumps(summary, allow_nan=False)


def test_invalid_optional_values_are_reported_not_used_or_imputed():
    result = split([
        row(1, 21, "fp", delta_recent_inner=float("nan"), delta_old_inner=0.0,
            future_residual=float("inf")),
        row(2, 22, "reasonable_tp", delta_recent_inner=1.0, delta_old_inner=0.0,
            future_residual=None),
    ])
    for name in ("inner_contrast", "future_residual"):
        summary = result["paired_summary"][name]
        assert summary["n"] == 0
        assert summary["mean"] is None
        assert summary["median"] is None
        assert summary["ci95_mean"] is None
        assert summary["unavailable_pairs"] == {"missing": 0, "invalid": 1}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("index", -1, "out_of_range"),
        ("index", 1.5, "invalid_type"),
        ("logit", float("nan"), "nonfinite"),
        ("logit", float("inf"), "nonfinite"),
        ("coverage", 1.1, "out_of_range"),
        ("coverage", -0.01, "out_of_range"),
        ("motion_recent", -0.01, "out_of_range"),
        ("motion_old", float("inf"), "nonfinite"),
        ("delta_recent", None, "missing"),
        ("delta_old", "0", "invalid_type"),
    ],
)
def test_invalid_main_fields_are_reported_and_ineligible(field, value, reason):
    result = split([row(1, 21, "fp", **{field: value}), row(2, 22, "reasonable_tp")])
    assert result["n_eligible"] == {"fp": 0, "reasonable_tp": 1}
    assert result["n_unmatched"] == {"fp": 0, "reasonable_tp": 1}
    assert result["excluded"]["invalid_main"]["fp"] == [
        {"shot": 1, "fields": {field: reason}}
    ]


def test_missing_main_key_and_missing_class_tag_are_explicitly_reported():
    missing_main = row(1, 21, "fp")
    del missing_main["coverage"]
    missing_class = row(2, 22, "fp")
    del missing_class["class_tag"]
    result = split([missing_main, missing_class])
    assert result["excluded"]["invalid_main"]["fp"][0]["fields"] == {"coverage": "missing"}
    assert result["excluded"]["invalid_class_tag"] == [{"shot": 2, "reason": "missing"}]


@pytest.mark.parametrize("bad_field", ["shot", "rank"])
@pytest.mark.parametrize("bad_value", [None, True, 1.2, "1"])
def test_invalid_identity_and_rank_are_rejected(bad_field, bad_value):
    invalid = row(1, 21, "fp")
    invalid[bad_field] = bad_value
    with pytest.raises(ValueError, match=bad_field):
        match_summary([invalid])


def test_duplicate_shots_are_rejected_even_outside_a_split_or_in_another_class():
    with pytest.raises(ValueError, match="duplicate.*shot"):
        match_summary([row(1, 21, "fp"), row(1, 361, "other")])


def test_bootstrap_is_paired_percentile_mean_fixed_seed_and_json_ready():
    rows = []
    for i, effect in enumerate([1.0, 2.0, 10.0]):
        rows.extend([
            row(10 + i, 21 + 2 * i, "fp", logit=0.1 * i),
            row(20 + i, 22 + 2 * i, "reasonable_tp", logit=0.1 * i,
                delta_recent=effect),
        ])
    result = match_summary(rows)
    summary = result["splits"]["full_after20"]["paired_summary"]["contrast"]
    draws = np.random.default_rng(131).integers(0, 3, size=(10000, 3))
    expected = np.quantile(np.array([1.0, 2.0, 10.0])[draws].mean(axis=1), [0.025, 0.975])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(13.0 / 3.0)
    assert summary["median"] == 2.0
    assert summary["ci95_mean"] == pytest.approx(expected.tolist())
    assert result["bootstrap"] == {
        "replicates": 10000, "seed": 131, "confidence": 0.95,
        "method": "paired percentile of mean", "rematched": False,
        "interpretation": "descriptive, conditional on the fixed matches",
        "single_pair_ci": "degenerate at the observed difference",
    }
    json.dumps(result, allow_nan=False)


def _gate_rows(front_pairs=6, back_pairs=6, tp_effects=None):
    rows = []
    effects = iter(tp_effects if tp_effects is not None else itertools.repeat(2.0))
    for base_rank, count, base_shot in ((21, front_pairs, 100), (191, back_pairs, 200)):
        for i in range(count):
            rows.append(row(base_shot + 2 * i, base_rank + 2 * i, "fp"))
            rows.append(row(base_shot + 2 * i + 1, base_rank + 2 * i + 1,
                            "reasonable_tp", delta_recent=next(effects)))
    return rows


def test_tentative_gate_passes_only_with_size_sign_and_ci_requirements():
    result = match_summary(_gate_rows())
    gate = result["gate"]
    assert gate["status"] == "passed_tentative"
    assert gate["passed"] is True
    assert gate["promotes_method"] is False
    assert all(gate["checks"].values())
    assert result["splits"]["full_after20"]["paired_summary"]["future_residual"]["n"] == 0


def test_gate_reports_insufficient_pairs_even_when_means_are_positive():
    gate = match_summary(_gate_rows(front_pairs=8, back_pairs=4))["gate"]
    assert gate["status"] == "insufficient_pairs"
    assert not gate["passed"]
    assert not gate["checks"]["back_at_least_5_pairs"]


def test_gate_also_requires_full_pair_count_when_both_halves_are_large_enough():
    gate = match_summary(_gate_rows(front_pairs=5, back_pairs=6))["gate"]
    assert gate["status"] == "insufficient_pairs"
    assert not gate["checks"]["full_at_least_12_pairs"]
    assert gate["checks"]["front_at_least_5_pairs"]
    assert gate["checks"]["back_at_least_5_pairs"]


def test_future_residual_is_diagnostic_and_cannot_change_the_gate():
    rows = _gate_rows()
    original_gate = match_summary(rows)["gate"]
    with_future = [dict(r, future_residual=-1000.0 if r["class_tag"] == "reasonable_tp"
                        else 1000.0) for r in rows]
    result = match_summary(with_future)
    assert result["gate"] == original_gate
    future = result["splits"]["full_after20"]["paired_summary"]["future_residual"]
    assert future["n"] == 12
    assert future["mean"] == -2000.0


def test_gate_requires_positive_descriptive_ci_lower_bound():
    gate = match_summary(_gate_rows(tp_effects=[-7.0] + [2.0] * 11))["gate"]
    assert gate["checks"]["full_mean_contrast_positive"]
    assert gate["checks"]["front_mean_contrast_positive"]
    assert gate["checks"]["back_mean_contrast_positive"]
    assert not gate["checks"]["full_ci_lower_positive"]
    assert gate["status"] == "not_passed"


def test_gate_requires_each_half_positive_independently():
    gate = match_summary(_gate_rows(tp_effects=[-1.0] * 6 + [5.0] * 6))["gate"]
    assert gate["checks"]["full_mean_contrast_positive"]
    assert not gate["checks"]["front_mean_contrast_positive"]
    assert gate["checks"]["back_mean_contrast_positive"]
    assert not gate["passed"]


def test_input_order_and_effect_values_do_not_change_matching():
    rows = _gate_rows(front_pairs=3, back_pairs=3)
    first = match_summary(rows)
    shuffled = rows.copy()
    random.Random(131).shuffle(shuffled)
    assert match_summary(shuffled) == first
    changed = [dict(r, delta_recent=r["delta_recent"] + r["shot"],
                    delta_old_inner=float("nan"), future_residual=None) for r in shuffled]
    result = match_summary(changed)
    for name in first["splits"]:
        assert result["splits"][name]["pairs"] == first["splits"][name]["pairs"]


def test_empty_inputs_and_finite_arithmetic_overflow_handling():
    result = match_summary([])
    assert result["gate"]["status"] == "insufficient_pairs"
    for section in result["splits"].values():
        assert section["n_pairs"] == 0
        for stat in section["paired_summary"].values():
            assert stat["n"] == 0
            assert stat["mean"] is None and stat["ci95_mean"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="nonfinite"):
        match_summary([
            row(1, 21, "fp", delta_recent=-1e308),
            row(2, 22, "reasonable_tp", delta_recent=1e308),
        ])


def _independent_cost(tp, fp):
    if (tp["index"] < 320) != (fp["index"] < 320):
        return None
    differences = [
        abs(tp["logit"] - fp["logit"]) / 0.25,
        abs(math.log((tp["index"] + 128) / (fp["index"] + 128))) / math.log(2),
        abs(tp["coverage"] - fp["coverage"]) / 0.05,
        abs(math.log((tp["motion_recent"] + 0.01) / (fp["motion_recent"] + 0.01))) / math.log(3),
        abs(math.log((tp["motion_old"] + 0.01) / (fp["motion_old"] + 0.01))) / math.log(3),
    ]
    return sum(differences) / 5 if max(differences) <= 1 else None


def test_small_random_instances_agree_with_exhaustive_lexicographic_oracle():
    rng = random.Random(731)
    largest_match = 0
    for _ in range(24):
        n = rng.choice([2, 3, 4])
        fps = [row(i + 1, 21 + i, "fp", logit=rng.uniform(-0.3, 0.3),
                   coverage=rng.uniform(0.47, 0.53), index=rng.choice([100, 128, 200, 320]),
                   motion_recent=rng.uniform(0.03, 0.10), motion_old=rng.uniform(0.03, 0.10))
               for i in range(n)]
        tps = [row(i + 11, 30 + i, "reasonable_tp", logit=rng.uniform(-0.3, 0.3),
                   coverage=rng.uniform(0.47, 0.53), index=rng.choice([100, 128, 200, 320]),
                   motion_recent=rng.uniform(0.03, 0.10), motion_old=rng.uniform(0.03, 0.10))
               for i in range(n)]
        candidates = []
        for count in range(n + 1):
            for ti in itertools.combinations(range(n), count):
                for fi in itertools.permutations(range(n), count):
                    costs = [_independent_cost(tps[t], fps[f]) for t, f in zip(ti, fi)]
                    if all(c is not None for c in costs):
                        candidates.append((count, sum(costs)))
        best_count = max(c[0] for c in candidates)
        largest_match = max(largest_match, best_count)
        best_cost = min(cost for count, cost in candidates if count == best_count)
        result = split(fps + tps)
        assert result["n_pairs"] == best_count
        assert sum(p["cost"] for p in result["pairs"]) == pytest.approx(best_cost)
    assert largest_match >= 3
