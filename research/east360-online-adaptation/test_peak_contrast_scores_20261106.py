"""Synthetic, CPU-only checks for the paired peak audit."""

import json

import numpy as np
import pytest

from peak_contrast_scores_20261106 import score_pairs


def make_inputs(n_pairs=1, n_arms=1):
    native = np.zeros((n_pairs, 2, 32), dtype=float)
    positive = np.zeros((n_pairs, 32), dtype=bool)
    positive[:, [0, 1]] = True
    data = {
        "native": native,
        "future": np.zeros((n_pairs, 2, 3, n_arms, 32), dtype=float),
        "anchor_delta": np.zeros((n_pairs, 2, 3, n_arms), dtype=float),
        "norms": np.ones((n_pairs, 2, 3, n_arms), dtype=float),
        "positive": positive,
        "protect": np.zeros((n_pairs, 32), dtype=bool),
        "native_timely": np.zeros(n_pairs, dtype=bool),
        "threshold_logit": 1.0,
        "arm_names": [f"arm{i}" for i in range(n_arms)],
        "pair_ids": np.column_stack(
            (np.arange(n_pairs) + 10, np.arange(n_pairs) + 100)
        ),
    }
    return data


def repeat_native(data):
    p, _, seeds, arms, _ = data["future"].shape
    data["future"] = np.broadcast_to(
        data["native"][:, :, None, None, :], (p, 2, seeds, arms, 32)
    ).copy()


def action(result, pair=0, seed=0, arm=0):
    return result["pairs"][pair]["seed_actions"][seed][arm]


def test_peak_displacement_uses_difference_of_maxima_and_matched_offsets():
    data = make_inputs()
    data["native"][0, 0, :2] = [0.4, -2.0]
    data["native"][0, 1, :2] = [0.3, -2.0]
    data["native"][0, 1, 17] = 4.0
    data["native"][0, 0, 25] = np.nan  # invalid padding, outside both masks
    repeat_native(data)
    data["future"][0, 0, 0, 0, :2] = [0.1, 0.8]
    data["future"][0, 1, 0, 0, :2] = [0.0, 0.5]
    data["future"][0, 1, 0, 0, 17] = 4.2

    result = score_pairs(**data)
    row = action(result)
    assert result["pairs"][0]["positive_offsets"] == [0, 1]
    assert row["danger_positive"]["shape_delta"] == pytest.approx(0.4)
    assert row["danger_positive"]["shape_delta"] != pytest.approx(2.8)
    assert row["safe_aligned"]["shape_delta"] == pytest.approx(0.2)
    assert row["shape_contrast"] == pytest.approx(0.2)
    assert row["safe_full"]["actual_peak"] == pytest.approx(4.2)
    assert row["safe_full"]["score_only_peak"] == pytest.approx(4.0)
    assert not row["safe_induced_raw"]  # it was already crossing
    json.dumps(result, allow_nan=False)


def test_uniform_anchor_shifts_cancel_but_raw_crossing_remains_visible():
    data = make_inputs()
    data["native"][:] = -0.2
    data["native"][0, 1] = -2.0
    repeat_native(data)
    data["anchor_delta"][0, 0] = 0.4
    data["anchor_delta"][0, 1] = -0.3
    data["future"][0, 0] += 0.4
    data["future"][0, 1] -= 0.3
    data["threshold_logit"] = 0.0

    result = score_pairs(**data)
    row = action(result)
    for key in ("danger_positive", "safe_aligned", "safe_full"):
        assert row[key]["actual_peak"] == pytest.approx(row[key]["score_only_peak"])
        assert row[key]["shape_delta"] == pytest.approx(0.0)
    assert row["new_danger_crossing_raw"]
    assert not row["new_danger_crossing_shape"]
    assert not row["feasible"]
    counts = result["arms"][0]["seeds"][0]["counts"]
    assert counts["new_danger_crossing_raw"] == 1
    assert counts["new_danger_crossing_shape"] == 0
    assert counts["new_danger_crossing_eligible"] == 1


def test_extreme_unobserved_danger_padding_never_enters_arithmetic():
    data = make_inputs()
    data["native"][0, 0, 2:] = np.inf
    data["future"][0, 0, :, :, :2] = -1e308
    data["future"][0, 0, :, :, 2:] = 1e308
    data["anchor_delta"][0, 0] = -1e308
    row = action(score_pairs(**data))
    assert row["danger_positive"]["shape_delta"] == 0
    assert row["danger_positive"]["score_only_peak"] == -1e308


def test_empty_protection_is_not_reported_as_observed_preservation():
    row = action(score_pairs(**make_inputs()))
    assert row["danger_protect"] is None
    assert row["protection_preserved"] is None
    assert row["protection_satisfied"]
    assert row["lost_protected_crossing"] is None


def test_feasibility_is_paired_per_action_and_preserves_capacity_evidence():
    data = make_inputs(4, 2)
    data["protect"][1, 5] = True
    data["native"][1, 0, 5] = 1.1
    data["native_timely"][1] = True
    repeat_native(data)

    # Different arms can demonstrate capacity in different seeds, but they
    # must not be mixed within a seed or represented as a learned policy.
    data["future"][0, 0, 0, 0, 0] = 0.06
    data["future"][0, 0, 1, 1, 0] = 0.06
    # Three distinct failures: matched safe damage, a full-block crossing,
    # and loss of the protected danger crossing.
    data["future"][1, 0, :, 0, 0] = 0.06
    data["future"][1, 1, 0, 0, 0] = 0.000002
    data["future"][1, 1, 1, 0, 9] = 1.0
    data["future"][1, 0, 2, 0, 5] = 0.9
    # Both inequalities are inclusive at the specified boundaries.
    data["future"][2, 0, 2, 1, 0] = 0.05
    data["future"][2, 1, 2, 1, 0] = 0.000001

    result = score_pairs(**data)
    capacity = result["capacity"]
    assert capacity["candidate_pairs_per_seed"] == [1, 1, 1]
    assert capacity["candidate_pairs_any_seed"] == 2
    assert capacity["candidate_pairs_at_least_two_seeds"] == 1
    assert capacity["no_solution_pair_ids"] == [[11, 101], [13, 103]]
    assert result["pairs"][0]["feasible_arms_by_seed"] == [["arm0"], ["arm1"], []]
    assert result["pairs"][0]["feasible_seed_count"] == 2
    assert [arm["candidate_pairs_at_least_two_seeds"] for arm in result["arms"]] == [
        0,
        0,
    ]
    assert action(result, 2, 2, 1)["feasible"]
    assert not action(result, 1, 0, 0)["feasible"]
    assert action(result, 1, 0, 0)["danger_shape_pass"]
    assert not action(result, 1, 0, 0)["safe_shape_pass"]
    assert action(result, 1, 1, 0)["safe_induced_raw"]
    assert not action(result, 1, 1, 0)["safe_no_new_crossing"]
    assert not action(result, 1, 1, 0)["feasible"]
    protected = action(result, 1, 2, 0)
    assert not protected["protection_preserved"]
    assert protected["lost_protected_crossing"]
    assert protected["lost_native_timely_protected_crossing"]
    assert not protected["feasible"]
    assert result["arms"][0]["seeds"][2]["counts"]["lost_protected_crossing"] == 1
    assert result["arms"][0]["seeds"][2]["counts"]["protection_satisfied"] == 3


def test_feasibility_boundaries_are_not_implicitly_relaxed():
    data = make_inputs(2, 2)
    data["future"][0, 0, 0, 0, 0] = np.nextafter(0.05, 0.0)
    data["future"][0, 0, 0, 1, 0] = 0.05
    data["future"][1, 0, 0, :, 0] = 0.05
    data["future"][1, 1, 0, 0, 0] = 1e-6
    data["future"][1, 1, 0, 1, 0] = np.nextafter(1e-6, np.inf)
    result = score_pairs(**data)
    assert not action(result, 0, 0, 0)["feasible"]
    assert action(result, 0, 0, 1)["feasible"]
    assert action(result, 1, 0, 0)["feasible"]
    assert not action(result, 1, 0, 1)["feasible"]


def test_crossings_suppressions_and_losses_are_raw_and_shape_specific():
    data = make_inputs(4)
    data["native"][:, 0, :2] = 0.9
    data["native"][:, 1, :] = -1.0
    data["native"][0, 1, 10] = 0.95
    data["native"][1, 1, 10] = 1.1
    data["native"][2, 0, 0] = 1.0
    data["native_timely"][3] = True
    repeat_native(data)
    data["future"][:, 0, :, :, 0] = 1.1
    data["future"][2, 0, :, :, :2] = 0.8
    data["anchor_delta"][0, :, :, :] = 0.2
    data["future"][0, 1, :, :, 10] = 1.1
    data["future"][1, 1, :, :, 10] = 0.9
    data["future"][1, 0, :, :, 0] = 1.2

    result = score_pairs(**data)
    a0, a1, a2, a3 = (action(result, pair=i) for i in range(4))
    assert a0["new_danger_crossing_raw"] and not a0["new_danger_crossing_shape"]
    assert a0["safe_induced_raw"] and not a0["safe_induced_shape"]
    assert a1["new_danger_crossing_raw"] and a1["new_danger_crossing_shape"]
    assert a1["safe_suppressed_raw"] and a1["safe_suppressed_shape"]
    assert a2["lost_danger_positive_crossing_raw"]
    assert a2["lost_danger_positive_crossing_shape"]
    assert not a2["new_danger_crossing_raw"]  # it was already crossing
    assert not a3["new_danger_crossing_raw"]  # already timely elsewhere
    assert a3["lost_protected_crossing"] is None  # no evaluable mask
    counts = result["arms"][0]["seeds"][0]["counts"]
    assert counts["native_miss_or_tardy"] == 3
    assert counts["new_danger_crossing_eligible"] == 2
    assert counts["new_danger_crossing_raw"] == 2
    assert counts["new_danger_crossing_shape"] == 1
    assert counts["safe_induced_raw"] == 1
    assert counts["safe_suppressed_raw"] == 1
    assert counts["lost_danger_positive_crossing_raw"] == 1
    summary = result["arms"][0]["seeds"][0]
    assert summary["rate_denominators"]["new_danger_crossing_raw"] == 2
    assert (
        summary["rate_denominators"]["new_danger_crossing_raw_per_miss_or_tardy"] == 3
    )
    assert summary["rates"]["new_danger_crossing_raw"] == 1.0
    assert summary["rates"][
        "new_danger_crossing_raw_per_miss_or_tardy"
    ] == pytest.approx(2 / 3)


def test_bootstrap_resamples_whole_independent_pairs_with_all_three_seeds():
    data = make_inputs(4, 2)
    effects = np.array([[1, 2, 3], [-1, 0, 1], [4, 2, -3], [0, 1, 2]], dtype=float)
    data["positive"][:, 1] = False
    data["future"][:, 0, :, 0, 0] = effects
    data["future"][:, 0, :, 1, 0] = effects * 2
    summary = score_pairs(**data)["arms"][0]["shape_contrast"]
    per_pair = effects.mean(axis=1)
    indices = np.random.default_rng(61106).integers(0, 4, size=(2000, 4))
    expected = np.percentile(per_pair[indices].mean(axis=1), [2.5, 97.5])
    assert summary["three_seed_mean"] == pytest.approx(per_pair.mean())
    assert summary["per_seed_mean"] == pytest.approx(effects.mean(axis=0))
    assert summary["per_seed_positive_proportion"] == pytest.approx(
        (effects > 0).mean(axis=0)
    )
    assert summary["bootstrap_ci_95"] == pytest.approx(expected)
    assert summary["bootstrap_resamples"] == 2000
    assert summary["bootstrap_seed"] == 61106

    # The same pair resampling is used independently of the order of arms.
    data["future"] = data["future"][:, :, :, ::-1, :]
    data["anchor_delta"] = data["anchor_delta"][:, :, :, ::-1]
    data["norms"] = data["norms"][:, :, :, ::-1]
    data["arm_names"].reverse()
    assert score_pairs(**data)["arms"][1]["shape_contrast"] == summary


def test_empty_pairs_have_null_statistics_and_no_fake_capacity():
    result = score_pairs(**make_inputs(0, 2))
    assert result["n_pairs"] == 0
    assert result["pairs"] == []
    assert result["capacity"]["candidate_pairs_per_seed"] == [0, 0, 0]
    assert result["capacity"]["candidate_pairs_at_least_two_seeds"] == 0
    assert result["capacity"]["no_solution_pair_ids"] == []
    for arm in result["arms"]:
        assert arm["shape_contrast"]["three_seed_mean"] is None
        assert arm["shape_contrast"]["bootstrap_ci_95"] is None
        assert arm["shape_contrast"]["per_seed_mean"] == [None, None, None]
        assert arm["seeds"][0]["rates"]["feasible"] is None
    json.dumps(result, allow_nan=False)


def test_empty_arm_set_keeps_every_pair_as_no_solution():
    result = score_pairs(**make_inputs(2, 0))
    assert result["arms"] == []
    assert result["capacity"]["candidate_pairs_per_seed"] == [0, 0, 0]
    assert result["capacity"]["no_solution_pair_ids"] == [[10, 100], [11, 101]]
    assert all(pair["feasible_seed_count"] == 0 for pair in result["pairs"])


@pytest.mark.parametrize(
    "key,mutate",
    [
        ("native", lambda d: d["native"][:, :, :31]),
        ("future", lambda d: d["future"][:, :, :2]),
        ("anchor_delta", lambda d: d["anchor_delta"][:, :, :, :0]),
        ("norms", lambda d: d["norms"][0]),
        ("positive", lambda d: d["positive"].astype(int)),
        ("positive", lambda d: np.zeros_like(d["positive"])),
        ("protect", lambda d: d["protect"].astype(int)),
        ("native_timely", lambda d: d["native_timely"].astype(int)),
        ("threshold_logit", lambda d: np.inf),
        ("threshold_logit", lambda d: True),
        ("arm_names", lambda d: ["one", "one"]),
        ("arm_names", lambda d: [1]),
        ("arm_names", lambda d: []),
        ("pair_ids", lambda d: d["pair_ids"].astype(float)),
        ("pair_ids", lambda d: np.array([[1, 2], [1, 3]])),
        ("pair_ids", lambda d: np.array([[1, 2], [3, 2]])),
        ("pair_ids", lambda d: np.array([[1, 2], [1, 2]])),
    ],
)
def test_rejects_invalid_shapes_types_names_and_reused_pairs(key, mutate):
    data = make_inputs(2)
    data[key] = mutate(data)
    with pytest.raises(ValueError):
        score_pairs(**data)


@pytest.mark.parametrize("key", ["native", "future", "anchor_delta", "norms"])
def test_rejects_nonfinite_values_at_required_positions(key):
    data = make_inputs()
    data[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_pairs(**data)


def test_safe_full_block_and_danger_protection_must_be_finite():
    data = make_inputs()
    data["native"][0, 1, 20] = np.inf  # safe full block, outside positive mask
    with pytest.raises(ValueError, match="finite"):
        score_pairs(**data)
    data = make_inputs()
    data["protect"][0, 20] = True
    data["native"][0, 0, 20] = 1.0
    data["future"][0, 0, 0, 0, 20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_pairs(**data)


def test_protect_must_witness_native_crossing_and_threshold_is_inclusive():
    data = make_inputs()
    data["protect"][0, 5] = True
    data["native"][0, 0, 5] = 0.99
    with pytest.raises(ValueError, match="protect"):
        score_pairs(**data)
    data["native"][0, 0, 5] = 1.0
    repeat_native(data)
    result = score_pairs(**data)
    assert result["pairs"][0]["native_danger_protect_peak"] == 1.0
    assert action(result)["protection_preserved"]
    assert not action(result)["lost_protected_crossing"]


def test_nonfinite_derived_arithmetic_is_rejected():
    data = make_inputs()
    data["native"][0, 0, :2] = -1e308
    data["future"][0, 0, :, :, :2] = 1e308
    with pytest.raises(ValueError, match="finite"):
        score_pairs(**data)


def test_pair_ids_can_coincide_across_but_not_within_strata():
    data = make_inputs(2)
    data["pair_ids"] = np.array([[1, 2], [2, 1]])
    assert [p["pair_id"] for p in score_pairs(**data)["pairs"]] == [[1, 2], [2, 1]]


def test_every_seed_and_arm_matches_independent_direct_calculation():
    rng = np.random.default_rng(731)
    data = make_inputs(7, 3)
    data["native"] = rng.normal(size=data["native"].shape)
    data["future"] = rng.normal(size=data["future"].shape)
    data["anchor_delta"] = rng.normal(scale=0.25, size=data["anchor_delta"].shape)
    data["positive"] = rng.random((7, 32)) < 0.3
    data["positive"][:, 0] = True
    data["native_timely"] = rng.random(7) < 0.4
    data["threshold_logit"] = 0.2
    original = {
        key: value.copy()
        for key, value in data.items()
        if isinstance(value, np.ndarray)
    }
    result = score_pairs(**data)

    for p in range(7):
        mask = data["positive"][p]
        native_danger = data["native"][p, 0, mask].max()
        native_safe = data["native"][p, 1, mask].max()
        native_safe_full = data["native"][p, 1].max()
        for seed in range(3):
            for arm in range(3):
                row = action(result, p, seed, arm)
                danger_future = data["future"][p, 0, seed, arm]
                safe_future = data["future"][p, 1, seed, arm]
                danger_delta = data["anchor_delta"][p, 0, seed, arm]
                safe_delta = data["anchor_delta"][p, 1, seed, arm]
                danger_shape = (
                    danger_future[mask] - danger_delta
                ).max() - native_danger
                safe_shape = (safe_future[mask] - safe_delta).max() - native_safe
                safe_cross = native_safe_full < 0.2 <= safe_future.max()
                assert row["danger_positive"]["actual_peak"] == pytest.approx(
                    danger_future[mask].max()
                )
                assert row["safe_aligned"]["actual_peak"] == pytest.approx(
                    safe_future[mask].max()
                )
                assert row["safe_full"]["actual_peak"] == pytest.approx(
                    safe_future.max()
                )
                assert row["danger_positive"]["score_only_peak"] == pytest.approx(
                    native_danger + danger_delta
                )
                assert row["safe_aligned"]["score_only_peak"] == pytest.approx(
                    native_safe + safe_delta
                )
                assert row["danger_positive"]["shape_delta"] == pytest.approx(
                    danger_shape
                )
                assert row["safe_aligned"]["shape_delta"] == pytest.approx(safe_shape)
                assert row["shape_contrast"] == pytest.approx(danger_shape - safe_shape)
                assert row["safe_induced_raw"] == safe_cross
                assert row["feasible"] == (
                    danger_shape >= 0.05 and safe_shape <= 1e-6 and not safe_cross
                )
                eligible = not data["native_timely"][p] and native_danger < 0.2
                assert row["new_danger_crossing_raw"] == (
                    eligible and danger_future[mask].max() >= 0.2
                )
                assert row["new_danger_crossing_shape"] == (
                    eligible and (danger_future[mask] - danger_delta).max() >= 0.2
                )
    for key, expected in original.items():
        np.testing.assert_array_equal(data[key], expected)


def test_duplicate_arm_names_are_rejected_even_when_length_matches():
    data = make_inputs(1, 2)
    data["arm_names"] = ["same", "same"]
    with pytest.raises(ValueError, match="arm_names"):
        score_pairs(**data)


def test_protection_and_feasibility_match_a_direct_multi_pair_oracle():
    rng = np.random.default_rng(2301)
    data = make_inputs(12, 3)
    data["native"] = rng.normal(size=data["native"].shape)
    data["future"] = rng.normal(size=data["future"].shape)
    data["anchor_delta"] = rng.normal(scale=0.25, size=data["anchor_delta"].shape)
    data["positive"] = rng.random((12, 32)) < 0.25
    data["positive"][:, 0] = True
    data["protect"][::2, 5] = True
    data["protect"][::2, 9] = True
    data["native"][::2, 0, 5] = 1.0
    data["native_timely"] = rng.random(12) < 0.5
    result = score_pairs(**data)

    for p in range(12):
        pos, prot = data["positive"][p], data["protect"][p]
        native_d = data["native"][p, 0, pos].max()
        native_s = data["native"][p, 1, pos].max()
        native_whole = data["native"][p, 1].max()
        for seed in range(3):
            for arm in range(3):
                row = action(result, p, seed, arm)
                future_d = data["future"][p, 0, seed, arm]
                future_s = data["future"][p, 1, seed, arm]
                delta_d = data["anchor_delta"][p, 0, seed, arm]
                delta_s = data["anchor_delta"][p, 1, seed, arm]
                has_protection = bool(prot.any())
                protected = bool(future_d[prot].max() >= 1) if has_protection else None
                expected = (
                    (future_d[pos] - delta_d).max() - native_d >= 0.05
                    and (future_s[pos] - delta_s).max() - native_s <= 1e-6
                    and not (native_whole < 1 <= future_s.max())
                    and (protected is None or protected)
                )
                assert row["protection_preserved"] is protected
                assert row["protection_satisfied"] == (protected is None or protected)
                assert row["lost_protected_crossing"] == (
                    None if protected is None else not protected
                )
                assert row["lost_native_timely_protected_crossing"] == (
                    not protected
                    if protected is not None and data["native_timely"][p]
                    else None
                )
                assert row["feasible"] == expected
