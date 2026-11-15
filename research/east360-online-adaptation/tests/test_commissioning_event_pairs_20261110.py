import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_event_pairs_20261110 import (
    fit_axis,
    match,
    pair_agreement,
)


def row(shot, family, endpoint, context=(0.0, 0.0, 0.0), feature=(0.0, 0.0)):
    return {
        "shot": shot,
        "family": family,
        "endpoint": endpoint,
        "context": np.array(context, dtype=np.float64),
        "feature": np.array(feature, dtype=np.float64),
    }


def pair(shot, family, difference, safe_feature=None):
    difference = np.asarray(difference, dtype=np.float64)
    safe_feature = (
        np.zeros_like(difference)
        if safe_feature is None
        else np.asarray(safe_feature, dtype=np.float64)
    )
    return (
        row(shot, family, 100, feature=safe_feature + difference),
        row(10000 + shot, family, 100, feature=safe_feature),
    )


@pytest.mark.parametrize("valid_endpoint", [50, 150])
def test_matching_enforces_family_and_inclusive_time_bound_and_allows_reuse(valid_endpoint):
    positives = [row(20, 1, 100), row(10, 1, 100), row(30, 2, 100)]
    eligible = row(9, 1, valid_endpoint)
    safe = [row(1, 1, 49), row(2, 1, 151), row(0, 0, 100), eligible]

    pairs = match(positives, safe)

    assert len(pairs) == 2
    assert pairs[0][0] is positives[1] and pairs[1][0] is positives[0]
    assert all(s is eligible for _, s in pairs)
    # Pair count alone must not be mistaken for independent shot counts.
    assert len({p["shot"] for p, _ in pairs}) == 2
    assert len({s["shot"] for _, s in pairs}) == 1


def test_context_scale_uses_all_safe_rows_and_per_coordinate_mad_floor():
    positive = row(1, 1, 100)
    x_difference = row(2, 1, 100, context=(0.1, 0, 0))
    y_difference = row(3, 1, 101, context=(0, 0.04, 0))
    # These rows cannot match, but they determine the safe-only x MAD of 5.
    other_families = [
        row(10 + i, 2, 1000, context=(x, 0, 0))
        for i, x in enumerate((-10, -5, 0, 5, 10))
    ]

    result = match([positive], [y_difference, *other_families, x_difference])

    assert len(result) == 1
    assert result[0][1] is x_difference


def test_context_scale_uses_the_specified_mad_multiplier():
    positive = row(1, 1, 100)
    x_difference = row(2, 1, 100, context=(1.2, 0, 0))
    y_difference = row(3, 1, 100, context=(0, 0.05, 0))
    other_families = [
        row(10 + i, 2, 1000, context=(x, 0, 0))
        for i, x in enumerate((-2, -1, 0, 1, 2))
    ]
    unmatchable_positives = [row(100 + i, 3, 2000) for i in range(20)]

    # x MAD is 1 and y MAD is zero: (1.2/1.4826)^2 < (0.05/0.05)^2.
    # Including the unmatchable positive contexts in the scale would instead
    # collapse the x MAD to zero and select y_difference.
    result = match([positive, *unmatchable_positives], [x_difference, y_difference, *other_families])

    assert len(result) == 1
    assert result[0][1] is x_difference


def test_endpoint_penalty_and_floor_are_applied_in_source_coordinates():
    positive = row(1, 1, 100)
    same_time = row(8, 1, 100, context=(0.04, 0, 0))
    same_context = row(9, 1, 150)

    # The distances are (0.04/0.05)^2 = 0.64 and (50/50)^2 = 1.
    result = match([positive], [same_context, same_time])

    assert result[0][1] is same_time


def test_ties_and_output_are_deterministic_for_reordered_input():
    positives = [row(12, 1, 100), row(11, 1, 100)]
    preferred = row(8, 1, 90, feature=(1, 0))
    same_id = row(8, 1, 90, feature=(2, 0))
    safe = [row(9, 1, 90), row(8, 1, 110), same_id, preferred]

    forward = match(positives, safe)
    backward = match(positives[::-1], safe[::-1])

    assert [p["shot"] for p, _ in forward] == [11, 12]
    assert [p["shot"] for p, _ in backward] == [11, 12]
    assert all(s is preferred for _, s in forward)
    assert all(s is preferred for _, s in backward)


def test_match_returns_no_pair_when_no_candidate_is_eligible():
    positive = row(1, 4, 100)
    assert match([], []) == []
    assert match([positive], []) == []
    assert match([positive], [row(2, 4, 151), row(3, 3, 100)]) == []


def test_weighted_ridge_uses_original_feature_scale_and_family_mean_loss():
    pairs = [
        pair(1, 0, [2, 0], safe_feature=[7, -3]),
        pair(2, 0, [2, 0], safe_feature=[-4, 6]),
        pair(3, 1, [0, 1], safe_feature=[9, 2]),
    ]

    axis = fit_axis(pairs)

    # Family masses are 1/2 each: (diag(2, .5) + I) beta = (1, .5).
    np.testing.assert_allclose(axis, np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
    assert axis.dtype == np.float32
    assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-6)


def test_zero_direction_and_empty_input_are_explicit():
    pairs = [pair(1, 0, [1, 0]), pair(2, 1, [-1, 0])]
    np.testing.assert_array_equal(fit_axis(pairs), np.zeros(2, dtype=np.float32))
    assert pair_agreement(pairs, np.zeros(2)) == 0.5
    with pytest.raises(ValueError, match="pair"):
        fit_axis([])
    with pytest.raises(ValueError, match="pair"):
        pair_agreement([], np.zeros(2))


def test_pair_agreement_weights_families_equally_and_scores_ties_half():
    pairs = [
        pair(1, 0, [1, 0]),
        pair(2, 0, [-1, 0]),
        pair(3, 0, [0, 2]),
        pair(4, 1, [2, 0]),
    ]
    assert pair_agreement(pairs, np.array([1.0, 0.0])) == pytest.approx(0.75)
    assert pair_agreement(pairs, np.array([-1.0, 0.0])) == pytest.approx(0.25)


def test_matching_to_fit_and_agreement_excludes_ineligible_rows():
    positives = [
        row(1, 0, 100, feature=(2, 0)),
        row(2, 1, 100, feature=(0, 1)),
        row(3, 5, 100, feature=(-100, -100)),
    ]
    safe = [
        row(11, 0, 100),
        row(12, 1, 100),
        row(13, 0, 151, feature=(100, 100)),
        row(14, 5, 151),
    ]
    pairs = match(positives, safe)

    assert [(p["shot"], s["shot"]) for p, s in pairs] == [(1, 11), (2, 12)]
    axis = fit_axis(pairs)
    np.testing.assert_allclose(axis, np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
    assert pair_agreement(pairs, axis) == 1.0


def test_invalid_pair_shapes_and_nonfinite_match_context_do_not_silently_fit():
    with pytest.raises(ValueError, match="feature"):
        fit_axis([(row(1, 0, 100, feature=(1, 2)), row(2, 0, 100, feature=(0,)))])
    with pytest.raises(ValueError, match="context"):
        match([row(1, 0, 100)], [row(2, 0, 100, context=(np.nan, 0, 0))])


def test_shuffled_control_preserves_covariance_balances_signs_and_is_reproducible(monkeypatch):
    families = [0] * 4 + [1] * 3 + [2, 3, 4, 5]
    basis = np.eye(len(families), dtype=np.float64)
    pairs = [pair(i + 1, family, basis[i]) for i, family in enumerate(families)]
    original_features = [(p["feature"].copy(), s["feature"].copy()) for p, s in pairs]
    systems = []
    seeds = []
    solve = np.linalg.solve
    default_rng = np.random.default_rng

    def record_solve(lhs, rhs):
        systems.append((lhs.copy(), rhs.copy()))
        return solve(lhs, rhs)

    def record_rng(seed):
        seeds.append(seed)
        return default_rng(seed)

    monkeypatch.setattr(np.linalg, "solve", record_solve)
    monkeypatch.setattr(np.random, "default_rng", record_rng)
    real_axis = fit_axis(pairs)
    control_axis = fit_axis(pairs, shuffled=True)
    control_reordered = fit_axis(pairs[::-1], shuffled=True)

    assert seeds == [131, 131]
    np.testing.assert_array_equal(systems[0][0], systems[1][0])
    np.testing.assert_allclose(systems[1][0], systems[2][0], atol=0, rtol=0)
    np.testing.assert_allclose(control_axis, control_reordered, atol=0, rtol=0)
    assert not np.array_equal(real_axis, control_axis)
    np.testing.assert_allclose(np.abs(systems[1][1]), systems[0][1], atol=0, rtol=0)
    signs = np.sign(systems[1][1])
    for family in set(families):
        assert abs(signs[np.array(families) == family].sum()) <= 1
    assert set(signs[-4:]) == {-1, 1}
    for (positive, safe), (positive_before, safe_before) in zip(pairs, original_features):
        np.testing.assert_array_equal(positive["feature"], positive_before)
        np.testing.assert_array_equal(safe["feature"], safe_before)
