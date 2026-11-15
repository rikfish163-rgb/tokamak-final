import json

import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_roughness_20261110 as roughness
from experiments.risk_aligned_online.commissioning_synchronized_roughness_20261110 import (
    fit_reference,
    summarize_pair,
)


START = roughness._START
STRIDE = roughness._STRIDE
HORIZON = roughness._HORIZON
RELEASE = START + HORIZON + 1
ARMS = ("static", "raw_matched", "aligned", "shifted", "stale")


def _shot(amplitudes, *, slope=0.02):
    """Provide independent blocks with a specified median absolute curvature."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    length = START + (len(amplitudes) - 1) * STRIDE + HORIZON + 2
    z = np.broadcast_to(slope * np.arange(length, dtype=np.float64)[:, None], (length, 2)).copy()
    signs = np.where(np.arange(HORIZON) % 2 == 0, 1.0, -1.0)
    for i, amplitude in enumerate(amplitudes):
        q = START + i * STRIDE
        z[q + 1:q + HORIZON + 1] += signs[:, None] * amplitude / 4
    return z, np.ones_like(z, dtype=bool)


def _copy_shot(shot):
    return tuple(a.copy() for a in shot)


def _reference(shot, log_ref=None):
    if log_ref is None:
        log_ref = np.zeros(2)
    return fit_reference([shot, _copy_shot(shot)], log_ref)[0]


def _manual_crossfit(standardized):
    """Calculate the two scalar regressions from explicit alternating folds."""
    residual = np.empty_like(standardized)
    for target, predictor in ((0, 1), (1, 0)):
        for train, test in ((slice(None, None, 2), slice(1, None, 2)),
                            (slice(1, None, 2), slice(None, None, 2))):
            x = standardized[train, predictor]
            y = standardized[train, target]
            coefficient = np.dot(x, y) / (np.dot(x, x) + 1.0)
            residual[test, target] = (standardized[test, target]
                                      - coefficient * standardized[test, predictor])
    return residual


def test_reference_weights_shots_equally_counts_only_joint_blocks_and_floors():
    long = _shot([[1, 0], [1, 0], [9, 0]], slope=0)
    short = _shot([[16, 0]], slope=0)
    unavailable = _shot([[100, 100]], slope=0)
    unavailable[1][:] = False
    offsets, receipt = fit_reference([long, short, unavailable], np.log([2, 3]))

    assert offsets.shape == (3, 2)
    np.testing.assert_allclose(offsets[0], np.log([4, 0.001]), atol=1e-12)
    np.testing.assert_allclose(offsets[:, 1], np.log(0.001), atol=1e-12)
    assert receipt["provided_shots"] == 3
    assert receipt["eligible_shots"] == 2
    assert receipt["eligible_blocks"] == 4
    json.dumps(receipt, allow_nan=False)


def test_pairwise_triplets_not_independent_channel_counts_and_exact_threshold():
    shot = _shot([[2, 3]])
    z, mask = shot
    mask[START + 1:START + HORIZON + 1] = False
    mask[START + 1:START + 15, 0] = True
    mask[START + 12:START + 26, 1] = True
    with pytest.raises(ValueError, match="two|2"):
        fit_reference([shot, _copy_shot(shot)], np.zeros(2))
    features, stats = summarize_pair(z, mask, np.zeros(2), np.zeros((3, 2)))
    assert stats["mature_blocks"] == 1
    assert stats["eligible_blocks"] == stats["emitted_updates"] == 0
    assert stats["median_abs_curvature_coherence"] is None
    assert stats["median_raw_scale"] is None
    json.dumps(stats, allow_nan=False)
    for arm in ARMS:
        np.testing.assert_array_equal(features[arm], features["static"])

    mask[START + 1:START + 15] = True  # exactly 12 joint triplets
    _, stats = summarize_pair(z, mask, np.zeros(2), np.zeros((3, 2)))
    assert stats["eligible_blocks"] == stats["emitted_updates"] == 1
    mask[START + 14, 1] = False
    _, stats = summarize_pair(z, mask, np.zeros(2), np.zeros((3, 2)))
    assert stats["eligible_blocks"] == 0


def test_nonfinite_values_are_unavailable_and_masked_values_do_not_leak():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(220, 2))
    mask = np.ones_like(z, dtype=bool)
    mask[132:136, 0] = False
    mask[174:178, 1] = False
    bad = z.copy()
    bad[~mask] = np.nan
    bad[140, 1] = np.inf
    actual_mask = mask.copy()
    actual_mask[140, 1] = False
    offsets = np.zeros((3, 2))
    expected, expected_stats = summarize_pair(z, actual_mask, np.zeros(2), offsets)
    actual, actual_stats = summarize_pair(bad, mask, np.zeros(2), offsets)
    for arm in ARMS:
        np.testing.assert_array_equal(actual[arm], expected[arm])
    assert actual_stats == expected_stats


def test_standardized_ridge_crossfit_and_roll_use_exact_joint_rows():
    rng = np.random.default_rng(20261110)
    z = rng.normal(size=(RELEASE + 1, 2))
    mask = np.ones_like(z, dtype=bool)
    mask[START + 5, 0] = False
    mask[START + 16, 1] = False
    log_ref = np.log([0.4, 2.5])
    offset, receipt = fit_reference([(z, mask), (z.copy(), mask.copy())], log_ref)

    block = z[START + 1:START + HORIZON + 1]
    availability = mask[START + 1:START + HORIZON + 1]
    joint = np.all(availability[:-2] & availability[1:-1] & availability[2:], axis=1)
    curvature = np.diff(block, n=2, axis=0)
    observed = curvature[joint]
    scale = np.exp(log_ref)
    standardized = observed / scale
    shifted = standardized.copy()
    shifted[:, 1] = np.roll(shifted[:, 1], len(shifted) // 2)
    np.testing.assert_array_equal(np.sort(shifted[:, 1]), np.sort(standardized[:, 1]))
    estimates = np.stack((observed, _manual_crossfit(standardized) * scale,
                          _manual_crossfit(shifted) * scale))
    expected = np.log(np.maximum(0.001, np.median(np.abs(estimates), axis=1)))
    np.testing.assert_allclose(offset, expected, rtol=1e-12, atol=1e-12)
    assert receipt["eligible_blocks"] == 2

    # Shifting before selecting joint rows is a different, prohibited control.
    wrong = curvature.copy() / scale
    wrong[:, 1] = np.roll(wrong[:, 1], np.count_nonzero(joint) // 2)
    wrong_log = np.log(np.maximum(0.001, np.median(
        np.abs(_manual_crossfit(wrong[joint]) * scale), axis=0)))
    assert np.max(np.abs(wrong_log - expected[2])) > 0.05


def test_synchronized_signal_is_suppressed_more_than_shifted_control():
    rng = np.random.default_rng(213)
    z = rng.normal(size=(START + 5 * STRIDE + HORIZON + 2, 2))
    z[:, 1] = 1.7 * z[:, 0] + 0.01 * rng.normal(size=len(z))
    mask = np.ones_like(z, dtype=bool)
    _, stats = summarize_pair(z, mask, np.log([1, 1.7]), np.zeros((3, 2)))
    raw = np.asarray(stats["median_raw_scale"])
    aligned = np.asarray(stats["median_aligned_scale"])
    shifted = np.asarray(stats["median_shifted_scale"])
    assert stats["eligible_blocks"] == 6
    assert stats["median_abs_curvature_coherence"] > 0.99
    assert np.all(aligned < 0.15 * raw)
    assert np.all(shifted > 0.5 * raw)
    json.dumps(stats, allow_nan=False)


def test_static_and_first_raw_update_match_original_contrast_and_huber_rule():
    z, mask = _shot([[16, 4 * np.exp(0.4)], [4, 0.001]])
    log_ref = np.full(2, np.log(4))
    offsets = np.full((3, 2), np.log(4))
    features, stats = summarize_pair(z, mask, log_ref, offsets)
    assert tuple(features) == ARMS
    static, _ = roughness.summarize(z, mask, (log_ref, np.ones(2, bool)), mode="static")
    np.testing.assert_array_equal(features["static"], static)
    for arm in ARMS:
        assert features[arm].dtype == np.float32
        assert features[arm].shape == z.shape
        np.testing.assert_array_equal(features[arm][:RELEASE], static[:RELEASE])

    contrast = roughness._contrast(z, mask)
    first_state = log_ref + np.array([0.25 * np.log(2), 0.1])
    expected_first = np.arcsinh(contrast[RELEASE] / np.exp(first_state))
    np.testing.assert_allclose(features["raw_matched"][RELEASE], expected_first, rtol=1e-6)
    second_state = first_state - 0.25 * np.array([
        (first_state[0] - log_ref[0]) + 0.1 * (first_state[0] - log_ref[0]),
        np.log(2) + 0.1 * (first_state[1] - log_ref[1]),
    ])
    expected_second = np.arcsinh(contrast[RELEASE + STRIDE] / np.exp(second_state))
    np.testing.assert_allclose(features["raw_matched"][RELEASE + STRIDE], expected_second,
                               rtol=1e-6, atol=1e-7)
    assert stats["mature_blocks"] == stats["eligible_blocks"] == stats["emitted_updates"] == 2


def test_all_active_arms_recenter_a_reference_block_and_share_release_schedule():
    reference = _shot([[1.1, 1.7]])
    log_ref = np.array([-0.4, 0.5])
    offsets = _reference(reference, log_ref)
    shifted_offsets = offsets - 0.2
    features, stats = summarize_pair(*reference, log_ref, shifted_offsets)
    contrast = roughness._contrast(*reference)
    expected = np.arcsinh(contrast[RELEASE] / np.exp(log_ref + 0.05))
    for arm in ARMS[1:]:
        np.testing.assert_array_equal(features[arm][:RELEASE], features["static"][:RELEASE])
        np.testing.assert_allclose(features[arm][RELEASE], expected, rtol=1e-6)
    assert stats["eligible_blocks"] == stats["emitted_updates"] == 1


def test_stale_starts_aligned_then_uses_previous_eligible_target_across_gap():
    reference = _shot([[1, 1]])
    offsets = _reference(reference)
    offsets[1] -= 0.2
    z, mask = _shot([[1, 1], [100, 100], [100, 100]])
    q2 = START + STRIDE
    mask[q2 + 1:q2 + 14] = False
    features, stats = summarize_pair(z, mask, np.zeros(2), offsets)
    r3 = RELEASE + 2 * STRIDE

    np.testing.assert_array_equal(features["aligned"][:r3], features["stale"][:r3])
    assert stats["mature_blocks"] == 3
    assert stats["eligible_blocks"] == stats["emitted_updates"] == 2
    contrast = roughness._contrast(np.where(mask, z, 0), mask)
    expected_stale_state = 0.05 - 0.25 * ((0.05 - 0.2) + 0.1 * 0.05)
    np.testing.assert_allclose(features["stale"][r3],
                               np.arcsinh(contrast[r3] / np.exp(expected_stale_state)), rtol=1e-6)
    assert not np.allclose(features["aligned"][r3], features["stale"][r3])


def test_raw_state_clamps_at_four_times_prior_scale_in_both_directions():
    z, mask = _shot([[4 * np.exp(10), 0.001]] * 20)
    log_ref = np.full(2, np.log(4))
    features, stats = summarize_pair(z, mask, log_ref, np.full((3, 2), np.log(4)))
    t = RELEASE + 19 * STRIDE
    contrast = roughness._contrast(z, mask)
    np.testing.assert_allclose(features["raw_matched"][t],
                               np.arcsinh(contrast[t] / [16, 1]), rtol=1e-6, atol=1e-7)
    assert stats["emitted_updates"] == 20


def test_prefix_truncation_future_mutation_and_per_call_reset():
    rng = np.random.default_rng(14)
    z = rng.normal(size=(310, 2))
    mask = rng.random(z.shape) > 0.06
    log_ref = np.array([-0.5, 0.8])
    offsets = np.array([[-1, -0.3], [-2, -0.5], [-1.2, 0.4]])
    prior_copy, offset_copy = log_ref.copy(), offsets.copy()
    expected, _ = summarize_pair(z, mask, log_ref, offsets)
    for cutoff in (39, START, RELEASE - 2, RELEASE - 1, RELEASE,
                   RELEASE + STRIDE - 1, RELEASE + STRIDE, 239):
        changed_z, changed_mask = z.copy(), mask.copy()
        changed_z[cutoff + 1:] = rng.normal(1000, 200, changed_z[cutoff + 1:].shape)
        changed_mask[cutoff + 1:] = ~changed_mask[cutoff + 1:]
        mutated, _ = summarize_pair(changed_z, changed_mask, log_ref, offsets)
        truncated, _ = summarize_pair(z[:cutoff + 1], mask[:cutoff + 1], log_ref, offsets)
        for arm in ARMS:
            np.testing.assert_array_equal(mutated[arm][:cutoff + 1], expected[arm][:cutoff + 1])
            np.testing.assert_array_equal(truncated[arm], expected[arm][:cutoff + 1])
    again, _ = summarize_pair(z, mask, log_ref, offsets)
    for arm in ARMS:
        np.testing.assert_array_equal(again[arm], expected[arm])
    np.testing.assert_array_equal(log_ref, prior_copy)
    np.testing.assert_array_equal(offsets, offset_copy)


def test_complete_block_without_next_row_is_mature_but_has_no_emitted_update():
    shot = _shot([[2, 3]])
    log_ref = np.zeros(2)
    offsets = np.zeros((3, 2))
    before = tuple(a[:RELEASE - 1] for a in shot)
    complete = tuple(a[:RELEASE] for a in shot)
    _, s0 = summarize_pair(*before, log_ref, offsets)
    _, s1 = summarize_pair(*complete, log_ref, offsets)
    _, s2 = summarize_pair(*shot, log_ref, offsets)
    assert (s0["mature_blocks"], s0["eligible_blocks"], s0["emitted_updates"]) == (0, 0, 0)
    assert (s1["mature_blocks"], s1["eligible_blocks"], s1["emitted_updates"]) == (1, 1, 0)
    assert (s2["mature_blocks"], s2["eligible_blocks"], s2["emitted_updates"]) == (1, 1, 1)


def test_invalid_inputs_and_duplicate_shot_objects_are_rejected():
    shot = _shot([[1, 2]])
    z, mask = shot
    with pytest.raises(ValueError, match="two|2"):
        fit_reference([], np.zeros(2))
    with pytest.raises(ValueError, match="two|2"):
        fit_reference([shot], np.zeros(2))
    with pytest.raises(ValueError, match="distinct|duplicate"):
        fit_reference([shot, shot], np.zeros(2))
    with pytest.raises(ValueError):
        fit_reference([shot, _copy_shot(shot)], np.zeros(3))
    with pytest.raises(ValueError):
        summarize_pair(z[:, :1], mask[:, :1], np.zeros(2), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        summarize_pair(z, mask[:-1], np.zeros(2), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        summarize_pair(z, mask, np.array([0, np.nan]), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        summarize_pair(z, mask, np.zeros(2), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        summarize_pair(z, mask, np.zeros(2), np.full((3, 2), np.nan))
