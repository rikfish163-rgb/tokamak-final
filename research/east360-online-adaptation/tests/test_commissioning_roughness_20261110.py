import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_roughness_20261110 import (
    coordinates,
    fit_prior,
    summarize,
)


START = 127
STRIDE = 32
HORIZON = 25


def _shot(block_roughness, *, slope=0.01):
    """Make blocks with known absolute second differences on a linear trend."""
    block_roughness = np.asarray(block_roughness, dtype=np.float64)
    if block_roughness.ndim == 1:
        block_roughness = block_roughness[:, None]
    blocks, channels = block_roughness.shape
    length = START + (blocks - 1) * STRIDE + HORIZON + 2
    z = np.broadcast_to(slope * np.arange(length, dtype=np.float64)[:, None],
                        (length, channels)).copy()
    signs = np.where(np.arange(HORIZON) % 2 == 0, 1.0, -1.0)
    for i, roughness in enumerate(block_roughness):
        q = START + i * STRIDE
        z[q + 1:q + HORIZON + 1] += signs[:, None] * roughness / 4.0
    return z, np.ones_like(z, dtype=bool)


def _contrast(z, t, channel=0):
    return z[t - 15:t + 1, channel].mean() - z[t - 63:t - 15, channel].mean()


def test_coordinates_normalize_in_float64_and_apply_caller_mask():
    values = np.array([[2.0, np.nan, 9.0], [4.0, np.inf, -50.0], [6.0, 3.0, 99.0]])
    available = np.array([[True, True, False], [True, True, False], [False, True, False]])
    z, mask = coordinates(values, available, np.array([2, 1, 8]), np.array([2, 4, 5]))
    np.testing.assert_array_equal(mask, [[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(z, [[0, 0, 0], [np.arcsinh(1), 0, 0],
                                   [0, np.arcsinh(0.5), 0]])
    assert z.dtype == np.float64 and mask.dtype == bool


def test_prior_is_median_within_then_equally_across_shots_and_floors():
    long = _shot([[4, 0, 1], [4, 0, 1], [64, 0, 1]])
    short = _shot([[16, 0, 1]])
    short[1][:, 2] = False
    prior, supported = fit_prior([long, short])
    np.testing.assert_array_equal(supported, [True, True, False])
    np.testing.assert_allclose(prior[:2], [np.log(8), np.log(0.001)], atol=1e-12)
    assert prior[2] == 0
    _, only_one_shot = fit_prior([long])
    assert not only_one_shot.any()

    # Incomplete blocks do not enter a reference, even if their values are large.
    one_block = tuple(a[:START + HORIZON + 1] for a in long)
    partial = tuple(a[:START + STRIDE + HORIZON] for a in long)
    first_prior = fit_prior([one_block, short])
    partial_prior = fit_prior([partial, short])
    for a, b in zip(first_prior, partial_prior):
        np.testing.assert_array_equal(a, b)


def test_prior_triplets_require_three_observations_and_at_least_twelve():
    first = _shot([[4, 4]])
    second = _shot([[4, 4]])
    for z, mask in (first, second):
        mask[START + 1:START + HORIZON + 1] = False
        mask[START + 1:START + 15, 0] = True  # 14 points give 12 triplets.
        mask[START + 1:START + 14, 1] = True  # 13 points give 11 triplets.
        z[~mask] = np.nan
    prior, supported = fit_prior([first, second])
    np.testing.assert_array_equal(supported, [True, False])
    assert prior[0] == pytest.approx(np.log(4))
    assert prior[1] == 0

    # Fourteen observations scattered apart do not become adjacent triplets.
    z, mask = _shot([[4]])
    mask[START + 1:START + HORIZON + 1] = False
    mask[START + 1:START + 26:2] = True
    mask[START + 2, 0] = True
    _, supported = fit_prior([(z, mask), (z, mask)])
    assert not supported[0]


def test_linear_trend_does_not_change_curvature_reference():
    flat = _shot([[4], [16]], slope=0)
    trend = _shot([[4], [16]], slope=2.5)
    log_flat, supported_flat = fit_prior([flat, tuple(a.copy() for a in flat)])
    log_trend, supported_trend = fit_prior([trend, tuple(a.copy() for a in trend)])
    np.testing.assert_array_equal(supported_flat, supported_trend)
    np.testing.assert_allclose(log_flat, log_trend, atol=1e-12)


def test_static_contrast_uses_available_means_and_minimum_counts():
    z = np.full((64, 3), 999.0)
    mask = np.zeros_like(z, dtype=bool)
    z[:24] = 1.0
    z[48:56] = 4.0
    mask[:24] = True
    mask[48:56] = True
    mask[0, 1] = False     # 23 preceding observations
    mask[48, 2] = False    # 7 recent observations
    z[~mask] = np.nan
    prior = (np.full(3, np.log(2)), np.ones(3, dtype=bool))
    features, stats = summarize(z, mask, prior, mode="static")
    np.testing.assert_allclose(features[-1], [np.arcsinh(1.5), 0, 0], rtol=1e-6)
    assert features.dtype == np.float32 and features.shape == z.shape
    assert stats == {"update_blocks": 0, "changed_rows": 0, "mature_blocks": 0}


def test_static_all_rows_match_a_direct_masked_window_calculation():
    rng = np.random.default_rng(17)
    z = rng.normal(size=(205, 3))
    mask = rng.random(z.shape) > 0.28
    z[77, 0] = np.nan  # Nonfinite values are never observations.
    prior = (np.array([-0.5, 0.3, 1.2]), np.array([True, True, False]))
    actual, _ = summarize(z, mask, prior, mode="static")
    expected = np.zeros_like(z, dtype=np.float32)
    for t in range(len(z)):
        for channel in range(2):
            recent = [z[i, channel] for i in range(max(0, t - 15), t + 1)
                      if mask[i, channel] and np.isfinite(z[i, channel])]
            past = [z[i, channel] for i in range(max(0, t - 63), max(0, t - 15))
                    if mask[i, channel] and np.isfinite(z[i, channel])]
            if len(recent) >= 8 and len(past) >= 24:
                expected[t, channel] = np.arcsinh((np.mean(recent) - np.mean(past)) /
                                                  np.exp(prior[0][channel]))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_huber_update_and_regularization_are_analytic_and_causal():
    # Both the saturated and quadratic Huber regimes are exercised.
    z, mask = _shot([[16, 4 * np.exp(0.4)], [4, 0.001]])
    prior = (np.full(2, np.log(4)), np.ones(2, dtype=bool))
    static, static_stats = summarize(z, mask, prior, mode="static")
    real, stats = summarize(z, mask, prior, mode="real")
    first, first_stats = summarize(z, mask, prior, mode="first")
    release1, release2 = START + 26, START + STRIDE + 26
    first_state = np.log(4) + np.array([0.25 * np.log(2), 0.1])
    second_state = first_state - 0.25 * np.array([
        (first_state[0] - np.log(4)) + 0.1 * (first_state[0] - np.log(4)),
        np.log(2) + 0.1 * (first_state[1] - np.log(4)),
    ])
    np.testing.assert_array_equal(real[:release1], static[:release1])
    np.testing.assert_array_equal(first[:release2], real[:release2])
    for channel in range(2):
        expected_first = np.arcsinh(_contrast(z, release1, channel) / np.exp(first_state[channel]))
        expected_second = np.arcsinh(_contrast(z, release2, channel) / np.exp(second_state[channel]))
        expected_held = np.arcsinh(_contrast(z, release2, channel) / np.exp(first_state[channel]))
        assert real[release1, channel] == pytest.approx(expected_first, rel=1e-6)
        assert real[release2, channel] == pytest.approx(expected_second, rel=1e-6)
        assert first[release2, channel] == pytest.approx(expected_held, rel=1e-6)
    assert static_stats["mature_blocks"] == stats["mature_blocks"] == 2
    assert static_stats["update_blocks"] == 0
    assert stats["update_blocks"] == 2 and first_stats["update_blocks"] == 1
    assert stats["changed_rows"] == np.count_nonzero(np.any(real != static, axis=1))


def test_complete_final_block_is_mature_without_a_future_feature():
    z, mask = _shot([[16]])
    prior = (np.array([np.log(4)]), np.array([True]))
    partial, partial_stats = summarize(z[:-2], mask[:-2], prior, mode="real")
    complete, complete_stats = summarize(z[:-1], mask[:-1], prior, mode="real")
    released, released_stats = summarize(z, mask, prior, mode="real")
    np.testing.assert_array_equal(complete[:-1], partial)
    np.testing.assert_array_equal(released[:-1], complete)
    assert partial_stats == {"update_blocks": 0, "changed_rows": 0, "mature_blocks": 0}
    assert complete_stats == {"update_blocks": 1, "changed_rows": 0, "mature_blocks": 1}
    assert released_stats == {"update_blocks": 1, "changed_rows": 1, "mature_blocks": 1}


def test_first_freezes_independently_per_supported_channel():
    z, mask = _shot([[16, 16, 16], [16, 16, 16], [4, 4, 4]])
    mask[START + 1:START + 14, 1] = False  # First block: fewer than 12 triplets.
    prior = (np.full(3, np.log(4)), np.array([True, True, False]))
    static, _ = summarize(z, mask, prior, mode="static")
    first, stats = summarize(z, mask, prior, mode="first")
    release1, release2 = START + 26, START + STRIDE + 26
    np.testing.assert_array_equal(first[:release1], static[:release1])
    np.testing.assert_array_equal(first[:release2, 1], static[:release2, 1])
    assert first[release1, 0] != static[release1, 0]
    assert first[release2, 1] != static[release2, 1]
    np.testing.assert_array_equal(first[:, 2], 0)
    assert stats["mature_blocks"] == 3 and stats["update_blocks"] == 2


def test_first_informed_zero_gradient_still_freezes_the_channel():
    z, mask = _shot([[4], [16]], slope=0)
    prior = (np.array([np.log(4)]), np.array([True]))
    static, _ = summarize(z, mask, prior, mode="static")
    first, stats = summarize(z, mask, prior, mode="first")
    real, _ = summarize(z, mask, prior, mode="real")
    np.testing.assert_array_equal(first, static)
    assert real[-1, 0] != static[-1, 0]
    assert stats["update_blocks"] == 1 and stats["changed_rows"] == 0


def test_lag_uses_immediately_previous_block_only_when_both_are_estimable():
    z, mask = _shot([[16], [2], [32], [8]])
    prior = (np.array([np.log(4)]), np.array([True]))
    static, _ = summarize(z, mask, prior, mode="static")
    lag, lag_stats = summarize(z, mask, prior, mode="lag")
    release2 = START + STRIDE + 26
    np.testing.assert_array_equal(lag[:release2], static[:release2])
    expected = np.arcsinh(_contrast(z, release2) / (4 * np.exp(0.25 * np.log(2))))
    assert lag[release2, 0] == pytest.approx(expected, rel=1e-6)
    assert lag_stats["update_blocks"] == 3

    # A missing middle target cannot be skipped in favor of an older block.
    mask[START + STRIDE + 1:START + STRIDE + 14] = False
    missing_lag, missing_stats = summarize(z, mask, prior, mode="lag")
    missing_static, _ = summarize(z, mask, prior, mode="static")
    release4 = release2 + 2 * STRIDE
    np.testing.assert_array_equal(missing_lag[:release4], missing_static[:release4])
    expected4 = np.arcsinh(_contrast(z, release4) / (4 * np.exp(0.25 * np.log(2))))
    assert missing_lag[release4, 0] == pytest.approx(expected4, rel=1e-6)
    assert missing_stats["mature_blocks"] == 4 and missing_stats["update_blocks"] == 1


def test_log_state_clamps_on_both_sides_of_the_prior():
    z, mask = _shot([[4 * np.exp(10), 0.001]] * 20)
    prior = (np.full(2, np.log(4)), np.ones(2, dtype=bool))
    real, stats = summarize(z, mask, prior, mode="real")
    t = START + 19 * STRIDE + 26
    expected = [np.arcsinh(_contrast(z, t, 0) / 16),
                np.arcsinh(_contrast(z, t, 1) / 1)]
    np.testing.assert_allclose(real[t], expected, rtol=1e-6, atol=1e-7)
    assert stats["update_blocks"] == 20


@pytest.mark.parametrize("mode", ["static", "real", "first", "lag"])
def test_future_mutation_truncation_reset_and_masked_values(mode):
    rng = np.random.default_rng(20261110)
    z = rng.normal(size=(310, 3))
    mask = rng.random(z.shape) > 0.18
    prior = (np.array([-1.0, 0.0, 1.0]), np.array([True, True, False]))
    saved_prior = tuple(x.copy() for x in prior)
    expected, _ = summarize(z, mask, prior, mode=mode)
    contaminated = z.copy()
    contaminated[~mask] = np.nan
    contaminated[0, 0] = z[0, 0] if mask[0, 0] else np.inf
    actual, _ = summarize(contaminated, mask, prior, mode=mode)
    np.testing.assert_array_equal(actual, expected)
    for a, b in zip(fit_prior([(z, mask), (z, mask)]),
                    fit_prior([(contaminated, mask), (contaminated, mask)])):
        np.testing.assert_array_equal(a, b)

    for cutoff in [39, 127, 151, 152, 153, 183, 184, 185, 239]:
        changed_z, changed_mask = z.copy(), mask.copy()
        changed_z[cutoff + 1:] = rng.normal(1000, 200, changed_z[cutoff + 1:].shape)
        changed_mask[cutoff + 1:] = ~changed_mask[cutoff + 1:]
        changed, _ = summarize(changed_z, changed_mask, prior, mode=mode)
        truncated, _ = summarize(z[:cutoff + 1], mask[:cutoff + 1], prior, mode=mode)
        np.testing.assert_array_equal(changed[:cutoff + 1], expected[:cutoff + 1])
        np.testing.assert_array_equal(truncated, expected[:cutoff + 1])
    again, _ = summarize(z, mask, prior, mode=mode)
    np.testing.assert_array_equal(again, expected)
    for before, after in zip(saved_prior, prior):
        np.testing.assert_array_equal(before, after)


def test_bad_inputs_are_rejected():
    z, mask = _shot([[4]])
    prior = (np.zeros(1), np.ones(1, bool))
    with pytest.raises(ValueError):
        coordinates(z, mask, np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError):
        coordinates(z, mask[:-1], np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        fit_prior([])
    with pytest.raises(ValueError):
        fit_prior([(z, mask), (np.zeros((len(z), 2)), np.ones((len(z), 2), bool))])
    with pytest.raises(ValueError):
        summarize(z, mask, prior, mode="invalid")
    with pytest.raises(ValueError):
        summarize(z, mask, (np.zeros(2), np.ones(2, bool)), mode="real")
