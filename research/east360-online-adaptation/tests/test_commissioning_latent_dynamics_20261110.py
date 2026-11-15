import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_latent_dynamics_20261110 import (
    HORIZON,
    LAG,
    START,
    STRIDE,
    causal_drift,
    fit_prior,
)


def linear_shot(length, slopes=(1.0,)):
    time = np.arange(length, dtype=np.float64)[:, None]
    hidden = time * np.asarray(slopes, dtype=np.float64)[None, :]
    return hidden, np.ones(length, dtype=bool)


def test_constants_and_known_slope_ridge_prior():
    assert (LAG, HORIZON, START, STRIDE) == (32, 25, 127, 32)
    hidden, observed = linear_shot(250, slopes=(0.5, -2.0, 0.0))
    gain, precision = fit_prior([(hidden, observed)])

    energy = 221.0 * np.square([0.5, -2.0, 0.0])
    np.testing.assert_allclose(precision, energy + 0.001)
    np.testing.assert_allclose(gain, energy / (energy + 0.001))
    assert gain.shape == precision.shape == (3,)
    assert np.all(precision > 0)


def test_prior_gives_each_ended_shot_equal_mass_not_each_block():
    short = np.zeros((START + HORIZON + 1, 1))
    short[START - LAG, 0] = -LAG
    short[START + 1 : START + HORIZON + 1, 0] = np.arange(1, 26)

    long = np.zeros((START + STRIDE + HORIZON + 1, 1))
    long[START - LAG, 0] = -LAG
    long[START + 1 : START + HORIZON + 1, 0] = 2 * np.arange(1, 26)
    long[START + STRIDE, 0] = LAG
    long[START + STRIDE + 1 :, 0] = LAG + 2 * np.arange(1, 26)

    gain, precision = fit_prior(
        [(short, np.ones(len(short), bool)), (long, np.ones(len(long), bool))]
    )
    np.testing.assert_allclose(precision, [221.001])
    np.testing.assert_allclose(gain, [1.5 * 221 / 221.001])


def test_prior_masks_targets_and_bounds_per_channel_gain():
    hidden = np.zeros((START + HORIZON + 1, 2))
    hidden[START - LAG] = -LAG
    hidden[START + 1 :, 0] = 4 * np.arange(1, 26)
    hidden[START + 1 :, 1] = -4 * np.arange(1, 26)
    observed = np.zeros(len(hidden), bool)
    observed[[START - LAG, START, START + 1, START + HORIZON]] = True

    gain, precision = fit_prior([(hidden, observed)])
    np.testing.assert_allclose(precision, [(1 + 625) / 2 + 0.001] * 2)
    np.testing.assert_array_equal(gain, [2.0, -2.0])


def test_prior_ignores_unfinished_last_block():
    second_origin = START + STRIDE
    hidden, observed = linear_shot(second_origin + HORIZON + 1)
    hidden[second_origin + 1 :] += 500
    early_prior = fit_prior([(hidden[: second_origin + 1], observed[: second_origin + 1])])
    partial_prior = fit_prior([(hidden[:-1], observed[:-1])])
    complete_prior = fit_prior([(hidden, observed)])

    for early, partial in zip(early_prior, partial_prior):
        np.testing.assert_array_equal(early, partial)
    np.testing.assert_allclose(early_prior[0], [221 / 221.001])
    np.testing.assert_array_equal(complete_prior[0], [2.0])


def test_static_features_apply_prior_only_to_observed_lag_pairs():
    hidden, observed = linear_shot(110, slopes=(1.0, -2.0))
    observed[40] = False
    prior = (np.array([0.5, -0.25]), np.ones(2))
    drift, stats = causal_drift(hidden, observed, prior, mode="static")

    np.testing.assert_array_equal(drift[:LAG], 0)
    np.testing.assert_array_equal(drift[40], 0)
    np.testing.assert_array_equal(drift[40 + LAG], 0)
    np.testing.assert_allclose(drift[41], [12.5, 12.5])
    assert stats["changed_steps"] == stats["mature_blocks"] == 0


def test_release_boundary_and_partial_block():
    prior = (np.zeros(1), np.ones(1))
    hidden, observed = linear_shot(START + HORIZON + 2)

    partial, partial_stats = causal_drift(hidden[:-2], observed[:-2], prior, mode="real")
    complete, complete_stats = causal_drift(hidden[:-1], observed[:-1], prior, mode="real")
    released, released_stats = causal_drift(hidden, observed, prior, mode="real")

    np.testing.assert_array_equal(partial, 0)
    np.testing.assert_array_equal(complete, 0)
    np.testing.assert_array_equal(released[:-1], complete)
    np.testing.assert_allclose(released[-1], [25 * 221 / 222], rtol=1e-6)
    assert partial_stats["mature_blocks"] == 0
    assert complete_stats["mature_blocks"] == released_stats["mature_blocks"] == 1
    assert complete_stats["changed_steps"] == 0
    assert released_stats["changed_steps"] == 1
    assert released.dtype == np.float32


def test_posterior_uses_prior_precision_and_gain():
    hidden, observed = linear_shot(START + HORIZON + 2)
    prior = (np.array([0.2]), np.array([2.0]))
    drift, _ = causal_drift(hidden, observed, prior, mode="real")
    np.testing.assert_allclose(drift[-2], [25 * 0.2])
    np.testing.assert_allclose(drift[-1], [25 * (2 * 0.2 + 221) / (2 + 221)], rtol=1e-6)


def test_future_mutation_and_truncation_do_not_change_earlier_features_or_prior():
    rng = np.random.default_rng(194)
    hidden = rng.normal(size=(310, 3))
    observed = rng.random(310) > 0.2
    prior = (np.array([0.3, -0.4, 0.8]), np.array([1.0, 2.0, 0.5]))
    original_prior = tuple(part.copy() for part in prior)
    cutoff = 193

    changed_hidden = hidden.copy()
    changed_hidden[cutoff + 1 :] = 1000 * rng.normal(size=changed_hidden[cutoff + 1 :].shape)
    changed_observed = observed.copy()
    changed_observed[cutoff + 1 :] = ~changed_observed[cutoff + 1 :]

    for mode in ("static", "real", "reverse"):
        original, _ = causal_drift(hidden, observed, prior, mode=mode)
        changed, _ = causal_drift(changed_hidden, changed_observed, prior, mode=mode)
        prefix, _ = causal_drift(hidden[: cutoff + 1], observed[: cutoff + 1], prior, mode=mode)
        np.testing.assert_array_equal(original[: cutoff + 1], changed[: cutoff + 1])
        np.testing.assert_array_equal(original[: cutoff + 1], prefix)
    for original, current in zip(original_prior, prior):
        np.testing.assert_array_equal(original, current)


def test_static_real_reverse_order_and_real_order_forecast_metrics():
    hidden, observed = linear_shot(START + STRIDE + HORIZON + 1)
    prior = (np.zeros(1), np.ones(1))
    results = {mode: causal_drift(hidden, observed, prior, mode=mode)
               for mode in ("static", "real", "reverse")}
    first_release = START + HORIZON + 1
    np.testing.assert_array_equal(results["static"][0], 0)
    np.testing.assert_allclose(results["real"][0][first_release], [25 * 221 / 222], rtol=1e-6)
    np.testing.assert_allclose(results["reverse"][0][first_release], [25 * 117 / 222], rtol=1e-6)

    for mode, (_, stats) in results.items():
        gain_at_second_origin = {"static": 0, "real": 221 / 222, "reverse": 117 / 222}[mode]
        assert stats["mature_blocks"] == 2
        assert stats["forecast_weight"] == 50
        assert stats["squared_hold_error_sum"] == pytest.approx(2 * 5525)
        assert stats["squared_forecast_error_sum"] == pytest.approx(
            5525 * (1 + (1 - gain_at_second_origin) ** 2)
        )


def test_reverse_control_reverses_future_mask_and_latents_together():
    hidden, _ = linear_shot(START + HORIZON + 2)
    observed = np.zeros(len(hidden), bool)
    observed[[START - LAG, START, START + 1, START + HORIZON + 1 - LAG,
              START + HORIZON + 1]] = True
    prior = (np.zeros(1), np.ones(1))

    real, real_stats = causal_drift(hidden, observed, prior, mode="real")
    reverse, reverse_stats = causal_drift(hidden, observed, prior, mode="reverse")
    np.testing.assert_allclose(real[-1], [25 / 2], rtol=1e-6)
    np.testing.assert_allclose(reverse[-1], [25 * 25 / 626], rtol=1e-6)
    assert real_stats["forecast_weight"] == reverse_stats["forecast_weight"] == 1
    assert real_stats["squared_forecast_error_sum"] == reverse_stats["squared_forecast_error_sum"] == 1


def test_forecast_weight_counts_observed_scalar_targets_and_ignores_masked_values():
    hidden, observed = linear_shot(START + HORIZON + 1, slopes=(1, 2))
    observed[START + 1] = False
    hidden[START + 1] = [1e9, -1e9]
    prior = (np.ones(2), np.ones(2))

    for mode in ("static", "real", "reverse"):
        _, stats = causal_drift(hidden, observed, prior, mode=mode)
        assert stats["mature_blocks"] == 1
        assert stats["forecast_weight"] == 48
        assert stats["squared_forecast_error_sum"] == pytest.approx(0)
        assert stats["squared_hold_error_sum"] == pytest.approx(5 * (5525 - 1))

    gain, precision = fit_prior([(hidden, observed)])
    energy = (5525 - 1) / 24 * np.array([1, 4])
    np.testing.assert_allclose(precision, energy + 0.001)
    np.testing.assert_allclose(gain, energy / (energy + 0.001))


def test_decay_has_half_life_four_valid_updates():
    slopes = [1, -1, 1, 0, 2]
    last_origin = START + (len(slopes) - 1) * STRIDE
    hidden = np.zeros((last_origin + HORIZON + 2, 1))
    hidden[START - LAG, 0] = -LAG
    for index, slope in enumerate(slopes):
        origin = START + index * STRIDE
        hidden[origin, 0] = index * LAG
        hidden[origin + 1 : origin + HORIZON + 1, 0] = index * LAG + slope * np.arange(1, 26)
    hidden[-1, 0] = LAG
    observed = np.ones(len(hidden), bool)

    drift, stats = causal_drift(hidden, observed, (np.zeros(1), np.ones(1)), mode="real")
    weights = np.power(2.0, -np.arange(4, -1, -1) / 4)
    gain = 221 * np.dot(weights, slopes) / (1 + 221 * np.sum(weights))
    np.testing.assert_allclose(drift[-1], [25 * gain], rtol=1e-6)
    assert stats["mature_blocks"] == 5


def test_ineligible_block_does_not_count_as_a_decay_update():
    third_origin = START + 2 * STRIDE
    hidden = np.zeros((third_origin + HORIZON + 2, 1))
    hidden[START - LAG, 0] = -LAG
    hidden[START + 1 : START + HORIZON + 1, 0] = np.arange(1, 26)
    hidden[START + STRIDE, 0] = LAG
    hidden[third_origin, 0] = 2 * LAG
    hidden[third_origin + 1 : third_origin + HORIZON + 1, 0] = 2 * LAG - np.arange(1, 26)
    hidden[-1, 0] = LAG
    observed = np.ones(len(hidden), bool)
    observed[START + STRIDE + 1 : START + STRIDE + HORIZON + 1] = False

    drift, stats = causal_drift(hidden, observed, (np.zeros(1), np.ones(1)), mode="real")
    decay = 2 ** (-1 / 4)
    gain = 221 * (decay - 1) / (1 + 221 * (decay + 1))
    np.testing.assert_allclose(drift[-1], [25 * gain], rtol=1e-6)
    assert stats["mature_blocks"] == 2


def test_reset_no_observation_and_masked_pairs():
    hidden, observed = linear_shot(260, slopes=(1, -1))
    prior = (np.array([0.2, -0.4]), np.ones(2))
    first, first_stats = causal_drift(hidden, observed, prior, mode="real")
    _, _ = causal_drift(-hidden, observed, prior, mode="reverse")
    repeated, repeated_stats = causal_drift(hidden, observed, prior, mode="real")
    np.testing.assert_array_equal(first, repeated)
    assert first_stats == repeated_stats

    missing = np.zeros(len(hidden), bool)
    gain, precision = fit_prior([(hidden, missing)])
    np.testing.assert_array_equal(gain, [0, 0])
    np.testing.assert_array_equal(precision, [0.001, 0.001])
    drift, stats = causal_drift(hidden, missing, prior, mode="real")
    np.testing.assert_array_equal(drift, 0)
    assert stats == {"changed_steps": 0, "mature_blocks": 0,
                     "squared_forecast_error_sum": 0.0, "squared_hold_error_sum": 0.0,
                     "forecast_weight": 0.0}

    observed[START - LAG] = False
    blocked, blocked_stats = causal_drift(hidden[: START + HORIZON + 2],
                                         observed[: START + HORIZON + 2],
                                         (np.zeros(2), np.ones(2)), mode="real")
    np.testing.assert_array_equal(blocked, 0)
    assert blocked_stats["mature_blocks"] == 0

    no_targets = np.zeros(START + HORIZON + 2, bool)
    no_targets[[START - LAG, START, START + HORIZON + 1 - LAG,
                START + HORIZON + 1]] = True
    drift, stats = causal_drift(hidden[: len(no_targets)], no_targets,
                                (np.zeros(2), np.ones(2)), mode="real")
    np.testing.assert_array_equal(drift, 0)
    assert stats["mature_blocks"] == stats["forecast_weight"] == 0


def test_input_contract_errors():
    hidden, observed = linear_shot(200, slopes=(1, 2))
    with pytest.raises(ValueError):
        fit_prior([])
    with pytest.raises(ValueError):
        fit_prior([(hidden, observed[:-1])])
    with pytest.raises(ValueError):
        causal_drift(hidden, observed, (np.zeros(2), np.ones(2)), mode="other")
    with pytest.raises(ValueError):
        causal_drift(hidden, observed, (np.zeros(1), np.ones(1)), mode="real")
    with pytest.raises(ValueError):
        causal_drift(hidden, observed, (np.zeros(2), np.zeros(2)), mode="real")
