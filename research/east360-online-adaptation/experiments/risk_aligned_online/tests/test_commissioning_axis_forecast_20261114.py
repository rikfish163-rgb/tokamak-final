import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_axis_forecast_20261114 as axis


def _signal(length=240):
    t = np.arange(length, dtype=np.float64)
    values = np.column_stack(
        [np.sin(t * (0.025 + c * 0.001) + c / 7) + t * 0.002 for c in range(23)]
    )
    return {"values": values, "available": np.ones_like(values, dtype=bool)}


def _copy(signal):
    return {key: value.copy() for key, value in signal.items()}


def test_target_is_centered_five_bin_shape_scaled_by_observed_past():
    signal = _signal()
    q = 127
    signal["available"][q - 20, 21] = False
    signal["values"][q - 18, 21] = np.nan
    means = np.array([2.0, 3.0, 1.0, -2.0, 6.0])
    signal["values"][q + 1 : q + 26, 21] = np.repeat(means, 5)

    design, issued, target, mature = axis.make_samples(signal, [q])
    past = signal["values"][q - 125 : q + 1, 21]
    observed = signal["available"][q - 125 : q + 1, 21] & np.isfinite(past)
    scale = np.std(past[observed])
    expected = np.arcsinh((means - means.mean()) / scale)
    expected -= expected.mean()

    assert axis.TARGET_INDEX == 21
    assert design.shape == (1, axis.D) and design.dtype == np.float64
    assert issued.dtype == mature.dtype == np.bool_
    assert issued.tolist() == mature.tolist() == [True]
    assert target.dtype == np.float64
    np.testing.assert_allclose(target[0], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(target.sum(axis=1), 0, atol=1e-14)


@pytest.mark.parametrize("q", [127, 151, 180])
def test_issued_features_do_not_change_with_future_forgery_or_truncation(q):
    signal = _signal()
    design, issued, _, mature = axis.make_samples(signal, [q])
    assert issued[0] and mature[0]

    forged = _copy(signal)
    rng = np.random.default_rng(q)
    forged["values"][q + 1 :] = rng.normal(size=forged["values"][q + 1 :].shape) * 1e6
    forged["available"][q + 1 :] = False
    changed_design, changed_issued, _, changed_mature = axis.make_samples(forged, [q])
    prefix = {key: value[: q + 1] for key, value in signal.items()}
    prefix_design, prefix_issued, _, prefix_mature = axis.make_samples(prefix, [q])

    np.testing.assert_array_equal(changed_design, design)
    np.testing.assert_array_equal(prefix_design, design)
    np.testing.assert_array_equal(changed_issued, issued)
    np.testing.assert_array_equal(prefix_issued, issued)
    assert not changed_mature[0] and not prefix_mature[0]


def test_maturity_needs_the_25_future_points_and_the_extra_arrival():
    signal = _signal()
    q = 127
    only_25 = {key: value[: q + 26] for key, value in signal.items()}
    with_arrival = {key: value[: q + 27].copy() for key, value in signal.items()}
    with_arrival["values"][q + 26, :] = np.nan
    with_arrival["available"][q + 26, :] = False
    d25, i25, y25, m25 = axis.make_samples(only_25, [q])
    d26, i26, y26, m26 = axis.make_samples(with_arrival, [q])
    np.testing.assert_array_equal(d25, d26)
    np.testing.assert_array_equal(i25, i26)
    assert i25[0] and not m25[0] and m26[0]
    assert np.isnan(y25).all() and np.isfinite(y26).all()

    for step in range(1, 26):
        missing = _copy(with_arrival)
        missing["available"][q + step, 21] = False
        _, issued, target, mature = axis.make_samples(missing, [q])
        assert issued[0] and not mature[0] and np.isnan(target).all()
    missing = _copy(with_arrival)
    missing["values"][q + 13, 21] = np.nan
    assert not axis.make_samples(missing, [q])[3][0]


def test_issued_and_mature_validity_are_separate():
    signal = _signal()
    q = 127
    # Remove the first five points of the most recent independent input window.
    signal["available"][q - 24 : q - 19, 11] = False
    design, issued, target, mature = axis.make_samples(signal, [q])
    assert not issued[0] and mature[0]
    assert np.isnan(design).all() and np.isfinite(target).all()

    signal = _signal()
    signal["available"][q - 125 : q + 1, 21] = False
    signal["available"][q, 21] = True
    design, issued, target, mature = axis.make_samples(signal, [q])
    assert not issued[0] and not mature[0]
    assert np.isnan(design).all() and np.isnan(target).all()


def test_all_noninput_channels_and_target_history_are_excluded_from_design():
    signal = _signal()
    q = 127
    baseline, valid, target, mature = axis.make_samples(signal, [q])
    altered = _copy(signal)
    ignored = [c for c in range(23) if c not in (7, 8, 9, 11, 18, 21)]
    altered["values"][:, ignored] = np.nan
    altered["available"][:, ignored] = False
    d, i, y, m = axis.make_samples(altered, [q])
    np.testing.assert_array_equal(d, baseline)
    np.testing.assert_array_equal(i, valid)
    np.testing.assert_array_equal(y, target)
    np.testing.assert_array_equal(m, mature)

    altered["values"][: q + 1, 21] = altered["values"][: q + 1, 21] * 3 + 17
    d, i, _, _ = axis.make_samples(altered, [q])
    np.testing.assert_array_equal(d, baseline)
    np.testing.assert_array_equal(i, valid)


def test_independent_features_match_three_25_point_windows():
    signal = _signal()
    t = np.arange(len(signal["values"]), dtype=np.float64)
    for channel, slope in ((11, 2.0), (18, -3.0), (9, 1.5), (7, -0.5), (8, 0.75)):
        signal["values"][:, channel] = slope * t + 5
    design, valid, _, _ = axis.make_samples(signal, [127])
    assert valid[0] and design[0, 0] == 1.0
    time_scale = np.std(np.arange(126, dtype=np.float64))
    for name, sign in (("n=1 amplitude", 1), ("v_loop", -1), ("ip", 1)):
        for window in range(3):
            start = axis.FEATURE_NAMES.index(f"{name}:w{window}_start")
            change = axis.FEATURE_NAMES.index(f"{name}:w{window}_change")
            np.testing.assert_allclose(
                design[0, start], np.arcsinh(sign * (-9.5 + 25 * window) / time_scale)
            )
            np.testing.assert_allclose(design[0, change], np.arcsinh(sign * 20 / time_scale))
    for name in ("dx_a", "dy_a"):
        for window in range(3):
            column = axis.FEATURE_NAMES.index(f"{name}:w{window}_abs_change")
            np.testing.assert_allclose(design[0, column], np.arcsinh(20 / time_scale))


@pytest.mark.parametrize("a, b", [(2.75, 11.0), (-3.5, 4.0)])
def test_target_positive_affine_invariance_and_negative_odd_symmetry(a, b):
    signal = _signal()
    q = [127, 152]
    baseline, valid, target, mature = axis.make_samples(signal, q)
    changed = _copy(signal)
    changed["values"][:, 21] = a * changed["values"][:, 21] + b
    design2, valid2, target2, mature2 = axis.make_samples(changed, q)
    np.testing.assert_array_equal(design2, baseline)
    np.testing.assert_array_equal(valid2, valid)
    np.testing.assert_array_equal(mature2, mature)
    assert valid.all() and mature.all()
    np.testing.assert_allclose(target2, np.sign(a) * target, rtol=1e-12, atol=1e-12)


def test_independent_inputs_are_relative_and_position_proxies_ignore_sign():
    signal = _signal()
    baseline, valid, _, _ = axis.make_samples(signal, [127, 160])
    changed = _copy(signal)
    for c in (7, 8, 9, 11, 18):
        changed["values"][:, c] = 2.5 * changed["values"][:, c] + 7
    design2, valid2, _, _ = axis.make_samples(changed, [127, 160])
    np.testing.assert_allclose(design2, baseline, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(valid2, valid)

    changed = _copy(signal)
    changed["values"][:, (7, 8)] *= -2
    design2, _, _, _ = axis.make_samples(changed, [127, 160])
    np.testing.assert_allclose(design2, baseline, rtol=1e-12, atol=1e-12)

    changed = _copy(signal)
    changed["values"][:, 11] *= -1
    design2, _, _, _ = axis.make_samples(changed, [127, 160])
    n1_columns = [i for i, name in enumerate(axis.FEATURE_NAMES) if name.startswith("n=1 amplitude:")]
    other_columns = [i for i in range(axis.D) if i not in n1_columns]
    assert len(n1_columns) == 6
    np.testing.assert_allclose(design2[:, n1_columns], -baseline[:, n1_columns], atol=1e-12)
    np.testing.assert_array_equal(design2[:, other_columns], baseline[:, other_columns])


def test_constant_past_uses_finite_floor_and_keeps_signed_future_shape():
    signal = _signal()
    q = 127
    signal["values"][: q + 1, 21] = 3
    signal["values"][q + 1 : q + 26, 21] = np.repeat([1, 2, 3, 4, 5], 5)
    design, issued, target, mature = axis.make_samples(signal, [q])
    assert issued[0] and mature[0]
    assert np.isfinite(design).all() and np.isfinite(target).all()
    assert target[0, 0] < 0 < target[0, -1]
    np.testing.assert_allclose(target.sum(axis=1), 0, atol=1e-13)


def test_empty_queries_and_invalid_query_contract():
    signal = _signal()
    design, issued, target, mature = axis.make_samples(signal, [])
    assert design.shape == (0, axis.D) and design.dtype == np.float64
    assert target.shape == (0, 5) and target.dtype == np.float64
    assert issued.shape == mature.shape == (0,)
    assert issued.dtype == mature.dtype == np.bool_
    for q in ([126], [240], [128, 127], [127, 127], [127.0], [[127]]):
        with pytest.raises((TypeError, ValueError)):
            axis.make_samples(signal, q)
    with pytest.raises((TypeError, ValueError)):
        axis.make_samples({**signal, "outcome": 1}, [127])


def test_fixed_ridge_weights_each_nonempty_shot_equally_including_intercept():
    x1 = np.array([[1.0, 2.0], [1.0, -1.0]])
    y1 = np.array([[2.0, -1.0, 0.0, 1.0, -2.0], [-1.0, 0.0, 1.0, -1.0, 1.0]])
    x2 = np.tile([1.0, 3.0], (17, 1))
    y2 = np.tile([-2.0, 3.0, 0.0, -1.0, 0.0], (17, 1))
    gram = (x1.T @ x1 / len(x1) + x2.T @ x2 / len(x2)) / 2
    cross = (x1.T @ y1 / len(x1) + x2.T @ y2 / len(x2)) / 2
    expected = np.linalg.solve(gram + 0.10 * np.eye(2), cross)
    expected -= expected.mean(axis=1, keepdims=True)
    coefficient = axis.fit_weighted(iter([(x1, y1), (x2, y2)]))
    repeated = axis.fit_weighted([(np.repeat(x1, 7, axis=0), np.repeat(y1, 7, axis=0)), (x2, y2)])
    np.testing.assert_allclose(coefficient, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(repeated, expected, rtol=1e-13, atol=1e-13)
    assert coefficient.dtype == np.float64
    np.testing.assert_allclose(coefficient.sum(axis=1), 0, atol=1e-14)
    prediction = axis.predict(x1, coefficient)
    assert prediction.shape == (2, 5) and prediction.dtype == np.float64
    np.testing.assert_allclose(prediction.sum(axis=1), 0, atol=1e-14)


def test_fit_rejects_no_usable_shots_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        axis.fit_weighted([])
    with pytest.raises(ValueError):
        axis.fit_weighted([(np.empty((0, 2)), np.empty((0, 5)))])
    with pytest.raises(ValueError):
        axis.fit_weighted([(np.array([[1.0, np.nan]]), np.zeros((1, 5)))])
    coefficient = axis.fit_weighted([(np.empty((0, 2)), np.empty((0, 5))), (np.ones((1, 2)), np.zeros((1, 5)))])
    assert coefficient.shape == (2, 5)


def _identifiable_shot(strength):
    signal = _signal(160)
    q = 127
    t = np.arange(len(signal["values"]), dtype=np.float64)
    signal["values"][:, 11] = np.sin(t * 0.23) + strength * np.clip((t - 102) / 25, 0, 1)
    design, issued, _, _ = axis.make_samples(signal, [q])
    assert issued[0]
    feature = design[0, axis.FEATURE_NAMES.index("n=1 amplitude:w2_change")]
    past = signal["values"][q - 125 : q + 1, 21]
    magnitude = np.std(past) * np.sinh(0.7 * feature)
    signal["values"][q + 1 : q + 26, 21] = np.repeat([-magnitude, 0, 0, 0, magnitude], 5)
    design, issued, target, mature = axis.make_samples(signal, [q])
    assert issued[0] and mature[0]
    return design, target


def test_synthetic_signed_target_is_identifiable_on_held_out_shots():
    train = [_identifiable_shot(a) for a in np.linspace(-4, 4, 41)]
    test = [_identifiable_shot(a) for a in np.linspace(-3.8, 3.8, 20)]
    coefficient = axis.fit_weighted(iter(train))
    x_test = np.concatenate([packet[0] for packet in test])
    y_test = np.concatenate([packet[1] for packet in test])
    predicted = axis.predict(x_test, coefficient)
    assert np.mean((predicted - y_test) ** 2) < 0.02
    assert np.mean(np.sign(predicted[:, -1]) == np.sign(y_test[:, -1])) >= 0.95
    np.testing.assert_allclose(predicted.sum(axis=1), 0, atol=1e-14)
