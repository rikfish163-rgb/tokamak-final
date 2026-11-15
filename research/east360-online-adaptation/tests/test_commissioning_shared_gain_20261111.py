import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_shared_gain_20261111 import (
    fit_window, gain_delta, replay,
)


def inputs():
    rng = np.random.default_rng(51)
    h = rng.normal(size=(300, 4)).astype(np.float32)
    q = np.array([127, 159, 191, 223, 255, 287])
    weights = rng.normal(size=(50, 4)).astype(np.float32) * .25
    forecast = rng.normal(size=(len(q), 25, 2)).astype(np.float32)
    jac = weights * np.tanh(h[q[0]])
    target = forecast.copy()
    target[0] += (jac @ np.array([.20, -.08, .10, .04])).reshape(25, 2)
    target[1:] += rng.normal(scale=.30, size=target[1:].shape)
    mask = np.ones_like(target, dtype=bool)
    mask[0, ::3, 1] = False
    return h, q, weights, forecast, target, mask


def test_observed_window_update_is_bounded_and_reduces_robust_error():
    rng = np.random.default_rng(13)
    jac = rng.normal(size=(80, 6))
    jac[:, -1] = 0.
    residual = jac @ np.array([.12, -.14, .10, .01, 0., 0.])
    u = fit_window(jac, residual, radius=.15)
    assert np.linalg.norm(u) == pytest.approx(.15)
    assert u[-1] == 0.
    assert np.linalg.norm(residual - jac @ u) < np.linalg.norm(residual)
    np.testing.assert_array_equal(fit_window(jac, 0.*residual, radius=.15), np.zeros(6))


def test_no_target_before_arrival_and_reset_for_each_shot():
    h, q, w, f, y, m = inputs()
    full, rec = replay(h, q, w, f, y, m)
    early, data = replay(h[:153], q[:1], w, f[:1], y[:1], m[:1])
    assert not data['release'].size
    assert not early.any()
    assert not full[:, :153].any()
    assert rec['release'][0] == 153
    assert full[:, 153].any()
    altered = y.copy()
    altered[1:] += 20.
    cf, _ = replay(h, q, w, f, altered, m)
    np.testing.assert_array_equal(full[:, :185], cf[:, :185])
    np.testing.assert_array_equal(replay(h, q, w, f, y, m)[0], full)


def test_order_control_preserves_dose_and_missing_targets_do_not_matter():
    h, q, w, f, y, m = inputs()
    state, data = replay(h, q, w, f, y, m)
    invalid = np.where(m, y, np.nan)
    np.testing.assert_array_equal(replay(h, q, w, f, invalid, m)[0], state)
    np.testing.assert_allclose(np.linalg.norm(state[0], axis=1),
                               np.linalg.norm(state[1], axis=1), atol=1e-7)
    assert not np.allclose(state[0], state[1])
    assert data['accepted_fit_error'][0] < data['before_fit_error'][0]


def test_same_state_changes_forecast_and_risk_feature_before_readout():
    h, q, w, f, y, m = inputs()
    state, data = replay(h, q, w, f, y, m)
    phi = np.tanh(h[q[1]])
    shifted = h[q[1]] + gain_delta(h[q[1]], state[0, q[1]])
    np.testing.assert_allclose(shifted - h[q[1]], phi * state[0, q[1]], atol=1e-7)
    assert not np.array_equal(shifted, h[q[1]])
    expected = f[1].ravel() + w @ (shifted - h[q[1]])
    np.testing.assert_allclose(data['issued_true'][1].ravel(), expected, atol=1e-6)
    np.testing.assert_array_equal(data['issued_true'][0], f[0])
