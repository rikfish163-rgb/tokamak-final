import numpy as np

from experiments.risk_aligned_online.commissioning_operating_mechanism_20261111 import controls


def test_feedback_controls_hold_exact_magnitude_or_delay():
    s = np.arange(77*20, dtype=np.float32).reshape(77, 20)
    a = controls(s)
    np.testing.assert_array_equal(a['real'], s)
    assert not a['lagged'][:32].any()
    np.testing.assert_array_equal(a['lagged'][32:], s[:-32])
    np.testing.assert_array_equal(a['sign_flip'], -s)
    np.testing.assert_array_equal(a['slope_flip'][:, ::2], s[:, ::2])
    np.testing.assert_array_equal(a['slope_flip'][:, 1::2], -s[:, 1::2])
    for k, v in controls(s[:45]).items():
        np.testing.assert_array_equal(v, a[k][:45])


def test_short_prefix_has_no_lagged_feedback():
    a = controls(np.ones((15, 20), np.float32))
    assert not a['lagged'].any()
    assert a['real'].all()
