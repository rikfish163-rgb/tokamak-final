import numpy as np

from point_confirmation_20261105 import confirm, gated


def test_causal_confirmation_never_uses_later_points():
    shadow = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert confirm(shadow, 0.5, 2).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    later = shadow.copy()
    later[5:] = 1.0
    assert np.array_equal(confirm(shadow, 0.5, 2)[:5], confirm(later, 0.5, 2)[:5])


def test_gated_prediction_keeps_every_static_crossing_and_requires_same_direction():
    base = np.array([0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
    active = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    shadow = np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1])
    same = gated(base, active, shadow, 0.5, 0)
    delayed = gated(base, active, shadow, 0.5, 2)
    np.testing.assert_array_equal(same, [0.8, 0.1, 0.9, 0.1, 0.1, 0.1])
    np.testing.assert_array_equal(delayed, [0.8, 0.1, 0.9, 0.9, 0.9, 0.1])


def test_equal_length_timed_trace_and_nonnegative_lookback_required():
    import pytest

    with pytest.raises(ValueError, match="length"):
        gated(np.zeros(2), np.zeros(3), np.zeros(2), 0.5, 0)
    with pytest.raises(ValueError, match="lookback"):
        confirm(np.zeros(2), 0.5, -1)
