import numpy as np

from within_shot_motion_calibration_20261106 import prequential, matched_context


def test_no_update_before_the_forecast_completes():
    origins = np.array([0, 10, 26, 52])
    hold = np.zeros((4, 25, 23))
    x = np.ones((*hold.shape, 1))
    y = np.ones_like(hold)
    m = np.ones_like(hold, dtype=bool)
    a, used = prequential(origins, hold, x, y, m)
    np.testing.assert_array_equal(used, [0, 0, 1, 3])
    assert (a[:2] == 0).all() and a[2, 0, 0] > 0.9
    y[1] = -30  # the second block only completes after origin 26
    b, _ = prequential(origins, hold, x, y, m)
    np.testing.assert_array_equal(a[:3], b[:3])
    assert b[3, 0, 0] != a[3, 0, 0]


def test_norm_matching_uses_older_direction_and_common_nonzero_support():
    raw = np.zeros((4, 25, 23))
    raw[1:, 0, 3] = 1
    raw[2:, 1, 3] = 2
    raw[3, 0, 4] = 5
    true, lag = matched_context(raw)
    assert not lag[:2].any() and not true[:2].any()
    assert true[2, 1, 3] == lag[2, 0, 3] == 2
    assert not true[3, :, 4].any()
    np.testing.assert_allclose(
        np.linalg.norm(true, axis=1), np.linalg.norm(lag, axis=1)
    )
