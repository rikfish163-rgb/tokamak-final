import numpy as np
import pytest

from ordered_motion_shrink_20261106 import causal, oracle_scale


def test_lag_uses_only_fully_matured_targets_at_its_clock():
    origins = np.array([0, 10, 20, 25, 30, 45, 55, 85])
    pred = np.ones((len(origins), 1, 1, 1))
    target = np.full(pred[:, 0].shape, 0.4)
    mask = np.ones_like(target, dtype=bool)
    step, count = causal(origins, pred, target, mask)
    np.testing.assert_array_equal(count, [0, 0, 0, 1, 1, 3, 5, 7])
    np.testing.assert_allclose(step[:, 0], [0, 0, 0, 0, 0, 0.4, 0.4, 0.4])
    target[2:] *= -1
    altered, _ = causal(origins, pred, target, mask)
    np.testing.assert_array_equal(step[:5], altered[:5])
    assert altered[5] < step[5]


def test_each_arm_has_its_own_past_and_support():
    origins = np.arange(7) * 32
    target = np.ones((7, 2, 1))
    pred = np.stack((2 * target, -2 * target, 0.1 * target), axis=1)
    mask = np.ones_like(target, dtype=bool)
    mask[1] = False
    mask[:, 1] = False
    target[~mask] = np.nan
    step, count = causal(origins, pred, target, mask)
    np.testing.assert_array_equal(count, [0, 1, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(step[4:], np.tile([0.5, 0.0, 1.0], (3, 1)))
    with pytest.raises(ValueError, match="clock"):
        causal(origins[::-1], pred, target, mask)


def test_oracle_solves_convex_huber_and_never_uses_missing_cells():
    target = np.array([[[-0.3], [-8.0]], [[0.3], [np.nan]], [[1.0], [1.0]]])
    mask = np.array([[[True], [False]], [[True], [False]], [[False], [False]]])
    start = np.array([[[-1.0], [np.nan]], [[0.0], [np.inf]], [[0.0], [0.0]]])
    delta = np.array([[[2.0], [np.nan]], [[0.0], [np.inf]], [[0.0], [0.0]]])
    assert oracle_scale(start, delta, target, mask) == pytest.approx(0.35)
    assert oracle_scale(start, -delta, target, mask) == 0.0
    assert oracle_scale(start, 0.2 * delta, target, mask) == 1.0


def test_oracle_gives_supported_windows_equal_weight_not_cells():
    y = np.array([[[0.0], [0.0]], [[1.0], [1.0]]])
    mask = np.array([[[True], [False]], [[True], [True]]])
    assert oracle_scale(np.zeros_like(y), np.ones_like(y), y, mask) == pytest.approx(
        0.5
    )
