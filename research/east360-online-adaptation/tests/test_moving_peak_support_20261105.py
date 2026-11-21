import numpy as np
import pytest

from moving_peak_support_20261105 import nearest_rms


def test_nearest_feature_uses_only_given_support_and_fixed_scale():
    support = np.array([[0.0, 0.0], [2.0, 2.0]])
    scale = np.array([1.0, 2.0])
    distance, j = nearest_rms(np.array([2.0, 4.0]), support, scale)
    assert j == 1 and distance == pytest.approx(1 / np.sqrt(2))
    assert nearest_rms(support[0], support, scale) == (0.0, 0)


def test_malformed_support_is_not_silently_used():
    with pytest.raises(ValueError):
        nearest_rms(np.zeros(2), np.empty((0, 2)), np.ones(2))
    with pytest.raises(ValueError):
        nearest_rms(np.zeros(2), np.ones((2, 2)), np.array([1.0, 0.0]))
