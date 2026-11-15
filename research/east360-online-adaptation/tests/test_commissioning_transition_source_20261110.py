import numpy as np

from experiments.risk_aligned_online.commissioning_transition_source_20261110 import decision_points


def test_source_sampling_keeps_only_legal_points_with_two_mature_windows():
    y = np.zeros(280, bool)
    y[[180, 193, 194, 197, 200, 244, 269]] = True
    valid = np.arange(len(y)) < 250
    np.testing.assert_array_equal(decision_points(y, valid), [191, 193, 200, 239, 244])
