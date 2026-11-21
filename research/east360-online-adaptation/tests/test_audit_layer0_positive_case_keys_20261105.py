import numpy as np

from audit_layer0_positive_case_keys_20261105 import key_match


def test_match_reports_one_coherent_nearest_key_and_three_distances():
    key = np.zeros(192)
    targets = np.ones((2, 192))
    targets[1, :64] = 0.0
    result = key_match(key, targets, 1.0)
    assert result["index"] == 1
    np.testing.assert_allclose(result["block_rms"], [0, 1, 1])
    assert abs(result["weight"] - (1 - (2 / 3) / 4) ** 2) < 1e-12
