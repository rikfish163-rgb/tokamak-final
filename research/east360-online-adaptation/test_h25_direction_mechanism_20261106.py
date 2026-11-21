import numpy as np

from run_h25_direction_mechanism_20261106 import records_for, shared_grid


def test_original_protection_positive_and_safe_use_different_targets():
    z = np.arange(125, dtype=np.float64)
    m = np.array([63, 95])
    p, q = np.zeros((2, 32), bool), np.zeros((2, 32), bool)
    p[0, 14:21] = True
    q[0, :5] = True
    r = records_for(1, True, 68, None, z, m, p, q, False)
    assert [(x["role"], x["target"]) for x in r] == [("positive", 84), ("protect", 68)]
    r = records_for(2, False, -1, 63, z, m, p & False, q & False, False)
    assert [(x["role"], x["target"]) for x in r] == [("safe", 95)]


def test_common_scale_checks_both_h25_arms_and_both_signs_nonmonotonically():
    # Each action owns a pair of opposite signs; both H25 arms get the same step.
    delta = np.zeros((5, 2, 2, 3))
    delta[0, 1, 1, 2] = 0.11
    delta[1, 0, 0, 0] = 0.12
    assert shared_grid(delta, 0.1) == 2
    delta[2:, 0, 1, 1] = 0.12
    assert shared_grid(delta, 0.1) is None
