import numpy as np

from adjacent_patch_compare_20261106 import (
    common_spaces,
    common_step,
    equal_rms_factors,
    future_blocks,
)


def test_equal_rank_never_grants_more_directions_to_a_control():
    e = np.eye(8)
    bank = np.stack((e[:6], e[2:], e[:6], np.vstack((e[:4], e[:2]))))
    q, n = common_spaces(bank)
    assert n == 4 and len(q) == 4
    for v in q:
        np.testing.assert_allclose(v.T @ v, np.eye(n), atol=1e-12)


def test_induced_rms_is_matched_without_amplifying_a_small_or_zero_arm():
    v = np.array([[1.0, 2.0, 4.0, 3.0], [1.0, 2.0, 4.0, 0.0]])
    f, good = equal_rms_factors(v)
    np.testing.assert_allclose(f * v, [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(good, [True, False])


def test_risk_guard_uses_both_signs_all_arms_and_does_not_assume_monotonicity():
    d = np.zeros((3, 4, 5, 2, 3))
    d[0, 3, 0, 1, 2] = 0.101
    d[0, 1, 2, 0, 0] = 0.2
    d[1, 2, :, 1, 1] = -0.11
    np.testing.assert_array_equal(
        common_step(d, np.array([True, True, False])), [1, -1, -1]
    )


def test_next_block_excludes_maturity_and_never_rewrites_a_previous_block():
    blocks = future_blocks(np.array([63, 95, 127]), 130)
    assert blocks[0].tolist() == list(range(64, 96))
    assert blocks[1].tolist() == list(range(96, 128))
    assert blocks[2].tolist() == [128, 129]
