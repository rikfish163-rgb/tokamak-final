import numpy as np

from run_h25_increment_mechanism_20261106 import (
    cosine_summary,
    four_spaces,
    paired_shot_effect,
)


def test_common_rank_across_both_losses_and_both_feedback_orders():
    e = np.eye(8)
    qs, rank = four_spaces(e[:2], e[2:5], e[3:6], e[5:7], e[4:5])
    assert rank == 1
    for q in qs:
        assert q.shape == (8, 3)
        np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(q[:, :2] @ q[:, :2].T, e[:2].T @ e[:2])


def test_paired_bootstrap_weights_shots_and_not_seed_window_repeats():
    values = np.array([[1, 1], [1, 1], [-2, -2]])
    result = paired_shot_effect(values, [7, 7, 9])
    assert result["mean"] == -0.5 and result["n_shots"] == 2
    assert result["resample_range"] == [-2, 1]


def test_order_cosine_is_conditioned_on_common_nonzero_families():
    u = np.ones((1, 3, 2))
    v = np.array([[[-1, -1], [0, 0], [1, 1]]])
    found = cosine_summary(u, v, u, u, np.array([[True, True, False]]))
    assert found["n_vectors"] == 1
    assert abs(found["median_level"] + 1) < 1e-12
    assert abs(found["median_increment"] - 1) < 1e-12
