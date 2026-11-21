import numpy as np

from causal_risk_tangent_20261104 import remove_anchor_directions


def test_projection_preserves_norm_bound_and_past_first_order_scores():
    J = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g = np.array([1.0, 2.0, 3.0])
    d = remove_anchor_directions(J, g)
    np.testing.assert_allclose(d, [0.0, 0.0, 3.0], atol=1e-13)
    np.testing.assert_allclose(J @ d, 0, atol=1e-13)
    assert np.linalg.norm(d) <= np.linalg.norm(g)


def test_zero_and_rank_zero_preserved():
    a = np.zeros((4, 3))
    np.testing.assert_array_equal(
        remove_anchor_directions(a, np.array([1, 2, 3.0])), [1, 2, 3]
    )
    np.testing.assert_array_equal(remove_anchor_directions(a, np.zeros(3)), 0)
