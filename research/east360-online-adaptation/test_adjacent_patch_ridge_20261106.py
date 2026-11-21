import numpy as np

from run_adjacent_patch_ridge_20261106 import ridge_directions


def test_linear_feedback_does_not_amplify_null_or_historical_coordinates():
    j = np.array([[1.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]])
    p = np.diag([1.0, 1.0, 0.0])
    r = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, 4.0]])
    u = ridge_directions(j, r, p)
    np.testing.assert_allclose(u[0], -u[1])
    np.testing.assert_allclose(u[:, 2], 0.0)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 0.35)


def test_unobservable_action_is_exact_no_write():
    assert not ridge_directions(np.zeros((2, 3)), np.ones((2, 2)), np.eye(3)).any()
