import numpy as np

from run_output_trust_mix_capacity_20261106 import (
    orthogonal_columns,
    projected,
    optimize,
)


def test_projection_preserves_reached_linear_outputs_and_stays_in_bank():
    q = orthogonal_columns(np.eye(5)[:3])
    reached = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=float)
    g = np.array([1, 1, 2, 0, 5], dtype=float)
    step = q @ projected(q, reached, g)
    np.testing.assert_allclose(reached @ step, np.zeros(2), atol=1e-10)
    assert step[2] > 0 and step[3] == step[4] == 0


def test_continuous_mix_obeys_both_signs_and_radius_in_linear_fixture():
    # Reach costs x in both signs, future benefits x+y. A mixed direction can
    # retain the future effect without violating the observed-output box.
    matrix = np.array(
        [[1, 0], [1, 0], [0, 0], [1, 1], [-1, 0], [-1, 0], [0, 0]], dtype=float
    )
    weights, _ = optimize(lambda w: (matrix @ w, matrix), np.zeros(3), np.zeros(2))
    assert weights[0] <= 0.1000001
    assert np.linalg.norm(weights) <= 0.3500001
    assert weights.sum() > 0.43
