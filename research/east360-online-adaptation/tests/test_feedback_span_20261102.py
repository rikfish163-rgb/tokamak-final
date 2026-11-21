import numpy as np

from experiments.risk_aligned_online.risk_conditioned_feedback_20261102.span_audit import (
    cone_projection,
)


def test_nonnegative_cone_projection_is_not_the_signed_span():
    gradients = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    target = np.array([-1.0, 2.0, 3.0])
    result = cone_projection(gradients, target)
    np.testing.assert_allclose(result["cosine"], 2 / np.sqrt(14), atol=1e-12)
    np.testing.assert_allclose(
        result["normalized_projection"], [0, 2 / np.sqrt(14), 0], atol=1e-12
    )
    assert result["nonzero_channels"] == 2
    assert result["kkt_residual"] < 1e-10


def test_empty_and_polar_cones_report_no_nonzero_direction():
    for g, r in [(np.zeros((2, 3)), np.ones(3)), (np.eye(3), -np.ones(3))]:
        result = cone_projection(g, r)
        assert result["cosine"] == 0
        assert not result["direction_defined"]


def test_positive_channel_and_task_scaling_preserves_cone():
    g = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [1.0, 1.0, -0.2]])
    r = np.array([0.1, 1.0, -0.4])
    first = cone_projection(g, r)
    second = cone_projection(g * np.array([0.01, 20.0, 3.0])[:, None], r * 10)
    np.testing.assert_allclose(first["cosine"], second["cosine"], atol=1e-10)
