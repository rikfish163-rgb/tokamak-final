import numpy as np
import pytest

from experiments.risk_aligned_online.lora_stream_20261102.feedback_audit import paired


def test_identical_gradient_has_no_target_sensitivity():
    assert paired(np.array([1.0, 0]), np.array([1.0, 0])) == dict(
        cosine=1.0, relative_difference=0.0
    )


def test_orthogonal_gradient_difference_is_not_an_additive_loss_decomposition():
    result = paired(np.array([1.0, 0]), np.array([0.0, 1]))
    assert result["cosine"] == 0
    assert result["relative_difference"] == pytest.approx(np.sqrt(2))


def test_zero_reference_is_explicitly_undefined():
    assert paired(np.zeros(2), np.ones(2)) == dict(
        cosine=None, relative_difference=None
    )
    assert paired(np.ones(2), np.zeros(2))["cosine"] is None


def test_invalid_vector_rejected():
    with pytest.raises(ValueError):
        paired(np.array([float("nan")]), np.zeros(1))
