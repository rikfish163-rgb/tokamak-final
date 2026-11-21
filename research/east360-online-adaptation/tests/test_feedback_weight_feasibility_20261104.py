import numpy as np
import pytest

from experiments.risk_aligned_online.mask_separated_intra_20261103.weight_feasibility import (
    bounded_minimax,
)


def test_minimax_respects_nonnegative_bounded_common_channel_budget():
    utilities = np.array([[1.0, -2.0, 999.0]] * 3)
    result = bounded_minimax(utilities, np.array([True, True, False]))
    np.testing.assert_allclose(result["weights"], [0, 2, 0])
    np.testing.assert_allclose(result["weighted_group_estimates"], [-4] * 3)
    assert result["worst_group_estimate"] == pytest.approx(-4)


@pytest.mark.parametrize("bad", ["empty", "nan", "shape"])
def test_minimax_fails_on_unsupported_or_invalid_analysis(bad):
    matrix, active = np.zeros((3, 2)), np.ones(2, dtype=bool)
    if bad == "empty":
        active[:] = False
    elif bad == "nan":
        matrix[0, 0] = np.nan
    else:
        matrix = np.zeros((2, 2))
    with pytest.raises(ValueError):
        bounded_minimax(matrix, active)
