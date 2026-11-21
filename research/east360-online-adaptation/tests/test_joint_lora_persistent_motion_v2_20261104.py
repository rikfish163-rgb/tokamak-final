import numpy as np
import pytest

from joint_lora_persistent_motion_v2_20261104 import verify_historical_view


def test_historical_check_does_not_compare_the_analytic_mapped_view():
    original = np.array([0.89, 0.40])
    historical = np.array([0.8900001, 0.4000001])
    mapped = np.array([0.94, 0.55])
    gaps = verify_historical_view(original, historical, mapped)
    assert gaps["historical"] < 3e-6
    assert gaps["analysis_vs_historical"] > 0.1
    with pytest.raises(ValueError, match="historical"):
        verify_historical_view(original, mapped, historical)
