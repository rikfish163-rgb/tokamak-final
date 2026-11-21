import numpy as np
import pytest
from scipy.special import expit, logit

from layer0_direction_scalar_20261106 import scalar_shadow


def test_scalar_control_matches_anchor_and_only_uses_anchor_difference():
    a = np.linspace(0.2, 0.5, 10)
    b = expit(logit(a) + 0.3)
    shadow = scalar_shadow(a, b, first=3, stride=2)
    b[4::2] = 0.99  # not an anchor; cannot change the control
    np.testing.assert_array_equal(shadow, scalar_shadow(a, b, first=3, stride=2))
    np.testing.assert_array_equal(shadow[:3], a[:3])
    np.testing.assert_array_equal(shadow[3::2], b[3::2])
    np.testing.assert_allclose(logit(shadow[3:]) - logit(a[3:]), 0.3)


def test_saturation_is_a_mechanism_limit_not_a_tuned_clipping_rule():
    a = np.array([0.3, 0.4, 0.5, 0.6])
    b = a.copy()
    b[2] = 1.0
    with pytest.raises(ValueError):
        scalar_shadow(a, b, first=2, stride=2)
