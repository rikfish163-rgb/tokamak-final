import numpy as np
import pytest
from experiments.risk_aligned_online.local_transition_20261023.controls import fit_scale


def test_fixed_scale_closed_form_and_invalid_reference():
    u=np.arange(1,65,dtype=float).reshape(4,16)
    assert fit_scale(u,.37*u)==pytest.approx(.37,abs=1e-14)
    assert fit_scale(u,2*u)==1.
    assert fit_scale(u,-u)==0.
    with pytest.raises(ValueError):fit_scale(u,np.ones((2,16)))
    with pytest.raises(ValueError):fit_scale(u,u*np.nan)
    with pytest.raises(ValueError):fit_scale(np.zeros_like(u),u)
