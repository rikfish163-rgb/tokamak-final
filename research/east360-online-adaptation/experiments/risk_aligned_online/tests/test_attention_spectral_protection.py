import numpy as np
import pytest

from experiments.risk_aligned_online.attention_spectral_20261029.protection import safe_weights


def test_cannot_compensate_safe_harm_with_disruptive_gain():
    gains=np.array([[2.,-1.],[-1.,2.]])
    w,info=safe_weights(gains,np.array([True,False]),np.array([0,0]))
    # Safe constraint requires w1>=w0/2; maximizing2*w0-w1 gives(1,.5).
    np.testing.assert_allclose(w,[1,.5],atol=1e-8)
    assert (gains@w>=-1e-9).all() and np.isclose(info['objective'],1.5)


def test_zero_is_reported_when_no_direction_can_avoid_harm():
    gains=np.array([[1.,2.],[-1.,-2.]])
    w,info=safe_weights(gains,np.array([True,False]),np.array([0,1]))
    np.testing.assert_allclose(w,0,atol=1e-9)
    assert info['objective']==0


def test_trace_budget_and_input_validation():
    gains=np.ones((2,8));groups=np.repeat([0,1],4);flags=np.array([True,False])
    w,_=safe_weights(gains,flags,groups)
    assert np.isclose(w.sum(),4) and (w>=0).all() and (w<=1).all()
    for group in [0,1]:assert w[groups==group].sum()<=2+1e-9
    with pytest.raises(ValueError):safe_weights(gains,np.ones(2,bool),groups)
    with pytest.raises(ValueError):safe_weights(gains*np.nan,flags,groups)
