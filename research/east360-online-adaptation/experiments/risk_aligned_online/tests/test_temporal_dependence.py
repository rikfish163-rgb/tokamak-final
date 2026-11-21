import numpy as np
import pytest

from experiments.risk_aligned_online.temporal_dependence_20261023.controls import flatten_window, causal_windows


def test_flatten_keeps_observed_medians_missing_values_and_other_columns():
    x=np.arange(2*5*23,dtype=np.float32).reshape(2,5,23)
    m=np.ones_like(x,dtype=bool);m[0,:2,12]=False;m[1,:,12]=False
    x[0,0,12]=np.nan;x[1,:,12]=-999
    original=x.copy();out=flatten_window(x,m,(12,))
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(out[:,:,np.arange(23)!=12],x[:,:,np.arange(23)!=12])
    np.testing.assert_array_equal(out[~m],x[~m])
    assert np.all(out[0,2:,12]==np.median(x[0,2:,12]))


def test_flatten_preserves_constants_and_is_idempotent():
    x=np.broadcast_to(np.arange(23,dtype=np.float32),(2,128,23)).copy()
    m=np.ones_like(x,dtype=bool)
    np.testing.assert_array_equal(flatten_window(x,m,tuple(range(23))),x)
    rng=np.random.default_rng(1);x=rng.normal(size=x.shape).astype(np.float32)
    y=flatten_window(x,m,tuple(range(23)))
    np.testing.assert_array_equal(flatten_window(y,m,tuple(range(23))),y)


def test_windowing_and_flatten_do_not_access_future():
    x=np.arange(350*23,dtype=np.float32).reshape(350,23);m=np.ones_like(x,dtype=bool)
    a,am=causal_windows(dict(values=x,available=m),128)
    assert a.shape==(350,128,23) and not am[0,:127].any()
    np.testing.assert_array_equal(a[200],x[73:201])
    changed=x.copy();changed[201:]=1e6
    b,bm=causal_windows(dict(values=changed,available=m),128)
    np.testing.assert_array_equal(flatten_window(a[:201],am[:201],(12,)),
                                  flatten_window(b[:201],bm[:201],(12,)))


def test_invalid_inputs_do_not_silently_produce_interventions():
    x=np.ones((1,128,23),np.float32);m=np.ones_like(x,dtype=bool)
    with pytest.raises(ValueError):flatten_window(x,m,(23,))
    with pytest.raises(ValueError):flatten_window(x,m,(12,12))
    x[0,0,1]=np.nan
    with pytest.raises(ValueError):flatten_window(x,m,(12,))
    with pytest.raises(ValueError):causal_windows(dict(values=x[0],available=m[0],outcome=1),128)
