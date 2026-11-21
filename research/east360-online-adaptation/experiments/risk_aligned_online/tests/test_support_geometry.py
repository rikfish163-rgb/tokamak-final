import numpy as np

from experiments.risk_aligned_online.support_geometry_20261023.analysis import (
    descriptor, covariance, nearest_sq, boundary_shifts,
)
from experiments.risk_aligned_online.simple_data import window


def test_descriptor_masks_and_future_independence():
    s=dict(values=np.ones((500,23),np.float32),available=np.ones((500,23),bool))
    s['available'][:,2]=False;s['values'][:,2]=999.
    c=np.zeros(23);scale=np.ones(23)
    x,m=window(s,63,128);a=descriptor(x,m,c,scale)
    assert a.shape==(69,) and np.isfinite(a).all()
    assert a[2]==a[25]==a[48]==0 and a[46]==.5
    s['values'][64:]=777.
    xx,mm=window(s,63,128)
    np.testing.assert_array_equal(a,descriptor(xx,mm,c,scale))


def test_covariance_and_nearest_match_scalar_calculation():
    x=np.array([[1.,0.],[2.,1.],[5.,4.]])
    w=np.array([.25,.25,.5]);mu=(x*w[:,None]).sum(0)
    expected=((x-mu)*w[:,None]).T@(x-mu)
    expected=.5*expected+.5*np.eye(2)*np.trace(expected)/2
    center,cov=covariance(x,w)
    np.testing.assert_allclose(center,mu);np.testing.assert_allclose(cov,expected)
    q=np.array([[0.,0.],[2.,3.]])
    d=nearest_sq(q,x,cov)
    manual=np.array([min((v-r)@np.linalg.solve(cov,v-r) for r in x) for v in q])
    np.testing.assert_allclose(d,manual)
    _,constant=covariance(np.ones((3,2)),np.ones(3)/3)
    assert np.linalg.eigvalsh(constant).min()>0


def test_boundary_probe_changes_only_observed_target_channel():
    x=np.zeros((128,23),np.float32);x[:,0]=np.sinh(3.)
    m=np.ones_like(x,dtype=bool);m[:10,0]=False;m[:,2]=False
    low=np.full(23,-1.);high=np.full(23,1.);low[2]=high[2]=np.nan
    probes,shift=boundary_shifts(x,m,low,high,np.zeros(23),np.ones(23))
    assert probes.shape==(24,128,23) and np.isclose(shift[0],-2,atol=1e-6) and np.count_nonzero(shift)==1
    np.testing.assert_array_equal(probes[0,:,1:],x[:,1:])
    np.testing.assert_array_equal(probes[0,:10,0],x[:10,0])
    np.testing.assert_allclose(probes[0,10:,0],np.sinh(1.),rtol=1e-6)
    np.testing.assert_array_equal(probes[1],x)
    np.testing.assert_array_equal(probes[-1],probes[0])
