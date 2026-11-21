import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.feedback_capacity_20261024.learning import (
    logistic_fit, linear_score, fit_normalization, prefix_statistics)


def test_regularized_separation_and_stationarity():
    x=np.array([[-2.],[-1.],[1.],[2.]])
    y=np.array([0.,0.,1.,1.]);w=np.ones(4)/4
    b,trace=logistic_fit(x,y,w,device='cpu',intercept=True)
    assert b[0]>0 and abs(b[1])<1e-10
    assert trace[-1]['gradient_max']<=1e-8
    assert all(a['objective']>=c['objective']-1e-14 for a,c in zip(trace,trace[1:]))
    bt=torch.tensor(b,requires_grad=True)
    xx=torch.tensor(np.c_[x,np.ones(4)])
    loss=torch.nn.functional.binary_cross_entropy_with_logits(xx@bt,torch.tensor(y))+.05*bt[0]**2
    assert torch.autograd.grad(loss,bt)[0].abs().max()<1e-8


def test_zero_feedback_is_exact_and_offset_is_used():
    x=np.zeros((4,3));y=np.array([0.,0.,1.,1.]);w=np.ones(4)/4
    offset=np.array([-.5,-.2,.2,.5])
    b,trace=logistic_fit(x,y,w,offset=offset,intercept=False,device='cpu')
    np.testing.assert_array_equal(b,np.zeros(3))
    np.testing.assert_array_equal(offset+linear_score(x,b,False),offset)
    expected=np.mean(np.logaddexp(0,offset)-y*offset)
    assert abs(trace[-1]['objective']-expected)<1e-12


def test_invalid_fit_rejected():
    for y,w in [(np.array([0.,2.]),np.array([.5,.5])),
                (np.array([0.,1.]),np.array([-.1,1.1]))]:
        with pytest.raises(ValueError):logistic_fit(np.ones((2,2)),y,w,device='cpu')


def test_normalization_masks_missing_and_does_not_center_delta():
    x=np.array([[1.,2.],[3.,2.]])
    d=np.ones((2,5,23));m=np.ones_like(d,dtype=bool);m[:,:,4]=False
    d[:,:,4]=np.nan
    n=fit_normalization(x,d,m,np.array([.25,.75]))
    np.testing.assert_allclose(n['mean'],[2.5,2.])
    np.testing.assert_allclose(n['dscale'],np.ones(23))
    assert np.isfinite(n['scale']).all()


def test_prefix_stats_do_not_read_after_endpoint_and_preserve_missingness():
    x=np.arange(150*23,dtype=np.float32).reshape(150,23)/100
    m=np.ones_like(x,dtype=bool);m[:,2]=False
    c=np.zeros(23);s=np.ones(23);ends=np.array([127,139])
    a=prefix_statistics({'values':x,'available':m},ends,c,s,device='cpu')
    xx=x.copy();xx[140:]=1e6
    b=prefix_statistics({'values':xx,'available':m},ends,c,s,device='cpu')
    np.testing.assert_array_equal(a,b)
    assert a.shape==(2,115)
    np.testing.assert_array_equal(a[:,np.arange(2,115,23)],0.)
    with pytest.raises(ValueError):prefix_statistics({'values':x,'available':m},np.array([126]),c,s,device='cpu')
