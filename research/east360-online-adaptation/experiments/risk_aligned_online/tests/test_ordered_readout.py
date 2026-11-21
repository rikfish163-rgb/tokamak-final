import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from experiments.risk_aligned_online.ordered_readout_20261025.learning import (
    decompose, minnorm, accepted, group_values,
)


def test_readout_components_are_orthogonal_and_shape_rejects_common_offset():
    torch.manual_seed(12)
    x=torch.randn(7,92,dtype=torch.double)
    a=decompose(x,'shape');b=decompose(x,'level')
    torch.testing.assert_close(a+b,x,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close((a*b).sum(-1),torch.zeros(7,dtype=torch.double),atol=1e-13,rtol=0)
    offset=torch.randn(7,1,23,dtype=torch.double).expand(7,4,23).reshape(7,92)
    torch.testing.assert_close(decompose(x+offset,'shape'),a,atol=1e-14,rtol=1e-14)
    assert not decompose(offset,'shape').abs().max()>1e-14
    reverse=x.reshape(7,4,23).flip(1).reshape(7,92)
    torch.testing.assert_close(decompose(reverse,'level'),b,atol=1e-14,rtol=1e-14)
    with pytest.raises(ValueError):decompose(x,'unknown')


def test_minimum_norm_solver_known_solutions_and_stationarity():
    g=np.eye(5)
    a,info=minnorm(g)
    np.testing.assert_allclose(a,np.ones(5)/5,atol=1e-12,rtol=0)
    assert info['minimum_directional_derivative']>0
    a,info=minnorm(np.array([[1.,0.],[-1.,0.],[0.,2.]]))
    np.testing.assert_allclose(a@np.array([[1.,0.],[-1.,0.],[0.,2.]]),0,atol=1e-12)
    assert info['norm_squared']<1e-20
    a,info=minnorm(np.zeros((5,8)))
    assert info['norm_squared']==0 and np.isclose(a.sum(),1)
    with pytest.raises(ValueError):minnorm(np.array([[np.nan]]))


def test_enumeration_matches_independent_constrained_optimizer():
    rng=np.random.default_rng(131)
    for _ in range(12):
        g=rng.normal(size=(5,12));a,info=minnorm(g);q=g@g.T
        other=minimize(lambda x:x@q@x,np.ones(5)/5,jac=lambda x:2*q@x,
            bounds=[(0,1)]*5,constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)},
            method='SLSQP',options={'ftol':1e-13,'maxiter':500})
        assert other.success
        assert abs(a@q@a-other.fun)<1e-9
        assert info['kkt_gap']<1e-9


def test_guard_never_accepts_average_improvement_that_harms_one_group():
    old=np.ones(5)
    new=np.array([.9,.9,.9,.9,1.01])
    assert accepted(old,new,'mean')
    assert not accepted(old,new,'guard')
    assert accepted(old,np.array([.9,1,1,1,.99]),'guard')
    assert not accepted(old,old,'guard')
    with pytest.raises(ValueError):accepted(old,np.array([1,1,np.nan,1,1]),'mean')


def test_grouping_protects_each_disruptive_identity_and_balances_safe_shots():
    records=[dict(shot=4,flag=True,loss=4.),dict(shot=1,flag=True,loss=1.),
             dict(shot=2,flag=False,loss=2.),dict(shot=3,flag=False,loss=6.)]
    ids,values=group_values(records)
    assert ids==[1,4,'safe_mean']
    np.testing.assert_array_equal(values,[1,4,4])
