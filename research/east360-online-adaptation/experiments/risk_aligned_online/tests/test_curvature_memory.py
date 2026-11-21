import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.curvature_memory_20261024.learning import (
    bernoulli_rows, solve_directions, functional_match,
)


def test_bernoulli_curvature_is_expected_fisher_not_observed_error_square():
    j=torch.tensor([[1.,2.],[3.,-1.]],dtype=torch.float64)
    p=torch.tensor([.2,.8],dtype=torch.float64);q=torch.tensor([.4,.6],dtype=torch.float64)
    u=bernoulli_rows(j,p,q)
    expected=sum(q[i]*(p[i]*(p[i]-1)**2+(1-p[i])*p[i]**2)*torch.outer(j[i],j[i]) for i in range(2))
    torch.testing.assert_close(u.T@u,expected,atol=1e-14,rtol=1e-14)
    empirical=sum(q[i]*p[i]**2*torch.outer(j[i],j[i]) for i in range(2))
    assert not torch.allclose(expected,empirical)


def test_dual_solve_matches_dense_and_damping_condition_bound():
    torch.manual_seed(3)
    u=torch.randn(5,11,dtype=torch.float64);g=torch.randn(4,11,dtype=torch.float64)
    full,diagonal,info=solve_directions(g,u)
    f=u.T@u;identity=torch.eye(11,dtype=torch.float64)
    expected=torch.linalg.solve(f+info['damping']*identity,g.T).T
    torch.testing.assert_close(full,expected,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(diagonal,g/(f.diag()+info['damping']),atol=1e-12,rtol=1e-12)
    eigen=torch.linalg.eigvalsh(f+info['damping']*identity)
    assert float(eigen[-1]/eigen[0])<=101.000000001
    with pytest.raises(ValueError):solve_directions(g,u*0)


def test_common_sensitive_direction_is_suppressed_not_merely_rescaled():
    u=torch.tensor([[10.,0.],[0.,1.]],dtype=torch.float64)
    g=torch.tensor([[1.,1.]],dtype=torch.float64)
    full,_,_=solve_directions(g,u)
    assert full[0,1]/full[0,0]>40
    raw_length=.01
    matched=functional_match(g[0],full[0],u,raw_length)
    assert 0<matched<raw_length
    a=g[0]/g[0].norm()*matched;b=full[0]/full[0].norm()*raw_length
    torch.testing.assert_close((u@a).square().sum(),(u@b).square().sum(),atol=1e-15,rtol=1e-12)


def test_nonfinite_geometry_and_unidentified_zero_update_fail_explicitly():
    j=torch.eye(3,dtype=torch.float64);p=torch.full((3,),.5,dtype=torch.float64);q=torch.ones(3,dtype=torch.float64)/3
    with pytest.raises(ValueError):bernoulli_rows(j,p*3,q)
    with pytest.raises(ValueError):bernoulli_rows(j,p,q*2)
    with pytest.raises(ValueError):solve_directions(j,j*float('nan'))
    assert functional_match(j[0]*0,j[0]*0,j,.01)==0.
    with pytest.raises(ValueError):functional_match(j[0]*0,j[0],j,.01)


def test_linear_transform_commutes_with_historical_mixture():
    torch.manual_seed(5);u=torch.randn(7,15,dtype=torch.float64);g=torch.randn(9,15,dtype=torch.float64)
    w=torch.rand(9,dtype=torch.float64);w/=w.sum()
    bank,_,_=solve_directions(g,u);direct,_,_=solve_directions((w@g)[None],u)
    torch.testing.assert_close(w@bank,direct[0],rtol=1e-12,atol=1e-12)


def test_updated_stream_has_no_future_influence_and_parent_is_immutable():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.trajectory_memory_20261024.learning import make_bank
    from experiments.risk_aligned_online.curvature_memory_20261024.learning import predict
    model=small_model();rng=np.random.default_rng(19);history=[]
    before={k:v.clone() for k,v in model.state_dict().items()}
    for sid,flag in enumerate([False,False,True,True]):
        x=rng.normal(size=(300,23)).astype('float32')
        history.append(dict(shot=sid,values=x,available=np.ones_like(x,bool),times=np.arange(300)/1000,
                            valid=np.ones(300,bool),is_disrupt=flag,down=.3 if flag else None))
    bank=make_bank(model,history,device='cpu')
    # Synthetic SPD geometry isolates runtime causality. Bernoulli construction
    # and dense-solve correctness are tested separately above.
    torch.manual_seed(9);u=torch.randn(8,bank['gradients'].shape[1],dtype=torch.float64)
    full,diagonal,_=solve_directions(bank['gradients'],u)
    bank.update(u=u.float(),full=full.float(),diagonal=diagonal.float())
    x=rng.normal(size=(300,23)).astype('float32');signal=dict(values=x,available=np.ones_like(x,bool))
    a,ra=predict(model,signal,bank,device='cpu');b,rb=predict(model,signal,bank,device='cpu')
    assert ra==rb
    for arm in a:np.testing.assert_array_equal(a[arm],b[arm])
    x[256:]=1e4;c,rc=predict(model,signal,bank,device='cpu')
    assert ra==rc
    for arm in a:
        np.testing.assert_array_equal(a[arm][:256],c[arm][:256])
        np.testing.assert_array_equal(a[arm][:128],a['off'][:128])
    assert any(not np.array_equal(a[arm],a['off']) for arm in a if arm!='off')
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(signal,y=1),bank,device='cpu')


def test_full_stream_gate_requires_functional_control_advantage():
    from experiments.risk_aligned_online.curvature_memory_20261024.run import eligible
    from experiments.risk_aligned_online.curvature_memory_20261024.learning import ARMS
    rows={a:dict(tpr=None,fpr=0.,J=None,n_safe=1,n_disrupt=0) for a in ('off',)+ARMS}
    assert not eligible(rows)
    rows={a:dict(tp=126,fp=36,tpr=.7,fpr=.2,J=.5,n_safe=180,n_disrupt=180) for a in rows}
    rows['full'].update(tp=144,tpr=.8,J=.6)
    assert eligible(rows)
    rows['raw_kl'].update(tp=144,tpr=.8,J=.6)
    assert not eligible(rows)


def test_equal_count_scores_cannot_pass_due_to_float_roundoff():
    from experiments.risk_aligned_online.curvature_memory_20261024.run import eligible
    from experiments.risk_aligned_online.curvature_memory_20261024.learning import ARMS
    rows={a:dict(tp=132,fp=36,tpr=132/180,fpr=.2,J=96/180,n_safe=180,n_disrupt=180)
          for a in ('off',)+ARMS}
    rows['full']['J']=float(np.nextafter(rows['full']['J'],np.inf))
    assert not eligible(rows)
