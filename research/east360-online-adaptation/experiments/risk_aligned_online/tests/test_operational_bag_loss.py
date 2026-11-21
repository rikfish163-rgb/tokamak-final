import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.operational_bag_loss_20261025.learning import (
    candidate_ends, bag_loss, aggregate, historical_gate,
)
from experiments.risk_aligned_online.finite_conditional_loss_20261025.learning import (
    LossPolicy, policy_step, outer_gradient,
)
from experiments.risk_aligned_online.history_tangent_20261023.learning import flat


def test_strict_operational_window_not_point_label_window():
    times=np.array([0.,3.833,3.970,3.971,4.,4.001])
    np.testing.assert_array_equal(candidate_ends(times,True,4.),[0,1])
    np.testing.assert_array_equal(candidate_ends(times,False,None),np.arange(6))
    with pytest.raises(ValueError):candidate_ends(np.array([3.971,4.]),True,4.)
    # The 167ms endpoint is timely regardless of its zero125ms point label.
    threshold=.12540414929389954
    z=torch.tensor([-3.,-1.,5.,6.,7.,8.],requires_grad=True)
    loss=bag_loss(z[candidate_ends(times,True,4.)].max(),True,threshold)
    loss.backward()
    assert z.grad[1]<0 and torch.count_nonzero(z.grad)==1


def test_margin_matches_fixed_probability_threshold_and_directions():
    threshold=.12540414929389954
    b=np.log(threshold/(1-threshold))
    for flag in (False,True):
        z=torch.tensor(b,requires_grad=True,dtype=torch.double)
        loss=bag_loss(z,flag,threshold)
        assert float(loss.detach())==pytest.approx(np.log(2))
        loss.backward();assert float(z.grad)==pytest.approx(-.5 if flag else .5)
    with pytest.raises(ValueError):bag_loss(torch.tensor(0.),True,1.)


def test_equal_shots_then_classes_and_asymmetric_harm():
    rows=[dict(shot=1,flag=True,delta=.2,gradient=np.array([1.])),
          dict(shot=2,flag=False,delta=-.4,gradient=np.array([3.])),
          dict(shot=3,flag=False,delta=.1,gradient=np.array([5.]))]
    value,g,_=aggregate(rows)
    assert value==pytest.approx(.5*.4+.5*(-.4+.2)/2)
    np.testing.assert_allclose(g,[.5*2+.25*(3+10)])
    with pytest.raises(ValueError):aggregate(rows[:1])
    with pytest.raises(ValueError):aggregate(rows+[rows[0]])


def test_finite_chain_and_static_prefix_no_credit():
    torch.manual_seed(131)
    parts=torch.randn(23,7,dtype=torch.double)
    c=torch.randn(23,dtype=torch.double);mask=torch.ones(23,dtype=torch.bool)
    policy=LossPolicy('context',device='cpu');theta=torch.randn(7,dtype=torch.double)
    step=policy_step(parts,policy(c,mask));q=torch.randn(7,dtype=torch.double)
    z=((theta-step)*q).sum().sin()
    ref=flat(torch.autograd.grad(bag_loss(z,True,.1254),tuple(policy.parameters())))
    changed=(theta-step.detach()).requires_grad_()
    risk=torch.autograd.grad(bag_loss((changed*q).sum().sin(),True,.1254),changed)[0]
    actual=outer_gradient(policy,c,mask,parts,parts.norm(dim=1).max(),risk)
    torch.testing.assert_close(actual,ref,atol=1e-12,rtol=1e-10)
    # A static prefix has no dependency on policy; its policy derivative is0.
    grad=outer_gradient(policy,c,mask,parts,parts.norm(dim=1).max(),torch.zeros_like(theta))
    assert torch.count_nonzero(grad)==0


def test_historical_gate_rejects_hidden_disruption_harm():
    rows=[dict(flag=True,delta=1e-8),dict(flag=True,delta=-1.),dict(flag=False,delta=-2.)]
    metrics=dict(tp=2,fp=0,very_early_over3000=0)
    assert not historical_gate(rows,metrics,metrics)
    rows[0]['delta']=0.
    assert historical_gate(rows,metrics,metrics)
    assert not historical_gate(rows,dict(metrics,very_early_over3000=1),metrics)


def test_runtime_causality_reset_norms_and_label_rejection():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.operational_bag_loss_20261025.learning import predict
    model=small_model();before={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(23).normal(size=(400,23)).astype('float32')
    signal=dict(values=x,available=np.ones_like(x,bool))
    policies={k:LossPolicy(k,device='cpu').export() for k in ('global','context')}
    old=policies['context'];curves,updates=predict(model,signal,policies,old,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm][:256],curves['off'][:256])
    for arm,rs in updates.items():
        assert [r['first_affected_prediction'] for r in rs]==[256,384]
        assert all(r['actual_step_norm']<=.01001 for r in rs)
        if arm in ('uniform','permuted','hold'):
            np.testing.assert_allclose([r['requested_step_norm'] for r in rs],[r['main_requested_norm'] for r in rs],atol=1e-9,rtol=1e-6)
    future=dict(signal,values=x.copy());future['values'][330:]=1234.
    changed,_=predict(model,future,policies,old,device='cpu')
    repeat,_=predict(model,signal,policies,old,device='cpu')
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:330],changed[arm][:330])
        np.testing.assert_array_equal(curves[arm],repeat[arm])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    with pytest.raises(ValueError):predict(model,dict(signal,is_disrupt=True),policies,old,device='cpu')
