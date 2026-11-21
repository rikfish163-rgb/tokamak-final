import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.finite_conditional_loss_20261025.learning import (
    LossPolicy,feedback_context,outer_gradient,aggregate_objective,
)


def test_policy_capacity_masks_and_bounds():
    global_policy=LossPolicy('global',device='cpu');context=LossPolicy('context',device='cpu')
    assert sum(p.numel() for p in global_policy.parameters())==24
    assert sum(p.numel() for p in context.parameters())==47
    e=torch.linspace(-1,1,23,dtype=torch.float64);mask=torch.ones(23,dtype=torch.bool);mask[2]=False
    a=global_policy(e,mask);b=context(e,mask)
    torch.testing.assert_close(a,b);assert float(a.detach().sum())==pytest.approx(.5) and a[2]==0
    with torch.no_grad():context.slope[4]=1
    assert context(e,mask)[4]!=b[4]
    torch.testing.assert_close(global_policy(-e,mask),a)
    assert not context(e,torch.zeros_like(mask)).any()


def test_exact_outer_gradient_matches_full_autograd_nonlinear_loss():
    torch.manual_seed(17)
    policy=LossPolicy('context',device='cpu')
    parts=torch.randn(23,3,dtype=torch.float64);scale=parts.norm(dim=1).max()
    e=torch.randn(23,dtype=torch.float64).tanh();mask=torch.ones(23,dtype=torch.bool)
    origin=torch.tensor([.2,-.1,.3],dtype=torch.float64)
    coeff=policy(e,mask);theta=origin-.01*(coeff@parts)/scale
    # Nonlinear task; derivative must be evaluated AFTER the actual inner step.
    loss=torch.exp(theta[0]+2*theta[1])+theta[2]**4
    reference=torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(policy.parameters()),retain_graph=True)])
    risk,=torch.autograd.grad(loss,theta)
    computed=outer_gradient(policy,e,mask,parts,scale,risk)
    torch.testing.assert_close(computed,reference,atol=1e-14,rtol=1e-12)


def test_harm_cost_is_applied_after_equal_shot_block_averaging():
    rows=[]
    for group in ('safe','early','danger'):
        for sid,values in ((1,[-2.,4.]),(2,[-2.]*8)):
            rows.extend(dict(shot=sid,group=group,delta=v,gradient=np.array([1.,3.])) for v in values)
    loss,gradient,details=aggregate_objective(rows)
    # First shot mean+1 becomes+2; second mean-2 stays-2. Same mass per shot.
    assert loss==pytest.approx(0)
    np.testing.assert_allclose(gradient,[1.5,4.5])
    assert len(details)==6


def test_feedback_context_only_arrived_targets_and_finite_missing_mask():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model=small_model();fast=initial_encoder(model)
    signal=dict(values=np.random.default_rng(19).normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    signal['available'][:,2]=False
    c,observed=feedback_context(model,signal,256,fast,hold=False,device='cpu')
    signal['values'][256:]=1234.
    after,obs=feedback_context(model,signal,256,fast,hold=False,device='cpu')
    torch.testing.assert_close(c,after,rtol=0,atol=0);assert torch.equal(obs,observed)
    assert not observed[2] and c[2]==0 and torch.isfinite(c).all()
    fake,_=feedback_context(model,signal,256,fast,hold=True,device='cpu')
    assert not torch.equal(c,fake)
    with pytest.raises(ValueError):feedback_context(model,dict(signal,label=1),256,fast,hold=False,device='cpu')


def test_stream_causality_step_matching_and_parent_reset():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.finite_conditional_loss_20261025.learning import predict
    model=small_model();old={k:v.clone() for k,v in model.state_dict().items()}
    values=np.random.default_rng(23).normal(size=(400,23)).astype('float32')
    signal=dict(values=values,available=np.ones_like(values,bool))
    policies={k:LossPolicy(k,device='cpu').export() for k in ('global','context')}
    curves,updates=predict(model,signal,policies,np.ones(23)/23,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm][:256],curves['off'][:256])
    for arm,rows in updates.items():
        assert [r['first_affected_prediction'] for r in rows]==[256,384]
        assert all(r['actual_step_norm']<=.01001 for r in rows)
        if arm in ('uniform','permuted','hold'):
            np.testing.assert_allclose([r['requested_step_norm'] for r in rows],
                [r['main_requested_norm'] for r in rows],atol=1e-9,rtol=1e-6)
    changed={**signal,'values':values.copy()};changed['values'][330:]=101.
    later,_=predict(model,changed,policies,np.ones(23)/23,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm][:330],later[arm][:330])
    again,_=predict(model,signal,policies,np.ones(23)/23,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm],again[arm])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,old[k],rtol=0,atol=0)
