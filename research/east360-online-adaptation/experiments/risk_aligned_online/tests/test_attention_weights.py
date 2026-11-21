import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_weights_20261029.learning import weighted_state,operational_objective,validate_fit_ids


def test_weighted_state_matches_uniform_and_zero_feedback():
    torch.manual_seed(3)
    g=torch.randn(4,23,512,dtype=torch.float64);w=torch.zeros(23,dtype=torch.float64)
    s=weighted_state(g,w)
    expected=-.05*g.sum(1)/g.sum(1).norm(dim=1,keepdim=True)
    torch.testing.assert_close(s,expected)
    assert torch.equal(weighted_state(torch.zeros_like(g),w),torch.zeros(4,512,dtype=torch.float64))
    with pytest.raises(ValueError):weighted_state(g,w[:22])


def test_weight_gradient_is_actual_finite_update_gradient():
    torch.manual_seed(4)
    g=torch.randn(2,23,5,dtype=torch.float64);w=torch.randn(23,dtype=torch.float64,requires_grad=True)
    target=torch.randn(2,5,dtype=torch.float64)
    f=lambda v:((weighted_state(g,v)-target)**2).sum()
    grad,=torch.autograd.grad(f(w),w)
    eps=1e-5
    for i in (0,6,12,22):
        d=torch.zeros_like(w);d[i]=eps
        torch.testing.assert_close(grad[i],(f(w+d)-f(w-d))/(2*eps),atol=1e-8,rtol=1e-5)


def test_operational_objective_weights_shots_not_windows_and_penalizes_harm():
    peak=torch.tensor([1.,-1.,1.],dtype=torch.float64,requires_grad=True)
    before=torch.zeros(3,dtype=torch.float64);flag=torch.tensor([True,False,False])
    objective,delta=operational_objective(peak,before,flag,.5)
    transformed=delta+delta.clamp_min(0)
    expected=.5*transformed[0]+.25*transformed[1:].sum()
    torch.testing.assert_close(objective,expected)
    assert delta[0]<0 and delta[1]<0 and delta[2]>0
    with pytest.raises(ValueError):operational_objective(peak,before,torch.ones(3,dtype=torch.bool),.5)


def test_policy_fit_can_only_use_ended_cold_prefix():
    order=list(range(100,460))
    validate_fit_ids(order[:20],order)
    validate_fit_ids(order[:16],order)
    with pytest.raises(ValueError):validate_fit_ids([order[20]],order)
    with pytest.raises(ValueError):validate_fit_ids([order[0],order[0]],order)
    with pytest.raises(ValueError):validate_fit_ids([],order)


def test_training_gradient_combination_matches_deployed_loss_gradient():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_lowrank_20261029.model import AttentionAdapter
    from experiments.risk_aligned_online.attention_weights_20261029.learning import feedback_parts,feedback_states
    torch.set_num_threads(1);torch.manual_seed(8);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=AttentionAdapter(base);rng=np.random.default_rng(8)
    signal=dict(values=rng.normal(size=(256,23)).astype('float32'),available=np.ones((256,23),bool))
    weights=torch.randn(23);parts,_=feedback_parts(net,signal,None)
    states,_=feedback_states(net,signal,None,weights)
    torch.testing.assert_close(states['learned_reset'],weighted_state(parts[None],weights)[0],atol=1e-7,rtol=1e-4)
    torch.testing.assert_close(states['uniform_reset'],weighted_state(parts[None],torch.zeros(23))[0],atol=1e-7,rtol=1e-4)
    signal['available'][:]=False
    empty,info=feedback_states(net,signal,None,weights)
    assert all(torch.count_nonzero(s)==0 for s in empty.values()) and info['observed']==0


def test_complete_prefix_causality_and_shot_reset():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_lowrank_20261029.model import AttentionAdapter
    from experiments.risk_aligned_online.attention_weights_20261029.learning import predict
    torch.set_num_threads(1);torch.manual_seed(9);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=AttentionAdapter(base);rng=np.random.default_rng(9)
    signal=dict(values=rng.normal(size=(416,23)).astype('float32'),available=np.ones((416,23),bool))
    weights=torch.randn(23);first,info=predict(net,signal,None,weights);repeat,ri=predict(net,signal,None,weights)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][288:]+=3
    after,_=predict(net,changed,None,weights)
    for k in first:
        assert np.array_equal(first[k],repeat[k])
        assert np.array_equal(first[k][:288],after[k][:288])
        np.testing.assert_allclose(first[k][:256],first['cold'][:256],atol=1e-7,rtol=1e-5)
    assert info==ri and all(r['updates']==2 and r['max_state_norm']<=.050001 for r in info.values())
    with pytest.raises(ValueError):predict(net,dict(signal,y=1),None,weights)
