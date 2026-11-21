"""Internal conditional weight adaptation, not a risk-output correction."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.routed_ffn import fit_router
from experiments.risk_aligned_online.routed_ffn_weight import RoutedWeightFfnModel
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.redesign_20261004.inter import postshot_update


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case(routing='local'):
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    model = RoutedWeightFfnModel(base, fit_router(torch.randn(256,128)), routing=routing)
    x = torch.randn(2,128,23); available = torch.ones_like(x,dtype=torch.bool)
    return base, model, x, available


def test_zero_weight_update_exactly_preserves_both_source_heads():
    base, model, x, available = case()
    reference = FfnModel(base)(x,available,torch.zeros(16576,requires_grad=True))
    actual = model(x,available)
    with torch.no_grad(): detached = model(x,available)
    for key in reference:
        torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0)
        torch.testing.assert_close(detached[key],reference[key],rtol=0,atol=0)
        assert not detached[key].requires_grad
    assert not base.encoder.layers[-1].linear2._forward_hooks


def test_weight_increment_is_linear_in_hidden_value_not_a_constant_offset():
    _, model, _, _ = case('uniform')
    hidden = torch.zeros(1,2,128); hidden[...,0] = torch.tensor([1.,3.])
    working = torch.zeros_like(model.memory); working[0,0,0]=2.
    result = model.increment(hidden,working)
    expected = torch.zeros(1,2,64); expected[...,0]=torch.tensor([2.,6.])/np.sqrt(8.)
    torch.testing.assert_close(result,expected)
    assert model.memory.shape == (8,64,128)
    assert model.temporary_dim == 65536


@pytest.mark.parametrize('routing',['local','uniform'])
def test_gate_norm_budget_is_equal_and_source_keys_are_frozen(routing):
    _, model, _, _ = case(routing)
    hidden = model.router_center+model.router_keys*model.router_scale
    gates = model.route(hidden)
    torch.testing.assert_close(gates.norm(dim=-1),torch.ones(8))
    assert ((gates>0).sum(-1) == (2 if routing=='local' else 8)).all()
    assert not model.router_keys.requires_grad


def test_signal_gradient_changes_both_tasks_but_never_parent_parameters():
    base, model, x, available = case()
    saved = {k:v.clone() for k,v in base.state_dict().items()}
    before = model(x,available)
    fast = model.initial_fast(2).requires_grad_()
    loss = model(x,available,fast)['forecast'].square().mean()
    gradient, gm = torch.autograd.grad(loss,(fast,model.memory))
    assert gradient.norm()>0 and gm.norm()>0
    update = (-.05*gradient[:1]/gradient.norm()).expand_as(fast)
    after = model(x,available,update)
    assert not torch.equal(before['logit'],after['logit'])
    assert not torch.equal(before['forecast'],after['forecast'])
    for k,v in saved.items(): torch.testing.assert_close(base.state_dict()[k],v,rtol=0,atol=0)


def test_signal_stream_uses_arrived_targets_and_disposes_temporary_weights():
    _, model, x, available = case()
    saved = {k:v.clone() for k,v in model.state_dict().items()}
    shot = dict(device='EAST',shot=7,values=x[0,:61].numpy(),
                available=available[0,:61].numpy(),times=np.arange(61)/1000.)
    first,audit = predict_shot(model,shot,True)
    second,_ = predict_shot(model,shot,True)
    np.testing.assert_array_equal(first,second)
    assert audit['fast_parameter_count']==65536 and audit['fast_discarded']
    attempt = audit['attempts'][0]
    assert attempt['fast_at_origin']['l2']==0.
    assert attempt['maturity_index']==attempt['predictions_sealed_through_index']==25
    if attempt['accepted']: assert attempt['effective_index']==26
    for k,v in saved.items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_revealed_outcome_changes_only_persistent_internal_weights():
    base, model, x, available = case()
    saved = {k:v.clone() for k,v in base.state_dict().items()}
    shot = dict(device='EAST',shot=7,values=x[0,:61].numpy(),available=available[0,:61].numpy(),
        times=np.arange(61)/1000.,y=np.zeros(61),valid=np.ones(61,bool),
        is_disrupt=False,down=None,horizon=125.,_sealed_scores=np.full(61,.9))
    result=postshot_update(model,shot,[],'cpu',np.random.default_rng(131),.7)
    assert result['accepted'] and result['scope']['parameter_count']==65536
    assert result['scope']['parameter_names']==['memory']
    assert model.memory.norm()>0 and not model.initial_fast(1).any()
    for k,v in saved.items(): torch.testing.assert_close(base.state_dict()[k],v,rtol=0,atol=0)


def test_rejects_mixed_or_nonfinite_state_and_removes_hooks_on_failure():
    base, model, x, available = case()
    bad=model.initial_fast(2); bad[1,0]=.1
    with pytest.raises(ValueError,match='same shot'): model(x,available,bad)
    bad=model.initial_fast(2); bad[:,0]=float('nan')
    with pytest.raises(ValueError,match='finite'): model(x,available,bad)
    with torch.inference_mode(),pytest.raises(RuntimeError,match='inference_mode'): model(x,available)
    with pytest.raises(ValueError): model(x[:,:127],available[:,:127])
    assert not base.encoder.layers[-1].linear2._forward_hooks
