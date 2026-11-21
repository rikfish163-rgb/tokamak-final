"""Source-state-conditioned internal updates: scope, causality, not efficacy."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.routed_ffn import RoutedFfnModel, fit_router
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.redesign_20261004.inter import postshot_update


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def case(routing='local'):
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    router = fit_router(torch.randn(256, 128), clusters=8)
    model = RoutedFfnModel(base, router, routing=routing)
    x = torch.randn(2, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    return base, model, x, a


def test_zero_is_the_parent_and_nograd_keeps_the_same_dispatch():
    base, model, x, a = case()
    reference = FfnModel(base)(x, a, torch.zeros(16576, requires_grad=True))
    result = model(x, a)
    with torch.no_grad(): inference = model(x, a)
    for key in reference:
        torch.testing.assert_close(result[key], reference[key], rtol=0, atol=0)
        torch.testing.assert_close(inference[key], reference[key], rtol=0, atol=0)
        assert not inference[key].requires_grad
    assert not base.encoder.layers[-1].linear2._forward_hooks


def test_source_keys_are_fixed_and_local_routing_is_sparse_normalized():
    _, model, _, _ = case()
    raw = model.router_keys * model.router_scale + model.router_center
    local = model.route(raw)
    assert torch.isfinite(local).all()
    torch.testing.assert_close(local.sum(-1), torch.ones(8))
    assert ((local > 0).sum(-1) == 2).all()
    assert (local.argmax(-1) == torch.arange(8)).all()
    uniform = RoutedFfnModel(model.base, dict(center=model.router_center,
        scale=model.router_scale, keys=model.router_keys), routing='uniform')
    torch.testing.assert_close(uniform.route(raw), torch.full((8, 8), 1/8))
    assert not model.router_keys.requires_grad


def test_gradients_change_both_tasks_without_changing_source_or_storing_fast():
    base, model, x, a = case()
    original = {k: v.clone() for k, v in base.state_dict().items()}
    before = model(x, a)
    fast = model.initial_fast(2).requires_grad_()
    loss = model(x, a, fast)['forecast'].square().mean()
    gm, gf = torch.autograd.grad(loss, (model.memory, fast))
    assert gm.norm() > 0 and gf.norm() > 0
    with torch.no_grad(): model.memory.copy_(-.1 * gm / gm.norm())
    after = model(x, a)
    assert not torch.equal(before['logit'], after['logit'])
    assert not torch.equal(before['forecast'], after['forecast'])
    assert model.temporary_dim == 512
    assert list(model.persistent_parameters()) == [model.memory]
    assert not model.initial_fast(1).any()
    for k, v in original.items(): torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)


def test_actual_signal_updates_reset_and_do_not_mutate_registered_state():
    _, model, x, a = case()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shot = dict(device='EAST', shot=7, values=x[0, :61].numpy(),
        available=a[0, :61].numpy(), times=np.arange(61)/1000.)
    first, audit = predict_shot(model, shot, True)
    second, _ = predict_shot(model, shot, True)
    np.testing.assert_array_equal(first, second)
    assert audit['fast_parameter_count'] == 512 and audit['fast_discarded']
    attempt = audit['attempts'][0]
    assert attempt['fast_at_origin']['l2'] == 0.
    assert attempt['maturity_index'] == attempt['predictions_sealed_through_index'] == 25
    if attempt['accepted']: assert attempt['effective_index'] == 26
    for k, v in state.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_revealed_label_updates_only_the_internal_memory():
    base, model, x, a = case()
    original = {k: v.clone() for k, v in base.state_dict().items()}
    shot = dict(device='EAST', shot=7, values=x[0, :61].numpy(), available=a[0, :61].numpy(),
        times=np.arange(61)/1000., y=np.zeros(61), valid=np.ones(61, bool),
        is_disrupt=False, down=None, horizon=125., _sealed_scores=np.full(61, .9))
    result = postshot_update(model, shot, [], 'cpu', np.random.default_rng(131), .7)
    assert result['accepted'] and result['scope']['parameter_count'] == 512
    assert result['scope']['parameter_names'] == ['memory']
    assert model.memory.norm() > 0
    for k, v in original.items(): torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)


def test_invalid_keys_and_mixed_fast_states_fail_without_retaining_hooks():
    base, model, x, a = case()
    with pytest.raises(ValueError, match='routing'): RoutedFfnModel(base, {}, routing='other')
    with pytest.raises(ValueError, match='finite'): fit_router(torch.full((16, 128), float('nan')))
    with pytest.raises(ValueError, match='variation'): fit_router(torch.ones(16, 128))
    fast = model.initial_fast(2); fast[1, 0] = .1
    with pytest.raises(ValueError, match='same shot'): model(x, a, fast)
    with torch.inference_mode(), pytest.raises(RuntimeError, match='inference_mode'): model(x, a)
    with pytest.raises(ValueError): model(x[:, :127], a[:, :127])
    assert not base.encoder.layers[-1].linear2._forward_hooks
