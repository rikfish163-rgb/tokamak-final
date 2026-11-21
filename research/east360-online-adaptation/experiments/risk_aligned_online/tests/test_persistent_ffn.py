import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.redesign_20261004.inter import postshot_update


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    x = torch.randn(2, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    return base, PersistentFfnModel(base), x, a


def test_zero_matches_original_ffn_and_nograd_uses_the_same_numerical_path():
    base, model, x, a = case()
    expected = FfnModel(base)(x, a, model.memory.new_zeros(16576, requires_grad=True))
    out = model(x, a, model.initial_fast(2).requires_grad_())
    with torch.no_grad():
        inference = model(x, a)
    for key in expected:
        torch.testing.assert_close(out[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(inference[key], expected[key], rtol=0, atol=0)
        assert not inference[key].requires_grad
    assert torch.is_grad_enabled()


def test_only_last_original_ffn_is_changed_and_fast_is_disposable():
    base, model, x, a = case()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    before = model(x, a)['logit'].detach()
    with torch.no_grad():
        model.memory[0] = .2
    fast = model.initial_fast(2).requires_grad_()
    out = model(x, a, fast)
    gm, gf = torch.autograd.grad(out['forecast'].square().mean(), (model.memory, fast))
    assert gm.norm() > 0 and gf.norm() > 0
    assert not torch.equal(before, out['logit'].detach())
    assert list(model.persistent_parameters()) == [model.memory]
    assert model.temporary_dim == 16576
    assert not model.initial_fast(1).any()
    assert all(not p.requires_grad for p in base.parameters())
    for k, v in state.items():
        torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)


def test_rejects_mixed_shot_fast_states_and_inference_mode():
    _, model, x, a = case()
    fast = model.initial_fast(2)
    fast[1, 0] = 1.
    with pytest.raises(ValueError, match='same shot'):
        model(x, a, fast)
    with torch.inference_mode(), pytest.raises(RuntimeError, match='inference_mode'):
        model(x, a)


def test_causal_intra_keeps_persistent_state_and_compact_receipts():
    _, model, x, a = case()
    with torch.no_grad():
        model.memory[0] = .1
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shot = dict(device='EAST', shot=7, values=x[0, :61].numpy(),
                available=a[0, :61].numpy(), times=np.arange(61)/1000.)
    scores, audit = predict_shot(model, shot, True)
    again, _ = predict_shot(model, shot, True)
    np.testing.assert_array_equal(scores, again)
    assert audit['fast_discarded'] and audit['fast_parameter_count'] == 16576
    first = audit['attempts'][0]
    receipt = first['fast_at_origin']
    assert receipt['count'] == 16576 and receipt['l2'] == 0.
    assert len(receipt['sha256']) == 64
    assert first['maturity_index'] == first['predictions_sealed_through_index'] == 25
    if first['accepted']:
        assert first['effective_index'] == 26
    for k, v in state.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_postshot_supervision_changes_memory_not_the_source():
    base, model, x, a = case()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    shot = dict(device='EAST', shot=7, values=x[0, :61].numpy(),
                available=a[0, :61].numpy(), times=np.arange(61)/1000.,
                y=np.zeros(61), valid=np.ones(61, bool), is_disrupt=False,
                down=None, horizon=125., _sealed_scores=np.full(61, .9))
    model.update_policy = 'error_only'
    result = postshot_update(model, shot, [], 'cpu', np.random.default_rng(131), .7)
    assert result['accepted'] and result['learned_neural_update']
    assert result['scope']['parameter_names'] == ['memory']
    assert result['scope']['parameter_count'] == 16576
    assert model.memory.norm() > 0 and not model.initial_fast(1).any()
    for k, v in state.items():
        torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)
