"""Density cannot supervise its own forecast through a hidden input shortcut."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cross_density import CrossDensityModel, density_loss, train_head
from experiments.risk_aligned_online.io23 import RAW23
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.embedding_model import EmbeddingModel
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot

D = RAW23.index('ne_nG')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case():
    torch.manual_seed(71)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    parent = EmbeddingModel(base)
    model = CrossDensityModel(parent)
    x = torch.randn(2, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    return parent, model, x, a


def test_zero_state_preserves_original_risk_bitwise_and_frozen_source():
    parent, model, x, a = case()
    original = {k:v.clone() for k,v in parent.state_dict().items()}
    expected = parent(x, a, parent.initial_fast(2).requires_grad_())['logit']
    actual = model(x, a, model.initial_fast(2).requires_grad_())['logit']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for k,v in original.items(): torch.testing.assert_close(parent.state_dict()[k], v, rtol=0, atol=0)


def test_auxiliary_features_never_see_density_values_or_availability():
    _, model, x, a = case()
    before = model.cross_features(x, a)
    x[:, :, D] = 1e20; a[:, :, D] = False
    after = model.cross_features(x, a)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert after.shape == (2, 152) and not model.base.risk_head._forward_pre_hooks


def test_source_mnm_cannot_be_an_auxiliary_shortcut_absent_on_east():
    _, model, x, a = case(); before = model.cross_features(x, a)
    x[:, :, RAW23.index('MNM')] = -1e10; a[:, :, RAW23.index('MNM')] = False
    torch.testing.assert_close(model.cross_features(x, a), before, rtol=0, atol=0)


def test_direct_forecast_has_no_density_history_dependency_and_exact_inverse():
    _, model, x, a = case(); f = model.initial_fast(2).requires_grad_()
    before = model(x, a, f)
    moved = x.clone(); moved[:, :, D] += 20
    torch.testing.assert_close(model(moved, a, f)['forecast'][:, :, D],
                               before['forecast'][:, :, D], rtol=0, atol=0)
    f = torch.full((2, 1), -2., requires_grad=True)
    out = model(x, a, f)
    torch.testing.assert_close(out['forecast'][:, :, D], before['forecast'][:, :, D] + 2,
                               rtol=0, atol=0)
    gradient, = torch.autograd.grad(out['logit'].sum(), f)
    assert gradient.norm() > 0 and torch.isfinite(gradient).all()


def test_density_loss_only_uses_density_targets_and_never_moves_the_truth():
    _, model, x, a = case(); p = model(x, a, model.initial_fast(2).requires_grad_())['forecast']
    truth = x[:, :25].clone().requires_grad_(); observed = a[:, :25].clone()
    loss = model.intra_forecast_loss(p, truth, observed)
    altered = truth.detach().clone(); altered[:, :, [i for i in range(23) if i != D]] = 1e10
    torch.testing.assert_close(model.intra_forecast_loss(p, altered, observed), loss, rtol=0, atol=0)
    g, = torch.autograd.grad(loss, truth, allow_unused=True)
    assert g is None
    observed[:, :, D] = False
    assert float(model.intra_forecast_loss(p, altered, observed).detach()) == 0.


def test_offset_changes_only_density_embedding_and_is_bounded():
    parent, model, x, a = case(); seen = []
    h = parent.base.embedding.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    model(x, a, model.initial_fast(2).requires_grad_()); model(x, a, torch.full((2,1), -2., requires_grad=True))
    h.remove()
    # Every model call first makes risk then independently masked auxiliary.
    first, changed = seen[0].reshape(2,16,8,46), seen[2].reshape(2,16,8,46)
    delta = changed - first; allowed = torch.zeros(46, dtype=torch.bool); allowed[D] = True
    assert not delta[:, :, :, ~allowed].any() and delta[:, :, :, D].norm() > 0
    assert model.offset(torch.tensor([[1e6], [-1e6]])).tolist() == [[12.], [-12.]]


def test_causal_stream_restores_model_discards_fast_and_never_uses_future_prefix():
    _, model, x, a = case(); values = x[0, :61].numpy().copy()
    # Force a measurable density bias while leaving risk weights untouched.
    values[:, D] = 100.
    s = dict(device='EAST', shot=1, values=values, available=a[0,:61].numpy(), times=np.arange(61)/1000.)
    state = {k:v.clone() for k,v in model.state_dict().items()}
    static, _ = predict_shot(model, s, False, stride=25, step=1., radius=12., objective='forward_holdout')
    scores, audit = predict_shot(model, s, True, stride=25, step=1., radius=12., objective='forward_holdout')
    np.testing.assert_array_equal(static[:26], scores[:26])
    assert audit['fast_parameter_count'] == 1 and audit['accepted_updates'] > 0 and audit['fast_discarded']
    for r in audit['attempts']:
        if r['accepted']: assert r['effective_index'] == r['maturity_index'] + 1
    for k,v in state.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    repeated, _ = predict_shot(model, s, True, stride=25, step=1., radius=12., objective='forward_holdout')
    np.testing.assert_array_equal(scores, repeated)


@pytest.mark.parametrize('fast', [torch.zeros(2, 2), torch.full((2,1), float('nan'))])
def test_invalid_fast_state_is_rejected(fast):
    _, model, x, a = case()
    with pytest.raises(ValueError): model(x, a, fast)


def test_source_auxiliary_training_cannot_change_original_classifier_or_scaler():
    parent, model, x, a = case()
    original = {k:v.clone() for k,v in parent.state_dict().items()}
    features = model.cross_features(x, a).detach()
    target = torch.ones(2, 25); mask = torch.ones_like(target, dtype=torch.bool)
    old_head = {k:v.clone() for k,v in model.cross_head.state_dict().items()}
    train_head(model, features, target, mask, steps=3)
    assert any(not torch.equal(model.cross_head.state_dict()[k], v) for k,v in old_head.items())
    for k,v in original.items(): torch.testing.assert_close(parent.state_dict()[k], v, rtol=0, atol=0)
    assert all(not p.requires_grad for p in parent.parameters())


def test_loss_has_no_target_when_density_is_unavailable_and_checks_horizon():
    p = torch.ones(2, 25, requires_grad=True); t = torch.zeros(2,25)
    mask = torch.zeros(2,25, dtype=torch.bool)
    loss = density_loss(p,t,mask); g, = torch.autograd.grad(loss,p)
    assert float(loss.detach()) == 0. and not g.any()
    with pytest.raises(ValueError): density_loss(p[:,:12],t[:,:12],mask[:,:12])
