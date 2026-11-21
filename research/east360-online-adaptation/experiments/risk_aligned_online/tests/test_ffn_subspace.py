from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.ffn_subspace import (
    FfnSubspaceModel, one_step, meta_step, risk_objective)
from experiments.risk_aligned_online.redesign_20261004.intra import (
    predict_shot, predict_source_diagnostic)
from experiments.risk_aligned_online.ffn_subspace_train import source_screen


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23))
    return FfnSubspaceModel(base, rank=8, seed=131)


def example():
    torch.manual_seed(7)
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    return dict(x=x, a=a, qx=x + .1, qa=a.clone(),
                truth=torch.randn(1, 25, 23),
                mask=torch.ones(1, 25, 23, dtype=torch.bool), y=torch.ones(1))


def test_zero_state_is_original_ffn_and_basis_is_orthonormal(model):
    e = example()
    expected = FfnModel(deepcopy(model.base))
    old = expected(e['x'], e['a'], expected.initial_delta())
    new = model(e['x'], e['a'], model.initial_fast(1).requires_grad_())
    for name in ('logit', 'forecast'):
        torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)
    torch.testing.assert_close(model.basis.T @ model.basis, torch.eye(8), atol=2e-6, rtol=0)
    assert model.temporary_dim == 8


def test_seed_and_physical_parameter_budget_are_fixed(model):
    other = FfnSubspaceModel(deepcopy(model.base), rank=8, seed=131)
    torch.testing.assert_close(model.basis, other.basis, rtol=0, atol=0)
    q = torch.randn(8)
    torch.testing.assert_close((model.basis @ q).norm(), q.norm(), rtol=2e-6, atol=2e-6)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 16576 * 8


def test_nonzero_state_changes_internal_outputs_without_mutating_parent(model):
    e = example()
    state = {k: v.clone() for k, v in model.base.state_dict().items()}
    q = model.initial_fast(1).requires_grad_()
    before = model(e['x'], e['a'], q)
    after = model(e['x'], e['a'], q + .4)
    assert not torch.equal(before['logit'], after['logit'])
    assert not torch.equal(before['forecast'], after['forecast'])
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in state.items())


def test_one_step_reads_signals_not_label(model):
    class NoLabel(dict):
        def __getitem__(self, key):
            if key in ('y', 'qx', 'qa', 'down', 'is_disrupt'):
                raise AssertionError('inner update accessed outcome/later query')
            return super().__getitem__(key)
    fast, row = one_step(model, NoLabel(example()))
    assert fast.shape == (1, 8)
    assert torch.isfinite(fast).all()
    assert fast.norm() <= 1.00001
    assert isinstance(row['accepted'], bool)


@pytest.mark.parametrize('mode', ['risk', 'signal', 'risk_protect'])
def test_meta_gradient_changes_basis_only(model, mode):
    original = {k: v.clone() for k, v in model.base.state_dict().items()}
    basis = model.basis.detach().clone()
    optimizer = torch.optim.Adam([model.basis], lr=.001)
    row = meta_step(model, optimizer, [example()], mode=mode)
    assert row['basis_grad_norm'] > 0
    assert not torch.equal(basis, model.basis)
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in original.items())
    assert all(p.grad is None for p in model.base.parameters())
    torch.testing.assert_close(model.basis.T @ model.basis, torch.eye(8), atol=2e-6, rtol=0)


def test_independent_shot_states_in_one_batch_are_rejected(model):
    e = example()
    fast = torch.zeros(2, 8); fast[1, 0] = 1
    with pytest.raises(ValueError, match='same shot'):
        model(e['x'].expand(2, -1, -1), e['a'].expand(2, -1, -1), fast)


def test_missing_targets_and_invalid_rank_fail_explicitly(model):
    e = example(); e['mask'].zero_()
    with pytest.raises(ValueError, match='observed'):
        one_step(model, e)
    with pytest.raises(ValueError, match='rank'):
        FfnSubspaceModel(deepcopy(model.base), rank=0)
    with pytest.raises(ValueError, match='rank'):
        FfnSubspaceModel(deepcopy(model.base), rank=True)


def test_generic_causal_replay_resets_and_never_rewrites_prefix(model):
    rng = np.random.default_rng(42)
    shot = dict(device='EAST', values=rng.normal(size=(60, 23)).astype(np.float32),
                available=np.ones((60, 23), bool), times=np.arange(60) / 1000.)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    static, _ = predict_shot(model, shot, intra=False)
    updated, audit = predict_shot(model, shot, intra=True)
    repeated, again = predict_shot(model, shot, intra=True)
    np.testing.assert_allclose(updated[:26], static[:26], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(updated, repeated)
    assert audit['attempts'] == again['attempts']
    assert audit['fast_discarded']
    assert audit['fast_parameter_count'] == 8
    assert audit['model_parameter_change_norm'] == 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert all(not r['accepted'] or r['effective_index'] == r['maturity_index'] + 1
               for r in audit['attempts'])


def test_source_screen_requires_both_classes_and_both_controls():
    def report(after, negative=.4, positive=.4):
        return dict(risk_before=.5, risk_after=after,
                    classes={str(i): dict(risk_before=.5, risk_after=v)
                             for i, v in enumerate((negative, positive))})
    reports = dict(random=report(.49), signal=report(.48), risk=report(.4))
    assert source_screen(reports)['passed']
    reports['risk'] = report(.4, positive=.51)
    assert 'source_class_1_risk_harmed' in source_screen(reports)['reasons']
    reports['risk'] = report(.48)
    assert 'not_better_than_signal' in source_screen(reports)['reasons']


def test_wrong_optimizer_cannot_modify_original_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    with pytest.raises(ValueError, match='only the update basis'):
        meta_step(model, optimizer, [example()], mode='risk')


def test_nonregression_outer_cost_doubles_harm_not_benefit():
    before = torch.tensor(.5)
    harmful = torch.tensor(.6, requires_grad=True)
    benefit = torch.tensor(.4, requires_grad=True)
    harmed_cost = risk_objective(harmful, before, protect=True)
    helped_cost = risk_objective(benefit, before, protect=True)
    torch.testing.assert_close(harmed_cost, torch.tensor(.7))
    torch.testing.assert_close(helped_cost, torch.tensor(.4))
    assert torch.autograd.grad(harmed_cost, harmful)[0] == 2
    assert torch.autograd.grad(helped_cost, benefit)[0] == 1


def test_source_whole_shot_requires_explicit_offline_mode(model):
    shot = dict(device='J-TEXT', values=np.zeros((30, 23), np.float32),
                available=np.ones((30, 23), bool), times=np.arange(30) / 1000.)
    with pytest.raises(ValueError, match='EAST only'):
        predict_shot(model, shot, intra=True)
    scores, audit = predict_source_diagnostic(model, shot, intra=True)
    assert len(scores) == 30
    assert audit['scope'] == 'OFFLINE_SOURCE_DIAGNOSTIC_NOT_JTEXT_ONLINE'
    assert audit['fast_discarded']
    assert audit['model_parameter_change_norm'] == 0


def test_offline_source_mode_cannot_be_used_for_east_or_enter_public_online_api(model):
    shot = dict(device='EAST', values=np.zeros((30, 23), np.float32),
                available=np.ones((30, 23), bool), times=np.arange(30) / 1000.)
    with pytest.raises(ValueError, match='J-TEXT'):
        predict_source_diagnostic(model, shot, intra=True)
    with pytest.raises(TypeError, match='offline_source'):
        predict_shot(model, shot, intra=True, offline_source=True)
