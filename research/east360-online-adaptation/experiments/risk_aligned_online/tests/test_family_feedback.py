from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.context_feedback import predict_shot, train_step
from experiments.risk_aligned_online.family_feedback import FAMILIES, FamilyFeedback, signal_target23
from experiments.risk_aligned_online.io23 import RAW23
from experiments.risk_aligned_online.simple23 import Config, Simple23


def case():
    torch.manual_seed(131)
    torch.set_num_threads(2)
    parent = Simple23(Config(window=32, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    parent.eval().requires_grad_(False)
    adapter = FamilyFeedback(parent.cfg.dim, patch=parent.cfg.patch, horizon=8)
    x = torch.randn(2, 32, 23)
    return parent, adapter, x, torch.ones_like(x, dtype=torch.bool)


def test_family_partition_keeps_every_recorded_channel_once():
    names = [name for group in FAMILIES.values() for name in group]
    assert sorted(names) == sorted(RAW23) and len(names) == len(set(names)) == 23
    _, physical, _, _ = case()
    shuffled = FamilyFeedback(grouping='shuffled')
    assert [len(g) for g in shuffled.groups] == [5, 6, 10, 2]
    assert sorted(sum((list(g) for g in shuffled.groups), [])) == list(range(23))
    assert shuffled.groups != physical.groups
    assert sum(p.numel() for p in physical.parameters()) == sum(p.numel() for p in shuffled.parameters())


def test_zero_state_is_exact_parent_and_nonzero_changes_input_tokens_not_head():
    parent, adapter, x, a = case()
    before = adapter(parent, x, a)
    assert torch.equal(before['logit'], parent(x, a)['logit'])
    state = adapter.initial_fast(2)
    state[:, :adapter.rank] = .5
    after = adapter(parent, x, a, state)
    assert before['trajectory'].shape == (2, 8, 23)
    assert not torch.equal(before['logit'], after['logit'])
    assert after['family_delta'][:, 1:].count_nonzero() == 0
    assert after['family_delta'][:, 0].count_nonzero() > 0
    # Unlike the previous common offset, the change depends on each patch.
    assert not torch.allclose(after['family_delta'][:, 0, 0], after['family_delta'][:, 0, 1])
    assert not parent.embedding._forward_hooks and not parent.encoder.layers[0]._forward_hooks


def test_missing_family_cannot_move_tokens_and_invalid_values_are_finite():
    parent, adapter, x, a = case()
    channels = adapter.groups[0]
    a[:, :, channels] = False
    x[:, :, channels] = float('nan')
    state = adapter.initial_fast(2)
    state[:, :adapter.rank] = 1.
    out = adapter(parent, x, a, state)
    assert out['family_delta'][:, 0].count_nonzero() == 0
    assert all(torch.isfinite(out[k]).all() for k in ('logit', 'trajectory', 'context'))


def test_complete_window_target_uses_only_origin_scale_and_available_channels():
    _, _, x, a = case()
    truth = x[:, -1:, :].expand(-1, 8, -1).clone()
    a[:, :, 3] = False
    x[:, :, 3] = float('nan')
    mask = torch.ones_like(truth, dtype=torch.bool)
    target, valid = signal_target23(x, a, truth, mask)
    assert target.shape == truth.shape and not valid[:, :, 3].any()
    assert target.count_nonzero() == 0
    changed = truth.clone()
    changed[:, 3, 9] += 5.
    altered, _ = signal_target23(x, a, changed, mask)
    assert altered[:, 3, 9].abs().min() > 0
    assert torch.equal(altered[:, :3], target[:, :3])
    assert torch.equal(altered[:, 4:], target[:, 4:])


def test_double_backward_updates_auxiliary_rule_without_changing_parent():
    parent, adapter, x, a = case()
    frozen = deepcopy(parent.state_dict())
    truth = torch.randn(2, 17, 23)
    episode = dict(x=x, a=a, truth=truth, mask=torch.ones_like(truth, dtype=torch.bool),
                   qx=x+.1, qa=a, y=torch.tensor([0., 1.]))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    result = train_step(parent, adapter, optimizer, episode, mode='aligned', unroll=2,
                        target_fn=signal_target23)
    assert result['inner_updates'] == 2 and result['fast_change'] > 0
    assert all(torch.equal(v, frozen[k]) for k, v in parent.state_dict().items())
    assert all(p.grad is None for p in parent.parameters())
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
               for block in adapter.down)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.controller.parameters())


def test_online_future_is_inaccessible_and_fast_state_resets_each_shot():
    parent, adapter, _, _ = case()
    values = np.random.default_rng(131).normal(size=(65, 23)).astype(np.float32)
    shot = dict(shot=1, values=values, available=np.ones_like(values, bool))
    scores, events = predict_shot(parent, adapter, shot, arm='learned', device='cpu',
                                 target_fn=signal_target23)
    changed = deepcopy(shot)
    changed['values'][31:] += 100.
    later, _ = predict_shot(parent, adapter, changed, arm='learned', device='cpu',
                            target_fn=signal_target23)
    np.testing.assert_array_equal(scores[:31], later[:31])
    repeated, _ = predict_shot(parent, adapter, shot, arm='learned', device='cpu',
                               target_fn=signal_target23)
    np.testing.assert_array_equal(scores, repeated)
    assert events and all(e['affects_from'] > e['observed_through'] for e in events)
    assert not parent.embedding._forward_hooks and not parent.encoder.layers[0]._forward_hooks


def test_update_is_independent_bounded_and_zero_without_matured_gradient():
    parent, adapter, x, a = case()
    context = adapter(parent, x, a)['context']
    state = adapter.initial_fast(2)
    assert torch.equal(adapter.update(state, torch.zeros_like(state), context), state)
    g = torch.randn_like(state)
    updated = adapter.update(state, g, context)
    torch.testing.assert_close(updated[:1], adapter.update(state[:1], g[:1], context[:1]))
    assert updated.norm(dim=-1).max() <= .5+1e-6
    assert adapter.update(torch.full_like(state, 20.), g, context).norm(dim=-1).max() <= 8.+1e-6


def test_forecast_control_parameters_do_not_depend_on_source_risk_label():
    parent, adapter, x, a = case()
    other = deepcopy(adapter)
    truth = torch.randn(2, 8, 23)
    episode = dict(x=x, a=a, truth=truth, mask=torch.ones_like(truth, dtype=torch.bool),
                   qx=x+.1, qa=a, y=torch.tensor([0., 1.]))
    altered = dict(episode, y=1.-episode['y'])
    for model, e in ((adapter, episode), (other, altered)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_step(parent, model, optimizer, e, mode='forecast', target_fn=signal_target23)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in adapter.state_dict().items())


def test_target_is_invariant_to_affine_file_units_without_erasing_observed_change():
    _, _, x, a = case()
    truth = torch.randn(2, 8, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    baseline, valid = signal_target23(x, a, truth, mask)
    factor = torch.linspace(.5, 10., 23)[None, None]
    shifted, changed_valid = signal_target23(x*factor+5., a, truth*factor+5., mask)
    torch.testing.assert_close(shifted, baseline, atol=1e-6, rtol=1e-5)
    assert torch.equal(valid, changed_valid)
    assert baseline.count_nonzero() > 0
    # This property concerns ONLY the auxiliary target, not parent risk-input
    # calibration, equilibrium matching or evidence of learnt plasma physics.


def test_forward_exception_removes_both_hooks():
    parent, adapter, x, a = case()

    def fail(_module, _args, _out):
        raise RuntimeError('deliberate downstream failure')

    handle = parent.risk_head.register_forward_hook(fail)
    try:
        with np.testing.assert_raises(RuntimeError):
            adapter(parent, x, a, torch.ones(2, adapter.state_dim))
    finally:
        handle.remove()
    assert not parent.embedding._forward_hooks and not parent.encoder.layers[0]._forward_hooks
