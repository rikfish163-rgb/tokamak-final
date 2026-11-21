import torch
import pytest

from experiments.risk_aligned_online.model import ModelConfig, RiskAlignedModel
from experiments.risk_aligned_online.learning import adapt, signal_loss


torch.set_num_threads(2)


def make_case():
    torch.manual_seed(13)
    cfg = ModelConfig(channels=3, window=16, patch=4, dim=16, heads=2,
                      rank=4, horizon=4, fast_lr=0.5)
    model = RiskAlignedModel(cfg, torch.zeros(3), torch.ones(3))
    x = torch.randn(2, 16, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    target = x[:, -1:, :] + torch.randn(2, 4, 3)
    return model, x, mask, target, torch.ones_like(target, dtype=torch.bool)


def test_direct_window_and_two_explicit_risk_horizons():
    model, x, mask, _, _ = make_case()
    result = model(x, mask)
    assert result['logits'].shape == (2, 2)
    assert result['forecast'].shape == (2, 4, 3)
    assert result['state'].shape == (2, 16)


def test_missing_values_never_influence_prediction():
    model, x, mask, _, _ = make_case()
    mask[:, :, 1] = False
    original = model(x, mask)
    changed = x.clone()
    changed[:, :, 1] = float('nan')
    altered = model(changed, mask)
    torch.testing.assert_close(original['logits'], altered['logits'])
    torch.testing.assert_close(original['forecast'], altered['forecast'])


def test_all_missing_window_is_finite_but_cannot_train_fast_state():
    model, x, mask, target, target_mask = make_case()
    mask[:] = False
    state, audit = adapt(model, x, mask, target, target_mask, create_graph=False)
    torch.testing.assert_close(state, model.initial_fast(2))
    assert torch.isfinite(model(x, mask)['logits']).all()
    assert audit['valid_count'].sum() == 0


def test_forecast_uses_observed_truth_and_does_not_backprop_through_target():
    model, x, mask, target, target_mask = make_case()
    target.requires_grad_(True)
    output = model(x, mask)
    losses, _, _ = signal_loss(output, target, target_mask)
    losses.sum().backward()
    assert target.grad is None
    assert model.forecast_head.weight.grad is not None


def test_fast_update_changes_internal_representation_not_persistent_weights():
    model, x, mask, target, target_mask = make_case()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old = model(x, mask)
    fast, audit = adapt(model, x, mask, target, target_mask, create_graph=False)
    new = model(x, mask, fast=fast)
    assert audit['gradient_norm'].max() > 0
    assert not torch.allclose(old['state'], new['state'], atol=1e-8)
    assert not torch.allclose(old['logits'], new['logits'], atol=1e-8)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_meta_objective_trains_signal_head_and_update_controller():
    model, x, mask, target, target_mask = make_case()
    fast, _ = adapt(model, x, mask, target, target_mask, create_graph=True)
    query = model(x + 0.2, mask, fast=fast)['logits']
    torch.nn.functional.binary_cross_entropy_with_logits(query, torch.ones_like(query)).backward()
    for module in (model.forecast_head, model.update_controller):
        norm = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        assert norm > 0, 'outer risk loss must train the adaptation rule'


def test_fast_state_is_bounded_and_no_observation_means_no_step():
    model, x, mask, target, target_mask = make_case()
    fast = model.initial_fast(2).detach().requires_grad_(True)
    for _ in range(5):
        fast, _ = adapt(model, x, mask, target * 1e5, target_mask,
                        fast=fast, create_graph=False)
        assert torch.linalg.vector_norm(fast - model.initial_fast(2), dim=-1).max() <= model.cfg.fast_radius + 1e-6
    target_mask[:] = False
    next_fast, _ = adapt(model, x, mask, target, target_mask, fast=fast, create_graph=False)
    torch.testing.assert_close(next_fast, fast)


def test_valid_targets_are_averaged_per_episode_not_across_other_shots():
    model, x, mask, target, target_mask = make_case()
    output = model(x, mask)
    baseline, _, _ = signal_loss(output, target, target_mask)
    target[1] += 1000
    changed, _, _ = signal_loss(output, target, target_mask)
    torch.testing.assert_close(baseline[0], changed[0])
    assert changed[1] > baseline[1]


def test_prediction_and_update_are_independent_of_other_batch_members():
    model, x, mask, target, target_mask = make_case()
    batch = model(x, mask)
    single = model(x[:1], mask[:1])
    torch.testing.assert_close(batch['logits'][:1], single['logits'], atol=1e-6, rtol=1e-5)
    batch_fast, _ = adapt(model, x, mask, target, target_mask)
    single_fast, _ = adapt(model, x[:1], mask[:1], target[:1], target_mask[:1])
    torch.testing.assert_close(batch_fast[:1], single_fast, atol=1e-6, rtol=1e-5)


def test_frozen_source_still_allows_fast_gradient():
    model, x, mask, target, target_mask = make_case()
    model.requires_grad_(False)
    fast, audit = adapt(model, x, mask, target, target_mask)
    assert audit['gradient_norm'].max() > 0
    assert not fast.requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_postshot_parameter_scope_excludes_source_temporal_blocks():
    model, _, _, _, _ = make_case()
    selected = {id(p) for p in model.persistent_parameters()}
    assert id(model.fast_init) in selected
    assert all(id(p) not in selected for p in model.blocks.parameters())
    assert all(id(p) not in selected for p in model.patch_embed.parameters())
    assert all(id(p) in selected for p in model.update_controller.parameters())


def test_forecast_reference_is_origin_history_not_future_truth():
    model, x, mask, target, target_mask = make_case()
    output = model(x, mask)
    last = output['reference_last'].clone()
    scale = output['reference_scale'].clone()
    signal_loss(output, target + 100, target_mask)
    torch.testing.assert_close(output['reference_last'], last)
    torch.testing.assert_close(output['reference_scale'], scale)
    torch.testing.assert_close(last, x[:, -1])


def test_constant_windows_support_finite_meta_gradient():
    model, x, mask, target, target_mask = make_case()
    x[:] = 0
    target[:] = 0
    fast, _ = adapt(model, x, mask, target, target_mask, create_graph=True)
    model(x, mask, fast=fast)['logits'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('kwargs', [dict(window=15), dict(heads=3),
                                  dict(fast_lr=0), dict(fast_radius=float('nan'))])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_fixed_control_is_explicit_ablation_not_a_hidden_fallback():
    model, x, mask, target, target_mask = make_case()
    _, audit = adapt(model, x, mask, target, target_mask, control='fixed')
    torch.testing.assert_close(audit['gate'], torch.ones(2))
    with pytest.raises(ValueError):
        adapt(model, x, mask, target, target_mask, control='unknown')


def test_joint_loss_and_meta_loss_can_share_the_inner_graph():
    model, x, mask, target, target_mask = make_case()
    support = model(x, mask)
    auxiliary, _, _ = signal_loss(support, target, target_mask)
    fast, _ = adapt(model, x, mask, target, target_mask, create_graph=True)
    logits = model(x + .1, mask, fast=fast)['logits']
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)) + .2 * auxiliary.mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_asinh_forecast_target_preserves_direction_without_extreme_scale_domination():
    cfg = ModelConfig(channels=3, window=16, patch=4, dim=16, heads=2,
                      horizon=4, forecast_space='asinh')
    model = RiskAlignedModel(cfg, torch.zeros(3), torch.ones(3))
    x = torch.ones(1, 16, 3)
    out = model(x, torch.ones_like(x, dtype=torch.bool))
    target = torch.full((1, 4, 3), 1e8)
    truth = torch.asinh((target - out['reference_last'][:, None]) / out['reference_scale'][:, None])
    out['forecast'] = truth.clone()
    loss, _, baseline = signal_loss(out, target, torch.ones_like(target, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.zeros(1))
    assert 0 < baseline.item() < 30


def test_forecast_coordinate_contract_is_explicit_and_rejects_unknown_space():
    assert ModelConfig().forecast_space == 'linear'
    with pytest.raises(ValueError, match='forecast_space'):
        ModelConfig(forecast_space='auto')
