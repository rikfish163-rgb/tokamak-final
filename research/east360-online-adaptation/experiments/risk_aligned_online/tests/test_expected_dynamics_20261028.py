import pytest
import torch

from experiments.risk_aligned_online.expected_dynamics_20261028.forecast import (
    observed_features, direct_examples, DirectForecast, forecast_screen,
)


def test_prefix_features_do_not_change_when_future_arrives_or_is_replaced():
    x = torch.randn(20, 23); a = torch.ones_like(x, dtype=torch.bool)
    f = observed_features(x, a)
    assert torch.equal(f[:7], observed_features(x[:7], a[:7]))
    changed = x.clone(); changed[7:] = 999
    assert torch.equal(f[:7], observed_features(changed, a)[:7])
    assert f.shape == (20, 92) and not f[0, 46:].any()


def test_masked_nan_never_becomes_a_target_or_feature():
    x = torch.randn(16, 23); a = torch.ones_like(x, dtype=torch.bool)
    a[4, 2] = False
    poisoned = x.clone(); poisoned[4, 2] = float('nan')
    f, target, mask = direct_examples(poisoned, a)
    expected = direct_examples(torch.where(a, x, 0.), a)
    for left, right in zip((f, target, mask), expected):
        assert torch.equal(left, right)
    assert torch.isfinite(f).all() and torch.isfinite(target).all()
    assert not mask[0, 3, 2]


def test_direct_target_order_and_zero_initialization():
    x = torch.arange(9, dtype=torch.float32)[:, None].expand(-1, 23)
    f, target, mask = direct_examples(x, torch.ones_like(x, dtype=torch.bool))
    assert target.shape == (5, 4, 23) and mask.all()
    assert torch.equal(target[:, :, 0], torch.tensor([1., 2., 3., 4.]).expand(5, -1))
    model = DirectForecast()
    assert not model(f).any()
    model(f).sub(target).square().mean().backward()
    assert model.net[-1].weight.grad.norm() > 0
    with pytest.raises(ValueError):
        direct_examples(x[:4], mask[:4, 0])


def test_average_improvement_does_not_hide_safe_failure():
    rows = [dict(disruptive=i >= 80, prediction=.9, persistence=1.) for i in range(160)]
    assert forecast_screen(rows)['eligible']
    for row in rows[:41]:
        row['prediction'] = 1.01
    assert not forecast_screen(rows)['eligible']
    with pytest.raises(ValueError):
        forecast_screen(rows[:-1])
