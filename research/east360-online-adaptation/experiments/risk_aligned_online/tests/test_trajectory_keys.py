import pytest
import torch

from experiments.risk_aligned_online.trajectory_keys_20261028.model import TrajectoryMemory


def pair():
    torch.manual_seed(131)
    a = TrajectoryMemory(64, torch.ones(4, 23), trajectory=False)
    b = TrajectoryMemory(64, torch.ones(4, 23), trajectory=True)
    b.load_state_dict(a.state_dict())
    return a, b


def test_initialization_matches_old_level_key_and_off_has_no_risk_delta():
    a, b = pair()
    x = torch.randn(12, 23); mask = torch.ones_like(x, dtype=torch.bool)
    for mode in ('memory', 'off'):
        torch.testing.assert_close(a.trace(x, mask, mode=mode)['delta'],
                                   b.trace(x, mask, mode=mode)['delta'], rtol=0, atol=0)
    assert b.trace(x, mask, mode='off')['delta'].count_nonzero() == 0


def test_trajectory_features_separate_equal_levels_with_opposite_arrived_changes():
    _, b = pair()
    x = torch.tensor([-1., 0., 1., 0.])[:, None].expand(-1, 23).clone()
    f = b.key_features(x, torch.ones_like(x, dtype=torch.bool))
    assert torch.equal(f[1, :46], f[3, :46])
    assert torch.equal(f[1, 46:69], -f[3, 46:69])
    assert not f[0, 69:].any()


def test_key_prefix_and_missing_data_invariance():
    _, b = pair()
    x = torch.randn(12, 23); mask = torch.ones_like(x, dtype=torch.bool)
    mask[3, 5] = False
    clean = torch.where(mask, x, 0.)
    f = b.key_features(clean, mask)
    assert torch.equal(f[:6], b.key_features(clean[:6], mask[:6]))
    assert f[3, 46+5] == f[4, 46+5] == 0
    poisoned = x.clone(); poisoned[~mask] = float('nan')
    torch.testing.assert_close(b.trace(poisoned, mask, mode='memory')['delta'],
                               b.trace(clean, mask, mode='memory')['delta'], rtol=0, atol=0)


def test_new_weights_receive_forecast_gradient_without_changing_parent_module():
    _, b = pair()
    x = torch.randn(16, 23); mask = torch.ones_like(x, dtype=torch.bool)
    b.trace(x, mask, mode='memory')['forecast_loss'].backward()
    assert b.key.weight.grad[:, 46:].norm() > 0
    with pytest.raises(ValueError):
        TrajectoryMemory(64, torch.ones(4, 23), trajectory='yes')
