import pytest
import torch

from experiments.risk_aligned_online.lifecycle_stream_20261025.learning import window_loss


def test_old_policy_punishes_high_early_disruptive_score_but_masked_policy_does_not():
    y = torch.tensor([0., 1.])
    z = torch.tensor([2., 1.], requires_grad=True)
    old = window_loss(z, y, True, 'all_points')
    old_gradient, = torch.autograd.grad(old, z)
    new = window_loss(z, y, True, 'disruptive_positive_only')
    new_gradient, = torch.autograd.grad(new, z)
    assert old_gradient[0] > 0
    assert new_gradient[0] == 0
    assert new_gradient[1] < 0


def test_safe_shot_and_default_policy_keep_existing_objective():
    z = torch.tensor([2., -1.])
    y = torch.zeros(2)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    assert torch.equal(window_loss(z, y, False, 'all_points'), expected)
    assert torch.equal(window_loss(z, y, False, 'disruptive_positive_only'), expected)


def test_missing_positive_and_unrecognized_policy_fail_visibly():
    with pytest.raises(ValueError, match='positive'):
        window_loss(torch.zeros(2), torch.zeros(2), True, 'disruptive_positive_only')
    with pytest.raises(ValueError, match='policy'):
        window_loss(torch.zeros(2), torch.zeros(2), False, 'typo')
