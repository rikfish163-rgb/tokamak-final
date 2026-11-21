import math

import pytest
import torch

from joint_motion_20261107 import OrderedTrajectory, components
from joint_trajectory_20261107 import JointTrajectory, joint_energy_score


def test_single_scenario_change_is_hand_calculated_joint_rms():
    encoded = torch.zeros(1, 1, 10, 2)
    target = torch.zeros(1, 10, 2)
    encoded[0, 0, [0, 1, 8, 9]] = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [6.0, 10.0]]
    )
    target[0, [0, 1, 8, 9]] = torch.tensor(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 3.0], [2.0, 4.0]]
    )
    probabilities = torch.ones(1, 1)
    available = torch.ones(1, 10, 2, dtype=torch.bool)

    level, change = components(encoded, probabilities, target, available)

    torch.testing.assert_close(level, joint_energy_score(encoded, probabilities, target, available))
    # Difference residuals are [2, 4] and [3, 3], without division by lag.
    torch.testing.assert_close(change, torch.tensor([math.sqrt(38.0 / 4.0)]))


def test_change_reuses_whole_trajectory_probabilities_and_spread_term():
    encoded = torch.zeros(1, 2, 9, 1)
    encoded[0, 1, 8, 0] = 4.0
    target = torch.zeros(1, 9, 1)
    target[0, 8, 0] = 1.0
    probabilities = torch.tensor([[0.25, 0.75]])
    available = torch.ones(1, 9, 1, dtype=torch.bool)

    _, change = components(encoded, probabilities, target, available)

    # Weighted target distances: 2.5; half the weighted pair distance: 0.75.
    torch.testing.assert_close(change, torch.tensor([1.75]))


def test_change_pair_requires_both_endpoints():
    encoded = torch.zeros(1, 1, 11, 1)
    target = torch.zeros(1, 11, 1)
    encoded[0, 0, 8:, 0] = torch.tensor([5.0, 100.0, 1000.0])
    target[0, 8, 0] = 2.0
    available = torch.ones(1, 11, 1, dtype=torch.bool)
    available[0, [2, 9], 0] = False

    _, change = components(encoded, torch.ones(1, 1), target, available)

    torch.testing.assert_close(change, torch.tensor([3.0]))


def test_channel_constant_offset_changes_level_but_not_change():
    time = torch.arange(25, dtype=torch.float32)
    target = torch.stack((time, 2.0 * time), dim=-1).unsqueeze(0)
    encoded = torch.zeros(1, 1, 25, 2)
    probabilities = torch.ones(1, 1)
    available = torch.ones(1, 25, 2, dtype=torch.bool)

    level, change = components(encoded, probabilities, target, available)
    shifted_level, shifted_change = components(
        encoded, probabilities, target + torch.tensor([2.0, -3.0]), available
    )

    assert not torch.equal(level, shifted_level)
    torch.testing.assert_close(change, shifted_change, rtol=0, atol=0)


def test_change_distinguishes_forward_and_reversed_target():
    target = torch.arange(25, dtype=torch.float32).reshape(1, 25, 1)
    encoded = target[:, None].clone()
    probabilities = torch.ones(1, 1)
    available = torch.ones(1, 25, 1, dtype=torch.bool)

    _, forward_change = components(encoded, probabilities, target, available)
    _, reversed_change = components(encoded, probabilities, target.flip(1), available)

    torch.testing.assert_close(forward_change, torch.tensor([0.0]))
    torch.testing.assert_close(reversed_change, torch.tensor([16.0]))


def test_change_breaks_a_level_score_tie_between_forward_and_reversed_targets():
    target = torch.arange(25, dtype=torch.float32).reshape(1, 25, 1)
    encoded = torch.zeros(1, 1, 25, 1)
    encoded[0, 0, :2, 0] = torch.tensor([11.0, -12.0])
    probabilities = torch.ones(1, 1)
    available = torch.ones(1, 25, 1, dtype=torch.bool)

    forward_level, forward_change = components(encoded, probabilities, target, available)
    reversed_level, reversed_change = components(encoded, probabilities, target.flip(1), available)

    torch.testing.assert_close(forward_level, reversed_level, rtol=0, atol=0)
    torch.testing.assert_close(forward_change, torch.tensor([math.sqrt(1337.0 / 17.0)]))
    torch.testing.assert_close(reversed_change, torch.tensor([math.sqrt(1369.0 / 17.0)]))
    assert forward_change < reversed_change


def test_missing_nans_and_all_missing_have_finite_first_gradients_without_mutation():
    torch.manual_seed(118)
    available = torch.ones(2, 12, 3, dtype=torch.bool)
    available[0, 0, 0] = False
    available[0, 1, 1] = False
    available[0, 8, 2] = False
    available[0, 10, 0] = False
    available[1] = False
    encoded = torch.randn(2, 4, 12, 3).masked_fill(~available[:, None], float("nan"))
    encoded.requires_grad_()
    target = torch.randn(2, 12, 3).masked_fill(~available, float("nan"))
    target.requires_grad_()
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    originals = tuple(value.detach().clone() for value in (encoded, probabilities, target, available))

    level, change = components(encoded, probabilities, target, available)
    clean_level, clean_change = components(
        torch.nan_to_num(encoded, nan=123.0),
        probabilities,
        torch.nan_to_num(target, nan=-456.0),
        available,
    )
    torch.testing.assert_close(level, clean_level)
    torch.testing.assert_close(change, clean_change)
    torch.testing.assert_close(level[1], torch.tensor(0.0))
    torch.testing.assert_close(change[1], torch.tensor(0.0))
    (level + change).sum().backward()

    for value in (encoded, probabilities, target):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[1]) == 0
    assert torch.count_nonzero(encoded.grad.masked_select(~available[:, None])) == 0
    assert torch.count_nonzero(target.grad[~available]) == 0
    for value, original in zip((encoded, probabilities, target, available), originals):
        torch.testing.assert_close(value.detach(), original, rtol=0, atol=0, equal_nan=True)


def test_ordered_model_loads_base_state_and_inherits_identical_forward():
    torch.manual_seed(119)
    center = torch.linspace(-1.0, 1.0, 23)
    scale = torch.linspace(0.5, 1.5, 23)
    base = JointTrajectory(center, scale, dim=64, scenarios=4).eval()
    ordered = OrderedTrajectory(center, scale, dim=64, scenarios=4, motion_weight=1.5).eval()

    incompatible = ordered.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.missing_keys == ["motion_weight"]
    assert incompatible.unexpected_keys == []
    assert OrderedTrajectory.forward is JointTrajectory.forward
    assert set(dict(ordered.named_buffers())) - set(dict(base.named_buffers())) == {"motion_weight"}
    assert ordered.motion_weight.ndim == 0
    assert torch.isfinite(ordered.motion_weight) and ordered.motion_weight >= 0
    assert ordered.motion_weight.item() == 1.5
    base_parameters = dict(base.named_parameters())
    ordered_parameters = dict(ordered.named_parameters())
    assert base_parameters.keys() == ordered_parameters.keys()
    assert sum(p.numel() for p in base_parameters.values() if p.requires_grad) == sum(
        p.numel() for p in ordered_parameters.values() if p.requires_grad
    )

    tokens = torch.randn(2, 17, 64)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    token_valid[0, :3] = False
    current = torch.randn(2, 23)
    current_available = torch.ones(2, 23, dtype=torch.bool)
    current_available[0, [2, 17]] = False
    current[~current_available] = float("nan")
    with torch.no_grad():
        expected = base(tokens, token_valid, current, current_available)
        actual = ordered(tokens, token_valid, current, current_available)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert actual["encoded"].shape == (2, 4, 25, 23)
    assert (actual["raw"][..., [17, 22]] >= 0).all()


def test_zero_weight_restores_exact_level_and_weighted_loss_is_per_sample():
    torch.manual_seed(120)
    encoded = torch.randn(2, 4, 25, 23)
    probabilities = torch.softmax(torch.randn(2, 4), dim=1)
    target = torch.randn(2, 25, 23)
    available = torch.rand(2, 25, 23) > 0.2
    level, change = components(encoded, probabilities, target, available)

    zero = OrderedTrajectory(torch.zeros(23), torch.ones(23), dim=8, motion_weight=0.0)
    weighted = OrderedTrajectory(torch.zeros(23), torch.ones(23), dim=8, motion_weight=2.5)
    assert torch.equal(zero.loss(encoded, probabilities, target, available), level)
    actual = weighted.loss(encoded, probabilities, target, available)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, level + 2.5 * change, rtol=0, atol=0)


def test_actual_loss_and_change_backpropagate_to_temporal_and_trajectory():
    torch.manual_seed(121)
    model = OrderedTrajectory(torch.zeros(23), torch.ones(23), dim=64)
    tokens = torch.randn(2, 17, 64)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    current = torch.randn(2, 23) * 0.2
    current_available = torch.ones(2, 23, dtype=torch.bool)
    current_available[1, 5:9] = False
    output = model(tokens, token_valid, current, current_available)
    target = torch.randn(2, 25, 23) * 0.3
    inputs = (output["encoded"], output["probabilities"], target, output["available"])
    _, change = components(*inputs)
    weights = (model.temporal.self_attn.in_proj_weight, model.trajectory.weight)
    change_gradients = torch.autograd.grad(change.mean(), weights, retain_graph=True)
    for gradient in change_gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    model.loss(*inputs).mean().backward()
    for weight in weights:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0


@pytest.mark.parametrize("lag", [0, -1, 25, 26, 1.5])
def test_invalid_lag_raises_value_error(lag):
    with pytest.raises(ValueError, match="lag"):
        components(
            torch.zeros(1, 1, 25, 1),
            torch.ones(1, 1),
            torch.zeros(1, 25, 1),
            torch.ones(1, 25, 1, dtype=torch.bool),
            lag=lag,
        )


@pytest.mark.parametrize("lag", [1, 24])
def test_valid_lag_boundaries_measure_differences_without_dividing_by_lag(lag):
    target = torch.arange(25, dtype=torch.float32).reshape(1, 25, 1)
    _, change = components(
        torch.zeros(1, 1, 25, 1),
        torch.ones(1, 1),
        target,
        torch.ones(1, 25, 1, dtype=torch.bool),
        lag=lag,
    )
    torch.testing.assert_close(change, torch.tensor([float(lag)]))


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), -float("inf"), [1.0]])
def test_motion_weight_must_be_a_finite_nonnegative_scalar(weight):
    with pytest.raises(ValueError, match="motion_weight"):
        OrderedTrajectory(torch.zeros(23), torch.ones(23), dim=8, motion_weight=weight)
