import torch

from experiments.risk_aligned_online.diagnostic_targets import coordinates, linear_extrapolation, trajectory_target


def observations():
    x = torch.arange(128.).reshape(1, 128, 1).expand(1, 128, 23).clone()
    return x, torch.ones_like(x, dtype=torch.bool)


def test_coordinate_is_past_only_and_affine_unit_invariant():
    x, available = observations()
    future = x[:, -1:] + torch.arange(1., 26.).reshape(1, 25, 1)
    mask = torch.ones_like(future, dtype=torch.bool)
    origin = coordinates(x, available)
    target, valid = trajectory_target(origin, future, mask)
    other = coordinates(x * 1000 + 700, available)
    converted, converted_valid = trajectory_target(other, future * 1000 + 700, mask)
    torch.testing.assert_close(converted, target)
    assert torch.equal(valid, converted_valid)
    reference = origin['scale'].clone()
    trajectory_target(origin, future * 10000, mask)
    assert torch.equal(origin['scale'], reference)


def test_constant_missing_and_insufficient_history_are_not_valid_targets():
    x, available = observations()
    x[:, :, 9] = 200000  # ip constant: its local scale is not identifiable
    available[:, -1, 7] = False  # dx_a reference missing
    available[:, :-15, 8] = False  # dy_a fewer than 16 past samples
    origin = coordinates(x, available)
    _, mask = trajectory_target(origin, x[:, -25:], available[:, -25:])
    assert not mask[:, :, :3].any()
    assert mask[:, :, 3:].all()


def test_missing_future_is_not_a_zero_training_observation():
    x, available = observations()
    origin = coordinates(x, available)
    future = x[:, -25:].clone()
    future[:, 2, 4] = float('nan')
    target, mask = trajectory_target(origin, future, torch.ones_like(future, dtype=torch.bool))
    assert not mask[:, 2, 3].any()
    assert torch.isfinite(target).all()


def test_regular_linear_ramp_has_no_false_diagnostic_innovation():
    x, available = observations()
    origin = coordinates(x, available)
    truth = x[:, -1:]+torch.arange(1., 26.).reshape(1, 25, 1)
    target, _ = trajectory_target(origin, truth, torch.ones_like(truth, dtype=torch.bool))
    prediction = linear_extrapolation(x, available, origin)
    torch.testing.assert_close(prediction, target)
