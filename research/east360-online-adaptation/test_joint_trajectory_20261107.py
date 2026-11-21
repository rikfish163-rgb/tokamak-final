import math

import torch
import torch.nn.functional as F

from joint_trajectory_20261107 import JointTrajectory, joint_energy_score


def test_single_scenario_is_masked_joint_rms():
    encoded = torch.tensor(
        [
            [[[3.0, 4.0, 50.0], [0.0, 0.0, 7.0]]],
            [[[-2.0, 5.0, 50.0], [7.0, 8.0, 3.0]]],
        ]
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]]
    )
    available = torch.tensor(
        [[[True, True, False], [False, False, True]], [[True, False, False], [False, True, False]]]
    )
    actual = joint_energy_score(encoded, torch.ones(2, 1), target, available)
    expected = torch.tensor([math.sqrt(74.0 / 3.0), math.sqrt(45.0 / 2.0)])
    torch.testing.assert_close(actual, expected)


def test_energy_score_weights_joint_distances_by_nonuniform_probabilities():
    encoded = torch.tensor([[[[0.0, 0.0]], [[3.0, 4.0]]]])
    probabilities = torch.tensor([[0.25, 0.75]])
    target = torch.tensor([[[0.0, 4.0]]])
    available = torch.ones(1, 1, 2, dtype=torch.bool)

    actual = joint_energy_score(encoded, probabilities, target, available)
    torch.testing.assert_close(actual, torch.tensor([2.3125 / math.sqrt(2.0)]))


def test_zero_observations_return_zero_with_finite_gradients():
    torch.manual_seed(11)
    encoded = torch.randn(2, 3, 2, 4, requires_grad=True)
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], requires_grad=True)
    target = torch.full((2, 2, 4), float("nan"), requires_grad=True)
    available = torch.zeros(2, 2, 4, dtype=torch.bool)

    score = joint_energy_score(encoded, probabilities, target, available)
    torch.testing.assert_close(score, torch.zeros(2))
    score.sum().backward()

    for value in (encoded, probabilities, target):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == 0


def test_scenario_and_probability_permutation_preserves_score():
    torch.manual_seed(12)
    encoded = torch.randn(2, 4, 3, 5)
    probabilities = torch.softmax(torch.randn(2, 4), dim=1)
    target = torch.randn(2, 3, 5)
    available = torch.rand(2, 3, 5) > 0.3
    permutation = torch.tensor([2, 0, 3, 1])

    actual = joint_energy_score(encoded, probabilities, target, available)
    reordered = joint_energy_score(
        encoded[:, permutation], probabilities[:, permutation], target, available
    )
    torch.testing.assert_close(actual, reordered)


def test_joint_score_distinguishes_dependence_with_identical_marginals():
    aligned = torch.tensor([[[[-1.0, -1.0]], [[1.0, 1.0]]]])
    opposed = torch.tensor([[[[-1.0, 1.0]], [[1.0, -1.0]]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[[1.0, 1.0]]])
    available = torch.ones(1, 1, 2, dtype=torch.bool)
    torch.testing.assert_close(aligned.sort(dim=1).values, opposed.sort(dim=1).values)

    aligned_score = joint_energy_score(aligned, probabilities, target, available)
    opposed_score = joint_energy_score(opposed, probabilities, target, available)
    torch.testing.assert_close(aligned_score, torch.tensor([0.5]))
    torch.testing.assert_close(opposed_score, torch.tensor([math.sqrt(2.0) - 0.5]))
    assert not torch.allclose(aligned_score, opposed_score)


def test_unobserved_nan_targets_do_not_change_score_or_gradients():
    torch.manual_seed(13)
    encoded = torch.randn(1, 3, 2, 3, requires_grad=True)
    probabilities = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    target = torch.tensor([[[1.0, float("nan"), 3.0], [float("nan"), 4.0, float("nan")]]], requires_grad=True)
    available = torch.tensor([[[True, False, True], [False, True, False]]])

    score = joint_energy_score(encoded, probabilities, target, available)
    replaced = joint_energy_score(encoded, probabilities, torch.nan_to_num(target, nan=123.0), available)
    torch.testing.assert_close(score, replaced)
    score.sum().backward()
    for value in (encoded, probabilities, target):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(target.grad[~available]) == 0


def test_model_contract_masks_inputs_and_initializes_distinct_paths():
    torch.manual_seed(14)
    center = torch.linspace(-1.0, 1.0, 23)
    scale = torch.linspace(0.5, 1.5, 23)
    model = JointTrajectory(center, scale, dim=8, scenarios=4)
    assert isinstance(model.temporal, torch.nn.TransformerEncoderLayer)
    assert model.temporal.self_attn.embed_dim == 8
    assert model.temporal.self_attn.num_heads == 4
    assert model.temporal.self_attn.batch_first
    assert model.temporal.linear1.out_features == 16
    assert model.temporal.dropout.p == 0.0
    assert model.temporal.norm_first
    assert model.temporal.activation is F.gelu
    torch.testing.assert_close(dict(model.named_buffers())["center"], center)
    torch.testing.assert_close(dict(model.named_buffers())["scale"], scale)

    tokens = torch.randn(2, 17, 8)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    token_valid[0, 0] = False
    token_valid[1, :5] = False
    current = torch.randn(2, 23)
    current_available = torch.ones(2, 23, dtype=torch.bool)
    current_available[0, [1, 17]] = False
    current_available[1, [4, 22]] = False
    current[~current_available] = float("nan")
    original = tuple(value.clone() for value in (tokens, token_valid, current, current_available))

    output = model(tokens, token_valid, current, current_available)
    assert set(output) == {"raw", "encoded", "probabilities", "available"}
    assert output["raw"].shape == (2, 4, 25, 23)
    assert output["encoded"].shape == (2, 4, 25, 23)
    assert output["probabilities"].shape == (2, 4)
    assert output["available"].shape == (2, 25, 23)
    assert torch.isfinite(output["raw"]).all()
    assert torch.isfinite(output["encoded"]).all()
    assert (output["raw"][..., [17, 22]] >= 0).all()
    assert (output["probabilities"] >= 0).all()
    torch.testing.assert_close(output["probabilities"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(output["available"], current_available[:, None, :].expand(-1, 25, -1))
    assert torch.count_nonzero(output["encoded"][:, 1:] - output["encoded"][:, :1]) > 0
    for value, before in zip((tokens, token_valid, current, current_available), original):
        torch.testing.assert_close(value, before, equal_nan=True)

    changed_tokens = tokens.clone()
    changed_tokens[~token_valid] += 1000.0
    changed_current = torch.where(current_available, current, torch.full_like(current, 1000.0))
    masked_output = model(changed_tokens, token_valid, changed_current, current_available)
    for name in ("raw", "encoded", "probabilities", "available"):
        torch.testing.assert_close(output[name], masked_output[name])


def test_model_uses_current_asinh_anchor_and_direct_nonnegative_variances():
    torch.manual_seed(15)
    center = torch.linspace(-1.0, 1.2, 23)
    scale = torch.linspace(0.5, 1.7, 23)
    model = JointTrajectory(center, scale, dim=8, scenarios=2)
    changes = torch.empty(2, 25, 23)
    changes[0].fill_(0.15)
    changes[1].fill_(-0.35)
    changes[..., 17] = -1.5
    changes[..., 22] = 0.7
    with torch.no_grad():
        model.trajectory.weight.zero_()
        model.trajectory.bias.copy_(changes.reshape(-1))

    tokens = torch.randn(2, 17, 8)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    current = center[None, :] + scale[None, :] * torch.tensor([[0.2], [-0.1]])
    current_available = torch.ones(2, 23, dtype=torch.bool)
    current_available[0, 1] = False
    current[0, 1] = float("nan")

    output = model(tokens, token_valid, current, current_available)
    anchor = torch.asinh((torch.where(current_available, current, center) - center) / scale)
    expected_raw = center + scale * torch.sinh(anchor[:, None, None, :] + changes[None, ...])
    expected_raw[..., 17] = scale[17] * F.softplus(changes[None, ..., 17])
    expected_raw[..., 22] = scale[22] * F.softplus(changes[None, ..., 22])
    torch.testing.assert_close(output["raw"], expected_raw)
    torch.testing.assert_close(output["encoded"], torch.asinh((expected_raw - center) / scale))
    assert (output["raw"][..., 0] < 0).all()


def test_real_energy_score_backpropagates_through_trajectory_and_temporal():
    torch.manual_seed(16)
    model = JointTrajectory(torch.zeros(23), torch.ones(23), dim=8, scenarios=4)
    tokens = torch.randn(2, 17, 8)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    token_valid[0, 3] = False
    current = torch.randn(2, 23) * 0.2
    current_available = torch.ones(2, 23, dtype=torch.bool)
    current_available[1, 5:9] = False
    output = model(tokens, token_valid, current, current_available)
    target = torch.randn(2, 25, 23) * 0.3

    score = joint_energy_score(
        output["encoded"], output["probabilities"].detach(), target, output["available"]
    )
    score.mean().backward()
    for weight in (model.temporal.self_attn.in_proj_weight, model.trajectory.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0
    assert model.probability.weight.grad is None


def test_inference_masks_nan_tokens_even_when_all_tokens_are_invalid():
    torch.manual_seed(17)
    model = JointTrajectory(torch.zeros(23), torch.ones(23), dim=8).eval()
    tokens = torch.randn(2, 17, 8)
    token_valid = torch.ones(2, 17, dtype=torch.bool)
    token_valid[0] = False
    token_valid[1, :5] = False
    tokens[~token_valid] = float("nan")
    current = torch.zeros(2, 23)
    current_available = torch.ones(2, 23, dtype=torch.bool)

    with torch.no_grad():
        output = model(tokens, token_valid, current, current_available)
        replaced = model(
            torch.where(token_valid[..., None], tokens, 0.0),
            token_valid,
            current,
            current_available,
        )

    for name in ("raw", "encoded", "probabilities", "available"):
        assert torch.isfinite(output[name]).all()
        torch.testing.assert_close(output[name], replaced[name])
