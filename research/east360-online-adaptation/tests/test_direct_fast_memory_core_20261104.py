"""CPU-only contract tests for direct diagnostic fast memory."""

import math

import pytest
import torch

from experiments.risk_aligned_online.direct_fast_memory_20261104.memory import (
    direct_target,
    dynamic_key,
    fast_step,
    innovation,
)


CONTINUOUS = tuple(c for c in range(23) if c not in (2, 10))
DTYPE = torch.float64


def blank_window(batch=1):
    return (torch.zeros(batch, 128, 23, dtype=DTYPE),
            torch.zeros(batch, 128, 23, dtype=torch.bool))


def blank_future(batch=1):
    return (torch.zeros(batch, 8, 23, dtype=DTYPE),
            torch.zeros(batch, 8, 23, dtype=torch.bool))


def blank_state(batch=1):
    return (torch.zeros(batch, 168, 70, dtype=DTYPE),
            torch.zeros(batch, 70, dtype=DTYPE),
            torch.zeros(batch, 168, dtype=DTYPE),
            torch.zeros(batch, 168, dtype=torch.bool))


def test_key_uses_actual_last_masked_means_and_last_sixteen_fraction():
    level, available = blank_window()
    for time, channel, value in ((112, 0, 1), (119, 0, 3), (120, 0, 5),
                                 (127, 0, 7), (112, 1, 4), (126, 2, 9)):
        level[0, time, channel] = value
        available[0, time, channel] = True
    # Outside the two eight-step summaries, and therefore not a key feature.
    level[0, 111, 0] = 300
    available[0, 111, 0] = True
    key = dynamic_key(level, available)
    expected = torch.zeros(1, 70, dtype=DTYPE)
    expected[0, 0] = 7
    expected[0, 23] = 4  # mean(5, 7) - mean(1, 3)
    expected[0, 46:49] = torch.tensor([4 / 16, 1 / 16, 1 / 16], dtype=DTYPE)
    expected[:, :69] /= math.sqrt(69)
    expected[:, 69] = 1
    torch.testing.assert_close(key, expected)
    assert key.shape == (1, 70)


def test_key_ignores_all_unobserved_values_including_nonfinite():
    level, available = blank_window(2)
    available[0, 127, 1] = True
    level[0, 127, 1] = 2
    expected = dynamic_key(level, available)
    contaminated = level.clone()
    contaminated[~available] = float('nan')
    contaminated[1, 0, 0] = float('inf')
    torch.testing.assert_close(dynamic_key(contaminated, available), expected)
    torch.testing.assert_close(expected[1, :69], torch.zeros(69, dtype=DTYPE))
    assert expected[1, 69] == 1


def test_key_rejects_nonfinite_observations_even_outside_summary():
    level, available = blank_window()
    available[0, 0, 4] = True
    level[0, 0, 4] = float('nan')
    with pytest.raises(ValueError):
        dynamic_key(level, available)


def test_target_is_time_major_future_minus_true_last_for_21_channels():
    level, available = blank_window()
    future, future_available = blank_future()
    level[0, -1] = torch.arange(23, dtype=DTYPE) * 2
    available[0, -1] = True
    for t in range(8):
        future[0, t] = torch.arange(23, dtype=DTYPE) * (t + 3) + t
    future_available[:] = True
    target, mask = direct_target(level, available, future, future_available)
    expected = (future[:, :, CONTINUOUS] - level[:, -1:, CONTINUOUS]).reshape(1, 168)
    torch.testing.assert_close(target, expected)
    assert mask.shape == target.shape == (1, 168)
    assert mask.dtype == torch.bool and mask.all()
    # Adjacent channel and time positions encode different actual values.
    assert target[0, 1] != target[0, 0]
    assert target[0, 21] != target[0, 0]


def test_target_validity_requires_each_future_and_actual_last_observation():
    level, available = blank_window()
    future, future_available = blank_future()
    available[:] = True
    future_available[:] = True
    # An earlier observation must not substitute for a missing last one.
    available[0, -1, 1] = False
    level[0, -2, 1] = 999
    future_available[0, 3, 4] = False
    target, mask = direct_target(level, available, future, future_available)
    mask = mask.reshape(1, 8, 21)
    target = target.reshape(1, 8, 21)
    assert not mask[0, :, CONTINUOUS.index(1)].any()
    assert not mask[0, 3, CONTINUOUS.index(4)]
    assert mask[0, 2, CONTINUOUS.index(4)]
    assert (target[~mask] == 0).all()


def test_target_ignores_unobserved_nans_and_does_not_derive_physical_units():
    level, available = blank_window()
    future, future_available = blank_future()
    level[:] = float('nan')
    future[:] = float('inf')
    level[0, -1, 0] = 1.25
    available[0, -1, 0] = True
    future[0, 0, 0] = 1.75
    future_available[0, 0, 0] = True
    target, mask = direct_target(level, available, future, future_available)
    assert mask.sum() == 1 and target[0, 0] == .5
    assert torch.isfinite(target).all() and (target[~mask] == 0).all()


@pytest.mark.parametrize('bad_field', ['past', 'future', 'excluded_future'])
def test_target_rejects_nonfinite_observed_inputs(bad_field):
    level, available = blank_window()
    future, future_available = blank_future()
    if bad_field == 'past':
        level[0, 0, 0], available[0, 0, 0] = float('inf'), True
    else:
        c = 2 if bad_field == 'excluded_future' else 0
        future[0, 0, c], future_available[0, 0, c] = float('nan'), True
    with pytest.raises(ValueError):
        direct_target(level, available, future, future_available)


def test_past_and_future_masks_block_gradients_from_missing_values():
    level, available = blank_window()
    future, future_available = blank_future()
    available[0, 127, 0] = True
    future_available[0, 0, 0] = True
    level.requires_grad_()
    future.requires_grad_()
    key = dynamic_key(level, available)
    target, _ = direct_target(level, available, future, future_available)
    (key.sum() + target.sum()).backward()
    assert (level.grad[~available] == 0).all()
    assert (future.grad[~future_available] == 0).all()
    assert level.grad[0, -1, 0] != 0 and future.grad[0, 0, 0] == 1


def test_fast_step_matches_exact_normalized_analytic_update_and_row_loss():
    torch.manual_seed(7)
    weight = torch.randn(2, 168, 70, dtype=DTYPE) * .01
    key = torch.randn(2, 70, dtype=DTYPE) * .2
    target = torch.randn(2, 168, dtype=DTYPE)
    mask = torch.rand(2, 168) > .4
    original = weight.clone()
    pred = torch.bmm(weight, key.unsqueeze(-1)).squeeze(-1)
    error = torch.where(mask, pred - target, 0)
    expected = weight - .5 * error.unsqueeze(-1) * key.unsqueeze(1) / (
        1 + key.square().sum(-1, keepdim=True)).unsqueeze(-1)
    expected_loss = .5 * error.square().sum(-1) / mask.sum(-1).clamp_min(1)
    updated, loss = fast_step(weight, key, target, mask, origin=weight, radius=100)
    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(weight, original)
    assert loss.shape == (2,)


def test_update_equals_autograd_per_output_gradient_without_dividing_by_168():
    torch.manual_seed(8)
    weight = (torch.randn(1, 168, 70, dtype=DTYPE) * .01).requires_grad_()
    key = torch.randn(1, 70, dtype=DTYPE) * .1
    target = torch.randn(1, 168, dtype=DTYPE)
    mask = torch.zeros(1, 168, dtype=torch.bool)
    mask[0, [0, 17, 167]] = True
    pred = torch.bmm(weight, key.unsqueeze(-1)).squeeze(-1)
    objective = .5 * ((pred - target).square() * mask).sum()
    gradient, = torch.autograd.grad(objective, weight)
    expected = weight - .4 * gradient / (1 + key.square().sum())
    result, _ = fast_step(weight, key, target, mask, lr=.4, radius=100, origin=weight)
    torch.testing.assert_close(result, expected)


def test_unprojected_update_strictly_reduces_observed_loss():
    weight, key, target, mask = blank_state()
    key[0, :3] = torch.tensor([.5, -1, 2], dtype=DTYPE)
    target[0, :2] = torch.tensor([1, -2], dtype=DTYPE)
    mask[0, :2] = True
    updated, before = fast_step(weight, key, target, mask, radius=100)
    _, after = fast_step(updated, key, target, mask, radius=100)
    assert after.item() < before.item()


def test_masked_finite_targets_do_not_change_update_and_empty_rows_do_not_move():
    weight, key, target, mask = blank_state(2)
    key[:, -1] = 1
    target[0, 0] = 3
    mask[0, 0] = True
    altered = target.clone()
    altered[~mask] = 12345
    first, loss = fast_step(weight, key, target, mask)
    second, second_loss = fast_step(weight, key, altered, mask)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(loss, second_loss)
    torch.testing.assert_close(first[1], weight[1], rtol=0, atol=0)
    assert loss[1] == 0


def test_per_row_projection_is_relative_to_origin_and_repeated_steps_stay_bounded():
    weight, key, target, mask = blank_state(2)
    origin = torch.full((168, 70), .1, dtype=DTYPE)
    weight = weight + origin
    key[:, -1] = 1
    target[0] = 100
    target[1] = -50
    mask[:] = True
    for _ in range(5):
        weight, _ = fast_step(weight, key, target, mask, radius=.3, origin=origin)
        norm = torch.linalg.vector_norm((weight - origin).flatten(1), dim=1)
        assert (norm <= .3 + 1e-12).all()
    torch.testing.assert_close(norm, torch.full((2,), .3, dtype=DTYPE))


def test_inactive_row_is_unchanged_even_if_caller_supplies_out_of_ball_state():
    weight, key, target, mask = blank_state()
    weight[0, 0, 0] = 5
    result, loss = fast_step(weight, key, target, mask, radius=2)
    torch.testing.assert_close(result, weight, rtol=0, atol=0)
    assert loss[0] == 0


def test_innovation_is_initially_zero_row_independent_and_reset_is_caller_owned():
    torch.manual_seed(9)
    shared_origin = torch.randn(168, 70, dtype=DTYPE) * .01
    state = shared_origin.expand(2, -1, -1).clone()
    key = torch.randn(2, 70, dtype=DTYPE) * .1
    target = torch.ones(2, 168, dtype=DTYPE)
    mask = torch.zeros(2, 168, dtype=torch.bool)
    mask[0, :5] = True
    initial = innovation(state, shared_origin, key)
    assert torch.equal(initial, torch.zeros_like(initial))
    updated, _ = fast_step(state, key, target, mask, origin=shared_origin)
    result = innovation(updated, shared_origin, key)
    assert result[0].abs().sum() > 0
    assert torch.equal(result[1], torch.zeros_like(result[1]))
    fresh_shot = shared_origin.unsqueeze(0).clone()
    assert torch.equal(innovation(fresh_shot, shared_origin, key[:1]), initial[:1])
    torch.testing.assert_close(state, shared_origin.expand_as(state))


def test_real_spatial_target_difference_produces_distinct_output_innovation():
    level, available = blank_window()
    future, future_available = blank_future()
    available[0, -1, :2] = True
    future_available[0, 0, :2] = True
    future[0, 0, 0] = 3
    future[0, 0, 1] = -1
    key = dynamic_key(level, available)
    target, mask = direct_target(level, available, future, future_available)
    weight, _, _, _ = blank_state()
    updated, _ = fast_step(weight, key, target, mask)
    change = innovation(updated, weight, key)
    assert change[0, 0] > 0 > change[0, 1]
    assert (change[0, 2:] == 0).all()


def test_fast_step_and_innovation_are_differentiable_in_all_floating_inputs():
    torch.manual_seed(10)
    weight = (torch.randn(1, 168, 70, dtype=DTYPE) * .001).requires_grad_()
    origin = (torch.randn(1, 168, 70, dtype=DTYPE) * .001).requires_grad_()
    key = (torch.randn(1, 70, dtype=DTYPE) * .1).requires_grad_()
    target = torch.randn(1, 168, dtype=DTYPE, requires_grad=True)
    mask = torch.ones(1, 168, dtype=torch.bool)
    updated, loss = fast_step(weight, key, target, mask, origin=origin, radius=100)
    objective = innovation(updated, origin, key).square().mean() + loss.mean()
    gradients = torch.autograd.grad(objective, (weight, origin, key, target))
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().sum() > 0 for g in gradients)


def test_simultaneously_active_batch_rows_match_independent_steps():
    torch.manual_seed(11)
    origin = torch.randn(3, 168, 70, dtype=DTYPE) * .001
    weight = origin.clone()
    key = torch.randn(3, 70, dtype=DTYPE) * .1
    target = torch.randn(3, 168, dtype=DTYPE) * 5
    mask = torch.rand(3, 168) > .3
    batched, losses = fast_step(weight, key, target, mask, origin=origin, radius=.2)
    for i in range(3):
        single, single_loss = fast_step(weight[i:i+1], key[i:i+1], target[i:i+1],
                                        mask[i:i+1], origin=origin[i:i+1], radius=.2)
        torch.testing.assert_close(batched[i:i+1], single)
        torch.testing.assert_close(losses[i:i+1], single_loss)
        torch.testing.assert_close(innovation(batched[i:i+1], origin[i:i+1], key[i:i+1]),
                                   innovation(single, origin[i:i+1], key[i:i+1]))


def test_projected_step_has_the_expected_directional_derivative():
    torch.manual_seed(12)
    weight, key, _, mask = blank_state()
    key[0] = torch.randn(70, dtype=DTYPE) * .1
    mask[:] = True
    base_target = torch.randn(1, 168, dtype=DTYPE)
    direction = torch.randn(1, 168, dtype=DTYPE)
    probe = torch.randn_like(weight)

    def objective(theta):
        updated, _ = fast_step(weight, key, base_target + theta * direction, mask,
                               origin=weight, radius=.02)
        return (updated * probe).sum()

    theta = torch.tensor(.2, dtype=DTYPE, requires_grad=True)
    derivative, = torch.autograd.grad(objective(theta), theta)
    eps = 1e-6
    finite_difference = (objective(theta.detach() + eps) - objective(theta.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-6, atol=1e-8)
    assert derivative.abs() > 0


def test_float32_cpu_path_preserves_dtype_and_bounded_innovation():
    level, available = blank_window()
    future, future_available = blank_future()
    level, future = level.float(), future.float()
    available[:] = True
    future_available[:] = True
    future[:] = 100
    key = dynamic_key(level, available)
    target, mask = direct_target(level, available, future, future_available)
    weight = torch.zeros(1, 168, 70, dtype=torch.float32)
    updated, loss = fast_step(weight, key, target, mask, origin=weight, radius=.2)
    assert key.dtype == target.dtype == updated.dtype == loss.dtype == torch.float32
    assert innovation(updated, weight, key).dtype == torch.float32
    assert torch.linalg.vector_norm(updated) <= .2 + 1e-6


@pytest.mark.parametrize('name,value', [
    ('lr', 0), ('lr', -1), ('lr', 1.01), ('lr', float('nan')),
    ('lr', float('inf')), ('lr', True), ('radius', 0), ('radius', -1),
    ('radius', float('nan')), ('radius', float('inf')), ('radius', False),
])
def test_step_rejects_invalid_hyperparameters(name, value):
    weight, key, target, mask = blank_state()
    with pytest.raises(ValueError):
        fast_step(weight, key, target, mask, **{name: value})


@pytest.mark.parametrize('field', ['weight', 'key', 'target', 'origin'])
def test_step_rejects_nonfinite_floating_inputs(field):
    weight, key, target, mask = blank_state()
    origin = weight.clone()
    {'weight': weight, 'key': key, 'target': target, 'origin': origin}[field].view(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        fast_step(weight, key, target, mask, origin=origin)


def test_bad_shapes_mask_types_dtypes_and_devices_are_rejected():
    level, available = blank_window()
    future, future_available = blank_future()
    weight, key, target, mask = blank_state()
    bad_calls = [
        lambda: dynamic_key(level[:, :-1], available[:, :-1]),
        lambda: dynamic_key(level, available.float()),
        lambda: dynamic_key(level.long(), available),
        lambda: direct_target(level, available, future[:, :-1], future_available[:, :-1]),
        lambda: direct_target(level, available, future, future_available.float()),
        lambda: direct_target(level, available, future.float(), future_available),
        lambda: fast_step(weight[:, :-1], key, target, mask),
        lambda: fast_step(weight, key[:, :-1], target, mask),
        lambda: fast_step(weight, key, target[:, :-1], mask),
        lambda: fast_step(weight, key, target, mask.float()),
        lambda: fast_step(weight, key.float(), target, mask),
        lambda: fast_step(weight, key, target, mask, origin=weight[:, :-1]),
        lambda: innovation(weight, weight, key[:, :-1]),
        lambda: innovation(weight, weight.float(), key),
        lambda: dynamic_key(torch.empty((1, 128, 23), device='meta'), available),
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('case', [
    'window_mask', 'window_level', 'future', 'future_mask', 'step_weight',
    'step_key', 'step_target', 'step_mask', 'step_origin', 'innovation_origin',
    'innovation_key',
])
def test_mismatched_devices_are_rejected_before_meta_finite_checks(case):
    level, available = blank_window()
    future, future_available = blank_future()
    weight, key, target, mask = blank_state()

    def meta_like(value):
        return torch.empty(value.shape, dtype=value.dtype, device='meta')

    calls = {
        'window_mask': lambda: dynamic_key(level, meta_like(available)),
        'window_level': lambda: dynamic_key(meta_like(level), available),
        'future': lambda: direct_target(level, available, meta_like(future), future_available),
        'future_mask': lambda: direct_target(level, available, future, meta_like(future_available)),
        'step_weight': lambda: fast_step(meta_like(weight), key, target, mask),
        'step_key': lambda: fast_step(weight, meta_like(key), target, mask),
        'step_target': lambda: fast_step(weight, key, meta_like(target), mask),
        'step_mask': lambda: fast_step(weight, key, target, meta_like(mask)),
        'step_origin': lambda: fast_step(weight, key, target, mask, origin=meta_like(weight)),
        'innovation_origin': lambda: innovation(weight, meta_like(weight), key),
        'innovation_key': lambda: innovation(weight, weight, meta_like(key)),
    }
    with pytest.raises(ValueError, match='same device'):
        calls[case]()
