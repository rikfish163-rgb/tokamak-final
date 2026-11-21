"""CPU-only contract tests for matured-target shadow commit decisions."""

import math

import pytest
import torch

from experiments.risk_aligned_online.shadow_commit_20261104.decision import choose_commit


def blank(batch=1, dtype=torch.float64):
    candidate = torch.zeros(batch, 168, dtype=dtype)
    active = torch.zeros_like(candidate)
    target = torch.zeros_like(candidate)
    mask = torch.zeros(batch, 168, dtype=torch.bool)
    return candidate, active, target, mask


def test_guard_uses_float64_observed_mse_per_row_and_strict_comparison():
    candidate, active, target, mask = blank(4, torch.float32)
    target[0, :3] = torch.tensor([1, -1, 2])
    candidate[0, :3] = torch.tensor([1, 1, 1])
    active[0, :3] = torch.tensor([0, -1, 2])
    mask[0, :3] = True
    target[1, :2] = torch.tensor([2, 1])
    candidate[1, :2] = torch.tensor([2, 1])
    active[1, :2] = torch.tensor([3, 3])
    mask[1, :2] = True
    candidate[2, 0], active[2, 0], target[2, 0], mask[2, 0] = 1, -1, 0, True
    original = tuple(x.clone() for x in (candidate, active, target, mask))

    accept, candidate_mse, active_mse, support = choose_commit(candidate, active, target, mask)
    assert torch.equal(accept, torch.tensor([False, True, False, False]))
    assert torch.equal(support, torch.tensor([3, 2, 1, 0], dtype=torch.int64))
    torch.testing.assert_close(candidate_mse, torch.tensor([5 / 3, 0, 1, 0], dtype=torch.float64))
    torch.testing.assert_close(active_mse, torch.tensor([1 / 3, 2.5, 1, 0], dtype=torch.float64))
    assert accept.dtype == torch.bool and candidate_mse.dtype == active_mse.dtype == torch.float64
    assert all(torch.equal(x, prior) for x, prior in zip((candidate, active, target, mask), original))


def test_delay_accepts_worse_and_equal_rows_only_when_supported():
    candidate, active, target, mask = blank(3)
    candidate[0, 0], active[0, 0], mask[0, 0] = 2, 1, True
    candidate[1, 0], active[1, 0], mask[1, 0] = 1, -1, True
    guard, cmse, amse, support = choose_commit(candidate, active, target, mask, policy='guard')
    delay, cmse2, amse2, support2 = choose_commit(candidate, active, target, mask, policy='delay')
    assert torch.equal(guard, torch.tensor([False, False, False]))
    assert torch.equal(delay, torch.tensor([True, True, False]))
    for x, y in ((cmse, cmse2), (amse, amse2), (support, support2)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_guard_accepts_arbitrarily_small_representable_improvement():
    candidate, active, target, mask = blank()
    candidate[0, 0] = 1 - 2 ** -40
    active[0, 0] = 1
    mask[0, 0] = True
    accept, cmse, amse, _ = choose_commit(candidate, active, target, mask)
    assert cmse[0] < amse[0] and accept[0]
    assert 0 < (amse - cmse).item() < 1e-10


def test_unobserved_nan_and_infinity_in_all_inputs_are_ignored():
    candidate, active, target, mask = blank(2)
    candidate[0, 0], active[0, 0], target[0, 0], mask[0, 0] = .5, 2, 1, True
    expected = choose_commit(candidate, active, target, mask)
    candidate[~mask] = float('nan')
    active[~mask] = float('inf')
    target[~mask] = float('-inf')
    actual = choose_commit(candidate, active, target, mask)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert torch.isfinite(actual[1]).all() and torch.isfinite(actual[2]).all()


@pytest.mark.parametrize('field', ['candidate', 'active', 'target'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_observed_values_are_rejected(field, value):
    candidate, active, target, mask = blank()
    mask[0, 0] = True
    {'candidate': candidate, 'active': active, 'target': target}[field][0, 0] = value
    with pytest.raises(ValueError):
        choose_commit(candidate, active, target, mask)


def test_no_support_is_not_accepted_and_errors_are_zero_for_both_policies():
    candidate, active, target, mask = blank(2)
    candidate.fill_(float('nan'))
    active.fill_(float('-inf'))
    target.fill_(float('nan'))
    for policy in ('guard', 'delay'):
        accept, cmse, amse, support = choose_commit(candidate, active, target, mask, policy=policy)
        assert not accept.any() and not support.any()
        assert torch.equal(cmse, torch.zeros(2, dtype=torch.float64))
        assert torch.equal(amse, torch.zeros(2, dtype=torch.float64))


def test_subtraction_and_mse_are_computed_after_float64_conversion():
    candidate, active, target, mask = blank(dtype=torch.float32)
    candidate[0, 0], active[0, 0], target[0, 0], mask[0, 0] = 3e38, 2e38, -3e38, True
    accept, cmse, amse, _ = choose_commit(candidate, active, target, mask)
    expected_c = (candidate[0, 0].double() - target[0, 0].double()).square()
    expected_a = (active[0, 0].double() - target[0, 0].double()).square()
    assert torch.isfinite(cmse).all() and torch.isfinite(amse).all()
    torch.testing.assert_close(cmse[0], expected_c)
    torch.testing.assert_close(amse[0], expected_a)
    assert not accept[0]


def test_large_finite_errors_with_representable_mse_do_not_overflow_early():
    candidate, active, target, mask = blank()
    candidate[0, 0], active[0, 0] = 1.5e154, 1.6e154
    mask[:] = True
    accept, cmse, amse, support = choose_commit(candidate, active, target, mask)
    assert support[0] == 168 and accept[0]
    assert torch.isfinite(cmse).all() and torch.isfinite(amse).all()
    torch.testing.assert_close(cmse[0], torch.tensor((1.5e154 / math.sqrt(168)) ** 2, dtype=torch.float64))
    torch.testing.assert_close(amse[0], torch.tensor((1.6e154 / math.sqrt(168)) ** 2, dtype=torch.float64))


def test_unrepresentable_mse_is_rejected_instead_of_producing_a_false_decision():
    candidate, active, target, mask = blank()
    candidate[0, 0], target[0, 0], mask[0, 0] = 1e308, -1e308, True
    with pytest.raises(ValueError, match='representable'):
        choose_commit(candidate, active, target, mask)


def test_existing_direct_target_and_mask_feed_the_decision_unchanged():
    from experiments.risk_aligned_online.direct_fast_memory_20261104.memory import direct_target

    level = torch.zeros(1, 128, 23, dtype=torch.float64)
    available = torch.zeros_like(level, dtype=torch.bool)
    future = torch.zeros(1, 8, 23, dtype=torch.float64)
    future_available = torch.zeros_like(future, dtype=torch.bool)
    available[0, -1, 0] = True
    future_available[0, 0, 0] = True
    future[0, 0, 0] = 2
    target, mask = direct_target(level, available, future, future_available)
    accept, cmse, amse, support = choose_commit(target.clone(), torch.zeros_like(target), target, mask)
    assert accept[0] and support[0] == 1 and cmse[0] == 0 and amse[0] == 4


def test_missing_values_have_finite_zero_gradients():
    candidate, active, target, mask = blank()
    mask[0, 0] = True
    candidate[0, 0], active[0, 0], target[0, 0] = 1, 2, 0
    candidate[~mask], active[~mask], target[~mask] = float('nan'), float('inf'), float('-inf')
    candidate.requires_grad_()
    active.requires_grad_()
    target.requires_grad_()
    accept, cmse, amse, _ = choose_commit(candidate, active, target, mask)
    assert accept[0]
    grads = torch.autograd.grad(cmse.sum() + amse.sum(), (candidate, active, target))
    assert all(torch.isfinite(g).all() and (g[~mask] == 0).all() for g in grads)
    assert grads[0][0, 0] == 2 and grads[1][0, 0] == 4 and grads[2][0, 0] == -6


def test_batch_rows_match_independent_calls():
    torch.manual_seed(1401)
    candidate = torch.randn(5, 168, dtype=torch.float64)
    active = torch.randn_like(candidate)
    target = torch.randn_like(candidate)
    mask = torch.rand(5, 168) > .4
    mask[2] = False
    for policy in ('guard', 'delay'):
        full = choose_commit(candidate, active, target, mask, policy=policy)
        for i in range(5):
            row = choose_commit(candidate[i:i+1], active[i:i+1], target[i:i+1],
                                mask[i:i+1], policy=policy)
            for x, y in zip(full, row):
                torch.testing.assert_close(x[i:i+1], y)


def test_empty_batch_returns_empty_matching_outputs():
    result = choose_commit(*blank(0))
    assert all(value.shape == (0,) for value in result)
    assert result[0].dtype == torch.bool and result[1].dtype == result[2].dtype == torch.float64
    assert result[3].dtype == torch.int64


@pytest.mark.parametrize('policy', ['Guard', 'always', '', None, 1, True])
def test_unknown_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        choose_commit(*blank(), policy=policy)


def test_bad_shapes_mask_and_mismatched_floating_dtypes_are_rejected():
    candidate, active, target, mask = blank()
    calls = [
        lambda: choose_commit(candidate[:, :-1], active, target, mask),
        lambda: choose_commit(candidate, active[:, :-1], target, mask),
        lambda: choose_commit(candidate, active, target[:, :-1], mask),
        lambda: choose_commit(candidate, active, target, mask[:, :-1]),
        lambda: choose_commit(candidate.reshape(168), active, target, mask),
        lambda: choose_commit(candidate, active.float(), target, mask),
        lambda: choose_commit(candidate, active, target.float(), mask),
        lambda: choose_commit(candidate.long(), active, target, mask),
        lambda: choose_commit(candidate, active, target, mask.float()),
        lambda: choose_commit(None, active, target, mask),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('field', ['candidate', 'active', 'target', 'mask'])
def test_mismatched_meta_device_is_rejected_before_reading_meta_values(field):
    candidate, active, target, mask = blank()
    inputs = dict(candidate=candidate, active=active, target=target, mask=mask)
    value = inputs[field]
    inputs[field] = torch.empty(value.shape, dtype=value.dtype, device='meta')
    with pytest.raises(ValueError, match='same device'):
        choose_commit(inputs['candidate'], inputs['active'], inputs['target'], inputs['mask'])
