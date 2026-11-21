"""CPU-only regression tests for the projected recursive ridge core."""

import pytest
import torch

from experiments.risk_aligned_online.direct_fast_memory_20261104.memory import (
    direct_target,
    dynamic_key,
    innovation,
)
from experiments.risk_aligned_online.direct_fast_memory_20261104.rls import (
    initial_covariance,
    ridge_step,
)


DTYPE = torch.float64


def state(batch=1, dtype=DTYPE):
    weight = torch.zeros(batch, 168, 70, dtype=dtype)
    covariance = initial_covariance(batch, weight)
    key = torch.zeros(batch, 70, dtype=dtype)
    target = torch.zeros(batch, 168, dtype=dtype)
    mask = torch.zeros(batch, 168, dtype=torch.bool)
    return weight, covariance, key, target, mask


def test_initial_covariance_is_independent_identity_for_each_shot_and_output():
    reference = torch.zeros(168, 70, dtype=DTYPE)
    covariance = initial_covariance(2, reference)
    eye = torch.eye(70, dtype=DTYPE)
    assert covariance.shape == (2, 168, 70, 70)
    assert covariance.dtype == reference.dtype and covariance.device == reference.device
    torch.testing.assert_close(covariance[0, 0], eye, rtol=0, atol=0)
    torch.testing.assert_close(covariance[1, 167], eye, rtol=0, atol=0)
    covariance[0, 0, 0, 0] = 2
    assert covariance[0, 1, 0, 0] == covariance[1, 0, 0, 0] == 1


def test_single_step_matches_analytic_gain_covariance_and_pre_update_loss():
    weight, covariance, key, target, mask = state()
    key[0, 0] = 2
    weight[0, 0, 0] = .5
    target[0, 0] = 3
    mask[0, 0] = True
    origin = weight.clone()
    actual_w, actual_p, loss = ridge_step(
        weight, covariance, key, target, mask, origin=origin, radius=100
    )
    expected_w = weight.clone()
    expected_w[0, 0, 0] += (3 - 1) * 2 / 5
    expected_p = covariance.clone()
    expected_p[0, 0, 0, 0] = 1 / 5
    torch.testing.assert_close(actual_w, expected_w)
    torch.testing.assert_close(actual_p, expected_p)
    assert loss.shape == (1,) and loss[0] == 2  # half of (3 - 1)^2


def test_sequential_updates_equal_closed_form_recursive_ridge_posterior():
    torch.manual_seed(101)
    initial = torch.randn(1, 168, 70, dtype=DTYPE) * .01
    weight = initial.clone()
    covariance = initial_covariance(1, initial)
    keys = torch.randn(7, 70, dtype=DTYPE) * .1
    targets = torch.randn(7, 168, dtype=DTYPE)
    mask = torch.ones(1, 168, dtype=torch.bool)
    for key, target in zip(keys, targets):
        weight, covariance, _ = ridge_step(
            weight, covariance, key[None], target[None], mask,
            origin=initial, radius=100,
        )
    precision = torch.eye(70, dtype=DTYPE) + keys.T @ keys
    expected_p = torch.linalg.solve(precision, torch.eye(70, dtype=DTYPE))
    expected_w = torch.linalg.solve(precision, initial[0].T + keys.T @ targets).T
    torch.testing.assert_close(weight[0], expected_w, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(covariance[0], expected_p.expand(168, -1, -1),
                               rtol=1e-11, atol=1e-11)
    torch.linalg.cholesky(covariance[0, [0, 167]])


def test_partial_masks_have_separate_per_output_ridge_posteriors():
    torch.manual_seed(102)
    initial = torch.randn(1, 168, 70, dtype=DTYPE) * .01
    weight, covariance = initial.clone(), initial_covariance(1, initial)
    keys = torch.randn(6, 70, dtype=DTYPE) * .1
    targets = torch.randn(6, 168, dtype=DTYPE)
    masks = torch.zeros(6, 168, dtype=torch.bool)
    masks[:, 0] = True
    masks[::2, 1] = True
    masks[1::2, 2] = True
    masks[3, 3] = True
    for key, target, mask in zip(keys, targets, masks):
        weight, covariance, _ = ridge_step(
            weight, covariance, key[None], target[None], mask[None],
            origin=initial, radius=100,
        )
    eye = torch.eye(70, dtype=DTYPE)
    for output in (0, 1, 2, 3, 167):
        selected = masks[:, output]
        x, y = keys[selected], targets[selected, output]
        precision = eye + x.T @ x
        expected_p = torch.linalg.solve(precision, eye)
        expected_w = torch.linalg.solve(precision, initial[0, output] + x.T @ y)
        torch.testing.assert_close(covariance[0, output], expected_p, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(weight[0, output], expected_w, rtol=1e-11, atol=1e-11)
    assert torch.equal(weight[0, 167], initial[0, 167])
    assert torch.equal(covariance[0, 167], eye)


def test_repeated_direction_reduces_gain_and_fixed_target_error():
    weight, covariance, key, target, mask = state()
    key[0, 0] = 1
    target[0, 0] = 1
    mask[0, 0] = True
    origin = weight.clone()
    gains, errors = [], []
    for _ in range(8):
        gains.append(float(covariance[0, 0, 0, 0] / (1 + covariance[0, 0, 0, 0])))
        errors.append(abs(1 - float(weight[0, 0, 0])))
        weight, covariance, loss = ridge_step(weight, covariance, key, target, mask,
                                              origin=origin, radius=100)
        assert torch.isfinite(loss).all()
    assert all(a > b > 0 for a, b in zip(gains, gains[1:]))
    assert all(a > b > 0 for a, b in zip(errors, errors[1:]))
    torch.testing.assert_close(covariance[0, 0, 0, 0], torch.tensor(1 / 9, dtype=DTYPE))


def test_noisy_repetitions_stay_finite_and_covariance_stays_positive():
    weight, covariance, key, target, mask = state()
    key[0, :2] = torch.tensor([1., -.5], dtype=DTYPE)
    mask[0, 0] = True
    origin = weight.clone()
    changes = []
    for i in range(20):
        target[0, 0] = 1 + (-.2 if i % 2 else .2)
        previous = weight.clone()
        weight, covariance, loss = ridge_step(weight, covariance, key, target, mask,
                                              origin=origin, radius=100)
        changes.append(float((weight - previous).norm()))
        assert torch.isfinite(weight).all() and torch.isfinite(covariance).all()
        assert torch.isfinite(loss).all()
    assert max(changes[-4:]) < max(changes[:4])
    torch.linalg.cholesky(covariance[0, 0])
    torch.testing.assert_close(covariance[0, 0], covariance[0, 0].T, rtol=0, atol=0)


def test_masked_outputs_and_an_unsupported_out_of_ball_row_are_exactly_unchanged():
    weight, covariance, key, target, mask = state(2)
    weight[1, 0, 0] = 5
    key[:, 0] = 1
    target[:, 0] = 3
    mask[0, 0] = True
    original_w, original_p = weight.clone(), covariance.clone()
    updated, new_p, loss = ridge_step(weight, covariance, key, target, mask,
                                     origin=torch.zeros(168, 70, dtype=DTYPE))
    assert torch.equal(updated[1], original_w[1])
    assert torch.equal(new_p[1], original_p[1]) and loss[1] == 0
    assert torch.equal(new_p[0, 1:], original_p[0, 1:])
    assert torch.equal(weight, original_w) and torch.equal(covariance, original_p)


def test_finite_masked_targets_do_not_affect_update_or_loss():
    weight, covariance, key, target, mask = state()
    key[0, 0], target[0, 0], mask[0, 0] = 1, 2, True
    modified = target.clone()
    modified[~mask] = 9999
    a = ridge_step(weight, covariance, key, target, mask, origin=weight)
    b = ridge_step(weight, covariance, key, modified, mask, origin=weight)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_active_rows_are_projected_about_shared_origin_and_covariance_still_updates():
    weight, covariance, key, target, mask = state(2)
    origin = torch.full((168, 70), .1, dtype=DTYPE)
    weight = weight + origin
    key[:, 0] = 1
    target[0], target[1] = 100, -100
    mask[:] = True
    for _ in range(4):
        weight, covariance, _ = ridge_step(weight, covariance, key, target, mask,
                                          origin=origin, radius=.3)
        norms = torch.linalg.vector_norm((weight - origin).flatten(1), dim=1)
        assert (norms <= .3 + 1e-12).all()
    torch.testing.assert_close(norms, torch.full((2,), .3, dtype=DTYPE))
    torch.testing.assert_close(covariance[:, 0, 0, 0], torch.full((2,), .2, dtype=DTYPE))


def test_projected_weights_are_not_the_unconstrained_ridge_solution():
    weight, covariance, key, target, mask = state()
    key[0, 0], target[0, 0], mask[0, 0] = 1, 10, True
    origin = weight.clone()
    for _ in range(2):
        weight, covariance, _ = ridge_step(weight, covariance, key, target, mask,
                                          origin=origin, radius=.1)
    torch.testing.assert_close(weight[0, 0, 0], torch.tensor(.1, dtype=DTYPE))
    torch.testing.assert_close(covariance[0, 0, 0, 0], torch.tensor(1 / 3, dtype=DTYPE))
    # Without projection, the identity-prior ridge solution would be 20 / 3.
    assert abs(float(weight[0, 0, 0]) - 20 / 3) > 6


def test_direct_real_target_updates_the_existing_innovation_interface():
    level = torch.zeros(1, 128, 23, dtype=DTYPE)
    available = torch.zeros_like(level, dtype=torch.bool)
    future = torch.zeros(1, 8, 23, dtype=DTYPE)
    future_available = torch.zeros_like(future, dtype=torch.bool)
    available[0, -1, :2] = True
    future_available[0, 0, :2] = True
    future[0, 0, :2] = torch.tensor([2., -1.], dtype=DTYPE)
    key = dynamic_key(level, available)
    target, mask = direct_target(level, available, future, future_available)
    weight, covariance, _, _, _ = state()
    updated, new_p, _ = ridge_step(weight, covariance, key, target, mask,
                                    origin=weight, radius=100)
    change = innovation(updated, weight, key)
    assert change[0, 0] > 0 > change[0, 1]
    assert (change[0, 2:] == 0).all()
    assert torch.equal(new_p[0, 2:], covariance[0, 2:])


def test_empty_batch_has_consistent_shapes():
    weight, covariance, key, target, mask = state(0)
    result, new_p, loss = ridge_step(weight, covariance, key, target, mask, origin=weight)
    assert result.shape == (0, 168, 70)
    assert new_p.shape == (0, 168, 70, 70)
    assert loss.shape == (0,)


def test_batch_rows_match_separate_steps_with_distinct_masks():
    torch.manual_seed(103)
    weight, covariance, key, target, mask = state(2)
    weight[:] = torch.randn_like(weight) * .01
    key[:] = torch.randn_like(key) * .1
    target[:] = torch.randn_like(target)
    mask[0, :10], mask[1, 5:20] = True, True
    origin = weight.clone()
    batched = ridge_step(weight, covariance, key, target, mask, origin=origin, radius=.2)
    for i in range(2):
        single = ridge_step(weight[i:i+1], covariance[i:i+1], key[i:i+1],
                            target[i:i+1], mask[i:i+1], origin=origin[i:i+1], radius=.2)
        for full, one in zip(batched, single):
            torch.testing.assert_close(full[i:i+1], one)


def test_cpu_float32_result_dtype_and_symmetry():
    weight, covariance, key, target, mask = state(dtype=torch.float32)
    key[0, :2] = torch.tensor([1., .5])
    target[0, 0] = 4
    mask[0, 0] = True
    updated, new_p, loss = ridge_step(weight, covariance, key, target, mask,
                                      origin=weight, radius=.3)
    assert updated.dtype == new_p.dtype == loss.dtype == torch.float32
    assert updated.device.type == new_p.device.type == 'cpu'
    assert torch.linalg.vector_norm(updated) <= .3 + 1e-6
    assert torch.equal(new_p[0, 0], new_p[0, 0].T)


@pytest.mark.parametrize('batch', [-1, 1.5, True, '1'])
def test_initial_covariance_rejects_invalid_batch(batch):
    with pytest.raises(ValueError):
        initial_covariance(batch, torch.zeros(1))


@pytest.mark.parametrize('reference', [None, 1, torch.ones(1, dtype=torch.int64),
                                       torch.tensor([float('nan')])])
def test_initial_covariance_rejects_invalid_reference(reference):
    with pytest.raises(ValueError):
        initial_covariance(1, reference)


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf'), True])
def test_ridge_step_rejects_invalid_radius(radius):
    weight, covariance, key, target, mask = state()
    with pytest.raises(ValueError):
        ridge_step(weight, covariance, key, target, mask, origin=weight, radius=radius)


@pytest.mark.parametrize('field', ['weight', 'covariance', 'key', 'target', 'origin'])
def test_ridge_step_rejects_all_nonfinite_inputs_even_when_masked(field):
    weight, covariance, key, target, mask = state()
    origin = weight.clone()
    values = dict(weight=weight, covariance=covariance, key=key, target=target, origin=origin)
    values[field].view(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        ridge_step(weight, covariance, key, target, mask, origin=origin)


@pytest.mark.parametrize('observed', [False, True])
def test_ridge_step_rejects_nonpositive_denominator(observed):
    weight, covariance, key, target, mask = state()
    covariance[0, 0, 0, 0] = -2
    key[0, 0] = 1
    mask[0, 0] = observed
    with pytest.raises(ValueError, match='denominator'):
        ridge_step(weight, covariance, key, target, mask, origin=weight)


def test_origin_is_mandatory_and_no_fallback_is_permitted():
    weight, covariance, key, target, mask = state()
    with pytest.raises(TypeError):
        ridge_step(weight, covariance, key, target, mask)
    with pytest.raises(ValueError):
        ridge_step(weight, covariance, key, target, mask, origin=None)


def test_bad_shapes_mask_and_floating_dtypes_are_rejected():
    weight, covariance, key, target, mask = state()
    calls = [
        lambda: ridge_step(weight[:, :-1], covariance, key, target, mask, origin=weight),
        lambda: ridge_step(weight, covariance[:, :-1], key, target, mask, origin=weight),
        lambda: ridge_step(weight, covariance[..., :-1], key, target, mask, origin=weight),
        lambda: ridge_step(weight, covariance, key[:, :-1], target, mask, origin=weight),
        lambda: ridge_step(weight, covariance, key, target[:, :-1], mask, origin=weight),
        lambda: ridge_step(weight, covariance, key, target, mask.float(), origin=weight),
        lambda: ridge_step(weight, covariance.float(), key, target, mask, origin=weight),
        lambda: ridge_step(weight, covariance, key.float(), target, mask, origin=weight),
        lambda: ridge_step(weight, covariance, key, target.float(), mask, origin=weight),
        lambda: ridge_step(weight, covariance, key, target, mask, origin=weight.float()),
        lambda: ridge_step(weight, covariance, key, target, mask, origin=weight[:, :-1]),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('field', ['weight', 'covariance', 'key', 'target', 'mask', 'origin'])
def test_mismatched_meta_device_rejected_without_reading_meta_values(field):
    weight, covariance, key, target, mask = state()
    values = dict(weight=weight, covariance=covariance, key=key, target=target,
                  mask=mask, origin=weight.clone())
    value = values[field]
    values[field] = torch.empty(value.shape, dtype=value.dtype, device='meta')
    with pytest.raises(ValueError, match='same device'):
        ridge_step(values['weight'], values['covariance'], values['key'],
                   values['target'], values['mask'], origin=values['origin'])
