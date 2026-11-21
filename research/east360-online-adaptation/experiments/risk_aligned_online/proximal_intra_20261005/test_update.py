"""CPU-only contracts for the unpromoted clipped/proximal update candidate."""
import math

import pytest
import torch

from experiments.risk_aligned_online.proximal_intra_20261005.update import (
    clipped_proximal_step,
)


def update(delta, gradient, **overrides):
    settings = dict(c=2.0, step=0.1, tau=0.260, dt=0.026, radius=1.0)
    settings.update(overrides)
    return clipped_proximal_step(delta, gradient, **settings)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_state_zero_gradient_has_no_invented_direction(dtype):
    delta = torch.zeros(7, dtype=dtype, device="cpu", requires_grad=True)
    result = update(delta, torch.zeros_like(delta))
    assert torch.equal(result, delta)
    assert result.data_ptr() != delta.data_ptr()
    assert result.dtype == dtype and result.device == delta.device


@pytest.mark.parametrize("decay_only", [False, True])
def test_no_evidence_decays_existing_state_and_keeps_zero_static(decay_only):
    for delta in (torch.zeros(3, dtype=torch.float64),
                  torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)):
        gradient = None if decay_only else torch.zeros_like(delta)
        result = update(delta, gradient, decay_only=decay_only)
        torch.testing.assert_close(result, math.exp(-0.1) * delta)


def test_no_observations_requires_explicit_decay_only_contract():
    delta = torch.zeros(3)
    with pytest.raises(ValueError, match="decay_only"):
        update(delta, None)
    with pytest.raises(ValueError, match="decay_only"):
        update(delta, torch.zeros_like(delta), decay_only=True)
    with pytest.raises(TypeError, match="decay_only"):
        update(delta, None, decay_only=1)


def test_zero_step_still_applies_clock_decay():
    delta = torch.tensor([0.3, -0.2], dtype=torch.float64)
    result = update(delta, torch.tensor([3.0, 4.0], dtype=delta.dtype), step=0.0)
    torch.testing.assert_close(result, math.exp(-0.1) * delta)


@pytest.mark.parametrize("multiple", [0.0, 1e-8, 0.5, 1.0, 100.0])
def test_formula_below_at_and_above_clip_threshold(multiple):
    delta = torch.tensor([0.2, -0.1], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=delta.dtype)
    result = update(delta, 2.0 * multiple * direction)
    expected = math.exp(-0.1) * (delta - 0.1 * min(multiple, 1.0) * direction)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-15)


def test_tiny_gradients_stay_proportional_and_large_gradients_saturate():
    delta = torch.zeros(2, dtype=torch.float64)
    gradient = torch.tensor([0.03, 0.04], dtype=delta.dtype)
    small = update(delta, gradient)
    torch.testing.assert_close(update(delta, gradient * 1e-9), small * 1e-9)
    clipped = update(delta, gradient * 100)
    torch.testing.assert_close(update(delta, gradient * 1000), clipped)
    assert float(clipped.norm()) == pytest.approx(0.1 * math.exp(-0.1))


def test_projection_is_after_both_signal_step_and_anchoring():
    delta = torch.tensor([0.3, 0.4], dtype=torch.float64)
    gradient = torch.tensor([-6.0, 0.0], dtype=delta.dtype)
    proposal = math.exp(-0.1) * (delta + torch.tensor([0.1, 0.0], dtype=delta.dtype))
    expected = proposal / proposal.norm() * 0.2
    torch.testing.assert_close(update(delta, gradient, radius=0.2), expected,
                               rtol=1e-14, atol=1e-15)
    # If projection were before decay, the norm would instead be 0.2*exp(-0.1).
    assert float(update(delta, gradient, radius=0.2).norm()) == pytest.approx(0.2)


@pytest.mark.parametrize("step", [0.01, 0.03, 0.10])
def test_constant_gradient_matches_closed_form_for_1800_steps(step):
    initial = torch.tensor([0.1, -0.05], dtype=torch.float64)
    delta = initial.clone()
    gradient = torch.tensor([3.0, 4.0], dtype=delta.dtype)
    direction = torch.tensor([0.6, 0.8], dtype=delta.dtype)
    decay = math.exp(-0.1)
    for k in range(1, 1801):
        delta = update(delta, gradient, step=step)
        accumulated = step * decay * (-math.expm1(-0.1 * k)) / (1 - decay)
        expected = decay**k * initial - accumulated * direction
        torch.testing.assert_close(delta, expected, rtol=2e-13, atol=2e-14)
        bound = decay**k * float(initial.norm()) + accumulated
        assert torch.isfinite(delta).all() and float(delta.norm()) <= bound + 2e-14
        assert float(delta.norm()) < 1.0
    assert float(delta.norm()) == pytest.approx(step / math.expm1(0.1))


@pytest.mark.parametrize("decay_only", [False, True])
def test_variable_dt_no_evidence_depends_on_elapsed_time_not_event_count(decay_only):
    initial = torch.tensor([0.3, -0.4], dtype=torch.float64)
    intervals = [0.001, 0.2, 0.037, 0.7, 0.062]
    delta = initial.clone()
    for dt in intervals:
        gradient = None if decay_only else torch.zeros_like(delta)
        delta = update(delta, gradient, dt=dt, decay_only=decay_only)
    expected = initial * math.exp(-sum(intervals) / 0.260)
    once = update(initial, None, dt=sum(intervals), decay_only=True)
    torch.testing.assert_close(delta, expected, rtol=2e-14, atol=1e-16)
    torch.testing.assert_close(delta, once, rtol=2e-14, atol=1e-16)


def test_variable_dt_with_evidence_matches_weighted_history():
    initial = torch.tensor([0.1, -0.2], dtype=torch.float64)
    intervals = [0.013, 0.26, 0.005, 0.9]
    gradients = torch.tensor([[0.2, 0.0], [-0.4, 0.3], [0.0, -0.5], [0.3, 0.4]],
                             dtype=initial.dtype)
    delta = initial.clone()
    for dt, gradient in zip(intervals, gradients):
        delta = update(delta, gradient, dt=dt)
    expected = math.exp(-sum(intervals) / 0.260) * initial
    for i, gradient in enumerate(gradients):
        expected -= math.exp(-sum(intervals[i:]) / 0.260) * 0.1 / 2.0 * gradient
    torch.testing.assert_close(delta, expected, rtol=2e-14, atol=1e-16)


def test_gradient_reversal_has_no_momentum_tail():
    delta = torch.zeros(2, dtype=torch.float64)
    gradient = torch.tensor([3.0, 4.0], dtype=delta.dtype)
    first = update(delta, gradient)
    second = update(first, -gradient)
    decay = math.exp(-0.1)
    torch.testing.assert_close(second, decay * 0.1 * (1 - decay) * gradient / 5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_norm_bound_under_large_alternating_and_zero_gradients(dtype):
    delta = torch.tensor([3.0, 4.0, -1.0], dtype=dtype, device="cpu")
    gradient = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
    for k in range(180):
        scale = [0.0, 1e-10, 1e10, -1e10][k % 4]
        delta = update(delta, scale * gradient, step=2.0, radius=0.3)
        assert torch.isfinite(delta).all()
        assert float(delta.double().norm()) <= 0.3 * (1 + 4 * torch.finfo(dtype).eps)


def test_very_large_finite_float32_gradient_is_clipped_not_lost():
    gradient = torch.tensor([3e38, -3e38], dtype=torch.float32)
    result = update(torch.zeros_like(gradient), gradient)
    expected = math.exp(-0.1) * 0.1 / math.sqrt(2) * torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(result, expected)


def test_small_calibration_does_not_overflow_step_over_c():
    gradient = torch.tensor([1e-310, 0.0], dtype=torch.float64)
    result = update(torch.zeros_like(gradient), gradient, c=1e-310, step=0.1)
    torch.testing.assert_close(result, torch.tensor([-0.1 * math.exp(-0.1), 0.0],
                                                    dtype=gradient.dtype))


def test_long_tau_approaches_clipped_sgd_and_long_gap_underflows_to_zero():
    delta = torch.tensor([0.2, -0.1], dtype=torch.float64)
    gradient = torch.tensor([3.0, 4.0], dtype=delta.dtype)
    torch.testing.assert_close(update(delta, gradient, tau=1e12),
                               delta - 0.1 * gradient / 5, rtol=1e-12, atol=1e-14)
    assert torch.equal(update(delta, gradient, dt=1e6), torch.zeros_like(delta))


@pytest.mark.parametrize("radius", [1.0, 0.01])
def test_noncontiguous_inputs_and_autograd_state_are_not_mutated(radius):
    storage = torch.tensor([0.2, 9.0, -0.3, 8.0], dtype=torch.float64,
                           requires_grad=True)
    delta = storage[::2]
    gradient = torch.tensor([3.0, 4.0], dtype=delta.dtype, requires_grad=True)
    before, grad_before = storage.detach().clone(), gradient.detach().clone()
    result = update(delta, gradient, radius=radius)
    torch.autograd.grad(result.sum(), (storage, gradient))
    assert torch.equal(storage.detach(), before)
    assert torch.equal(gradient.detach(), grad_before)
    assert storage.grad is None and gradient.grad is None
    assert result.data_ptr() != delta.data_ptr()


@pytest.mark.parametrize("case", ["zero", "unclipped", "clipped", "projected"])
def test_first_and_second_derivatives_away_from_clip_and_projection_kinks(case):
    delta = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    values = {"zero": [0.0, 0.0], "unclipped": [0.3, 0.4],
              "clipped": [3.0, 4.0], "projected": [0.3, 0.4]}
    gradient = torch.tensor(values[case], dtype=delta.dtype, requires_grad=True)
    radius = 0.05 if case == "projected" else 1.0
    function = lambda d, g: update(d, g, radius=radius)
    assert torch.autograd.gradcheck(function, (delta, gradient))
    assert torch.autograd.gradgradcheck(function, (delta, gradient))


@pytest.mark.parametrize("name", ["c", "step", "tau", "dt", "radius"])
@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_or_nonfinite_scalar_raises(name, bad):
    with pytest.raises(ValueError, match=name):
        update(torch.zeros(2), torch.zeros(2), **{name: bad})


@pytest.mark.parametrize("name", ["c", "tau", "dt", "radius"])
def test_strictly_positive_scalars_reject_zero(name):
    with pytest.raises(ValueError, match=name):
        update(torch.zeros(2), torch.zeros(2), **{name: 0.0})


@pytest.mark.parametrize("name", ["c", "step", "tau", "dt", "radius"])
@pytest.mark.parametrize("bad", [True, None, "1", torch.tensor(1.0)])
def test_non_real_or_tensor_hyperparameters_are_not_silently_coerced(name, bad):
    with pytest.raises(TypeError, match=name):
        update(torch.zeros(2), torch.zeros(2), **{name: bad})


@pytest.mark.parametrize("name", ["delta", "gradient"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_tensors_raise_even_if_step_zero_or_decay_underflows(name, bad):
    tensors = dict(delta=torch.zeros(2), gradient=torch.zeros(2))
    tensors[name][0] = bad
    with pytest.raises(FloatingPointError, match=name):
        update(**tensors, step=0.0, dt=1e6)


@pytest.mark.parametrize("bad", [torch.tensor(0.0), torch.zeros(1, 2), torch.empty(0)])
def test_delta_requires_one_nonempty_flat_vector(bad):
    with pytest.raises(ValueError, match="delta"):
        update(bad, torch.zeros_like(bad))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64,
                                  torch.float16, torch.bfloat16])
def test_unsupported_tensor_dtypes_raise(dtype):
    with pytest.raises(TypeError, match="delta"):
        update(torch.zeros(2, dtype=dtype), torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize("bad", [torch.zeros(3), torch.zeros(1, 2),
                                torch.zeros(2, dtype=torch.float64),
                                torch.empty(2, device="meta")])
def test_gradient_shape_dtype_and_device_must_match_delta(bad):
    with pytest.raises(ValueError, match="gradient"):
        update(torch.zeros(2), bad)


@pytest.mark.parametrize("name", ["delta", "gradient"])
def test_inputs_must_be_tensors(name):
    tensors = dict(delta=torch.zeros(2), gradient=torch.zeros(2))
    tensors[name] = [0.0, 0.0]
    with pytest.raises(TypeError, match=name):
        update(**tensors)


def test_nonfinite_intermediate_raises_without_mutating_inputs():
    maximum = torch.finfo(torch.float64).max
    delta = torch.tensor([maximum, 0.0], dtype=torch.float64)
    gradient = torch.tensor([-1.0, 0.0], dtype=delta.dtype)
    before, grad_before = delta.clone(), gradient.clone()
    with pytest.raises(FloatingPointError):
        update(delta, gradient, c=1.0, step=maximum)
    assert torch.equal(delta, before) and torch.equal(gradient, grad_before)
