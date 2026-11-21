import math

import pytest
import torch

from trajectory_teaching_loss_20261105 import teaching_loss


def _softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _reference(scores, leads, is_disrupted, mask, threshold, mode):
    """Scalar, per-shot specification independent of the vectorized loss."""
    observed = [(score, lead) for score, lead, keep in zip(scores, leads, mask) if keep]
    if mode == "point":
        terms = []
        for score, lead in observed:
            target = is_disrupted and 0 <= lead <= 125
            difference = score - threshold
            terms.append(_softplus(-difference if target else difference))
        return sum(terms) / len(terms)

    temperature = 0.5
    if is_disrupted:
        eligible_max = max(score for score, lead in observed if lead > 30)
        result = _softplus((threshold - eligible_max) / temperature)
        bins = ((30, 125), (125, 500), (500, 1500), (1500, 3000))
        bin_maxima = []
        for lower, upper in bins:
            in_bin = [score for score, lead in observed if lower < lead <= upper]
            if in_bin:
                bin_maxima.append(max(in_bin))
        if bin_maxima:
            normalized = temperature * math.log(
                sum(math.exp(score / temperature) for score in bin_maxima)
                / len(bin_maxima)
            )
            result += 0.25 * _softplus((threshold - normalized) / temperature)
        too_early = [score for score, lead in observed if lead > 3000]
        if too_early:
            x = (max(too_early) - threshold) / temperature
            result += 0.25 / (1.0 + math.exp(-x))
        return result

    sorted_scores = sorted((score for score, _ in observed), reverse=True)
    count = (len(sorted_scores) + 3) // 4
    main = _softplus((sorted_scores[0] - threshold) / temperature)
    top_losses = [_softplus((score - threshold) / temperature) for score in sorted_scores[:count]]
    return main + 0.5 * sum(top_losses) / count


@pytest.mark.parametrize("mode", ["point", "trajectory"])
def test_mixed_batch_matches_independent_per_shot_reference(mode):
    logits = torch.tensor(
        [[2.0, -0.2, 0.9, -1.0, 1.5, 0.4, -0.7, 4.0],
         [0.1, 1.2, -0.3, 0.6, 1.7, -0.9, 8.0, -8.0],
         [0.5, -1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )
    leads = torch.tensor(
        [[30.0, 31.0, 125.0, 126.0, 500.0, 1500.0, 3000.0, 3001.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [3001.0, 4000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    disrupted = torch.tensor([True, False, True])
    valid = torch.tensor(
        [[True, True, True, True, True, True, True, True],
         [True, False, True, True, True, True, False, False],
         [True, True, False, False, False, False, False, False]]
    )
    threshold = 0.37
    expected = torch.tensor(
        [_reference(s, t, d, m, threshold, mode)
         for s, t, d, m in zip(logits.tolist(), leads.tolist(), disrupted.tolist(), valid.tolist())],
        dtype=logits.dtype,
    )
    actual = teaching_loss(logits, leads, disrupted, valid, threshold, mode)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["point", "trajectory"])
def test_randomized_batch_matches_scalar_reference(mode):
    generator = torch.Generator().manual_seed(20261105)
    logits = torch.randn((24, 13), generator=generator, dtype=torch.float64)
    leads = 4200.0 * torch.rand((24, 13), generator=generator, dtype=torch.float64) - 100.0
    valid = torch.rand((24, 13), generator=generator) > 0.35
    disrupted = torch.arange(24) % 2 == 0
    valid[:, 0] = True
    leads[disrupted, 0] = 50.0
    threshold = -0.23
    expected = torch.tensor(
        [_reference(s, t, d, m, threshold, mode)
         for s, t, d, m in zip(logits.tolist(), leads.tolist(), disrupted.tolist(), valid.tolist())],
        dtype=logits.dtype,
    )
    actual = teaching_loss(logits, leads, disrupted, valid, threshold, mode)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_point_targets_include_zero_and_125_but_not_outside():
    logits = torch.ones((1, 5), dtype=torch.float64, requires_grad=True)
    leads = torch.tensor([[-0.001, 0.0, 125.0, 125.001, 500.0]], dtype=torch.float64)
    loss = teaching_loss(logits, leads, torch.tensor([True]), torch.ones_like(logits, dtype=torch.bool), 0.0, "point")
    loss.sum().backward()
    assert torch.equal(logits.grad < 0, torch.tensor([[False, True, True, False, False]]))


def test_safe_point_targets_ignore_arbitrary_leads():
    logits = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    leads = torch.tensor([[0.0, 50.0, 4000.0]])
    loss = teaching_loss(logits, leads, torch.tensor([False]), torch.ones_like(logits, dtype=torch.bool), 0.0, "point")
    loss.sum().backward()
    assert torch.all(logits.grad > 0)


def test_trajectory_strict_30_and_3000_boundaries_and_all_bin_endpoints():
    logits = torch.tensor([[10.0, -2.0, -1.0, -0.9, -0.8, -0.7, 0.4]], dtype=torch.float64, requires_grad=True)
    leads = torch.tensor([[30.0, 30.001, 125.0, 500.0, 1500.0, 3000.0, 3000.001]], dtype=torch.float64)
    mask = torch.ones_like(logits, dtype=torch.bool)
    actual = teaching_loss(logits, leads, torch.tensor([True]), mask, 0.2, "trajectory")
    expected = _reference(logits.detach().tolist()[0], leads.tolist()[0], True, mask.tolist()[0], 0.2, "trajectory")
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=logits.dtype), rtol=1e-12, atol=1e-12)
    actual.sum().backward()
    assert logits.grad[0, 0] == 0  # Exactly 30 ms is ineligible.
    assert logits.grad[0, 5] < 0  # Exactly 3000 ms is an allowed-bin target.

    # Isolate the early-alarm penalty: only the score at >3000 changes.
    first = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    teaching_loss(first, torch.tensor([[100.0, 3000.0]]), torch.tensor([True]), torch.ones_like(first, dtype=torch.bool), 0.0, "trajectory").sum().backward()
    teaching_loss(second, torch.tensor([[100.0, 3000.001]]), torch.tensor([True]), torch.ones_like(second, dtype=torch.bool), 0.0, "trajectory").sum().backward()
    assert first.grad[0, 1] < 0
    assert second.grad[0, 1] > 0


def test_allowed_early_bins_pull_up_while_point_objective_pushes_down():
    leads = torch.tensor([[126.0, 501.0, 1501.0]], dtype=torch.float64)
    valid = torch.ones_like(leads, dtype=torch.bool)
    disrupted = torch.tensor([True])
    point = torch.tensor([[0.3, 0.5, 0.9]], dtype=torch.float64, requires_grad=True)
    trajectory = point.detach().clone().requires_grad_()
    teaching_loss(point, leads, disrupted, valid, 0.0, "point").sum().backward()
    teaching_loss(trajectory, leads, disrupted, valid, 0.0, "trajectory").sum().backward()
    assert torch.all(point.grad > 0)
    assert torch.all(trajectory.grad < 0)


@pytest.mark.parametrize(
    ("dtype", "lead_values"),
    [(torch.uint8, [30, 125, 126, 200, 255]),
     (torch.bfloat16, [30, 1488, 1504, 2992, 3008])],
)
def test_lead_boundaries_are_compared_without_narrow_dtype_rounding(dtype, lead_values):
    logits = torch.tensor([[0.4, -0.7, 2.0, 0.3, -0.2]], dtype=torch.float64)
    leads = torch.tensor([lead_values], dtype=dtype)
    valid = torch.ones_like(logits, dtype=torch.bool)
    expected = _reference(logits.tolist()[0], leads.tolist()[0], True, valid.tolist()[0], 0.1, "trajectory")
    actual = teaching_loss(logits, leads, torch.tensor([True]), valid, 0.1, "trajectory")
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=logits.dtype), rtol=1e-12, atol=1e-12)


def test_very_large_finite_bin_logits_do_not_produce_nan_gradients():
    logits = torch.tensor([[2.0e38, 1.9e38]], dtype=torch.float32, requires_grad=True)
    leads = torch.tensor([[100.0, 200.0]])
    loss = teaching_loss(logits, leads, torch.tensor([True]), torch.ones_like(logits, dtype=torch.bool), 0.0, "trajectory")
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


def test_safe_auxiliary_backpropagates_to_top_quarter_not_only_maximum():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]], requires_grad=True)
    leads = torch.zeros_like(logits)
    loss = teaching_loss(logits, leads, torch.tensor([False]), torch.ones_like(logits, dtype=torch.bool), 0.0, "trajectory")
    loss.sum().backward()
    assert torch.all(logits.grad[0, :3] > 0)  # ceil(9 / 4) == 3.
    assert torch.count_nonzero(logits.grad[0, 3:]) == 0


@pytest.mark.parametrize("mode", ["point", "trajectory"])
def test_padding_does_not_change_loss_or_receive_gradients(mode):
    valid = torch.tensor([[True, False, True, False, True], [False, True, True, False, True]])
    leads = torch.tensor([[50.0, 0.0, 126.0, 0.0, 4000.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    changed_leads = torch.where(valid, leads, torch.tensor(torch.finfo(leads.dtype).max))
    original = torch.tensor([[0.1, -100.0, 0.5, 100.0, -0.2], [100.0, 0.9, -1.0, -100.0, 0.2]], requires_grad=True)
    changed = torch.where(valid, original.detach(), torch.tensor(torch.finfo(original.dtype).max)).requires_grad_()
    disrupted = torch.tensor([True, False])
    a = teaching_loss(original, leads, disrupted, valid, 0.3, mode)
    b = teaching_loss(changed, changed_leads, disrupted, valid, 0.3, mode)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    b.sum().backward()
    torch.testing.assert_close(original.grad, changed.grad)
    assert torch.count_nonzero(original.grad[~valid]) == 0
    assert torch.count_nonzero(changed.grad[~valid]) == 0


def test_absent_auxiliary_bins_and_too_early_windows_have_finite_gradients():
    logits = torch.tensor([[0.3, -0.5, 4.0], [0.1, 9.0, 8.0], [-0.2, 0.6, 7.0]], requires_grad=True)
    leads = torch.tensor([[3001.0, 5000.0, 0.0], [0.0, 0.0, 0.0], [100.0, 500.0, 0.0]])
    valid = torch.tensor([[True, True, False], [True, False, False], [True, True, False]])
    losses = teaching_loss(logits, leads, torch.tensor([True, False, True]), valid, 0.0, "trajectory")
    losses.sum().backward()
    assert torch.isfinite(losses).all()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


@pytest.mark.parametrize("mode", ["point", "trajectory"])
def test_autograd_matches_finite_differences_in_mixed_batch(mode):
    logits = torch.tensor(
        [[0.7, -0.6, 1.0, 0.1, -2.0],
         [0.2, 0.9, -0.3, 1.3, -0.8],
         [1.5, 0.6, -0.4, 0.3, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    leads = torch.tensor(
        [[3001.0, 4000.0, 0.0, 0.0, 0.0],
         [100.0, 200.0, 700.0, 2000.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    valid = torch.tensor(
        [[True, True, False, False, False],
         [True, True, True, True, False],
         [True, True, True, True, True]]
    )
    disrupted = torch.tensor([True, True, False])
    threshold = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda scores, cut: teaching_loss(scores, leads, disrupted, valid, cut, mode),
        (logits, threshold),
    )


@pytest.mark.parametrize("mode", ["point", "trajectory"])
@pytest.mark.parametrize("is_disrupted", [False, True])
def test_empty_valid_shot_fails_closed(mode, is_disrupted):
    with pytest.raises(ValueError, match="valid"):
        teaching_loss(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([is_disrupted]), torch.tensor([[False, False]]), 0.0, mode)


@pytest.mark.parametrize("mode", ["point", "trajectory"])
def test_disrupted_shot_without_lead_strictly_above_30_fails_closed(mode):
    with pytest.raises(ValueError, match="30"):
        teaching_loss(torch.zeros((2, 3)), torch.tensor([[100.0, 0.0, 0.0], [0.0, 30.0, 1000.0]]), torch.tensor([False, True]), torch.tensor([[True, False, False], [True, True, False]]), 0.0, mode)


def test_zero_length_window_axis_fails_closed():
    with pytest.raises(ValueError, match="valid"):
        teaching_loss(torch.zeros((1, 0)), torch.zeros((1, 0)), torch.tensor([False]), torch.zeros((1, 0), dtype=torch.bool), 0.0, "trajectory")


def test_malformed_shapes_and_mask_types_fail_visibly():
    with pytest.raises(ValueError, match="shapes"):
        teaching_loss(torch.zeros((1, 2)), torch.zeros((1, 1)), torch.tensor([True]), torch.ones((1, 2), dtype=torch.bool), 0.0, "trajectory")
    with pytest.raises(ValueError, match="disrupted"):
        teaching_loss(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([[True]]), torch.ones((1, 2), dtype=torch.bool), 0.0, "trajectory")
    with pytest.raises(TypeError, match="bool"):
        teaching_loss(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([True]), torch.ones((1, 2)), 0.0, "trajectory")
    with pytest.raises(TypeError, match="floating point"):
        teaching_loss(torch.zeros((1, 2), dtype=torch.int64), torch.zeros((1, 2)), torch.tensor([True]), torch.ones((1, 2), dtype=torch.bool), 0.0, "trajectory")
    with pytest.raises(ValueError, match="scalar"):
        teaching_loss(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([True]), torch.ones((1, 2), dtype=torch.bool), torch.tensor([0.0]), "trajectory")
    with pytest.raises(ValueError, match="same device"):
        teaching_loss(torch.zeros((1, 2)), torch.empty((1, 2), device="meta"), torch.tensor([True]), torch.ones((1, 2), dtype=torch.bool), 0.0, "trajectory")


def test_unknown_mode_fails_visibly():
    with pytest.raises(ValueError, match="mode"):
        teaching_loss(torch.zeros((1, 1)), torch.tensor([[50.0]]), torch.tensor([True]), torch.tensor([[True]]), 0.0, "unknown")
