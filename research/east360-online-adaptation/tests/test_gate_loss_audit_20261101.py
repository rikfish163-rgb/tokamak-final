import pytest
import torch

from experiments.risk_aligned_online.gate_loss_audit_20261101.core import (
    centered_loss,
    output_derivatives,
)


def test_smooth_l1_dc_ac_derivative_matches_autograd_and_sums():
    torch.manual_seed(11)
    p = torch.randn(2, 25, 23, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(p)
    m = torch.rand(p.shape) > 0.2
    u, dc, ac = output_derivatives(p, y, m)
    loss = (
        torch.nn.functional.smooth_l1_loss(p, y, reduction="none")
        .masked_fill(~m, 0)
        .sum()
        / m.sum()
    )
    actual = torch.autograd.grad(loss, p)[0]
    assert torch.allclose(u, actual, atol=1e-16, rtol=1e-14)
    assert torch.allclose(dc + ac, u, atol=1e-16, rtol=1e-14)
    assert not dc[~m].any() and not ac[~m].any()
    assert torch.allclose(
        ac.sum(1), torch.zeros(2, 23, dtype=torch.float64), atol=1e-16, rtol=0
    )


def test_saturated_same_sign_error_loses_time_component():
    p = torch.zeros(1, 25, 23, dtype=torch.float64)
    y = torch.linspace(5, 6, 25, dtype=torch.float64)[None, :, None].expand_as(p)
    mask = torch.ones_like(p, dtype=torch.bool)
    u, dc, ac = output_derivatives(p, y, mask)
    # Averaging25 copies of a nonbinary float need not be bit-identical.
    assert torch.allclose(u, dc, atol=1e-16, rtol=0)
    assert ac.abs().max() <= 1e-16
    assert centered_loss(p, y, mask) > 0


def test_centered_loss_has_no_channel_offset_and_no_missing_value_effect():
    torch.manual_seed(14)
    p = torch.randn(2, 25, 23, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(p)
    mask = torch.rand(p.shape) > 0.1
    loss = centered_loss(p, y, mask)
    shift = torch.randn(2, 1, 23, dtype=p.dtype)
    assert torch.allclose(
        centered_loss(p + shift, y - 2 * shift, mask), loss, atol=1e-15, rtol=0
    )
    missing = y.clone()
    missing[~mask] = float("nan")
    assert torch.equal(centered_loss(p, missing, mask), loss)
    gradient = torch.autograd.grad(loss, p)[0]
    assert torch.allclose(
        gradient.sum(1), torch.zeros(2, 23, dtype=p.dtype), atol=1e-16, rtol=0
    )
    assert not gradient[~mask].any()


def test_empty_and_nonfinite_observed_targets_are_not_silently_used():
    p = torch.zeros(1, 25, 23, requires_grad=True)
    mask = torch.zeros_like(p, dtype=torch.bool)
    y = torch.full_like(p, float("nan"))
    assert centered_loss(p, y, mask) == 0
    u, dc, ac = output_derivatives(p, y, mask)
    assert not u.any() and not dc.any() and not ac.any()
    mask[0, 0, 0] = True
    with pytest.raises(ValueError):
        output_derivatives(p, y, mask)
