import pytest
import torch

from experiments.risk_aligned_online.lora_stream_20261102.core import cap_shift


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_state_mixed_basis_derivative_equals_uncapped_interior(dtype):
    generator = torch.Generator().manual_seed(131)
    features = torch.randn(2, 16, 128, generator=generator, dtype=dtype)
    basis = torch.randn(4, 128, generator=generator, dtype=dtype, requires_grad=True)
    state = torch.zeros(2, 4, 64, dtype=dtype, requires_grad=True)
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[1, :3] = False
    target = torch.randn(2, 16, 64, generator=generator, dtype=dtype)
    raw = ((features @ basis.T / 2) @ state).masked_fill(~valid[:, :, None], 0.)
    capped, _ = cap_shift(raw, valid)
    actual_state, = torch.autograd.grad((capped*target).sum(), state, create_graph=True, retain_graph=True)
    expected_state, = torch.autograd.grad((raw*target).sum(), state, create_graph=True, retain_graph=True)
    actual_basis, = torch.autograd.grad(actual_state.square().sum(), basis, retain_graph=True)
    expected_basis, = torch.autograd.grad(expected_state.square().sum(), basis)
    assert torch.isfinite(actual_basis).all() and actual_basis.norm() > 0
    torch.testing.assert_close(actual_basis, expected_basis, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equivalent_cap_preserves_zero_interior_and_capped_forward(dtype):
    generator = torch.Generator().manual_seed(131)
    delta = torch.randn(3, 16, 64, generator=generator, dtype=dtype)
    delta *= torch.tensor([0., .01, 2.], dtype=dtype)[:, None, None]
    valid = torch.ones(3, 16, dtype=torch.bool)
    valid[1, :5] = False
    clean = delta.masked_fill(~valid[:, :, None], 0.)
    old_rms = clean.double().flatten(1).norm(dim=1)/(valid.sum(1)*64).double().sqrt()
    old_factor = (.25/old_rms.clamp_min(.25)).to(dtype)
    actual, rms = cap_shift(delta, valid)
    torch.testing.assert_close(actual, clean*old_factor[:, None, None], atol=2e-7 if dtype == torch.float32 else 1e-14, rtol=1e-6)
    torch.testing.assert_close(rms, old_rms*old_factor.double(), atol=1e-14, rtol=1e-6)
    assert not actual[0].any() and not actual[~valid].any()
    assert (rms <= .2500001).all()
