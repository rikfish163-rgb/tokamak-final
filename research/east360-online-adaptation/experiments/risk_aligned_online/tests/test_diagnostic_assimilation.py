import torch

from experiments.risk_aligned_online.diagnostic_assimilation import assimilate
from experiments.risk_aligned_online.diagnostic_adapter import DiagnosticAdapter


def test_arrived_trajectory_corrects_its_own_named_coordinates():
    adapter = DiagnosticAdapter()
    coefficients = torch.zeros(2, 12, requires_grad=True)
    true_coefficients = torch.tensor([[1., -.5]*6, [-.2, .7]*6])
    prediction = torch.einsum('bck,tk->btc', coefficients.reshape(2, 6, 2), adapter.time_basis)
    target = torch.einsum('bck,tk->btc', true_coefficients.reshape(2, 6, 2), adapter.time_basis)
    mask = torch.ones_like(target, dtype=torch.bool)
    current = torch.zeros(2, 12)
    updated = assimilate(current, prediction, target, mask, adapter.time_basis)
    assert (updated*true_coefficients).sum() > 0
    after = torch.einsum('bck,tk->btc', updated.reshape(2, 6, 2), adapter.time_basis)
    assert (after-target).square().mean() < (prediction-target).square().mean()
    updated.sum().backward()
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()


def test_unobserved_channels_do_not_move_and_batching_is_independent():
    adapter = DiagnosticAdapter()
    current = torch.zeros(2, 12)
    prediction = torch.zeros(2, 25, 6)
    target = torch.ones_like(prediction)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, 2] = False
    updated = assimilate(current, prediction, target, mask, adapter.time_basis)
    assert torch.equal(updated[:, 4:6], current[:, 4:6])
    for i in range(2):
        separate = assimilate(current[i:i+1], prediction[i:i+1], target[i:i+1], mask[i:i+1], adapter.time_basis)
        torch.testing.assert_close(updated[i:i+1], separate)
