import pytest
import torch

import paired_diagnostic_step_20261106 as old
import whole_block_update_20261106 as wb


def test_basis_separates_later_mix_from_mature_risk_normal():
    grads = torch.zeros(2, 6, 128)
    grads[0, 0, :2] = torch.tensor([2.0, 3.0])
    grads[0, 1, :3] = torch.tensor([1.0, -3.0, 1.0])
    anchor = torch.zeros(2, 128)
    anchor[0, 0] = 4
    anchor[1, 0] = 8
    grads[1, 1, 1] = 1
    basis = wb.split_bank(grads, anchor, 1.0)
    assert basis.shape == (2, 7, 128)
    assert (basis[:, :6, 0] == 0).all()
    assert (basis[:, 6, 1:] == 0).all()
    assert 0 < basis[0, 6, 0] <= 1
    assert basis[1, 6].count_nonzero() == 0
    assert basis[0, 0, 1] < 0
    assert basis[0, 1, 1] > 0


def test_zero_anchor_still_allows_observed_descent():
    grads = torch.randn(2, 6, 128, generator=torch.Generator().manual_seed(17))
    basis = wb.split_bank(grads, torch.zeros_like(grads[:, 0]), 0.1)
    expected = old.effects(grads.flatten(0, 1), 0.1).reshape_as(grads) / old.RADIUS
    torch.testing.assert_close(basis[:, :6], expected)
    assert basis[:, 6].count_nonzero() == 0


def test_nonfinite_or_nonpositive_bank_scale_is_rejected():
    grads, normal = torch.zeros(2, 6, 128), torch.zeros(2, 128)
    for ref in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            wb.split_bank(grads, normal, ref)
    grads[0, 1, 1] = float("nan")
    with pytest.raises(ValueError):
        wb.split_bank(grads, normal, 1.0)


def test_descriptor_masks_observations_and_uses_only_reached_risks():
    q = torch.ones(2, 11)
    pred = 3 * q
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[:, 3] = False
    q[:, 3], pred[:, 3] = float("nan"), float("nan")
    z0, zm = torch.tensor([1.0, 2.0]), torch.tensor([2.0, 0.0])
    inp = wb.describe(q, pred, mask, z0, zm)
    assert inp.shape == (2, 35) and torch.isfinite(inp).all()
    assert (inp[:, (3, 14, 25)] == 0).all()
    torch.testing.assert_close(inp[:, -1], torch.tensor([0.25, -0.5]))


def test_identity_initialization_learns_directions_and_stays_inside_radius():
    net = wb.Combiner()
    x = torch.ones(4, 35)
    bank = torch.zeros(4, 7, 128)
    bank[:, 0, 0] = 1.0
    bank[:, 6, 1] = 1.0
    initial = net(x, bank)
    assert initial.count_nonzero() == 0
    (-initial[:, 1].sum()).backward()
    with torch.no_grad():
        net.head.bias.sub_(net.head.bias.grad)
    result = net(x, bank)
    assert (result[:, 1] > 0).all()
    assert (result.norm(dim=1) < old.RADIUS).all()
    torch.testing.assert_close(wb.compose(bank, -net.coefficients(x)), -result)
