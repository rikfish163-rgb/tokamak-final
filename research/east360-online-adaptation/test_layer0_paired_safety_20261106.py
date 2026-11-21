import numpy as np
import torch

import run_layer0_paired_safety_20261106 as run


def test_same_view_control_keeps_marginals_but_breaks_original_pairs():
    view = np.array([0, 0, 1, 0, 1])
    order = run.shuffled_partners(view)
    np.testing.assert_array_equal(view[order], view)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(view)))
    assert not np.any(order == np.arange(len(view)))


def test_pair_loss_detects_safety_upshift_relative_to_danger():
    danger = torch.tensor([0.2, -0.05])
    safe = torch.tensor([0.4, -0.2])
    direct = run.contrast(danger, safe)
    swapped = run.contrast(danger, safe.flip(0))
    assert direct > 0
    assert swapped > direct


def test_reversed_order_reverses_actual_action_without_changing_norm():
    coeff = torch.tensor([[1., -1., .5]])
    gradient = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]])
    margin = torch.tensor([.4])
    actual = run.direction(coeff, gradient, margin)
    opposite = run.direction(coeff, -gradient, margin)
    torch.testing.assert_close(actual, -opposite)
    torch.testing.assert_close(actual.norm(dim=1), torch.tensor([run.RADIUS / 1.4]))


def test_write_never_reuses_late_data_or_padding():
    block, frame = run.write_grid(np.array([63, 95]), np.array([True, True]), 106)
    np.testing.assert_array_equal(frame[block == 0], np.arange(72, 96))
    np.testing.assert_array_equal(frame[block == 1], [104, 105])
