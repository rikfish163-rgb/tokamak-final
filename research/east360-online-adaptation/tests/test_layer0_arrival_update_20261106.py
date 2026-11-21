import pytest
import torch

from layer0_arrival_update_20261106 import arrival_indices, matched_descent


def test_each_read_reaches_anchor_and_never_the_next_sample():
    t = torch.tensor([135, 167])
    o, indices = arrival_indices(t, 400)
    torch.testing.assert_close(o, torch.tensor([110, 142]))
    torch.testing.assert_close(indices[:, 0], o + 1)
    torch.testing.assert_close(indices[:, -1], t)
    with pytest.raises(ValueError):
        arrival_indices(t, 167)


def test_matched_control_changes_direction_not_action_size():
    g = torch.tensor([[3.0, 4.0], [0.0, 0.0], [2.0, 0.0]])
    a = matched_descent(g, torch.tensor([0.6, 0.0, 0.2]))
    torch.testing.assert_close(a.norm(dim=1), torch.tensor([0.6, 0.0, 0.2]))
    assert (a * g).sum(1).tolist()[0] < 0
    with pytest.raises(ValueError):
        matched_descent(g, torch.tensor([0.6, 0.5, 0.2]))
