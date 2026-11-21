"""Contract tests for previous-shot and first-arrival controls."""

import pytest
import torch

from point_donor_controls_20261105 import targets


def test_donor_controls_preserve_target_norm_with_declared_tail_hold():
    real = torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.0, 0.3]])
    previous = torch.tensor([[0.1, 0.0], [0.0, -0.1]])
    current_state = torch.tensor([[1.1], [0.0], [0.9]])
    previous_state = torch.tensor([[0.0], [1.0]])
    controls, clock, near = targets(
        real, previous, current_state, previous_state, torch.ones(1)
    )
    assert clock.tolist() == [0, 1, 1]
    assert near.tolist() == [1, 0, 1]
    for key in ("previous_clock", "previous_state", "first_direction"):
        torch.testing.assert_close(controls[key].norm(dim=1), real.norm(dim=1))
    torch.testing.assert_close(controls["first_direction"][:, 0], real.norm(dim=1))
    torch.testing.assert_close(controls["first_full"], real[:1])
    torch.testing.assert_close(controls["previous_clock"][2], torch.tensor([0.0, -0.3]))


def test_later_current_windows_cannot_change_any_earlier_target():
    real = torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.0, 0.3]])
    old = torch.tensor([[0.1, 0.0], [0.0, -0.1]])
    cur = torch.tensor([[1.1], [0.0], [0.9]])
    prior = torch.tensor([[0.0], [1.0]])
    before, clock1, near1 = targets(real, old, cur, prior, torch.ones(1))
    real[-1] = torch.tensor([0.04, -0.1])
    cur[-1] = -100.0
    after, clock2, near2 = targets(real, old, cur, prior, torch.ones(1))
    for key in before:
        assert torch.equal(before[key][:2], after[key][:2])
    assert torch.equal(clock1[:2], clock2[:2]) and torch.equal(near1[:2], near2[:2])


def test_zero_first_direction_is_not_replaced_using_the_future():
    real = torch.tensor([[0.0, 0.0], [0.1, -0.1]])
    cur = torch.zeros((2, 1))
    old = torch.tensor([[0.2, 0.0], [0.0, 0.1]])
    controls, _, near = targets(real, old, cur, cur, torch.ones(1))
    assert near.tolist() == [0, 0]  # first index in a tie
    assert not controls["first_direction"].any()
    assert not controls["first_full"].any()


def test_absent_donor_is_a_visible_protocol_failure():
    with pytest.raises(ValueError, match="nonempty"):
        targets(
            torch.ones(1, 2),
            torch.zeros(0, 2),
            torch.ones(1, 1),
            torch.zeros(0, 1),
            torch.ones(1),
        )
