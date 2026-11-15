import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_lag_mixer_20261111 as lag
from experiments.risk_aligned_online import commissioning_lag_teacher_20261111 as teacher
from experiments.risk_aligned_online.commissioning_lag_guard_20261111 import (
    basis, guarded_loss, path_endpoints, signed_direction, shot_peaks,
)


def test_path_support_includes_each_release_interval_and_unmodifiable_peak():
    q = np.array([1, 12])
    r0, r1 = q + lag.H + 1
    z = np.full(r1+6, -2.)
    z[[4, r0+2, r1+3]] = [2., 3., 4.]
    ends = path_endpoints(z, q, r0)
    np.testing.assert_array_equal(ends, [4, r0, r0+2, r1+3])
    slots, _ = lag.delayed_slots(q, ends)
    np.testing.assert_array_equal(slots, [0, 1, 1, 2])


def test_whitening_stays_in_the_observed_span_and_damps_weak_modes():
    g = torch.zeros(1, 3, 2, 2)
    g[0, :2, 0, 0] = 1.
    g[0, 2, 1, 1] = .001
    anchor, modes = basis(g)
    torch.testing.assert_close(modes[0, 0], modes[0, 1])
    assert torch.count_nonzero(modes[..., 0, 1]) == 0
    assert torch.count_nonzero(modes[..., 1, 0]) == 0
    assert 0 < modes[0, 2, 1, 1] < .01
    torch.testing.assert_close(anchor.norm(dim=(1, 2)), torch.ones(1))


def test_same_real_feedback_span_can_reverse_with_equal_matrix_radius():
    g = torch.tensor([[[[1.]], [[0.]]], [[[0.]], [[0.]]]])
    anchor, modes = basis(g)
    keys = torch.ones(2, 1)
    neutral = signed_direction(anchor, modes, keys, torch.zeros(2, 1))
    reversed_ = signed_direction(anchor, modes, keys, torch.tensor([[-2.], [0.]]))
    assert neutral[0, 0, 0] > 0 > reversed_[0, 0, 0]
    torch.testing.assert_close(neutral.norm(dim=(1, 2)), reversed_.norm(dim=(1, 2)))
    assert reversed_[1].count_nonzero() == 0


def test_peak_is_risk_not_largest_delta_and_can_move_between_windows():
    before = torch.tensor([-.02, -.04, -.50, -.11, -.30])
    delta = torch.tensor([-.01, .06, .20, .01, .11])
    owner = torch.tensor([0, 0, 0, 1, 1])
    old, arg = shot_peaks(before, owner, 2)
    new, arg_new = shot_peaks(before+delta, owner, 2)
    torch.testing.assert_close(old, torch.tensor([-.02, -.11]))
    torch.testing.assert_close(new, torch.tensor([.02, -.10]))
    assert arg.tolist() == [0, 3]
    assert arg_new.tolist() == [1, 3]


def test_one_point_per_safe_shot_recovers_existing_direction_loss():
    delta = torch.tensor([[.03, .09], [-.05, -.11]])
    margins = torch.tensor([[-.08, -.02], [-.03, .10]])
    actual = guarded_loss(delta, margins[:, 1], delta[:, 1])
    torch.testing.assert_close(actual, teacher.direction_loss(delta, margins))


def test_guard_sees_new_worst_crossing_outside_the_matched_anchor():
    delta = torch.tensor([[.10, -.04]])
    before = torch.tensor([-.03])
    # The matched safe endpoint improves, but a different endpoint crosses.
    guard_delta = torch.tensor([.07], requires_grad=True)
    path = guarded_loss(delta, before, guard_delta)
    point = guarded_loss(delta, before, delta[:, 1])
    assert path > point
    path.backward()
    assert guard_delta.grad.item() > 0
