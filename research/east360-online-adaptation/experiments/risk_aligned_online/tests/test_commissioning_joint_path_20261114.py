import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_joint_path_20261114 as exp


def test_path_direction_changes_sign_without_changing_endpoints_or_increment_set():
    x = torch.zeros((1, 5, len(exp.GROUPS)))
    i, j = exp.PAIRS[0]
    x[0, 1:, i] = 1.
    x[0, 2:, j] = 1.
    other = x[:, :1]+x[:, -1:]-x.flip(1)
    torch.testing.assert_close(x[:, [0, -1]], other[:, [0, -1]])
    torch.testing.assert_close(x.diff(dim=1), other.diff(dim=1).flip(1))
    area = exp.path_area(x)
    assert area[0, 0] > .9
    torch.testing.assert_close(exp.path_area(other), -area)
    torch.testing.assert_close(exp.path_area(x+torch.arange(x.shape[-1])), area)


def test_earlier_control_never_takes_a_measurement_not_ready_at_issue():
    q = np.array([127, 156, 158, 191, 223, 255])
    ready = np.array([True, True, False, False, True, True])
    i, paired = exp.older_indices(q, ready)
    assert i.tolist() == [-1, -1, 0, 1, 1, 4]
    assert paired.tolist() == [False, False, False, False, True, True]


def test_real_and_reverse_differ_only_in_the_orthogonal_direction():
    origin = torch.tensor([[[2., 0., 0.]], [[2., 0., 0.]]])
    now = torch.tensor([[[1., 3., 0.]], [[1., 0., 0.]]])
    previous = torch.tensor([[[1., 0., 4.]], [[2., 1., 0.]]])
    a, b, c = exp.rotate(origin, now, previous, torch.tensor([.4, .4]))
    torch.testing.assert_close(a[0]+b[0], 2*(1-.4**2)**.5*origin[0])
    torch.testing.assert_close(a[0].norm(), c[0].norm())
    torch.testing.assert_close(a[0].norm(), origin[0].norm())
    for d in (a,b,c):
        torch.testing.assert_close(d[1], origin[1])


def test_stress_control_can_hold_angle_fixed_without_opening_an_undefined_axis():
    origin = torch.tensor([[[2., 0., 0.]], [[2., 0., 0.]]])
    now = torch.tensor([[[1., 3., 0.]], [[1., 0., 0.]]])
    previous = torch.tensor([[[1., 0., 4.]], [[2., 1., 0.]]])
    a,b,c = exp.rotate(origin, now, previous, exp.stress_strength(torch.tensor([True, True])))
    torch.testing.assert_close(a[0, 0], torch.tensor([3.**.5, 1., 0.]))
    torch.testing.assert_close(b[0, 0], torch.tensor([3.**.5, -1., 0.]))
    torch.testing.assert_close(c[0, 0], torch.tensor([3.**.5, 0., 1.]))
    for d in (a,b,c):
        torch.testing.assert_close(d[1], origin[1])
