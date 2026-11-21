import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_feedback_precision_20261113 as precision


def test_covariance_uses_the_same_bin_across_previous_windows_and_ignores_missing_rows():
    values = torch.tensor([[[[1., 2.], [90., -90.]], [[2., 4.], [90., -90.]],
                            [[100., 0.], [0., 100.]], [[3., 6.], [90., -90.]]]])
    count, c = precision.sample_covariance(values, torch.tensor([[True, True, False, True]]))
    torch.testing.assert_close(count, torch.tensor([3]))
    torch.testing.assert_close(c[0], torch.tensor([[.5, 1.], [1., 2.]]))


def test_issue_time_does_not_look_at_its_own_or_an_unreleased_window():
    # At 180 and 190 exactly three windows have arrived; a later row is invalid.
    q = np.array([100, 132, 148, 180, 190, 222])
    x = torch.arange(1., 7.).reshape(-1, 1, 1)
    eligible = np.array([True, True, True, True, False, True])
    count, c = precision.past_covariance(q, x, eligible, np.array([3, 4, 5]))
    torch.testing.assert_close(count, torch.tensor([3, 3, 4]))
    assert c[0].item() == 1.
    x[3:] = 1234.
    n2, c2 = precision.past_covariance(q, x, eligible, np.array([3, 4]))
    torch.testing.assert_close(n2, count[:2])
    torch.testing.assert_close(c2, c[:2])


def test_old_history_and_ineligible_rows_are_excluded():
    q = np.arange(12)*precision.lag.STRIDE
    x = torch.arange(12.).reshape(-1, 1, 1)
    x[:3] = 100.
    x[7] = -100.
    eligible = np.ones(len(q), bool)
    eligible[7] = False
    count, c = precision.past_covariance(q, x, eligible, np.array([11]))
    torch.testing.assert_close(count, torch.tensor([7]))
    expected = torch.var(x[[3, 4, 5, 6, 8, 9, 10], 0, 0])
    torch.testing.assert_close(c.flatten(), expected.reshape(1))


def test_full_precision_downweights_shared_movement_without_silencing_contrast():
    cov = torch.tensor([[[1., 1.], [1., 1.]]])
    x = torch.tensor([[[1., 1.], [1., -1.]]])
    diag = precision.apply_precision(x, cov, full=False)
    full = precision.apply_precision(x, cov, full=True)
    torch.testing.assert_close(diag, x / 4.)
    torch.testing.assert_close(full[0, 0], x[0, 0] / 5.5)
    torch.testing.assert_close(full[0, 1], x[0, 1] / 2.5)
    torch.testing.assert_close(precision.apply_precision(x, cov*0., full=True), x)


def test_same_time_basis_and_measurements_are_compared_in_current_and_early_conditions():
    cold = torch.eye(2)*.1
    local = torch.tensor([[[.3, .1], [.1, .4]], [[.1, -.05], [-.05, .2]]])
    n = torch.tensor([4, 8])
    now = torch.tensor([[[.5, .25]], [[.25, .5]]])
    old = -now.flip(0)
    (neutral, previous, stationary, stationary_old,
     diagonal, diagonal_old, full, changed_early) = precision.forces(now, old, cold, local, n)
    torch.testing.assert_close(neutral, now)
    torch.testing.assert_close(previous, old)
    torch.testing.assert_close(stationary, now/1.3)
    torch.testing.assert_close(stationary_old, old/1.3)
    mixture = (4*cold + n[:, None, None]*local)/(4+n[:, None, None])
    torch.testing.assert_close(diagonal, precision.apply_precision(now, mixture, full=False))
    torch.testing.assert_close(diagonal_old, precision.apply_precision(old, mixture, full=False))
    torch.testing.assert_close(full, precision.apply_precision(now, mixture, full=True))
    torch.testing.assert_close(changed_early, precision.apply_precision(old, mixture, full=True))
