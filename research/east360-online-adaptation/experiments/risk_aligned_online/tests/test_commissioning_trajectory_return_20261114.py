import torch

from experiments.risk_aligned_online import commissioning_trajectory_return_20261114 as trial


def test_dispersion_cannot_tell_earlier_and_later_departures_apart():
    # Both tracks begin and end at the same diagnostic level, with the same
    # five observations, but one has already started returning.
    x = torch.tensor([[[0., 0.], [3., -2.], [2., -1.5], [1., -.5], [0., 0.]]])
    shuffled = trial.reverse_middle(x)
    torch.testing.assert_close(shuffled[:, [0, -1]], x[:, [0, -1]])
    torch.testing.assert_close(shuffled.sort(dim=1).values, x.sort(dim=1).values)
    torch.testing.assert_close(trial.reverse_middle(shuffled), x)
    order, spread = trial.excursion_statistics(x)
    other, sham_spread = trial.excursion_statistics(shuffled)
    assert (order < 0).all() and (other > 0).all()
    torch.testing.assert_close(order, -other)
    torch.testing.assert_close(spread, sham_spread)


def test_diagnostic_excursion_depends_on_departure_not_signal_polarity():
    x = torch.tensor([[[0.], [.5], [2.], [2.], [3.]]])
    towards, spread = trial.excursion_statistics(x)
    backwards, _ = trial.excursion_statistics(x.flip(1))
    negative, same_spread = trial.excursion_statistics(-x)
    assert towards.item() > 0 and spread.item() > 0
    torch.testing.assert_close(towards, -backwards)
    torch.testing.assert_close(towards, negative)
    torch.testing.assert_close(spread, same_spread)


def test_quiet_track_has_zero_target_and_finite_derivative():
    x = torch.zeros(3, 5, 2, requires_grad=True)
    a, b = trial.excursion_statistics(x)
    direction, = torch.autograd.grad((a+b).sum(), x)
    assert torch.equal(a, torch.zeros_like(a)) and torch.equal(b, torch.zeros_like(b))
    assert torch.isfinite(direction).all()


def test_new_loss_changes_forecast_gradient_not_just_target_score():
    x = torch.tensor([[[0.], [.4], [.2], [1.1], [.1]]], requires_grad=True)
    target = torch.tensor([[[0.], [2.], [1.3], [.4], [0.]]])
    a, _ = trial.excursion_statistics(x)
    goal, _ = trial.excursion_statistics(target)
    sham_goal, _ = trial.excursion_statistics(trial.reverse_middle(target))
    ga, = torch.autograd.grad(a, x, grad_outputs=trial.feedback_force(a, goal), retain_graph=True)
    gb, = torch.autograd.grad(a, x, grad_outputs=trial.feedback_force(a, sham_goal))
    assert ga.norm().item() > 0 and gb.norm().item() > 0
    assert (ga*gb).sum().item() < 0


def test_same_coefficient_blend_for_all_forces_and_zero_residual():
    raw = torch.tensor([[[[3., 0.]]], [[[0., 9.]]], [[[0., 0.]]], [[[0., -2.]]]])
    directions = trial.blend_directions(raw)
    torch.testing.assert_close(directions[:, 0, 0],
                               torch.tensor([[3., 0.], [1., 1.], [1., 0.], [1., -1.]]))


def test_conditional_preference_is_the_squared_distance_difference():
    x = torch.tensor([[[4., 6.], [.5, -.2], [.2, .4], [2., -1.5], [9., 7.]]])
    y = torch.tensor([[[0., 0.], [2., 3.], [10., 10.], [.3, -.8], [0., 0.]]])
    pred, observed = (trial.soft_magnitude(v) for v in (x, y))
    shuffled = trial.reverse_middle(observed)
    exact = ((pred-shuffled).square()-(pred-observed).square()).sum((1, 2))
    p, t = (trial.interior_contrast(v) for v in (x, y))
    torch.testing.assert_close(2*(p*t).sum(1), exact)
    torch.testing.assert_close(t, -trial.interior_contrast(trial.reverse_middle(y)))
    x[:, [0, -1]] += 185.
    torch.testing.assert_close(p, trial.interior_contrast(x))


def test_preference_update_excludes_every_reversal_invariant_force():
    x = torch.tensor([[[0.], [.5], [.2], [1.5], [0.]]], requires_grad=True)
    y = torch.tensor([[[0.], [1.5], [.2], [.5], [0.]]])
    p, t = (trial.interior_contrast(v) for v in (x, y))
    f = trial.contrast_force(t)
    positive, = torch.autograd.grad(p, x, grad_outputs=f, retain_graph=True)
    negative, = torch.autograd.grad(p, x, grad_outputs=-f)
    assert positive.norm().item() > 0.
    torch.testing.assert_close(positive, -negative)
    torch.testing.assert_close(positive[:, [0, 2, 4]], torch.zeros_like(positive[:, [0, 2, 4]]))
    identical = y.clone()
    identical[:, 1] = y[:, 3]
    assert not trial.contrast_force(trial.interior_contrast(identical)).any()


def test_opposed_updates_keep_a_common_parent_and_weight():
    raw = torch.tensor([[[[3., 0.]]], [[[0., 9.]]], [[[0., -9.]]], [[[1., 0.]]]])
    directions = trial.blend_directions(raw, strength=torch.tensor([[.25],[.25],[1.]]))
    torch.testing.assert_close(directions[:, 0, 0],
                               torch.tensor([[3., 0.], [1., .25], [1., -.25], [2., 0.]]))
