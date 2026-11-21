import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_competing_forecasts_20261114 as trial


def test_window_must_be_timely_when_the_update_can_act():
    clock = np.arange(250) / 1000
    q = np.array([127, 159, 191])
    # Full handoff at 166, 198, 230 ms, giving 64, 32 and 0 ms.
    np.testing.assert_array_equal(trial.precursor_rows(q, clock, .230, 65, 10),
                                  [True, True, False])
    np.testing.assert_array_equal(trial.precursor_rows(q, clock, .228, 125, 30),
                                  [True, False, False])
    np.testing.assert_array_equal(trial.precursor_rows(q, clock, .230, 25, 10),
                                  [False, False, False])


def test_earlier_control_cannot_borrow_a_packet_arriving_after_issue():
    q = np.array([127, 159, 191, 255])
    before = trial.previous_packet(q, np.array([True, False, True, True]))
    np.testing.assert_array_equal(before, [-1, 0, -1, -1])


def test_competition_is_a_differentiable_bounded_diagnostic_not_a_risk_head():
    width = len(trial.group.USE)
    levels = torch.tensor([-.4, 0., .4])[:, None, None].expand(-1, 5, width).clone()
    levels.requires_grad_()
    normal = torch.full_like(levels, -.5)
    adverse = torch.full_like(levels, .5)
    scale = torch.ones(width)
    real = trial.relative_fit(levels, normal, adverse, scale)
    torch.testing.assert_close(real[1], torch.zeros(4))
    assert (real[0] < 0).all() and (real[2] > 0).all()
    torch.testing.assert_close(trial.relative_fit(levels, adverse, normal, scale), -real)
    torch.testing.assert_close(trial.relative_fit(levels, normal, normal, scale),
                               torch.zeros_like(real))
    real.sum().backward()
    assert (levels.grad > 0).all()


def test_regression_is_shot_balanced_and_cold_shrinks_to_source():
    x = np.repeat(np.array([[.1, -.1, .1]]), 5, axis=0)
    shape = (5, trial.TARGET_WIDTH)
    y = np.ones(shape)
    data = [(0, x, -y), (0, x[:1], y[:1]), (1, x, 2*y), (1, x[:1], 2*y[:1])]
    model = trial.fit_source(data)
    pred = model.forecast(x, model.source)
    np.testing.assert_allclose(pred[0], 0., atol=1e-6)
    np.testing.assert_allclose(pred[1], 2., atol=1e-6)
    cold = [(0, x, y), (1, x, -y)]
    mapped = model.forecast(x, model.fit_cold(cold))
    assert ((mapped[0] > 0.) & (mapped[0] < 1.)).all()
    assert ((mapped[1] > -1.) & (mapped[1] < 2.)).all()


def test_missing_contrast_preserves_direction_and_near_zero_is_not_amplified():
    legacy = torch.randn(2, 15, 4)
    force = torch.stack((torch.ones_like(legacy), -torch.ones_like(legacy),
                         1e-9*torch.ones_like(legacy)))
    mix = trial.compose(legacy, force, torch.tensor([False, True]))
    torch.testing.assert_close(mix[:, 0], legacy[:1].expand(3, -1, -1))
    assert not torch.allclose(mix[0, 1], mix[1, 1])
    unit = legacy[1] / legacy[1].norm()
    torch.testing.assert_close(mix[2, 1], unit, atol=1e-5, rtol=0.)
