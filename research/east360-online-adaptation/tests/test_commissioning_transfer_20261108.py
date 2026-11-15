import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_transfer_20261108 import (
    Rows,
    TARGET_FACTORIAL,
    alarm_loss,
    combine,
    contrast,
    fit_readouts,
    make_rows,
    shot_weights,
    transport,
)


def test_contrast_keeps_direction_but_not_a_sensor_gain_or_offset():
    x = np.tile(np.arange(80, dtype=np.float64)[:, None], (1, 23))
    x[:, 1] *= -2
    x[:, 2] = 17
    a = np.ones_like(x, dtype=bool)
    a[:, 3] = False
    r = contrast(x, a)
    np.testing.assert_array_equal(r[-1, :4], [1, -1, 0, 0])
    np.testing.assert_array_equal(r[:47], 0)
    np.testing.assert_array_equal(contrast(11 * x + 81, a), r)
    x[53:] = np.random.default_rng(131).normal(size=x[53:].shape)
    np.testing.assert_array_equal(contrast(x, a)[:53], r[:53])


def test_point_weights_do_not_count_a_long_shot_as_more_independent_evidence():
    shot = np.array([0, 0, 0, 1, 2, 2])
    y = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
    w = shot_weights(shot, y)
    np.testing.assert_allclose(np.bincount(shot, w), [.25, .25, .50])


def test_coral_changes_training_source_only_and_matches_conditional_centers():
    rng = np.random.default_rng(131)
    y = np.repeat([False, True], 40)
    shots = np.repeat(np.arange(8), 10)
    sx = rng.normal(size=(80, 2))
    tx = 2 * sx + np.where(y[:, None], [3, 5], [-3, -5])
    source = Rows(sx, shots, y, y, ~y)
    target = Rows(tx, shots, y, y, ~y)
    before = tx.copy()
    moved = transport(source, target, conditional=True)
    for label in (False, True):
        np.testing.assert_allclose(moved[y == label].mean(0), tx[y == label].mean(0),
                                   atol=1e-6)
    np.testing.assert_array_equal(target.x, before)
    assert not np.allclose(transport(source, target, conditional=False), moved)


def test_train_window_and_timely_bag_are_distinct():
    t = np.array([.850, .875, .950, .970, .980, 1.001])
    r = make_rows(np.ones((len(t), 2)), t, True, 1., 125., 30., 0)
    np.testing.assert_array_equal(r.y, [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(r.timely, [0, 1, 1, 0, 0])
    assert not r.safe.any()
    r2 = make_rows(np.ones((len(t), 2)), t, False, None, 125., 30., 1)
    merged = combine([r, r2])
    assert len(merged.x) == 11 and merged.safe.sum() == 6


def test_alarm_loss_cares_about_first_alarm_eligible_positive_and_full_safe_peak():
    z = torch.tensor([-.3, .1, 4., -.2, .4], requires_grad=True)
    # The third point is too late; it cannot help a timely alarm.
    loss = alarm_loss(z, [torch.tensor([0, 1])], [torch.tensor([3, 4])], 0.)
    (grad,) = torch.autograd.grad(loss, z)
    assert grad[1] < 0 < grad[4]
    torch.testing.assert_close(grad[[0, 2, 3]], torch.zeros(3))
    with pytest.raises(ValueError, match="timely"):
        alarm_loss(z, [], [torch.tensor([3, 4])], 0.)


def test_target_factorial_trains_without_reading_source_training_examples():
    from types import SimpleNamespace

    torch.manual_seed(131)
    base = SimpleNamespace(cfg=SimpleNamespace(dim=2), risk_head=torch.nn.Linear(2, 1))
    x = np.random.default_rng(131).normal(size=(12, 25))
    y = np.r_[np.zeros(6, bool), np.ones(6, bool)]
    target = Rows(x, np.repeat(np.arange(4), 3), y, y, ~y)
    heads, details = fit_readouts(base, None, target, .7, "cpu", arms=TARGET_FACTORIAL)
    assert set(heads) == set(TARGET_FACTORIAL)
    for name, (motion, alignment, alarm) in TARGET_FACTORIAL.items():
        assert alignment == "target_only"
        assert heads[name].shape == (26 if motion else 3,)
        assert (details[name]["alarm_loss"] > 0) == alarm
