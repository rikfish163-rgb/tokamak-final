import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_peak_feedback_20261114 as trial


def test_positive_bag_uses_the_label_window_and_strict_source_deadline():
    s = dict(is_disrupt=True, down=.200, times=np.array([.150, .175, .189, .190, .201]),
             y=np.array([0, 1, 1, 1, 0]), valid=np.array([True, True, True, True, False]))
    np.testing.assert_array_equal(trial.bag_ends(s), [1, 2])
    s['is_disrupt'] = False
    np.testing.assert_array_equal(trial.bag_ends(s), [0, 1, 2, 3])


def test_peak_loss_tracks_one_false_alarm_and_one_timely_crossing():
    z = torch.tensor([-3., 1., -2.], requires_grad=True)
    safe = trial.penalty(trial.aggregate(z, peak=True), 0., False)
    positive = trial.penalty(trial.aggregate(z, peak=True), 0., True)
    safe.backward(retain_graph=True)
    assert z.grad[1] > 0 and not z.grad[[0, 2]].any()
    z.grad.zero_()
    positive.backward()
    assert z.grad[1] < 0 and not z.grad[[0, 2]].any()
    assert safe > trial.penalty(z, 0., False).mean()


def test_transformation_cannot_write_a_state_without_diagnostic_feedback():
    grad = torch.arange(2 * 2 * trial.motion.C, dtype=torch.float32).reshape(2, 2, -1)
    grad[0] = 0.
    m = torch.zeros((2 * trial.motion.C,) * 2)
    torch.testing.assert_close(trial.remap(grad, m), grad)
    m[2, 0] = .5
    out = trial.remap(grad, m)
    assert not out[0].any()
    torch.testing.assert_close(out[1, 0, 0], grad[1, 0, 0] + .5 * grad[1, 0, 2])
