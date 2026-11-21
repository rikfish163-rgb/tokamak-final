import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.alarm_objective import timely_shot_loss


def test_only_timely_disruptive_points_and_max_safe_point_receive_gradients():
    logits = torch.tensor([[.1, .8, 9.], [-1., .2, -2.]], requires_grad=True)
    rows = [dict(is_disrupt=True, down=.030, times=np.array([.010, .020, .025])),
            dict(is_disrupt=False, down=None, times=np.array([0., .01, .02]))]
    loss = timely_shot_loss(logits, rows, threshold=.5)
    loss.backward()
    # Exactly 10 ms is excluded. The 5 ms point cannot rescue this shot.
    assert logits.grad[0, 0] < 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert logits.grad[1, 1] > 0
    assert torch.count_nonzero(logits.grad[1, [0, 2]]) == 0


def test_padding_is_not_an_observation_and_threshold_is_decision_origin():
    logits = torch.tensor([[0., 100.], [0., -100.]])
    rows = [dict(is_disrupt=True, down=.020, times=np.array([0.])),
            dict(is_disrupt=False, down=None, times=np.array([0.]))]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-np.log(3), -np.log(3)], dtype=torch.float32), torch.tensor([1., 0.]))
    assert torch.allclose(timely_shot_loss(logits, rows, threshold=.75), expected)


def test_no_actionable_prefix_is_rejected_not_silently_relabelled():
    with pytest.raises(ValueError, match='timely'):
        timely_shot_loss(torch.zeros(1, 1), [dict(is_disrupt=True, down=.005,
                                                 times=np.array([0.]))], threshold=.5)


def test_endpoint_labels_do_not_change_the_shot_objective():
    rows = [dict(is_disrupt=True, down=.050, times=np.array([0., .02]), y=np.zeros(2))]
    logits = torch.tensor([[.1, .2]])
    before = timely_shot_loss(logits, rows, threshold=.5)
    rows[0]['y'][:] = 1
    assert torch.equal(before, timely_shot_loss(logits, rows, threshold=.5))
