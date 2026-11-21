import numpy as np
import torch

from point_attribution_20261105 import fit_state_point, frozen_point_models


def test_state_control_optimizes_point_loss_with_ended_shot_weights():
    g = np.ones((4, 256))
    lead = np.array([[100.0, 200.0], [0.0, 0.0]])
    inputs = np.zeros((4, 84))
    inputs[:1, 0] = 1
    inputs[1:, 0] = -1
    bank = dict(
        state=inputs,
        z=np.zeros(4),
        g=g,
        lead=lead,
        valid=np.ones_like(lead, dtype=bool),
        disruptive=np.array([True, False]),
        shot_weight=np.array([0.7, 0.3]),
    )
    (scale, w), note = fit_state_point(bank, 0.0)
    assert scale.shape == (84,) and w.shape == (84, 256)
    assert note["after"] < note["before"]
    assert w[0].sum() > 0


def test_locked_point_is_identical_in_real_and_control_kernel_slots():
    point = (torch.ones(84), torch.ones(84, 256))
    state = (torch.zeros(84), torch.zeros(84, 256))
    items = frozen_point_models(point, state)
    assert items["point_real"] is items["trajectory_real"] is point
    assert items["trajectory_state"] is state
