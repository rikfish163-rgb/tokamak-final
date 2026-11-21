"""Matched-source learning and source-only gradient-scale calibration."""

from copy import deepcopy

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from joint_trajectory_20261107 import JointTrajectory
from run_joint_motion_20261107 import calibrate_weight, fit_step
from test_joint_stream_20261107 import base_model, signal


def episode(base):
    s = signal(190)
    e = j.sample_signals([s], j.eligible_origins([s]), np.random.default_rng(13), 3, 'cpu')
    e['tokens'], e['token_valid'], _ = j.features(base, e['x'], e['a'])
    return e


def test_zero_motion_weight_is_same_actual_training_step_as_level_control():
    base = base_model()
    level = JointTrajectory(base.center, base.scale, dim=8, scenarios=4)
    motion = OrderedTrajectory(base.center, base.scale, dim=8, scenarios=4, motion_weight=0.)
    motion.load_state_dict(level.state_dict(), strict=False)
    models = dict(level=level, motion=motion)
    opts = {k: torch.optim.AdamW(v.parameters(), lr=j.FIT_LR, weight_decay=0.) for k,v in models.items()}
    result = fit_step(models, opts, episode(base))
    assert result['level'] == result['motion']
    for key, value in level.state_dict().items():
        torch.testing.assert_close(value, motion.state_dict()[key], atol=0, rtol=0)


def test_gradient_scaling_uses_no_heldout_labels_and_changes_only_weight_buffer():
    base, s = base_model(), signal(190)
    s['y'] = np.ones(190)
    model = OrderedTrajectory(base.center, base.scale, dim=8, scenarios=4)
    before = deepcopy(model.state_dict())
    report = calibrate_weight(base, model, [s], batches=2, batch=2)
    assert report['weight'] > 0 and np.isfinite(report['weight'])
    assert report['batches'] == 2
    for key,value in model.state_dict().items():
        if key != 'motion_weight':
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    s['y'][:] = 0
    second = OrderedTrajectory(base.center, base.scale, dim=8, scenarios=4)
    second.load_state_dict(before)
    repeated = calibrate_weight(base, second, [s], batches=2, batch=2)
    assert repeated == report
