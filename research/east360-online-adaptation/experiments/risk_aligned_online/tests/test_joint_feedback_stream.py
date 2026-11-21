"""Whole-VAL diagnostic must neither read outcomes nor rewrite old scores."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
from experiments.risk_aligned_online.joint_feedback_20261022.diagnose_stream import predict_pair


def fixture():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = PairedFeedback(Simple23(Config(window=16, patch=4, dim=8, heads=2,
        fast_site='tokens'), np.zeros(23), np.ones(23)), np.ones(23))
    model.prepare_decoder()
    rng = np.random.default_rng(32)
    shot = dict(values=rng.normal(size=(37,23)).astype('float32'),
                available=np.ones((37,23), bool))
    return model, shot


def test_updates_wait_for_observations_and_preserve_persistent_state():
    model, shot = fixture()
    state = copy.deepcopy(model.state_dict())
    scores, audit = predict_pair(model, shot, device='cpu', interval=8)
    np.testing.assert_array_equal(scores['static'][:16], scores['intra'][:16])
    assert audit['updates'] and all(u['first_affected_prediction'] > u['last_observation']
                                    for u in audit['updates'])
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in state.items())
    # Changing a later signal cannot change any earlier sealed prediction.
    changed = copy.deepcopy(shot); changed['values'][25:] += 11.
    new, _ = predict_pair(model, changed, device='cpu', interval=8)
    np.testing.assert_array_equal(new['intra'][:25], scores['intra'][:25])
    again, _ = predict_pair(model, shot, device='cpu', interval=8)
    np.testing.assert_array_equal(again['intra'], scores['intra'])


def test_label_access_and_invalid_schedule_rejected():
    model, shot = fixture()
    for key in ('y', 'down', 'is_disrupt', 'event_target', 'valid'):
        with pytest.raises(ValueError, match='outcome'):
            predict_pair(model, dict(shot, **{key:0}), device='cpu')
    for interval in (0,-1,1.5):
        with pytest.raises(ValueError, match='interval'):
            predict_pair(model, shot, device='cpu', interval=interval)
