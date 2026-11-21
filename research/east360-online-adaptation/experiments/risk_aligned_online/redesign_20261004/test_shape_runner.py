"""Pair ownership and causal timing in the shared predictor loop."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .intra import predict_shot
from .test_shape_intra import ShapeModel


def fixture(size=330):
    model = ShapeModel()
    model.cfg = SimpleNamespace(window=128, dim=64, horizon=25)
    model.initial_fast = lambda batch: model.center.new_zeros(batch, 128)
    values = np.repeat(model.center.numpy()[None], size, axis=0)
    signals = dict(values=values, available=np.ones_like(values, bool),
                   times=np.arange(size)/1000.)
    return model, signals


def test_shape_loop_pairs_only_matured_blocks_and_changes_only_later_scores():
    model, signals = fixture()
    scores, audit = predict_shot(model, signals, True, objective='shape')
    attempts = audit['attempts']
    assert attempts[0]['reason'] == 'needs_previous'
    assert attempts[1]['pair_origins'] == [0, 100] and attempts[1]['accepted']
    assert attempts[1]['effective_index'] == 126
    assert np.all(scores[:126] == .5) and np.any(scores[126:] != .5)
    assert audit['model_parameter_change_norm'] == audit['model_buffer_change_norm'] == 0
    assert audit['accepted_updates'] > 0
    assert audit['loss_weights']['masked_residual_shape'] == 1.


def test_pair_and_fast_are_reset_between_shots():
    model, signals = fixture()
    left, left_audit = predict_shot(model, signals, True, objective='shape')
    right, right_audit = predict_shot(model, signals, True, objective='shape')
    np.testing.assert_array_equal(left, right)
    assert right_audit['attempts'][0]['reason'] == 'needs_previous'
    assert left_audit['attempts'][1]['fast_before'] == right_audit['attempts'][1]['fast_before']


def test_shape_policy_does_not_use_future_suffix_for_earlier_scores():
    model, signals = fixture()
    left, _ = predict_shot(model, signals, True, objective='shape')
    changed = dict(signals, values=signals['values'].copy())
    changed['values'][250:] += 1000.
    right, _ = predict_shot(model, changed, True, objective='shape')
    np.testing.assert_array_equal(left[:250], right[:250])


@pytest.mark.parametrize('kwargs', [dict(stride=30), dict(step=.1), dict(radius=2.)])
def test_shape_contract_rejects_changed_experiment_parameters(kwargs):
    model, signals = fixture()
    with pytest.raises(ValueError, match='shape'):
        predict_shot(model, signals, True, objective='shape', **kwargs)
