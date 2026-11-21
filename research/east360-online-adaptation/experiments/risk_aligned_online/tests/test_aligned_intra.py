"""Input alignment must not alter the existing in-shot update experiment."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.context_feedback_20261022.model import ContextFeedback
from experiments.risk_aligned_online.context_feedback_20261022.stream import predict_pair
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform
from experiments.risk_aligned_online.warmstart_alignment_20261022.intra import predict_aligned_pair


def make_model():
    torch.set_num_threads(1); torch.manual_seed(131)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2, fast_site='tokens'),
                    np.zeros(23), np.ones(23))
    model = ContextFeedback(base, np.ones(23)); model.freeze_base()
    return model


def signal():
    return dict(values=np.random.default_rng(2).normal(size=(99, 23)).astype('float32'),
                available=np.ones((99, 23), bool))


def mapping():
    gain = np.ones(23); gain[12] = .002
    return dict(gain=gain.tolist(), offset=[0.] * 23, first_affected_index=20)


def test_warmup_uses_existing_raw_update_and_rejects_premature_mapping():
    model, s = make_model(), signal()
    expected, _ = predict_pair(model, s, device='cpu', use_event=False, uniform=True)
    actual, _ = predict_aligned_pair(model, s, None, index=19, device='cpu')
    for arm in expected: np.testing.assert_array_equal(actual[arm], expected[arm])
    with pytest.raises(ValueError, match='not yet available'):
        predict_aligned_pair(model, s, mapping(), index=19, device='cpu')
    with pytest.raises(ValueError, match='ended prefix'):
        predict_aligned_pair(model, s, None, index=20, device='cpu')


def test_after_prefix_equals_same_update_on_mapped_observations():
    model, s = make_model(), signal(); before = copy.deepcopy(model.state_dict())
    expected, _ = predict_pair(model, transform(s, mapping()), device='cpu',
                              use_event=False, uniform=True)
    actual, info = predict_aligned_pair(model, s, mapping(), index=20, device='cpu')
    for arm in expected: np.testing.assert_array_equal(actual[arm], expected[arm])
    assert info['updates'] and all(r['delta'] > 0 for r in info['updates'])
    for k, v in before.items(): assert torch.equal(v, model.state_dict()[k])
    np.testing.assert_array_equal(actual['static'][:32], actual['intra'][:32])


def test_aligned_update_cannot_use_future_values_or_outcome():
    model, s = make_model(), signal()
    original, _ = predict_aligned_pair(model, s, mapping(), index=20, device='cpu')
    changed = copy.deepcopy(s); changed['values'][55:] += 10
    other, _ = predict_aligned_pair(model, changed, mapping(), index=20, device='cpu')
    for arm in original:
        np.testing.assert_array_equal(original[arm][:55], other[arm][:55])
    with pytest.raises(ValueError, match='signals only'):
        predict_aligned_pair(model, dict(s, is_disrupt=True), mapping(), index=20, device='cpu')
