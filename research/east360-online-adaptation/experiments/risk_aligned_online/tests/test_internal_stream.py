import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.internal_dynamics_20261022.model import InternalDynamics
from experiments.risk_aligned_online.internal_dynamics_20261022.stream import predict


def fixture():
    torch.manual_seed(3); torch.set_num_threads(2)
    base = Simple23(Config(window=32, patch=8, dim=16, heads=2, layers=2, horizon=8, fast_site='tokens'),
                    np.zeros(23), np.ones(23))
    model = InternalDynamics(base, np.ones(23), np.ones(23, bool), rank=4)
    torch.nn.init.normal_(model.decoder.weight, std=.02)
    x = np.random.default_rng(8).normal(size=(65, 23)).astype('float32')
    return model, dict(device='EAST', shot=7, values=x, available=np.ones_like(x, bool), times=np.arange(65)/1000.)


@pytest.mark.parametrize('feedback_history', ['full', 'recent3'])
def test_stream_future_perturbation_and_reset(feedback_history):
    model, shot = fixture()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    a, audit = predict(model, shot, intra=True, device='cpu', feedback_history=feedback_history)
    altered = dict(shot, values=shot['values'].copy()); altered['values'][33:] += 1000.
    b, _ = predict(model, altered, intra=True, device='cpu', feedback_history=feedback_history)
    np.testing.assert_array_equal(a[:33], b[:33])
    repeat, _ = predict(model, shot, intra=True, device='cpu', feedback_history=feedback_history)
    np.testing.assert_array_equal(a, repeat)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in original.items())
    assert audit['updates']
    for update in audit['updates']:
        assert update['last_target_index'] <= update['now_index'] < update['first_affected_prediction']


def test_stream_outcome_not_accepted_and_static_has_no_updates():
    model, shot = fixture()
    with pytest.raises(ValueError, match='outcome'):
        predict(model, dict(shot, down=1.), intra=True, device='cpu')
    scores, audit = predict(model, shot, intra=False, device='cpu')
    assert len(scores) == 65 and not audit['updates']
