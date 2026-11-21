import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_adapt_20261022.model import Config, EventForecastModel
from experiments.risk_aligned_online.event_adapt_20261022.stream import predict_causal_shot


def setup():
    torch.manual_seed(19); torch.set_num_threads(2)
    model = EventForecastModel(Config(dim=8, rank=2, horizon=8, dilations=(1, 2)),
        np.zeros(23), np.ones(23), np.ones(23), np.ones(23, bool))
    s = dict(device='EAST', shot=1, values=np.random.default_rng(3).normal(size=(48, 23)).astype('float32'),
             available=np.ones((48, 23), bool), times=np.arange(48)/1000.)
    return model, s


def test_future_perturbation_cannot_change_past_updates_or_predictions():
    model, s = setup()
    a, audit = predict_causal_shot(model, s, intra=True, stride=4, lr=.1, radius=1., device='cpu')
    changed = dict(s, values=s['values'].copy()); changed['values'][25:] += 900
    b, other = predict_causal_shot(model, changed, intra=True, stride=4, lr=.1, radius=1., device='cpu')
    np.testing.assert_allclose(a[:25], b[:25], rtol=1e-6, atol=1e-7)
    early = [u for u in audit['updates'] if u['now_index'] < 25]
    other_early = [u for u in other['updates'] if u['now_index'] < 25]
    assert early == other_early
    assert all(u['last_target_index'] <= u['now_index'] < u['first_affected_prediction'] for u in audit['updates'])
    assert all(np.isfinite(u['issued_forecast_loss']) for u in audit['updates'])


def test_temporary_state_is_discarded_and_model_unchanged():
    model, s = setup(); original = {k: v.clone() for k, v in model.state_dict().items()}
    a, info = predict_causal_shot(model, s, intra=True, stride=4, lr=.1, radius=1., device='cpu')
    b, again = predict_causal_shot(model, s, intra=True, stride=4, lr=.1, radius=1., device='cpu')
    np.testing.assert_array_equal(a, b)
    assert info['accepted_updates'] > 0
    for k,v in original.items(): torch.testing.assert_close(v, model.state_dict()[k])


def test_static_blocked_path_equals_full_causal_forward():
    model, s = setup()
    a, info = predict_causal_shot(model, s, intra=False, stride=4, lr=.1, radius=1., device='cpu')
    with torch.no_grad():
        b = model.sequence(torch.tensor(s['values'])[None], torch.tensor(s['available'])[None])['logit'].sigmoid()[0]
    np.testing.assert_allclose(a, b.numpy(), rtol=1e-6, atol=1e-7)
    assert info['accepted_updates'] == 0


def test_prediction_api_refuses_outcome_and_offline_event_fields():
    model, s = setup()
    with pytest.raises(ValueError, match='outcome'):
        predict_causal_shot(model, dict(s, down=.04), intra=True, stride=4, lr=.1, radius=1., device='cpu')
