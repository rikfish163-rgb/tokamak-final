import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.first_feedback_20261023.learning import (
    schedule, predict, update_latest,
)
from experiments.risk_aligned_online.tests.test_increment_forecast import model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_first_step_moves_without_increasing_later_update_budget():
    assert schedule(800, 'early1') == [153, 384, 512, 640, 768]
    assert schedule(800, 'late1') == schedule(800, 'late4') == [256, 384, 512, 640, 768]
    assert schedule(153, 'early1') == []
    assert schedule(154, 'early1') == [153]
    assert schedule(256, 'late1') == []
    with pytest.raises(ValueError): schedule(800, 'unknown')


def test_latest_update_only_consumes_one_fully_arrived_target():
    _, m = model('increment')
    with torch.no_grad(): m.base.forecast_head.weight.normal_(0, .01)
    signal = dict(values=np.random.default_rng(10).normal(size=(153,23)).astype('float32'),
                  available=np.ones((153,23), bool))
    state, receipt = update_latest(m, signal, 153, initial_encoder(m), device='cpu')
    assert receipt['forecast_origins'] == [127]
    assert receipt['last_observation'] == 152
    assert receipt['first_affected_prediction'] == 153
    assert receipt['gradient_steps'] == 1 and receipt['observed_targets'] == 25*23
    assert 0 < receipt['parameter_delta_norm'] <= .00101
    with pytest.raises(ValueError): update_latest(m, signal, 152, initial_encoder(m), device='cpu')
    signal['available'][:] = False
    old = initial_encoder(m)
    new, receipt = update_latest(m, signal, 153, old, device='cpu')
    assert receipt['gradient_steps'] == 0
    for k in old: torch.testing.assert_close(new[k],old[k],rtol=0,atol=0)


def test_past_only_reset_and_no_persistent_parameter_updates():
    _, m = model('increment')
    with torch.no_grad(): m.base.forecast_head.weight.normal_(0, .01)
    before = {k:v.clone() for k,v in m.state_dict().items()}
    s = dict(values=np.random.default_rng(4).normal(size=(400,23)).astype('float32'),
             available=np.ones((400,23),bool))
    scores, receipts = predict(m,s,device='cpu')
    again, again_receipts = predict(m,s,device='cpu')
    assert receipts == again_receipts
    for k in scores: np.testing.assert_array_equal(scores[k],again[k])
    for arm in ('early1','late1','late4'):
        first = schedule(400,arm)[0]
        np.testing.assert_array_equal(scores['static'][:first],scores[arm][:first])
    s['values'][153:] = 999.
    changed, changed_receipts = predict(m,s,device='cpu')
    assert changed_receipts['early1'][0] == receipts['early1'][0]
    for k in scores: np.testing.assert_array_equal(changed[k][:153],scores[k][:153])
    for k,v in m.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with pytest.raises(ValueError): predict(m,dict(s,is_disrupt=True),device='cpu')
