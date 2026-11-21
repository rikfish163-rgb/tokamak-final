import copy

import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.aligned_state_20261024.model import (
    AlignedState, first_tokens, fixed_readout, observed_loss, update_fast,
)


def fixture():
    torch.manual_seed(31)
    cfg = Config(window=16, patch=4, dim=8, heads=2, layers=2,
                 horizon=3, fast_site='tokens', input_view='source')
    base = Simple23(cfg, torch.zeros(23), torch.ones(23)).eval()
    with torch.no_grad():
        base.fast_origin.copy_(torch.randn_like(base.fast_origin))
    model = AlignedState(copy.deepcopy(base), torch.ones(23)).eval()
    x = torch.randn(1, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :4] = False
    return base, model, x, mask


def test_identity_bridge_preserves_source_readout_and_learned_fast_origin():
    base, model, x, mask = fixture()
    with torch.no_grad():
        h, valid = first_tokens(base, x, mask)
        expected = base(x, mask)['logit']
        torch.testing.assert_close(fixed_readout(base, h, valid), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model(x, mask)['logit'], expected, rtol=1e-5, atol=1e-6)


def test_temporary_step_changes_encoder_not_parent_or_readout_and_can_reset():
    _, model, x, mask = fixture()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    fast = model.new_fast()
    out, info = update_fast(model, fast, x, mask, learning_rate=1e-3)
    assert info['gradient_steps'] == 1 and info['gradient_norm'] > 0
    assert any(not torch.equal(out[k], fast[k]) for k in fast)
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v, model.new_fast()[k]) for k,v in fast.items())
    assert not any('layers.1' in k or 'risk_head' in k or 'bridge' in k for k in out)
    # Carry this state forward within a shot, then reset from the fixed parent.
    next_fast, _ = update_fast(model, out, x, mask, learning_rate=1e-3)
    assert any(not torch.equal(next_fast[k], out[k]) for k in out)
    assert all(torch.equal(v, model.new_fast()[k]) for k,v in fast.items())


def test_loss_masks_missing_channels_and_uncertified_mode_number():
    _, model, x, mask = fixture()
    mask[..., 2] = False
    a = observed_loss(model, x, mask)
    changed = x.clone(); changed[..., 2] = float('nan')
    b = observed_loss(model, changed, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert model.reconstruction_channels.sum() == 22
    assert not model.reconstruction_channels[model.mode_number_index]
    from experiments.risk_aligned_online.io23 import RAW23
    assert model.mode_number_index == RAW23.index('mode_number_n')


def test_unobserved_input_is_not_a_training_example():
    _, model, x, mask = fixture()
    fast = model.new_fast()
    out, info = update_fast(model, fast, x, torch.zeros_like(mask), learning_rate=1e-3)
    assert info['gradient_steps'] == 0 and info['observations'] == 0
    assert all(torch.equal(v, out[k]) for k,v in fast.items())
    with pytest.raises(ValueError):
        update_fast(model, fast, x.repeat(2,1,1), mask.repeat(2,1,1), learning_rate=1e-3)


def test_fast_prediction_uses_the_supplied_state_without_mutation():
    _, model, x, mask = fixture()
    fast, _ = update_fast(model, model.new_fast(), x, mask, learning_rate=.01)
    with torch.no_grad():
        before = model(x, mask)['logit']
        after = model.predict_fast(x, mask, fast)['logit']
        assert not torch.equal(before, after)
        torch.testing.assert_close(model(x, mask)['logit'], before, rtol=0, atol=0)


def test_validation_is_not_passed_to_test_only_scorer(monkeypatch):
    import numpy as np
    from experiments.risk_aligned_online.aligned_state_20261024 import train
    monkeypatch.setattr(train.io23, 'source_roles', lambda: {'val':[1,2]})
    rows = [dict(shot=1,is_disrupt=True,down=.05,times=np.array([.01]),scores=np.array([.8])),
            dict(shot=2,is_disrupt=False,down=.05,times=np.array([.01]),scores=np.array([.1]))]
    def forbidden(*args, **kwargs):
        raise AssertionError('TEST-only scorer must not receive VAL')
    monkeypatch.setattr(train, 'score_fixed_test', forbidden)
    assert train.validation_score(rows,.5)['tp'] == 1
    with pytest.raises(ValueError): train.validation_score(rows[:1],.5)


def test_stream_prediction_cannot_see_future_and_resets_between_shots():
    import numpy as np
    from experiments.risk_aligned_online.aligned_state_20261024.stream import predict_stream
    _, model, x, mask = fixture()
    x = x.repeat(1,3,1)[0].numpy(); mask = mask.repeat(1,3,1)[0].numpy()
    signal = dict(values=x, available=mask)
    curves, updates = predict_stream(model,signal,device='cpu',interval=8,learning_rate=.001)
    assert all(u['first_affected_index'] > u['observed_end'] for u in updates)
    changed = dict(values=x.copy(),available=mask.copy()); changed['values'][24:] += 50.
    other, _ = predict_stream(model,changed,device='cpu',interval=8,learning_rate=.001)
    for arm in curves: np.testing.assert_array_equal(curves[arm][:24],other[arm][:24])
    again,_ = predict_stream(model,signal,device='cpu',interval=8,learning_rate=.001)
    for arm in curves: np.testing.assert_array_equal(curves[arm],again[arm])


def test_zero_timely_detections_in_target_calibration_stops_full_stream():
    from experiments.risk_aligned_online.aligned_state_20261024.east import calibration_viability
    records=[dict(is_disrupt=True,timely_peak=.981),dict(is_disrupt=True,timely_peak=.980),
             dict(is_disrupt=False,timely_peak=None)]
    result=calibration_viability(records,.9938)
    assert not result['allow_full_stream'] and result['timely_detected']==0
    assert not calibration_viability(records,None)['allow_full_stream']
    assert not calibration_viability(records[2:],.5)['allow_full_stream']
    records[0]['timely_peak']=.995
    assert calibration_viability(records,.9938)['allow_full_stream']
