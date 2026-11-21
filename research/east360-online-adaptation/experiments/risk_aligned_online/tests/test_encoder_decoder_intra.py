"""Separate reconstruction fit from actual risk-path adaptation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.aligned_encoder import predict_shot
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import signals
from experiments.risk_aligned_online.tests.test_cross_channel_stable import model


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('scope', ['encoder', 'decoder', 'both'])
def test_declared_scope_causal_reset_and_preserves_first_prediction(scope):
    m = model()
    s = signals(105)
    initial = {k: v.clone() for k, v in m.state_dict().items()}
    flags = {k: p.requires_grad for k, p in m.named_parameters()}
    p, audit = predict_shot(m, s, intra=True, scope=scope)
    changed = deepcopy(s)
    changed['values'][60:] += 11
    q, _ = predict_shot(m, changed, intra=True, scope=scope)
    again, _ = predict_shot(m, s, intra=True, scope=scope)
    static, _ = predict_shot(m, s, intra=False, scope=scope)
    np.testing.assert_array_equal(p[:60], q[:60])
    np.testing.assert_array_equal(p, again)
    np.testing.assert_allclose(p[:8], static[:8], atol=2e-7, rtol=0)
    assert audit['accepted_updates'] > 0
    assert audit['update_scope'] == scope
    assert audit['risk_head_updated'] is False
    assert audit['auxiliary_decoder_updated'] == (scope != 'encoder')
    expected = {'encoder', 'decoder'} if scope == 'both' else {scope}
    assert set(audit['parameter_group_distances']) == expected
    assert all(0 < d <= .500001 for d in audit['parameter_group_distances'].values())
    assert all(r['effective_index'] == r['observed_through'] + 1 for r in audit['attempts'])
    if scope == 'decoder':
        np.testing.assert_allclose(p, static, atol=2e-7, rtol=0)
        assert audit['risk_path_unchanged_before_reset']
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    assert flags == {k: p.requires_grad for k, p in m.named_parameters()}


def test_joint_first_encoder_step_is_not_rescaled_by_decoder_gradient():
    m = model()
    _, encoder = predict_shot(m, signals(9), intra=True, scope='encoder')
    _, both = predict_shot(m, signals(9), intra=True, scope='both')
    a, b = encoder['attempts'][0], both['attempts'][0]
    assert a['accepted'] and b['accepted']
    assert a['group_gradient_norms']['encoder'] == b['group_gradient_norms']['encoder']
    assert a['group_parameter_distances']['encoder'] == b['group_parameter_distances']['encoder']


@pytest.mark.parametrize('scope', ['decoder', 'both'])
def test_exception_after_update_restores_decoder_and_encoder(scope):
    m = model()
    initial = {k: v.clone() for k, v in m.state_dict().items()}
    real = m.four_losses
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('injected after accepted proposal')
        return real(*args, **kwargs)

    m.four_losses = fail
    with pytest.raises(RuntimeError, match='injected'):
        predict_shot(m, signals(45), intra=True, scope=scope)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)


def test_invalid_scope_and_label_input_rejected_before_mutation():
    m = model()
    with pytest.raises(ValueError, match='scope'):
        predict_shot(m, signals(), intra=True, scope='risk')
    with pytest.raises(ValueError, match='outcome'):
        predict_shot(m, dict(signals(), down=1.), intra=True, scope='both')


@pytest.mark.parametrize('scope', ['decoder', 'both'])
def test_missing_observations_and_short_shots_do_not_disappear(scope):
    m = model()
    s = signals(40)
    s['available'][:] = False
    p, audit = predict_shot(m, s, intra=True, scope=scope)
    assert len(p) == 40 and audit['accepted_updates'] == 0
    p, audit = predict_shot(m, signals(7), intra=True, scope=scope)
    assert len(p) == 7 and audit['attempted_updates'] == 0
