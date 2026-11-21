"""An ended-EAST auxiliary head must change updates, never the zero-fast risk."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel
from experiments.risk_aligned_online.ended_context_rule import causal_peak_logit
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel, train_pair


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(131)
    return DecoderFfnModel(Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                                   torch.zeros(23), torch.ones(23))).eval()


def shot(label=False):
    rng = np.random.default_rng(4)
    return dict(device='EAST', shot=12 if label else 13,
        values=rng.normal(size=(230, 23)).astype(np.float32),
        available=np.ones((230, 23), bool), times=np.arange(230)/1000.,
        valid=np.ones(230, bool), is_disrupt=label,
        down=.229 if label else None, horizon=125.,
        _sealed_scores=np.linspace(.1, .9, 230, dtype=np.float32))


def test_initial_head_and_ffn_paths_exactly_match_parent():
    m = model()
    old = PersistentFfnModel(m.base)
    x, a = torch.randn(1, 128, 23), torch.rand(1, 128, 23) > .2
    fast = torch.randn(1, m.temporary_dim, requires_grad=True)*.001
    out, ref = m(x, a, fast), old(x, a, fast)
    for key in out:
        torch.testing.assert_close(out[key], ref[key], rtol=0, atol=0)
    g, = torch.autograd.grad(out['forecast'].square().mean(), fast, retain_graph=True)
    h, = torch.autograd.grad(ref['forecast'].square().mean(), fast)
    torch.testing.assert_close(g, h, rtol=0, atol=0)


def test_decoder_change_leaves_current_risk_exactly_unchanged_and_base_intact():
    m = model()
    original = {k: v.clone() for k, v in m.base.state_dict().items()}
    x, a = torch.randn(1, 128, 23), torch.ones(1, 128, 23, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    before = m(x, a, fast)
    with torch.no_grad():
        m.decoder.weight.add_(.01)
        m.decoder.bias.add_(.1)
    after = m(x, a, fast)
    torch.testing.assert_close(after['logit'], before['logit'], rtol=0, atol=0)
    assert not torch.equal(after['forecast'], before['forecast'])
    for k, v in m.base.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['signal_only', 'risk_after_update'])
def test_only_decoder_learns_from_ended_pair(mode):
    m = model()
    original = {k: v.clone() for k, v in m.base.state_dict().items()}
    result = train_pair(m, [(shot(False), 180), (shot(True), 180)],
        torch.optim.Adam(m.decoder.parameters(), lr=.0003), boundary=.7, mode=mode)
    assert result['gradient_norm'] > 0 and result['decoder_delta_norm'] > 0
    if mode == 'risk_after_update':
        assert all(r['differentiated_updates'] > 0 for r in result['records'])
        assert result['risk_gradient_norm'] > 0
    assert not m.memory.any() and m.memory.grad is None
    for k, v in m.base.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)


def test_changed_decoder_replay_matches_emitted_causal_state():
    m = model()
    with torch.no_grad():
        m.decoder.bias.add_(.1)
    s = shot()
    scores, audit = predict_shot(m, s, intra=True)
    for query in (0, 25, 26, 125, 126, 180):
        logit, _ = causal_peak_logit(m, s, query)
        assert abs(float(logit.detach().sigmoid())-float(scores[query])) < 2e-6
    assert audit['fast_discarded'] and not m.memory.any()


def test_no_current_label_or_ended_head_training_changes_previous_scores():
    m = model()
    a, _ = predict_shot(m, shot(False), intra=True)
    b, _ = predict_shot(m, shot(True), intra=True)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('violation', ['not_sealed', 'not_east', 'wrong_optimizer'])
def test_train_requires_ended_east_and_only_auxiliary_optimizer(violation):
    m = model()
    pair = [(shot(False), 180), (shot(True), 180)]
    optimizer = torch.optim.Adam(m.decoder.parameters())
    if violation == 'not_sealed':
        del pair[0][0]['_sealed_scores']
    elif violation == 'not_east':
        pair[0][0]['device'] = 'J-TEXT'
    else:
        optimizer = torch.optim.Adam([m.memory])
    with pytest.raises(ValueError):
        train_pair(m, pair, optimizer, boundary=.7, mode='risk_after_update')


@pytest.mark.parametrize('violation', ['pending', 'future_history', 'not_sealed'])
def test_fit_rejects_unended_or_future_prefix(tmp_path, violation):
    from types import SimpleNamespace
    from experiments.risk_aligned_online.ended_decoder_experiment import fit_ended_prefix
    ids = list(range(30))
    history = [dict(shot=sid, _sealed_scores=np.zeros(2)) for sid in ids]
    provider = SimpleNamespace(ended_ids=ids, pending=None)
    if violation == 'pending':
        provider.pending = object()
    elif violation == 'future_history':
        history[-1]['shot'] = 31
    else:
        del history[0]['_sealed_scores']
    with pytest.raises(ValueError, match='sealed'):
        fit_ended_prefix(None, history, provider=provider, output=tmp_path,
                         head_steps=40, threshold=.7, mode='risk_after_update')


def test_head_budget_is_not_replaced_by_ordinary_inter_budget(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from experiments.risk_aligned_online import ended_decoder_experiment as experiment
    monkeypatch.setattr(experiment, 'fit_ended_prefix', lambda *a, **kw: kw['head_steps'])
    result = experiment.prefix_updater(None, {}, [],
        provider=SimpleNamespace(ended_ids=list(range(30))), output=tmp_path,
        head_steps=40, threshold=.7, mode='risk_after_update', steps=4)
    assert result == 40


def test_independent_count_excludes_exact30ms_and_post_disruption_alarm():
    from experiments.risk_aligned_online.verify_ended_decoder import classify
    times = np.asarray([.068, .070, .101])
    valid = np.ones(3, bool)
    late = classify(times, np.asarray([.1, .9, .9]), valid,
                    threshold=.7, disruptive=True, down=.1)
    assert late['alarm'] and late['lead_ms'] == 30. and not late['timely']
    timely = classify(times, np.asarray([.9, .1, .9]), valid,
                      threshold=.7, disruptive=True, down=.1)
    assert timely['timely'] and timely['lead_ms'] == 32.
    after = classify(times, np.asarray([.1, .1, .9]), valid,
                     threshold=.7, disruptive=True, down=.1)
    assert not after['alarm'] and not after['timely']


def test_independent_count_ignores_invalid_and_preserves_safe_alarm():
    from experiments.risk_aligned_online.verify_ended_decoder import classify
    times, scores = np.asarray([.01, .02]), np.asarray([.9, .9])
    safe = classify(times, scores, np.asarray([False, True]),
                    threshold=.7, disruptive=False, down=None)
    assert safe['alarm'] and not safe['timely'] and safe['lead_ms'] is None
