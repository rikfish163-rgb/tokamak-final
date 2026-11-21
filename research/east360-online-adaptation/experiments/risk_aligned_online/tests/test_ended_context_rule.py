import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.ended_context_rule import (
    ContextFfnModel, causal_peak_logit, fixed_signal_loss, peak_index, train_pair,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(131)
    return ContextFfnModel(Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                                   torch.zeros(23), torch.ones(23))).eval()


def shot(disruptive=False, size=230):
    rng = np.random.default_rng(4)
    return dict(device='EAST', shot=12 if disruptive else 13,
        values=rng.normal(size=(size, 23)).astype(np.float32),
        available=np.ones((size, 23), bool), times=np.arange(size)/1000.,
        valid=np.ones(size, bool), is_disrupt=disruptive,
        down=(size-1)/1000. if disruptive else None,
        horizon=125., _sealed_scores=np.linspace(.1, .9, size, dtype=np.float32))


def test_uniform_loss_and_gradient_exactly_match_parent_with_missing_channels():
    m = model()
    prediction = torch.randn(1, 25, 23, requires_grad=True)
    truth = torch.randn_like(prediction)
    available = torch.rand_like(truth) > .3
    truth[0, 0, 0] = float('nan')
    loss = fixed_signal_loss(m, prediction.detach(), truth, available)
    original = m.forecast_loss(prediction, truth, available)
    weighted = loss(prediction, truth, available)
    torch.testing.assert_close(weighted, original, rtol=0, atol=0)
    g1, = torch.autograd.grad(weighted, prediction, retain_graph=True)
    g2, = torch.autograd.grad(original, prediction)
    torch.testing.assert_close(g1, g2, rtol=0, atol=0)


def test_weights_are_fixed_for_the_same_candidate_acceptance_objective():
    m = model()
    with torch.no_grad():
        m.rule.net[-1].weight.normal_(std=.1)
    forecast = torch.randn(1, 25, 23)
    truth = torch.randn_like(forecast)
    available = torch.ones_like(truth, dtype=torch.bool)
    loss = fixed_signal_loss(m, forecast, truth, available)
    weights = loss.weights.clone()
    loss(forecast*3, truth, available)
    torch.testing.assert_close(loss.weights, weights, rtol=0, atol=0)
    assert (weights >= .25).all() and (weights <= 2).all()


def test_peak_uses_operational_lead_not_125ms_point_label():
    s = shot(True)
    scores = np.zeros(230, np.float32)
    scores[50], scores[199], scores[220] = .9, .99, 1.
    # index199 is exactly30ms: excluded; index50 is179ms: allowed.
    assert peak_index(s, scores) == 50


def test_meta_outer_risk_reaches_rule_but_not_source_or_persistent_ffn():
    m = model()
    before = {k: v.clone() for k, v in m.base.state_dict().items()}
    optimizer = torch.optim.Adam(m.rule.parameters(), lr=.003)
    pair = [(shot(False), 180), (shot(True), 180)]
    record = train_pair(m, pair, optimizer, boundary=.7)
    assert record['gradient_norm'] > 0
    assert record['rule_delta_norm'] > 0
    assert not m.memory.any() and m.memory.grad is None
    for k, v in m.base.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize('nonuniform', [False, True])
def test_causal_replay_matches_stream_and_maturity_emits_before_update(nonuniform):
    m = model()
    if nonuniform:
        with torch.no_grad():
            m.rule.net[-1].weight.normal_(std=.1)
    s = shot()
    scores, audit = predict_shot(m, s, intra=True, stride=100, step=.05, radius=1.)
    for query in (0, 25, 26, 125, 126, 180):
        logit, info = causal_peak_logit(m, s, query, create_graph=False)
        assert abs(float(logit.detach().sigmoid())-float(scores[query])) < 2e-6
        assert info['mature_events'] == len(range(0, query-25, 100))
    assert audit['fast_discarded'] and not m.memory.any()


def test_current_outcome_cannot_change_signal_only_predictions():
    m = model()
    a, b = shot(False), shot(True)
    scores_a, _ = predict_shot(m, a, intra=True)
    scores_b, _ = predict_shot(m, b, intra=True)
    np.testing.assert_array_equal(scores_a, scores_b)


def test_outer_requires_sealed_ended_east_and_rule_only_optimizer():
    m = model()
    pair = [(shot(False), 180), (shot(True), 180)]
    with pytest.raises(ValueError, match='rule'):
        train_pair(m, pair, torch.optim.Adam([m.memory]), boundary=.7)
    del pair[0][0]['_sealed_scores']
    with pytest.raises(ValueError, match='sealed'):
        train_pair(m, pair, torch.optim.Adam(m.rule.parameters()), boundary=.7)


def test_rule_budget_not_overwritten_by_common_inter_steps(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from experiments.risk_aligned_online import ended_context_experiment as experiment
    monkeypatch.setattr(experiment, 'fit_ended_prefix', lambda *a, **kw: kw['steps'])
    provider = SimpleNamespace(ended_ids=list(range(30)))
    # The runner always passes its ordinary inter budget (4). The meta-rule
    # budget (40) belongs to a DIFFERENT optimizer and must not be overridden.
    actual = experiment.prefix_updater(None, {}, [], provider=provider,
        output=tmp_path, rule_steps=40, steps=4, threshold=.7)
    assert actual == 40


@pytest.mark.parametrize('violation', ['pending', 'future_history', 'not_sealed'])
def test_prefix_training_rejects_premature_or_future_history(tmp_path, violation):
    from types import SimpleNamespace
    from experiments.risk_aligned_online.ended_context_experiment import fit_ended_prefix
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
        fit_ended_prefix(None, history, provider=provider, output=tmp_path, steps=40, threshold=.7)
