"""Regressions for the safe-prefix collapse seen on the natural EAST stream."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from . import inter
from .test_inter import ContextStub, shot, update


@pytest.mark.parametrize(('safe', 'score', 'category'), [(True, .1, 'TN'),
                                                       (False, .9, 'TP')])
def test_correct_shot_retains_experience_without_classification_step(safe, score, category):
    model = ContextStub()
    model.update_policy = 'error_only'
    current = shot(1, safe=safe, sealed=score)
    before = deepcopy(model.state_dict())
    result = update(model, current)
    assert result['reason'] == 'correct_shot_experience_only'
    assert result['sealed_category'] == category
    assert not result['accepted'] and not result['learned_neural_update']
    assert result['steps_completed'] == 0 and result['drift']['step_l2'] == 0
    assert 1 in model._inter_guard_cache
    for name, tensor in before.items():
        assert torch.equal(tensor, model.state_dict()[name])
    # The next error can still train on the previously correct safe experience.
    if safe:
        next_result = update(model, shot(2, sealed=.1), [current], steps=0)
        assert next_result['sampling']['history_negative'] == 32


@pytest.mark.parametrize(('safe', 'score', 'category'), [(True, .9, 'FP'),
                                                       (False, .1, 'FN')])
def test_sealed_error_proposes_a_persistent_neural_step(safe, score, category):
    model = ContextStub()
    model.update_policy = 'error_only'
    result = update(model, shot(1, safe=safe, sealed=score))
    assert result['sealed_category'] == category
    assert result['accepted'] and result['steps_completed'] == 4
    assert result['drift']['step_l2'] > 0


@pytest.mark.parametrize(('first_lead', 'category'), [(200., 'TP'), (30., 'TARDY'),
                                                     (30.0001, 'TP')])
def test_error_policy_uses_operational_first_alarm_not_125ms_training_pool(first_lead, category):
    model = ContextStub()
    model.update_policy = 'error_only'
    leads = np.asarray([200., 125., 80., 30.0001, 30., 0., -1.])
    current = shot(1, leads, sealed=.1)
    current['_sealed_scores'][np.flatnonzero(leads == first_lead)[0]:] = .9
    result = update(model, current)
    assert result['sealed_category'] == category
    assert (result['reason'] == 'correct_shot_experience_only') == (category == 'TP')


def test_postterminal_alarm_is_not_used_to_classify_disruption():
    model = ContextStub()
    model.update_policy = 'error_only'
    current = shot(1, [125., 80., 30., 0., -1.], sealed=.1)
    current['_sealed_scores'][-1] = .9
    assert update(model, current)['sealed_category'] == 'FN'


def test_crossings_guard_keeps_early_operational_tp_outside_training_window():
    model = ContextStub()
    model.guard_policy = 'crossings'
    old = shot(1, [200., 125., 80., 30., 0.], sealed=.1)
    old['_sealed_scores'][0] = .9
    result = update(model, shot(2), [old], steps=0)
    assert (1, 0) in {(r['shot'], r['end']) for r in result['guard']['endpoints']}
    assert result['guard']['policy'] == 'crossings'


@pytest.mark.parametrize(('kind', 'reason'), [('safe', 'new_safe_false_alarm'),
                                             ('timely', 'timely_tp_lost')])
def test_crossings_policy_still_rejects_actual_historical_crossing(kind, reason):
    model = ContextStub()
    model.guard_policy = 'crossings'
    if kind == 'safe':
        current = shot(2, logit=-1., context=3.)
        old = shot(1, safe=True, logit=-.005, context=1., sealed=.49)
    else:
        current = shot(2, safe=True, logit=1., context=3.)
        old = shot(1, logit=.01, context=1., sealed=.6)
    result = update(model, current, [old], lr=.4)
    assert not result['accepted'] and reason in result['reasons']
    assert 'historical_margin_degradation' not in result['reasons']


def test_crossings_policy_does_not_treat_a_confidence_change_as_an_error():
    model = ContextStub()
    model.guard_policy = 'crossings'
    current = shot(2)
    old = shot(1, safe=True, logit=-3., sealed=.1)
    records, _ = inter._prepare_records(model, current, [old], .5)
    rows, guard = inter._guard_rows(records)
    guard.update(threshold=.5, policy='crossings')
    batch = inter._windows(rows, model, 'cpu', torch.float32, inter.simple_data)
    # Measured candidate confidence rises by >.02 but remains safely below .5;
    # use a balanced baseline BCE that leaves room for this confidence change.
    baseline = np.full(len(rows), .02)
    guard['before_bce'] = .2
    _, reasons = inter._assess_guard(model, batch, guard, baseline)
    assert guard['max_margin_degradation'] > .02
    assert reasons == []
    guard['policy'] = 'margin'
    _, reasons = inter._assess_guard(model, batch, guard, baseline)
    assert reasons == ['historical_margin_degradation']


@pytest.mark.parametrize('attribute', ['update_policy', 'guard_policy'])
def test_unknown_update_and_guard_policies_rejected(attribute):
    model = ContextStub()
    setattr(model, attribute, 'invented')
    with pytest.raises(ValueError, match='policy'):
        update(model, shot(1))
