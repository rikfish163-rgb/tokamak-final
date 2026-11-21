"""A whole-model loss improvement must not be confused with fast contribution."""
import copy

import pytest

from experiments.risk_aligned_online.paired_gap_guard_20261031.rules import evaluate_guards


def values(number):
    return {'source_event':number, 'historical_S':number, 'historical_D':number}


def test_absolute_improvement_can_hide_worse_fast_contribution():
    result = evaluate_guards(values(1.), values(.8), values(.9), values(.5))
    assert result['original_pass'] and result['dual_absolute_pass']
    assert not result['paired_pass']
    assert set(result['gap_violations']) == set(values(1.))
    assert result['groups']['historical_S']['gap_change'] == pytest.approx(.2)


def test_dual_absolute_is_distinct_from_the_gap_condition():
    result = evaluate_guards(values(1.), values(.8), values(.9), values(.85))
    assert result['original_pass']
    assert not result['dual_absolute_pass'] and not result['paired_pass']
    assert not result['gap_violations']
    assert set(result['off_violations']) == set(values(1.))


def test_new_rule_never_accepts_an_original_rejection():
    result = evaluate_guards(values(1.), values(.8), values(1.01), values(.8))
    assert not any(result[key] for key in ('original_pass', 'dual_absolute_pass', 'paired_pass'))


def test_no_change_and_joint_improvement_are_allowed_without_mutation():
    args = (values(1.), values(1.1), values(.8), values(.95))
    original = copy.deepcopy(args)
    result = evaluate_guards(*args)
    assert all(result[key] for key in ('original_pass', 'dual_absolute_pass', 'paired_pass'))
    assert args == original
    result = evaluate_guards(values(.5), values(.3), values(.5), values(.3))
    assert result['paired_pass']


def test_missing_historical_class_is_not_fabricated():
    d = {'source_event':.1, 'historical_S':.4}
    result = evaluate_guards(d, d, d, d)
    assert set(result['groups']) == set(d) and result['paired_pass']


def test_tolerance_is_explicit_and_applies_to_all_comparisons():
    result = evaluate_guards(values(0.), values(0.), values(5e-8), values(0.))
    assert result['paired_pass']
    result = evaluate_guards(values(0.), values(0.), values(5e-8), values(0.), tolerance=0.)
    assert not result['paired_pass']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., None, True, '0.1'])
def test_invalid_losses_are_rejected(bad):
    first = values(.1)
    first['source_event'] = bad
    with pytest.raises(ValueError):
        evaluate_guards(first, values(.1), values(.1), values(.1))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, True, None])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_guards(values(.1), values(.1), values(.1), values(.1), tolerance=bad)


def test_inconsistent_group_keys_or_missing_source_are_errors():
    for bad in ({}, {'historical_S':.2}, {'source_event':.1, 'new_group':.2}):
        with pytest.raises(ValueError):
            evaluate_guards(bad, bad, bad, bad)
    with pytest.raises(ValueError):
        evaluate_guards(values(.1), {'source_event':.1}, values(.1), values(.1))
