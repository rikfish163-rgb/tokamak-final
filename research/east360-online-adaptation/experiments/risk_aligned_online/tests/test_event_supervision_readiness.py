"""Offline family annotations must not silently become recovery supervision."""
import pytest

from experiments.risk_aligned_online.event_prefix_probe import Event
from experiments.risk_aligned_online.event_supervision_readiness import summarize


def row(shot, flag=False, role='train'):
    return dict(shot=shot, role=role, disruptive=flag)


def meta(flag=False, status='event'):
    return dict(outcome=int(flag), status=status, cohort='training_pool')


def test_sparse_intervals_are_not_recovery_or_ordered_risk_stages():
    events = {1: [Event(1, 1, 3, .2, .3), Event(1, 2, 0, .5, .6)],
              2: [Event(2, 1, 3, .2, .4)]}
    result = summarize([row(1, True), row(2)], events,
                       {1: meta(True), 2: meta()})['train']
    assert result['shots_with_events'] == 2
    assert result['multi_event_shots'] == 1
    assert result['adjacent_event_pairs'] == 1
    assert result['adjacent_with_unannotated_gap'] == 1
    assert result['alphabet_descending_pairs'] == 1
    assert result['transition_counts']['D']['D>A'] == 1
    assert result['family_shots']['D'] == {'D': 1, 'S': 1}
    assert result['recovery_labels_available'] is False
    assert result['rows'][0]['unannotated_gap_seconds'] == pytest.approx(.2)


def test_unknown_missing_and_short_are_separate_not_safe_targets():
    result = summarize([row(1), row(2), row(3)], {},
                       {1: meta(status='no_event_detected'),
                        2: meta(status='excluded_short_record')})['train']
    assert result['status_counts'] == {'no_event_detected': 1,
                                     'excluded_short_record': 1, 'not_in_label_file': 1}
    assert result['shots_with_events'] == 0
    assert result['duration_ms'] is None
    assert result['multi_event_fraction_of_annotated_shots'] is None
    assert result['recovery_labels_available'] is False


def test_repeated_family_counts_shot_once_and_intervals_separately():
    events = {1: [Event(1, 1, 0, .2, .3), Event(1, 2, 0, .3, .4)]}
    r = summarize([row(1)], events, {1: meta()})['train']
    assert r['family_shots']['A']['S'] == 1
    assert r['family_intervals']['A']['S'] == 2
    assert r['adjacent_with_unannotated_gap'] == 0
    assert r['alphabet_equal_pairs'] == 1


@pytest.mark.parametrize('inventory,events,metadata', [
    ([row(1, role='test')], {}, {}),
    ([row(1), row(1)], {}, {}),
    ([row(1)], {}, {1: meta(True, 'no_event_detected')}),
    ([row(1)], {}, {1: meta()}),
    ([row(1)], {1: [Event(1, 1, 0, .2, .4), Event(1, 2, 1, .3, .5)]}, {1: meta()}),
])
def test_reject_wrong_role_identity_outcome_status_or_overlap(inventory, events, metadata):
    with pytest.raises(ValueError):
        summarize(inventory, events, metadata)
