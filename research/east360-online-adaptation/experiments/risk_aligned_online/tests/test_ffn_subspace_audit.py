import pytest

from experiments.risk_aligned_online.ffn_subspace_audit import partition_fragments, whole_source_screen


def test_negative_endpoints_are_split_by_actual_source_shot_outcome():
    rows = [dict(shot=s, label=y, risk_before=.5, risk_after=a)
            for s, y, a in ((1, 0, .4), (2, 0, .6), (2, 1, .3))]
    groups = partition_fragments(rows, {1: False, 2: True})
    assert set(groups) == {'safe', 'disrupt_prepositive', 'disrupt_positive'}
    assert all(g['n'] == 1 for g in groups.values())
    assert groups['safe']['risk_after'] == .4
    assert groups['disrupt_prepositive']['risk_after'] == .6
    assert groups['disrupt_positive']['risk_after'] == .3


def test_fragment_partition_rejects_unknown_shot_or_positive_safe_label():
    row = dict(shot=1, label=0, risk_before=.5, risk_after=.4)
    with pytest.raises(ValueError, match='outcome'):
        partition_fragments([row], {})
    with pytest.raises(ValueError, match='safe'):
        partition_fragments([dict(row, label=1)], {1: False})


def counts(tp, fp):
    return dict(tp=tp, fp=fp, n_disrupt=80, n_safe=80)


def test_whole_gate_does_not_accept_same_counts_or_trade_lost_tp_for_fewer_fp():
    results = dict(static=counts(73, 6), random=counts(73, 6), signal=counts(72, 6), risk=counts(73, 6))
    gate = whole_source_screen(results)
    assert not gate['passed'] and 'not_strictly_better_than_random' in gate['reasons']
    results['risk'] = counts(72, 3)
    gate = whole_source_screen(results)
    assert not gate['passed'] and 'source_static_tp_lost' in gate['reasons']
    results['risk'] = counts(74, 6)
    assert whole_source_screen(results)['passed']


def test_whole_gate_rejects_unmatched_cohorts():
    results = {name: counts(73, 6) for name in ('static', 'random', 'signal', 'risk')}
    results['signal']['n_safe'] = 81
    with pytest.raises(ValueError, match='same'):
        whole_source_screen(results)
