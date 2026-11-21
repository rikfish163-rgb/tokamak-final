import pytest

from experiments.risk_aligned_online.warning_lead_metrics import paired, summarize


def row(shot, lead, *, disruptive=True, safe_alarm=False):
    return dict(shot=shot, disruptive=disruptive, down_s=10. if disruptive else None,
                alarm=(lead is not None) if disruptive else safe_alarm, lead_ms=lead,
                timely=bool(disruptive and lead is not None and lead > 30))


def test_secondary_deadlines_are_strict_and_use_all_disruptions():
    rows = [row(i, lead) for i, lead in enumerate([15, 20, 25, 30, 31, 4000, None])]
    rows += [row(7, None, disruptive=False, safe_alarm=True), row(8, None, disruptive=False)]
    m = summarize(rows)
    assert [a['tp'] for a in m['by_deadline'].values()] == [5, 4, 3, 2]
    assert m['by_deadline']['30']['tpr'] == 2/7
    assert m['by_deadline']['30']['tp_at_most_3000ms'] == 1
    assert m['fp'] == 1 and m['fpr'] == .5
    assert m['all_alarmed_disruptions']['count'] == 6
    assert m['alarmed_disruptions_at_most_3000ms']['mean_ms'] == 24.2


def test_paired_average_compares_the_same_alarmed_shots():
    old = [row(i, x) for i, x in enumerate([10, 22, None, 5000])]
    new = [row(i, x) for i, x in enumerate([15, 30, 31, 9000])]
    m = paired(old, new)
    assert m['lead_gain_common_at_most_3000ms']['count'] == 2
    assert m['lead_gain_common_at_most_3000ms']['mean_ms'] == 6.5
    assert m['newly_alarmed_disruptions'] == [2]
    assert m['deadline_flips']['30']['gained'] == [2]
    assert m['deadline_flips']['25']['gained'] == [1, 2]
    with pytest.raises(ValueError, match='same'):
        paired(old, list(reversed(new)))
