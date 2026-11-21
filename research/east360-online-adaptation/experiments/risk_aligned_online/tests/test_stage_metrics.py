import copy

import pytest

from experiments.risk_aligned_online.stage_metrics import summarize_commissioning


def row(shot, disrupt, alarm=False, lead=None):
    return dict(shot=shot, is_disrupt=disrupt, alarm=alarm,
                timely=bool(disrupt and alarm and lead is not None and lead > 30), lead_ms=lead)


def test_hand_calculated_costs_and_segments():
    order = [dict(shot=i, proxy=float(i)) for i in range(1, 5)]
    predictions = [row(1, True, True, 3001), row(2, False, True),
                   row(3, True, True, 30), row(4, False)]
    before = copy.deepcopy((predictions, order))
    result = summarize_commissioning(predictions, order, 2, block_size=2)
    full = result['full']
    assert [full[k] for k in ('TP', 'FN', 'FP', 'TN')] == [1, 1, 1, 1]
    assert full['TPR'] == full['FPR'] == full['balanced_accuracy'] == .5
    assert full['J'] == 0
    assert full['timely_accuracy'] == .5
    assert full['early_over_3000'] == 1
    assert full['missed_proxy_sum'] == 3
    assert full['total_disrupt_proxy'] == 4
    assert full['normalized_missed_proxy'] == .75
    assert result['startup_cost_proxy'] == 1  # Includes timely startup disruption.
    assert result['post_startup_missed_cost_proxy'] == 3
    assert result['startup']['n'] == result['post_startup']['n'] == 2
    assert len(result['blocks']) == 2
    assert result['high_proxy_quartile']['manifest_shots'] == [4]
    assert result['cumulative'][-1]['missed_proxy_sum'] == 3
    assert result['cumulative'][-1]['startup_cost_proxy'] == 1
    assert (predictions, order) == before


def test_all_miss_single_class_and_large_budget():
    result = summarize_commissioning([row(1, True), row(2, True, True, 30)],
                                     [dict(shot=1, proxy=2), dict(shot=2, proxy=6)], 99)
    full = result['full']
    assert full['FN'] == 2 and full['TPR'] == 0 and full['FPR'] is None
    assert full['balanced_accuracy'] is None and full['J'] is None
    assert full['timely_accuracy'] == 0 and full['normalized_missed_proxy'] == 1
    assert result['startup_cost_proxy'] == 8
    assert result['post_startup']['n'] == 0
    assert result['post_startup_missed_cost_proxy'] == 0


def test_safe_only_empty_and_zero_denominators():
    safe = summarize_commissioning([row(1, False)], [dict(shot=1, proxy=0)], 0)['full']
    assert safe['TPR'] is None and safe['FPR'] == 0
    assert safe['timely_accuracy'] == 1 and safe['normalized_missed_proxy'] is None
    empty = summarize_commissioning([], [], 0)
    assert empty['full']['n'] == 0 and empty['full']['timely_accuracy'] is None
    assert empty['blocks'] == empty['cumulative'] == []
    zero = summarize_commissioning([row(1, True)], [dict(shot=1, proxy=0)], 0)
    assert zero['full']['normalized_missed_proxy'] is None


def test_quartile_uses_complete_manifest_not_observed_prefix():
    order = [dict(shot=i, proxy=i) for i in range(1, 9)]
    result = summarize_commissioning([row(1, False), row(2, False)], order, 0)
    assert result['high_proxy_quartile']['manifest_shots'] == [7, 8]
    assert result['high_proxy_quartile']['metrics']['n'] == 0
    assert result['manifest_count'] == 8 and result['observed_count'] == 2


@pytest.mark.parametrize('lead,timely', [(30, False), (30.000001, True), (3000, True)])
def test_strict_timing_boundary(lead, timely):
    result = summarize_commissioning([row(1, True, True, lead)], [dict(shot=1, proxy=1)], 0)
    assert result['full']['TP'] == int(timely)
    assert result['full']['early_over_3000'] == 0


@pytest.mark.parametrize('timely', [False, True])
def test_disruptive_alarm_requires_lead_even_if_not_timely(timely):
    prediction = row(1, True, True)
    prediction['timely'] = timely
    with pytest.raises(ValueError, match='lead_ms'):
        summarize_commissioning([prediction], [dict(shot=1, proxy=1)], 0)


@pytest.mark.parametrize('lead', [-1., -1e-12, float('nan'), float('inf')])
def test_disruptive_alarm_rejects_invalid_lead(lead):
    with pytest.raises(ValueError, match='lead_ms'):
        summarize_commissioning([row(1, True, True, lead)], [dict(shot=1, proxy=1)], 0)


def test_safe_alarm_without_lead_is_valid_false_positive():
    full = summarize_commissioning([row(1, False, True)], [dict(shot=1, proxy=1)], 0)['full']
    assert full['FP'] == 1 and full['TP'] == 0
    assert full['FPR'] == 1 and full['timely_accuracy'] == 0


def test_zero_lead_disruptive_alarm_is_valid_tardy():
    full = summarize_commissioning([row(1, True, True, 0.)], [dict(shot=1, proxy=1)], 0)['full']
    assert full['FN'] == 1 and full['TP'] == 0


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True])
def test_invalid_proxy(bad):
    with pytest.raises(ValueError):
        summarize_commissioning([], [dict(shot=1, proxy=bad)], 0)


@pytest.mark.parametrize('budget,block', [(-1, 20), (1.5, 20), (True, 20), (0, 0), (0, 1.5)])
def test_invalid_budgets(budget, block):
    with pytest.raises(ValueError):
        summarize_commissioning([], [], budget, block)


def test_reject_duplicate_identity_order_and_inconsistent_timing():
    order = [dict(shot=1, proxy=1), dict(shot=2, proxy=2)]
    for predictions, manifest in [([row(2, False)], order),
                                  ([row(1, False), row(1, False)], order),
                                  ([], [order[0], order[0]]),
                                  ([row(3, False)], order)]:
        with pytest.raises(ValueError):
            summarize_commissioning(predictions, manifest, 0)
    bad = row(1, True, True, 30)
    bad['timely'] = True
    with pytest.raises(ValueError, match='timely'):
        summarize_commissioning([bad], order, 0)
