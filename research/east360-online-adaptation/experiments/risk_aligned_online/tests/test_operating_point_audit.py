import numpy as np
import pytest

from experiments.risk_aligned_online.operating_point_audit import (
    peak_for_shot, best_at_count, timely_auc, at_threshold,
)
from experiments.risk_aligned_online.representation_analysis import metrics_for_rows


def test_strict_thirty_ms_and_no_post_event_score():
    t = np.array([.969, .970, 1., 1.001])
    assert peak_for_shot(t, [.4, .9, .95, 1.], True, 1.) == .4
    assert peak_for_shot([.970, 1.], [.9, .8], True, 1.) is None
    assert peak_for_shot(t, [.4, .9, .95, 1.], False, None) == 1.


def test_budget_keeps_threshold_ties_together():
    rows = [dict(disruptive=True, peak=.8), dict(disruptive=False, peak=.8),
            dict(disruptive=False, peak=.8), dict(disruptive=True, peak=.9)]
    point = best_at_count(rows, 1)
    assert (point['tp'], point['fp'], point['threshold']) == (1, 0, .9)
    point = best_at_count(rows, 2)
    assert (point['tp'], point['fp'], point['threshold']) == (2, 2, .8)


def test_auc_counts_ties_half_and_missing_timely_peak_as_failure():
    rows = [dict(disruptive=True, peak=.8), dict(disruptive=True, peak=None),
            dict(disruptive=False, peak=.8), dict(disruptive=False, peak=.5)]
    assert timely_auc(rows) == pytest.approx(1.5/4)


def test_all_zero_scores_can_select_no_alarm_sentinel():
    rows = [dict(disruptive=True, peak=0.), dict(disruptive=False, peak=0.)]
    point = best_at_count(rows, 0)
    assert point['tp'] == point['fp'] == 0
    assert point['threshold'] > 1.


def test_peak_validates_inputs():
    with pytest.raises(ValueError):
        peak_for_shot([1., .9], [.2, .3], False, None)
    with pytest.raises(ValueError):
        peak_for_shot([.9, 1.], [.2, np.nan], True, 1.)


def test_curves_require_both_classes_and_valid_integer_budget():
    rows = [dict(disruptive=True, peak=.4), dict(disruptive=False, peak=.2)]
    for invalid in (-1, 1.5, 2, True):
        with pytest.raises(ValueError):
            best_at_count(rows, invalid)
    with pytest.raises(ValueError):
        timely_auc(rows[:1])


def test_compressed_peak_counts_equal_full_first_alarm_evaluation():
    rng = np.random.default_rng(131)
    full = []
    for sid in range(12):
        times = np.arange(101)/1000.
        full.append(dict(shot=sid, is_disrupt=bool(sid % 2), down=.100,
                         times=times, scores=rng.integers(0, 11, 101)/10.))
    peaks = [dict(disruptive=r['is_disrupt'],
                  peak=peak_for_shot(r['times'], r['scores'], r['is_disrupt'], r['down']))
             for r in full]
    for threshold in np.r_[np.arange(11)/10., np.nextafter(1., 2.)]:
        compressed = at_threshold(peaks, threshold)
        reference = metrics_for_rows(full, threshold)
        assert (compressed['tp'], compressed['fp']) == (reference['tp'], reference['fp'])


def test_exact_budget_search_equals_exhaustive_counts():
    rng = np.random.default_rng(131)
    rows = [dict(disruptive=bool(i % 2), peak=float(rng.integers(0, 11)/10.))
            for i in range(40)]
    candidates = [at_threshold(rows, t) for t in np.r_[np.arange(11)/10., np.nextafter(1., 2.)]]
    for budget in range(21):
        expected = max((p for p in candidates if p['fp'] <= budget),
                       key=lambda p: (p['tp'], -p['fp'], p['threshold']))
        actual = best_at_count(rows, budget)
        assert {k: actual[k] for k in ('tp', 'fp', 'threshold')} == {
            k: expected[k] for k in ('tp', 'fp', 'threshold')}
