import numpy as np
import pytest

from experiments.risk_aligned_online.relational_attention_20261030.run import (
    schedule, classify_queries, shot_summary,
)


def test_schedule_uses_no_outcomes_and_strictly_later_queries():
    assert schedule(177) == []
    assert schedule(178) == [(127, 152, 177)]
    assert schedule(203) == [(127, 152, 177), (152, 177, 202)]
    assert all(o + 25 == m and m + 25 == q < 1000 for o, m, q in schedule(1000))


def test_east_strict30_and_fixed125_groups():
    times = np.array([.800, .875, .969, .970, .990, 1.001])
    assert classify_queries(times, True, 1.).tolist() == [
        'early', 'danger', 'danger', 'late', 'late', 'postterminal']
    assert classify_queries(times, False, None).tolist() == ['safe'] * 6


def test_summary_equal_shot_not_window_weight_and_records_zeros():
    records = [dict(shot=1, group='safe')] * 3 + [dict(shot=2, group='safe')]
    pred = dict(cold=np.zeros(4), real=np.array([-1., -1., -1., 1.]),
                support=np.array([True, True, True, False]))
    rows, groups = shot_summary(records, pred, ('real',))
    assert len(rows) == 2
    expected = (np.logaddexp(0., -1.) + np.logaddexp(0., 1.)) / 2 - np.log(2)
    assert groups['real']['safe']['mean_delta_bce'] == pytest.approx(expected)
    assert groups['real']['safe']['supported_shots'] == 1
    assert groups['real']['safe']['shots'] == 2


def test_summary_nonfinite_prediction_rejected():
    with pytest.raises(ValueError, match='finite'):
        shot_summary([dict(shot=1, group='safe')],
                     dict(cold=np.array([0.]), real=np.array([np.nan]), support=np.array([True])),
                     ('real',))
