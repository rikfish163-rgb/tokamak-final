import numpy as np

from experiments.risk_aligned_online.arrival_stream_20261030.diagnose import describe_shot


def test_first_alarm_before_update_cannot_be_retracted():
    times = np.arange(200) / 1000
    cold = np.full(200, .1)
    cold[2] = .8
    changed = cold.copy()
    changed[153:] = .01
    row = describe_shot(times, {'cold': cold, 'real': changed}, False, None, .7, 153)
    assert row['cold_error'] and row['early_sealed_false_alarm']
    assert row['arms']['real']['first_alarm_index'] == 2
    assert row['arms']['real']['margin'] == .8 - .7


def test_timely_margin_excludes_exact_thirty_ms_and_late_points():
    times = np.arange(201) / 1000
    cold = np.full(201, .1)
    cold[170] = .9
    real = cold.copy()
    real[169] = .8
    row = describe_shot(times, {'cold': cold, 'real': real}, True, .2, .7, 153)
    assert row['cold_error'] and row['timely_update_possible']
    assert row['arms']['cold']['lead_ms'] == 30
    assert row['arms']['cold']['margin'] == .1 - .7
    assert row['arms']['real']['lead_ms'] == 31
    assert row['arms']['real']['margin'] == .8 - .7


def test_short_shot_has_no_timely_update_opportunity():
    times = np.arange(100) / 1000
    cold = np.full(100, .1)
    row = describe_shot(times, {'cold': cold, 'real': cold}, True, .099, .7, None)
    assert row['cold_error'] and not row['timely_update_possible']
    assert not row['early_sealed_false_alarm']
