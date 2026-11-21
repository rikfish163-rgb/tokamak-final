import numpy as np

from experiments.risk_aligned_online.error_forensics import classify, prefix_features, record


def test_timely_is_strict_and_safe_alarm_is_error():
    assert classify(True, True, 30.) == 'TARDY'
    assert classify(True, True, 30.001) == 'TP'
    assert classify(True, False, None) == 'FN'
    assert classify(False, True, None) == 'FP'
    assert classify(False, False, None) == 'TN'


def test_postterminal_crossing_never_becomes_alarm():
    result = record(np.array([0., .010, .050]), np.array([.1, .1, .99]), True, .04, .7)
    assert result['category'] == 'FN'
    assert result['deadline_index'] == 0  # .010 is EXACTLY 30ms: not timely.
    safe = record(np.array([0., .010]), np.array([.1, .8]), False, None, .7)
    assert safe['deadline_index'] is None  # Safe shots have no disruption deadline.


def test_prefix_features_do_not_read_future_and_constant_is_not_missing():
    values = np.arange(80*23).reshape(80, 23).astype(float)
    available = np.ones_like(values, bool)
    values[:, 0] = 4.
    before = prefix_features(values, available, 30, np.zeros(23), np.ones(23))
    values[31:] = 1e25
    available[31:] = False
    assert prefix_features(values, available, 30, np.zeros(23), np.ones(23)) == before
    assert before['Bt']['constant']
    assert before['Bt']['available_fraction'] == 1.
    assert before['Bt']['log_variation'] < 1e-14
