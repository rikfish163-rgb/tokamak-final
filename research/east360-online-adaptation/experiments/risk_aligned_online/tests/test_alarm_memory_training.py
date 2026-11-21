from experiments.risk_aligned_online.alarm_memory_training import timing_summary


def test_timing_summary_uses_measured_first_alarm_not_all_crossings():
    result = {'predictions': [dict(timely=True, lead_ms=300.),
                              dict(timely=True, lead_ms=301.),
                              dict(timely=True, lead_ms=1001.),
                              dict(timely=False, lead_ms=None),
                              dict(timely=False, lead_ms=5.)]}
    assert timing_summary(result) == {'above_300ms': 2, 'above_1000ms': 1}
