import numpy as np

from experiments.risk_aligned_online.lora_stream_20261102.clock_audit import opportunity


def event(end, *, observed=10, accepted=True):
    return dict(
        target_last=end, first_affected=end + 1, observed=observed, accepted=accepted
    )


def test_a_128_point_clock_can_intersect_30_to_125_ms_stage():
    times = np.arange(1000) / 1000
    result = opportunity(times, 0.9, [event(i) for i in range(159, 1000, 128)])
    assert result["clock_endpoints"] == [799]
    assert result["observed_endpoints"] == result["accepted_endpoints"] == [799]


def test_same_clock_can_skip_the_whole_label_stage():
    times = np.arange(1000) / 1000
    result = opportunity(times, 0.95, [event(i) for i in range(159, 1000, 128)])
    assert result["clock_endpoints"] == []


def test_first_effect_must_be_strictly_more_than_30_ms_before_disruption():
    times = np.arange(300) / 1000
    assert opportunity(times, 0.19, [event(159)])["clock_endpoints"] == []
    assert opportunity(times, 0.191, [event(159)])["clock_endpoints"] == [159]


def test_no_observation_or_rejected_update_are_distinguished():
    times = np.arange(1000) / 1000
    none = opportunity(times, 0.9, [event(799, observed=0, accepted=False)])
    rejected = opportunity(times, 0.9, [event(799, accepted=False)])
    assert none["clock_endpoints"] == rejected["clock_endpoints"] == [799]
    assert none["observed_endpoints"] == []
    assert rejected["observed_endpoints"] == [799]
    assert none["accepted_endpoints"] == rejected["accepted_endpoints"] == []


def test_no_future_endpoint_and_no_warmup_updates():
    times = np.arange(160) / 1000
    assert opportunity(times, 0.2, [event(159)])["clock_endpoints"] == []
    assert opportunity(np.arange(1000) / 1000, 0.9, [])["clock_endpoints"] == []
