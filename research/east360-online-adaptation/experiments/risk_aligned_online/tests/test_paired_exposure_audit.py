import numpy as np
import pytest

from experiments.risk_aligned_online.paired_exposure_audit import (
    summarize_trace, prefix_features, summarize_safe,
)


def test_prefix_keeps_late_alarm_separate_from_full_record():
    out = summarize_trace([3., 3.001, 3.002, 3.003], [.1, .2, .8, .9], .8, [2, 5])
    assert out['alarm_index'] == 2
    assert out['start_s'] == 3.
    assert out['span_s'] == pytest.approx(.003)
    assert out['prefix']['2'] == dict(complete=True, points=2, alarm=False)
    assert out['prefix']['5'] == dict(complete=False, points=4, alarm=True)
    assert out['exceedance_fraction'] == .5


def test_one_point_trace_has_zero_span_and_can_alarm():
    out = summarize_trace([9.], [.8], .8, [2])
    assert out['span_s'] == 0
    assert out['alarm_index'] == 0
    assert not out['prefix']['2']['complete']


@pytest.mark.parametrize('times,scores,threshold', [
    ([], [], .5), ([1, 1], [.1, .2], .5), ([1, 2], [.1], .5),
    ([1], [np.nan], .5), ([1], [1.1], .5), ([1], [.1], np.nan),
])
def test_invalid_trace_rejected(times, scores, threshold):
    with pytest.raises(ValueError):
        summarize_trace(times, scores, threshold, [2])


def test_feature_summary_uses_observed_prefix_not_suffix_or_masked_fill():
    x = np.zeros((4, 23)); m = np.ones_like(x, dtype=bool)
    x[:2, 0] = [10, 20]; x[2:, 0] = 10000
    m[:2, 1] = False; x[:2, 1] = np.nan
    out = prefix_features(x, m, np.zeros(23), np.ones(23), 2)
    assert out[0]['median_source_coordinate'] == pytest.approx(np.median(np.arcsinh([10, 20])))
    assert out[0]['beyond_10_source_scales'] == .5
    assert out[1] == dict(available_fraction=0., median_source_coordinate=None,
                         beyond_10_source_scales=None)
    assert out[2]['median_source_coordinate'] == 0


def test_invalid_observed_feature_and_scale_rejected():
    x = np.zeros((2, 23)); m = np.ones_like(x, dtype=bool)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        prefix_features(x, m, np.zeros(23), np.ones(23), 2)
    with pytest.raises(ValueError):
        prefix_features(np.zeros((2, 23)), m, np.zeros(23), np.zeros(23), 2)


def test_safe_summary_conditions_complete_prefix_without_replacing_full_denominator():
    a = summarize_trace([0., 1., 2.], [.1, .1, .9], .8, [2])
    b = summarize_trace([0.], [.9], .8, [2])
    rows = [dict(disruptive=False, static=a, fixed=a),
            dict(disruptive=False, static=b, fixed=b),
            dict(disruptive=True, static=a, fixed=a)]
    out = summarize_safe(rows, [2])
    assert out['n_safe'] == 2
    assert out['arms']['static']['fp'] == 2
    assert out['prefix']['2']['all_available']['static_fp'] == 1
    assert out['prefix']['2']['complete_only']['n_safe'] == 1
    assert out['prefix']['2']['complete_only']['static_fp'] == 0


def test_empty_safe_group_has_no_invented_rate():
    out = summarize_safe([], [2])
    assert out['n_safe'] == 0
    assert out['arms']['fixed']['fpr'] is None
