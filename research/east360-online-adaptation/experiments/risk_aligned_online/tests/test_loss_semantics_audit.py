"""A correct early alarm need not be a positive fixed-window training point."""
import numpy as np

from experiments.risk_aligned_online.lifecycle_stream_20261025.loss_semantics_audit import describe


def test_early_timely_peak_has_opposed_point_label():
    result = describe(np.array([.8, .9, .97]), np.array([.9, .6, .99]),
                      True, 1., .7, np.array([0, 1]))
    assert result['timely']
    assert result['timely_peak_lead_ms'] == 200.
    assert result['timely_peak_point_label'] == 0
    assert result['point_gradient_at_timely_peak'] > 0
    assert result['positive_bag_gradient_at_timely_peak'] < 0
    assert result['selected_high_negative_count'] == 1


def test_positive_window_peak_is_not_opposed():
    result = describe(np.array([.8, .9]), np.array([.6, .9]),
                      True, 1., .7, np.array([0, 1]))
    assert result['timely_peak_point_label'] == 1
    assert result['point_gradient_at_timely_peak'] < 0
    assert result['selected_high_negative_count'] == 0


def test_strict_deadline_and_post_terminal_values_are_not_timely_peaks():
    result = describe(np.array([.97, 1.001]), np.array([.9, 1.]),
                      True, 1., .7, np.array([], dtype=int))
    assert not result['timely']
    assert result['timely_peak_index'] is None


def test_safe_shot_has_only_negative_targets_and_no_positive_bag_gradient():
    result = describe(np.array([.8, .9]), np.array([.9, .6]),
                      False, None, .7, np.array([0, 1]))
    assert result['alarm'] and not result['timely']
    assert result['positive_bag_gradient_at_timely_peak'] is None
    assert result['selected_high_negative_count'] == 1
