import numpy as np
import pytest

from experiments.risk_aligned_online.io23 import RAW23
from experiments.risk_aligned_online.operating_overlap_20261023.diagnostics import (
    describe_window, caliper_match, weighted_covariance, covariance_decomposition, qa_geometry_check,
)


def record():
    x = np.ones((256, 23), np.float32)
    x[:, RAW23.index('ip')] = 200_000.
    x[:, RAW23.index('Bt')] = 2.
    x[:, RAW23.index('qa_proxy')] = 3.
    return dict(values=x, available=np.ones_like(x, bool))


def test_descriptor_uses_only_arrived_window():
    s = record()
    before = describe_window(s, 127)
    s['values'][128:] = 1.e8
    assert describe_window(s, 127) == before
    assert before['ip_A'] == 200_000
    assert before['Bt_magnitude'] == 2
    assert before['relative_current_change'] == 0
    assert before['qa_Ip_over_Bt'] == 300_000


def test_relative_change_invariant_to_current_unit_and_polarity():
    s = record()
    s['values'][:, RAW23.index('ip')] = np.arange(256) + 100.
    a = describe_window(s, 127)
    s['values'][:, RAW23.index('ip')] *= -1000.
    b = describe_window(s, 127)
    assert b['relative_current_change'] == pytest.approx(a['relative_current_change'])
    assert b['ip_A'] == pytest.approx(1000*a['ip_A'])
    assert b['current_sign'] == -1


def test_missing_zero_and_outcome_fields_not_silently_imputed():
    s = record()
    s['available'][:32, RAW23.index('ip')] = False
    assert describe_window(s, 127)['excluded'] == 'insufficient_endpoint_current'
    s = record()
    s['values'][:, RAW23.index('Bt')] = 0.
    assert describe_window(s, 127)['excluded'] == 'nonpositive_magnitude'
    s = record()
    with pytest.raises(ValueError, match='signal-only'):
        describe_window(dict(s, down=0.2), 127)


def test_caliper_does_not_force_far_neighbour():
    source = np.array([[100., 2., 0.], [200., 2., .02]])
    target = np.array([[101., 2., 0.], [200., 2., 0.], [400., 2., 0.]])
    counts, nearest = caliper_match(source, target, .05, drift_tolerance=.01)
    assert counts.tolist() == [1, 0, 0]
    assert nearest.tolist() == [0, -1, -1]
    counts, nearest = caliper_match(source, target, .05, drift_tolerance=None)
    assert counts.tolist() == [1, 1, 0]


def test_total_covariance_law_with_unequal_weights():
    h = np.array([[1., 2.], [4., 0.], [-1., 1.], [2., 3.]])
    w = np.array([.1, .2, .3, .4])
    total, within, between = covariance_decomposition(h, w, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(total, weighted_covariance(h, w))
    np.testing.assert_allclose(total, within+between, atol=1e-14)
    assert np.linalg.eigvalsh(within).min() >= -1e-14


def test_covariance_translation_invariance_does_not_preserve_risk_level():
    h = np.array([[1., 2.], [4., 0.], [-1., 1.]])
    w = np.ones(3)/3
    np.testing.assert_allclose(weighted_covariance(h, w), weighted_covariance(h+10., w))
    assert not np.allclose(h[:, 0], (h+10.)[:, 0])


def test_constant_states_have_explicit_undefined_fraction():
    from experiments.risk_aligned_online.operating_overlap_20261023.run import covariance_summary
    rows = [dict(weight=.5, shot=s, relative_current_change=0.) for s in (1,2)]
    result = covariance_summary(rows, np.ones((2,3)))
    for entry in result.values():
        assert entry['between_fraction'] is None
        assert entry['fraction_status'] == 'undefined_zero_total_variance'


def test_geometry_check_is_algebra_not_q95_reconstruction():
    s = record()
    coefficient = 2*np.pi*.25**2/1.05
    s['values'][:, RAW23.index('qa_proxy')] = coefficient*2/200_000.
    result = qa_geometry_check(s, radius=.25, major_radius=1.05)
    assert result['valid_points'] == 256
    assert result['relative_error_quantiles_50_95_99_100'][-1] < 1e-6
    assert result['interpretation'] == 'empirical_formula_consistency_not_q95_or_extractor_certification'
