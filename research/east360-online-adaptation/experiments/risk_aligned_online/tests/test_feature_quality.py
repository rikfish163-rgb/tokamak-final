import numpy as np
import pytest

from experiments.risk_aligned_online.feature_quality import describe_channel


def test_constant_prefix_does_not_imply_constant_record():
    x = np.r_[np.ones(100), np.arange(100.)]
    out = describe_channel(x, np.ones(200, bool), center=0., scale=1.)
    assert out['prefix100_constant'] is True
    assert out['record_constant'] is False
    assert out['n_observed'] == 200


def test_missing_is_not_a_constant_signal():
    out = describe_channel(np.zeros(200), np.zeros(200, bool), center=0., scale=1.)
    assert out['record_constant'] is None
    assert out['prefix100_constant'] is None
    assert out['quantiles'] is None
    assert out['available_fraction'] == 0


def test_nonfinite_and_masked_values_never_enter_statistics():
    out = describe_channel(np.array([np.nan, np.inf, 999., 2., 2.]),
                           np.array([True, True, False, True, True]), center=1., scale=2.)
    assert out['n_observed'] == 2
    assert out['record_constant'] is True
    assert out['quantiles'] == [2.] * 5
    assert out['median_source_z'] == .5


def test_endpoints_included_and_empty_or_bad_scale_rejected():
    x = np.r_[np.zeros(999), 10.]
    out = describe_channel(x, np.ones(1000, bool), center=0., scale=1.)
    assert out['quantiles'][-1] == 10.
    assert out['sampled_points'] == 128
    with pytest.raises(ValueError):
        describe_channel(x, np.ones(1000, bool), center=0., scale=0.)
    with pytest.raises(ValueError):
        describe_channel(np.array([]), np.array([], bool), center=0., scale=1.)
