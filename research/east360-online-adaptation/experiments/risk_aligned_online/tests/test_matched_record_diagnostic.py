import numpy as np
import pytest

from experiments.risk_aligned_online.matched_preparation_20261023.record_diagnostic import summarize_record


def test_full_record_keeps_negative_zero_and_excludes_missing():
    s = dict(times=np.arange(4, dtype=float), values=np.ones((4, 23)),
             available=np.ones((4, 23), bool))
    s['values'][:, 0] = [-2., 0., 2., 999.]
    s['available'][3, 0] = False
    s['available'][:, 2] = False
    before = s['values'].copy()
    out = summarize_record(s)
    assert out['channels']['Bt']['quantiles'] == pytest.approx([-2., -1.6, 0., 1.6, 2.])
    assert out['channels']['Bt']['negative_fraction'] == pytest.approx(1/3)
    assert out['channels']['Bt']['zero_fraction'] == pytest.approx(1/3)
    assert out['channels']['MNM']['quantiles'] is None
    np.testing.assert_array_equal(s['values'], before)


def test_record_rejects_nonfinite_observation():
    s = dict(times=np.arange(2, dtype=float), values=np.ones((2, 23)),
             available=np.ones((2, 23), bool))
    s['values'][0, 0] = np.nan
    with pytest.raises(ValueError, match='valid23'):
        summarize_record(s)
