import numpy as np
import pytest

from experiments.risk_aligned_online.matched_preparation_20261023.prefix import summarize_prefix


def test_timestamp_boundary_mask_and_real_zero_are_respected():
    s = dict(times=np.array([4.,4.05,4.1]), values=np.ones((3,23)), available=np.ones((3,23),bool))
    s['values'][:,0] = [0.,2.,1000.]
    s['values'][0,1] = 999.; s['available'][0,1] = False
    out = summarize_prefix(s)
    assert out['observed_points'] == 2
    assert out['channels']['Bt']['median'] == 1.
    assert out['channels']['CIII']['median'] == 1.
    assert out['channels']['CIII']['available_fraction'] == .5
    s['available'][:2,2] = False
    assert summarize_prefix(s)['channels']['MNM']['median'] is None
    s['times'] = s['times'][::-1]
    with pytest.raises(ValueError,match='increasing'):
        summarize_prefix(s)
