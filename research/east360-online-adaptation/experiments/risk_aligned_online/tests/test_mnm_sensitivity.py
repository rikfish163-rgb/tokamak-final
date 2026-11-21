import numpy as np
import pytest

from experiments.risk_aligned_online.mnm_sensitivity_20261022.run import restore_mnm
from experiments.risk_aligned_online.io23 import RAW23


def test_restore_changes_only_observed_mnm_and_its_mask():
    x = np.ones((8,23), np.float32); m = np.ones_like(x, bool)
    c=RAW23.index('MNM'); x[:,c]=0.; m[:,c]=False
    raw=np.full_like(x, 2.5); obs=np.ones_like(m)
    raw[2,c]=np.nan; obs[2,c]=False
    s=dict(values=x,available=m)
    restored=restore_mnm(s,raw,obs)
    assert restored['values'][0,c]==2.5 and restored['available'][0,c]
    assert restored['values'][2,c]==0 and not restored['available'][2,c]
    cols=[i for i in range(23) if i!=c]
    np.testing.assert_array_equal(restored['values'][:,cols],x[:,cols])
    np.testing.assert_array_equal(restored['available'][:,cols],m[:,cols])
    assert (s['values'][:,c]==0).all() and not s['available'][:,c].any()
    raw[4:,c]=100
    other=restore_mnm(s,raw,obs)
    np.testing.assert_array_equal(other['values'][:4],restored['values'][:4])


def test_no_outcome_fields_or_mismatched_arrays_allowed():
    s=dict(values=np.zeros((2,23),np.float32),available=np.zeros((2,23),bool))
    with pytest.raises(ValueError): restore_mnm(dict(s,y=None),s['values'],s['available'])
    with pytest.raises(ValueError): restore_mnm(s,s['values'][:1],s['available'])
    x=s['values'].copy(); x[0,2]=np.inf
    with pytest.raises(ValueError): restore_mnm(s,x,np.ones_like(s['available']))
