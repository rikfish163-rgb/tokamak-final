import numpy as np

from experiments.risk_aligned_online import commissioning_causal_reference_20261114 as trial


def make_signal():
    x = np.random.default_rng(131).normal(size=(260,23)).astype(np.float32)
    return dict(values=x, available=np.ones_like(x,dtype=bool))


def test_positive_affine_changes_preserve_rank_but_not_absolute_measurements():
    x = make_signal()
    v,m = x['values'],x['available']
    end = np.array([127,223])
    scales = np.ones(23)
    a,_ = trial.describe(x,end,np.zeros(23),scales)
    b,_ = trial.describe(dict(values=np.float32(7)*v+np.float32(3),available=m),
                         end,np.zeros(23),scales)
    for k in ('recent_rank','opening_rank'):
        np.testing.assert_array_equal(a[k],b[k])
    assert np.max(abs(a['absolute']-b['absolute']))>1.


def test_reference_is_nonoverlapping_independent_of_suffix_and_reports_missingness():
    x = make_signal()
    v,m = x['values'],x['available']
    a,good = trial.describe(x,np.array([223]),np.zeros(23),np.ones(23))
    b,keep = trial.describe(dict(values=v[:224],available=m[:224]),
                            np.array([223]),np.zeros(23),np.ones(23))
    np.testing.assert_array_equal(good,keep)
    for k in a:
        np.testing.assert_array_equal(a[k],b[k])
    m[140,trial.CHANNELS[0]] = False  # preceding, not current, window
    _, changed = trial.describe(x,np.array([223]),np.zeros(23),np.ones(23))
    assert good.item() and not changed.item()


def test_midrank_does_not_report_constant_as_extreme():
    x = np.zeros((1,8,2))
    y = x.copy()
    y[:,:,1] = 1.
    np.testing.assert_array_equal(trial.rank_position(x,y)[0,0],[0.,-1.])
