import numpy as np

from experiments.risk_aligned_online import commissioning_signed_diagnostics_20261112 as paths


def test_coincidence_is_not_a_sum_of_marginal_changes():
    amplitude = np.array([[[.8, .4], [0., 0.], [0., 0.]]])
    falling_frequency = np.array([[.7, 0., 0.]])
    actual, shifted = paths.magnetic_pair(amplitude, falling_frequency, np.array([1]))
    np.testing.assert_allclose(actual, [[.7/3., .4/3.]])
    np.testing.assert_array_equal(shifted, 0.)


def test_radiation_voltage_control_reverses_the_relation_only():
    radiation = np.array([[1., 0., 0.], [0., .7, 0.]])
    voltage = np.array([[0., .5, 0.], [1., 0., 0.]])
    actual, reversed_ = paths.radiative_pair(radiation, voltage)
    np.testing.assert_array_equal(actual, [.25, 0.])
    np.testing.assert_array_equal(reversed_, [0., .35])


def fake():
    n = 320
    values = np.zeros((n,len(paths.NAMES)), dtype=float)
    r = np.arange(paths.FIRST, n, paths.STRIDE)
    q = r[1]-paths.H-1
    values[:,paths.FREQ] = 2000.
    values[:,paths.CURRENT] = 100000.
    values[q+1:q+9,paths.AMP] = .3
    values[q+1:q+9,paths.N1] = .6
    values[q+1:q+9,paths.FREQ] = 1800.
    return values,r,q


def test_zero_frequency_has_its_own_flag_and_cannot_drive_signed_evidence():
    x,r,q = fake()
    ok = np.ones_like(x,dtype=bool)
    pos = paths.observe_block(x,ok,x,r,rank=22)
    assert pos['mag_usable'][1] and not pos['zero_transition'][1]
    assert (pos['magnetic'][1]>0.).all()
    x[q+1:q+9,paths.FREQ] = 0.
    zero = paths.observe_block(x,ok,x,r,rank=22)
    assert not zero['mag_usable'][1] and zero['zero_transition'][1]
    np.testing.assert_array_equal(zero['magnetic'][1], 0.)
    assert (zero['naive'][1]>0.).all()
    assert zero['rad_usable'][1]  # no unnecessary censoring of other channels


def test_invalid_radiation_does_not_censor_valid_frequency():
    x,r,q = fake()
    mask = np.ones_like(x,dtype=bool)
    mask[q+25,paths.XSKEW] = False
    got = paths.observe_block(x,mask,x,r,rank=22)
    assert got['mag_usable'][1] and not got['rad_usable'][1]
    assert got['radiation'][1]==0.


def test_truncation_does_not_change_any_available_proxy():
    x,r,_ = fake()
    mask = np.ones_like(x,dtype=bool)
    stop = r[3]
    full = paths.observe_block(x,mask,x,r,rank=22)
    prefix = paths.observe_block(x[:stop],mask[:stop],x[:stop],r[r<stop],rank=22)
    for k,v in prefix.items():
        np.testing.assert_array_equal(full[k][r<stop],v)
