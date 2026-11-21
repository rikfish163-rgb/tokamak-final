import numpy as np

from experiments.risk_aligned_online import commissioning_weak_signal_probe_20261114 as probe


def test_weak_changes_survive_without_an_event_and_do_not_read_later_samples():
    t = np.arange(240)[:,None]
    values = t*1e-4*np.ones((1,8))
    mask = np.ones_like(values,dtype=bool)
    original, valid = probe.trajectory(values,mask,np.array([127,151]))
    short, _ = probe.trajectory(values[:152],mask[:152],np.array([127,151]))
    np.testing.assert_array_equal(original,short)
    assert valid.all()
    assert np.all(original.reshape(2,8,8)[...,1:] > 0)
    # A static offset can alter the level coordinate but not the seven signed differences.
    shifted,_ = probe.trajectory(values+.25,mask,np.array([127,151]))
    np.testing.assert_allclose(shifted.reshape(2,8,8)[...,1:],
                               original.reshape(2,8,8)[...,1:],atol=1e-7,rtol=0.)


def test_missing_input_does_not_make_a_waveform_appear_stationary():
    values = np.ones((150,8))
    mask = np.ones_like(values,dtype=bool)
    mask[20,5] = False
    x,known=probe.trajectory(values,mask,np.array([127,149]))
    assert known.tolist()==[False,True]
    assert len(x)==len(known)


def test_short_record_has_no_complete_trajectory():
    values = np.ones((80,8),dtype=np.float32)
    endpoints = probe.end_grid(len(values))
    x,known = probe.trajectory(values,np.ones_like(values,dtype=bool),endpoints)
    assert x.shape == (0,64)
    assert known.shape == (0,)


def test_fitting_mass_is_balanced_by_class_and_shot():
    shot=np.array([1,1,2,3,3,3])
    y=np.array([False,False,False,True,True,True])
    w=probe.training_weights(shot,y)
    np.testing.assert_allclose([w[shot==k].sum() for k in (1,2,3)],[.25,.25,.5])


def test_plus_32_witness_is_timestamped_after_the_actual_decision():
    endpoint=np.arange(127,240,8)
    ends,use=probe.later_rows(endpoint,np.array([0,2,11,14]))
    np.testing.assert_array_equal(ends,[4,6])
    np.testing.assert_array_equal(use,[0,1])
    np.testing.assert_array_equal(endpoint[ends]-endpoint[np.array([0,2,11,14])[use]],[32,32])
