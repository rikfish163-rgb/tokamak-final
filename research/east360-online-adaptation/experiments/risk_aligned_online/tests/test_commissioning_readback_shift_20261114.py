import numpy as np

from experiments.risk_aligned_online import commissioning_readback_shift_20261114 as study


def test_readback_calibration_never_uses_overlapping_or_unseen_packets():
    # Every previous fitting window ends at least 128 samples before evaluation.
    ends = np.arange(127, 385, 4)
    packet = np.array([127, 255, 383])
    at = study.calibration_slots(ends, packet)
    assert not at[ends<255].any()
    np.testing.assert_array_equal(at[ends>=255], np.where(ends[ends>=255]<383,1,2))
    np.testing.assert_array_equal(ends-packet[np.maximum(at-1,0)]>=128,at>0)


def test_aligned_residual_changes_readback_only_after_fitting_clock():
    ends = np.arange(127, 385, 4)
    z = np.broadcast_to(np.r_[np.ones(8),-np.ones(8)],(len(ends),16))[...,None].copy()
    residual = 2.*z
    out, n = study.local_corrections(z,residual,ends,np.ones(len(ends),bool))
    assert (out[:,n==0]==0).all()
    # Rolling eight target patches reverses the sign without changing mean/magnitude.
    np.testing.assert_allclose(out[0],0.)
    np.testing.assert_allclose(out[1],-out[2],atol=1e-12)
    assert (out[1,n>0]*residual[n>0]).sum()>0.


def test_causal_calibration_is_invariant_to_extension_and_keeps_missing_packets_out():
    rng = np.random.default_rng(10)
    ends = np.arange(127, 701, 4)
    z,t = (rng.normal(size=(len(ends),16,2)) for _ in range(2))
    eligible = np.ones(len(ends),bool)
    eligible[0] = False
    d,n = study.local_corrections(z,t,ends,eligible)
    assert not n[ends<383].any()
    a,b=study.local_corrections(z[:90],t[:90],ends[:90],eligible[:90])
    np.testing.assert_array_equal(b,n[:90])
    np.testing.assert_array_equal(a,d[:,:90])
