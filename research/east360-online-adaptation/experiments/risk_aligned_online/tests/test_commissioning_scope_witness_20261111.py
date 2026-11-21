import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_scope_witness_20261111 as scope


def test_positive_scope_is_strictly_before_deadline_and_on_common_support():
    t = np.array([.850,.875,.900,.940,.960,.970,.980])
    p = np.arange(len(t))
    good = np.array([1,1,1,0,1,1,1],dtype=bool)
    assert scope.witness_points(t,p,good,True,1.).tolist()==[1,2,4]


def test_safe_scope_keeps_separated_peaks_on_existing_support():
    t = np.arange(9)
    p = np.array([9,8,7,1,2,4,5,6,10])
    good = np.array([0,1,1,1,1,1,1,1,1],dtype=bool)
    assert scope.witness_points(t,p,good,False,None).tolist()==[1,4,6,8]


def test_same_family_tangent_does_not_modify_latest_value_masks_or_mean():
    w = scope.lag.W
    raw = torch.zeros((1,w,46))
    t = torch.arange(w)/8.
    raw[:, :, scope.lag.CHANNELS[0]] = t.sin()
    raw[:, :, 23:] = 1.
    g = torch.zeros_like(raw)
    g[:, :, scope.lag.CHANNELS[0]] = t.cos()
    d = scope.ascent_candidate(raw,g,~scope.lag.CROSS)
    assert d.abs().sum()>0.
    assert (d[:, -1]==0.).all() and (d[:, :, 23:]==0.).all()
    torch.testing.assert_close(d.sum(dim=1),torch.zeros((1,46)),atol=1e-5,rtol=0.)
