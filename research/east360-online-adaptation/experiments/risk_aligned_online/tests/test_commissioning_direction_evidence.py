import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_direction_evidence_20261111 import (
    decision_evidence, mix_direction, shift_evidence,
)


def test_evidence_uses_only_completed_targets_and_tracks_direction():
    q = np.arange(3)*32
    old = np.zeros((3,25,10),np.float32)
    native = np.ones_like(old)
    target = native.copy()
    target[1] = old[1]
    valid = np.ones_like(target,bool)
    all_e, stats = decision_evidence(q,old,native,target,valid,92)
    assert not all_e[:26].any()
    assert all_e[26,0] > 0 and all_e[58,0] < 0 and all_e[90,0] > 0
    assert stats['comparisons']==3 and stats['native_wins']==2
    target[-1] = -99.
    prefix, _ = decision_evidence(q,old,native,target,valid,76)
    np.testing.assert_array_equal(prefix,all_e[:76])
    lag = shift_evidence(all_e)
    assert not lag[:32].any()
    np.testing.assert_array_equal(lag[32:],all_e[:-32])


def test_unobserved_channels_never_supply_evidence():
    q = np.asarray([127])
    old = np.zeros((1,25,10),np.float32)
    new = np.ones_like(old)
    y = old.copy()
    m = np.ones_like(y,bool)
    m[:,:,4:] = False
    all_e,_ = decision_evidence(q,old,new,y,m,160)
    y[:,:,4:] = 999.
    b,_ = decision_evidence(q,old,new,y,m,160)
    np.testing.assert_array_equal(all_e,b)
    assert b[153,0]<0. and np.any(b[153,1:]==0.)


def test_direction_mixture_preserves_local_update_budget_and_zero_updates():
    old = torch.tensor([[2.,0.],[0.,2.],[0.,0.]])
    new = torch.tensor([[0.,2.],[2.,0.],[0.,0.]])
    m = mix_direction(old,new,torch.tensor([0.,1.,.5]))
    torch.testing.assert_close(m,torch.tensor([[2.,0.],[2.,0.],[0.,0.]]))
    equal = mix_direction(old,new,torch.full((3,),.5))
    torch.testing.assert_close(equal.norm(dim=1),old.norm(dim=1))
    assert (equal[:2,:]>0).all()
