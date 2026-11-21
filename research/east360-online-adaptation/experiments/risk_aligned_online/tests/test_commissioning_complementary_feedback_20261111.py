import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_complementary_feedback_20261111 as joint


def test_two_rotations_have_identical_realized_norm_and_angle():
    old = torch.tensor([[[2., 0., 0., 0.]], [[0., 4., 0., 0.]]])
    real = torch.tensor([[[7., 4., 0., 0.]], [[6., 2., 0., 0.]]])
    wrong = torch.tensor([[[4., 0., -3., 0.]], [[0., -3., 7., 0.]]])
    a, b, used = joint.equal_angle(old, real, wrong)
    assert used.tolist() == [True, True]
    n = old.flatten(1).norm(dim=1)
    for new in (a,b):
        torch.testing.assert_close(new.flatten(1).norm(dim=1),n)
        torch.testing.assert_close((new*old).flatten(1).sum(1)/n.square(),
                                   torch.full_like(n,1./np.sqrt(1.+joint.RATIO**2)))
        assert (new[...,3]==0.).all()
    assert (a[0,0,1]>0) and (b[0,0,2]<0)


def test_no_unique_direction_is_common_no_rotation():
    a = torch.tensor([[2.,0.,0.],[0.,0.,0.],[2.,0.,0.]])
    real = torch.tensor([[0.,5.,0.],[0.,1.,0.],[-1.,0.,0.]])
    wrong = torch.tensor([[8.,0.,0.],[0.,0.,4.],[0.,3.,0.]])
    new,control,used = joint.equal_angle(a,real,wrong)
    assert not used.any()
    torch.testing.assert_close(new,a,atol=0.,rtol=0.)
    torch.testing.assert_close(control,a,atol=0.,rtol=0.)


def test_amplitude_from_old_origin_cannot_bleed_into_new_phase_interval():
    old, new = np.array([127,159,191,223]), np.array([159,223])
    t = np.array([152,153,184,185,216,217,248,249])
    phase, amp = joint.common_slots(old,new,t)
    assert phase.tolist()==[0,1,1,2,2,3,3,4]
    assert amp.tolist()==[0,0,0,1,1,0,0,2]
    before,extra = joint.common_slots(old[:2],new[:1],t[:5])
    np.testing.assert_array_equal(before,phase[:5])
    np.testing.assert_array_equal(extra,amp[:5])


def test_no_common_mature_origin_cannot_create_a_rotation():
    phase, amp = joint.common_slots(np.array([127]),np.empty(0,dtype=int),np.array([152,153]))
    assert phase.tolist()==[0,1]
    assert not amp.any()
