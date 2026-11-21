import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_opposed_feedback_20261114 as study
from experiments.risk_aligned_online.commissioning_opposed_identification_20261114 import orientation_agreement


def test_paired_error_is_signed_and_invariant_to_additive_midpoints():
    high = torch.tensor([[[2.,.1],[.1,2.]]])
    low = -high
    truth = high*.3
    a = study.signed_fit(high,low,truth,dim=(1,2))
    torch.testing.assert_close(study.signed_fit(low,high,truth,dim=(1,2)),-a)
    torch.testing.assert_close(study.signed_fit(high+4,low+4,truth+4,dim=(1,2)),a)
    assert a.item()>0.


def test_matched_routes_share_absolute_physical_write_and_keep_cls_out():
    x = torch.randn(4,16,64)
    m = torch.ones(4,16,dtype=torch.bool)
    m[0,0] = False
    d = study.centered_unit(x,m)
    assert not d[0,0].any()
    torch.testing.assert_close(d.sum(1),torch.zeros(4,64),atol=2e-7,rtol=0.)
    torch.testing.assert_close(d.flatten(1).norm(dim=1),torch.ones(4))
    s = torch.tensor([[-1.,1.,0.,-1.],[1.,-1.,-1.,1.]])
    g = torch.tensor([.6,.8,0.,.5])
    p = study.action_vectors(s,g,d)
    torch.testing.assert_close(p.flatten(2).norm(dim=2),g.expand(2,-1))


def test_learning_collapses_to_zero_with_no_usable_past_outcome_pair():
    values = np.array([[1.,0.,1.,0.],[-1.,0.,-1.,0.]])
    targets = np.array([.7,-.7])
    bank = [(41,True,values,targets),(81,False,values,targets)]
    np.testing.assert_array_equal(study.teacher(bank,80),np.zeros(4))
    result = study.teacher(bank,81)
    assert result[0]>0. and result[2]>0.


def test_matched_wrong_packet_never_reads_next_slot():
    q = np.array([127,159,191,223,255,287])
    valid = np.array([False,True,True,False,True,True])
    older = study.earlier_qualified(q,valid)
    assert older.tolist()==[-1,-1,1,2,2,4]
    assert np.all(q[older[older>=0]]+study.OFFSET<=q[older>=0])


def test_anchor_effect_separates_earlier_and_late_samples_without_changing_labels():
    t = np.round(np.arange(220)/1000,9)
    z = np.zeros(len(t))
    z[193],z[171] = 99.,5.
    selected = study.teacher_points(t,z,np.ones(len(t),bool),True,.2)
    assert len(selected)>0 and not (selected==171).any() and not (selected==193).any()
    assert np.all((t[selected] < .17) & (t[selected] >= .075))


def test_orientation_counts_neither_cancelled_nor_equal_candidate_effects():
    plus = np.array([.25,.25,.25])
    minus = np.array([.75,.5,.25])
    z = np.array([.5,.5,.25])
    assert orientation_agreement(plus,minus,z,True)==dict(known=1,support=3,proportion=1.)
    assert orientation_agreement(plus,minus,z,False)==dict(known=1,support=3,proportion=0.)


def test_shared_budget_has_no_access_to_history_and_smoothly_switches_at_zero():
    phase = np.array([0.,.25,.5,.75,1.])
    cap = study.handoff_dose(np.ones(5),phase,np.ones(5,bool))
    np.testing.assert_array_equal(cap,[1.,.5,0.,.5,1.])
    np.testing.assert_array_equal(study.handoff_dose(np.ones(5),phase,np.zeros(5,bool)),phase)
    old,new = np.array([[1.,-1.,1.,-1.,1.]]),np.array([[-1.,1.,1.,1.,-1.]])
    route = study.handoff_sign(old,new,phase)
    np.testing.assert_array_equal(route,[[1.,-1.,1.,1.,-1.]])
    np.testing.assert_array_equal(study.handoff_sign(old*.5,new*.5,phase),route)
