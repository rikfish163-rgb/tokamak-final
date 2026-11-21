import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_motion_clock_dose_20261114 as trial


def test_calibration_cannot_see_the_packets_due_after_issue():
    pred = torch.full((5,5,8),2.)
    truth = torch.zeros((5,5,8))
    observed = np.ones(5,dtype=bool)
    q = np.array([127,150,158,159,192])
    w,n = trial.issued_gain(q,pred,truth,observed)
    torch.testing.assert_close(w[:2],torch.ones_like(w[:2]))
    assert n.tolist() == [0,0,1,1,3]
    altered = truth.clone()
    altered[1:] = 8
    other,_ = trial.issued_gain(q,pred,altered,observed)
    torch.testing.assert_close(other[:4],w[:4])
    assert (other[4]>w[4]).all()


def test_coefficients_absorb_gain_only_not_sign():
    q = np.arange(127,300,32)
    pred = torch.ones((len(q),5,8))
    y = -pred.clone()
    y[...,0],y[...,1] = .5,2.
    got,_ = trial.issued_gain(q,pred,y,np.ones(len(q),bool))
    assert (got[1:,0]>.5).all() and (got[1:,0]<1.).all()
    assert (got[1:,1]==1.).all()
    assert not got[1:,2:].any()


def test_equal_budget_is_an_explicit_amplification_of_the_diagnostic_directions():
    offsets = torch.tensor([[[[.01,.00]],[[.01,0.]]],[[[.00,-.02]],[[.00,0.]]]])
    actual,dose = trial.fixed_radius(offsets,torch.tensor([.6,2.]))
    torch.testing.assert_close(dose,torch.tensor([.6,0.]))
    torch.testing.assert_close(actual.flatten(2).norm(dim=-1),dose.repeat(2,1))
    assert actual[1,0,0,1]<0.
