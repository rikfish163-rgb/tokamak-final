import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_radiation_plane_20261114 as plane


def test_direct_radiation_and_mask_contributions_cannot_drive_the_edit():
    w = torch.randn(64, 8*46, generator=torch.Generator().manual_seed(17))
    proj = plane.Projection(w)
    x = torch.randn(2,16,8,46, generator=torch.Generator().manual_seed(19))
    x[...,23:] = 1.
    a = proj.coords(x.flatten(2))
    x[..., list(plane.RAD)] *= 5.
    torch.testing.assert_close(proj.coords(x.flatten(2)), a, atol=0, rtol=0)
    x[...,23:] = 0.
    for lane in plane.RAD:
        x[...,23+lane] = 1.
    assert not proj.coords(x.flatten(2)).any()
    x[...,23:] = 1.
    x[:,0,0,23+plane.RAD[2]] = 0.
    assert not proj.coords(x.flatten(2))[:,0,2].any()


def test_match_dose_never_amplifies_a_possible_write_or_uses_disjoint_support():
    writes = torch.tensor([[[[4.0]], [[0.0]]], [[[.1]], [[2.0]]], [[[2.0]], [[3.0]]]])
    result = plane.matched(writes, torch.tensor([.2, 1.0]), origin=torch.zeros_like(writes))
    size = result.flatten(2).norm(dim=-1)
    torch.testing.assert_close(size,torch.tensor([[.1,0.],[.1,0.],[.1,0.]]))


def test_causal_handoff_does_not_read_the_unreleased_or_disabled_update():
    q = np.array([127,159,191,223])
    s, previous, portion = plane.slots(q,np.array([157,158,190,191,195,222,254,255]),
                                      np.array([False,True,False,True]), patch=8)
    np.testing.assert_array_equal(s,[0,0,2,2,2,0,4,4])
    np.testing.assert_array_equal(previous,[0,0,0,0,0,0,0,0])
    np.testing.assert_allclose(portion[3:5],[.125,.625])
    _, old, _ = plane.slots(q,np.array([190]),np.array([True,True,False,False]),patch=8)
    np.testing.assert_array_equal(old,[1])
