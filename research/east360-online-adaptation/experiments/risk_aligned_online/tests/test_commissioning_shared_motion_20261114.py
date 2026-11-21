import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_shared_motion_20261114 as motion
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_fixed_25_point_basis_is_anchored_and_does_not_roll_predictions():
    knots = torch.arange(1.,6.).view(1,5,1).repeat(1,1,8)
    knots.requires_grad_()
    wave = motion.waveform(knots)
    assert wave.shape == (1,25,8)
    torch.testing.assert_close(wave[0,:,0], torch.arange(1.,26.)/5.)
    grad, = torch.autograd.grad(wave[0,5,0],knots)
    torch.testing.assert_close(grad[0,:,0],torch.tensor([.8,.2,0.,0.,0.]))


def test_missing_last_measurement_does_not_become_the_observed_anchor():
    x = torch.zeros(1,128,8)
    a = torch.ones_like(x,dtype=torch.bool)
    x[0,-2,0] = .25
    x[0,-1,0] = 90.
    a[0,-1,0] = False
    x[0,-1,1] = 3.
    x[0,-2,1] = 90.
    a[0,-2,1] = False
    d = motion.motion_view(x,a,torch.ones(8))
    torch.testing.assert_close(d.anchor[0,:2],torch.tensor([.25,3.]))
    assert d.known.all()
    e = motion.motion_view(torch.where(a,x,0.),a,torch.ones(8))
    torch.testing.assert_close(d.features,e.features)


def test_the_same_fast_latent_enters_both_forecast_and_frozen_attention():
    torch.manual_seed(131)
    base = Simple23(Config(),torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    c = motion.Carrier(64)
    torch.nn.init.normal_(c.linear.weight, std=.02)
    tau = torch.ones(8)
    x = torch.linspace(-1,1,128)[None,:,None].expand(2,128,23).contiguous()
    a = torch.ones_like(x, dtype=torch.bool)
    history = motion.input_view(base,x,a,tau)
    k0 = torch.zeros(2,5,8)
    state = torch.zeros(2,2,8,requires_grad=True)
    k1 = motion.shared_state(k0,state)
    score = motion.frozen_hidden(base,x,a,c.static(k1,history.known),None) @ base.risk_head.weight[0]
    loss = motion.sample_loss(k1,torch.ones(2,25,8),torch.ones(2,25,8,dtype=torch.bool)).sum()
    g, = torch.autograd.grad(loss,state,retain_graph=True)
    r, = torch.autograd.grad(score.sum(),state)
    assert g.abs().sum() > 0 and r.abs().sum() > 0
    assert all(v.grad is None for v in base.parameters())


def test_matched_actions_do_not_amplify_and_invalid_slots_do_not_borrow_future():
    raw = torch.tensor([[[[1.]],[[2.]]],[[[.25]],[[.5]]],[[[.5]],[[0.]]]])
    matched,used = motion.match_actions(raw,torch.tensor([.4,.4]))
    torch.testing.assert_close(matched.flatten(2).norm(dim=-1),torch.tensor([[.25,0.]]).expand(3,2))
    torch.testing.assert_close(used,torch.tensor([.25,0.]))
    q = np.array([127,159,191,223])
    state,prev,fraction = motion.available_states(q,np.array([190,191,197,222,254,255]),
                                                  np.array([False,True,True,False]))
    np.testing.assert_array_equal(state,[2,2,2,3,0,0])
    np.testing.assert_array_equal(prev,[0,0,0,2,0,0])
    np.testing.assert_allclose(fraction[:3],[0.,.125,.875])
