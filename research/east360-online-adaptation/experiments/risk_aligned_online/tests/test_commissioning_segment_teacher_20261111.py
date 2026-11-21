import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_segment_teacher_20261111 as seg
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_segment_respects_release_and_strict_warning_cutoff():
    t = np.arange(23)*.001
    release = np.array([4, 11, 17])
    slots = np.searchsorted(release, np.arange(len(t)), side='right')
    z = np.arange(len(t), dtype=float)
    b = seg.select_blocks(t, z, slots, True, .047)
    assert b.tolist() == [[12, 13, 14, 15, 16], [6, 7, 8, 9, 10]]
    assert np.all(slots[b] == slots[b[:,:1]])
    assert np.all(np.round((.047-t[b])*1000, 9) > 30.)
    assert seg.select_blocks(t[:4], z[:4], slots[:4], False, None).shape == (0,5)


def test_robust_score_distinguishes_spike_from_sustained_evidence():
    a = torch.tensor([[[.6, .6, 1.7, .6, .6], [.4, .4, .4, .4, .4]],
                      [[.1, .2, .3, .8, .5], [.1, .9, .1, .1, .1]]])
    pos = torch.tensor([True, False])
    torch.testing.assert_close(seg.summarize_blocks(a,pos,False), torch.tensor([1.7,.9]))
    torch.testing.assert_close(seg.summarize_blocks(a,pos,True), torch.tensor([.6,.9]))
    assert seg.segment_loss(a,pos,.7,True)>seg.segment_loss(a,pos,.7,False)


def test_adapter_does_not_write_cls_or_missing_or_first_patch():
    h = torch.tensor([[[0.,0.],[1.,2.],[3.,5.],[4.,9.],[7.,0.]]])  # CLS is last.
    valid = torch.tensor([[True,True,False,True]])
    delta = seg.token_change(h,valid,torch.tensor([[1.,2.]]))
    assert delta.shape == h.shape
    torch.testing.assert_close(delta[0,1],torch.tensor([1.,2.]).tanh()*torch.tensor([1.,2.]))
    assert torch.count_nonzero(delta[0,[0,2,3,4]])==0


def test_all_arms_have_same_actual_token_rms_and_zero_if_unavailable():
    d = torch.zeros(3,2,4,2)
    d[0,:,1,0] = 1.
    d[1,:,1,1] = 2.
    d[2,0,2,0] = 3.
    u, rms = seg.equal_dose(d,.10)
    torch.testing.assert_close(rms,torch.tensor([.10,0.]))
    torch.testing.assert_close(u.square().mean(dim=(2,3)).sqrt(),rms.expand(3,2))
    assert u[1,0,1,1]>0 and u[2,0,2,0]>0


def test_teacher_gradient_moves_through_second_encoder_block_only():
    torch.manual_seed(1)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,horizon=25),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3,16,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    direction = torch.randn(1,3,8).requires_grad_(True)
    h, dose = seg.adapted_hidden(base,x,mask,direction,.05)
    assert h.shape == (1,3,8)
    torch.testing.assert_close(dose,torch.full((3,),.05))
    (h[0,:,0].square().mean()).backward()
    assert direction.grad.abs().sum()>0
    assert not any(p.grad is not None for p in base.parameters())
    assert not base.encoder.layers[0]._forward_hooks
    assert not base.risk_head._forward_pre_hooks
