import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_frequency_feedback_20261112 as freq


def test_guard_checks_every_sample_in_the_difference_interval():
    x = np.full((125, len(freq.io23.RAW23)), 2., dtype=np.float32)
    m = np.ones_like(x,dtype=bool)
    q = np.array([10, 65])
    signal = dict(values=x,available=m)
    x[14,freq.RAW_F] = 0.
    m[78,freq.RAW_F] = False
    true, placebo = freq.masks(signal,q,rank=21)
    assert not true[0,:4].any() and true[0,4:].all()
    np.testing.assert_array_equal(np.flatnonzero(~true[1]), [8,9,10,11,12])
    np.testing.assert_array_equal(true.sum(1),placebo.sum(1))
    assert not np.array_equal(true,placebo)


def test_full_support_reproduces_legacy_loss_coordinate():
    y = torch.randn((3,freq.lag.H,len(freq.lag.CHANNELS)), generator=torch.Generator().manual_seed(3))
    full = torch.ones((len(y),freq.lag.H-freq.lag.LAG),dtype=torch.bool)
    torch.testing.assert_close(freq.motion(y,full),freq.lag.motion(y),atol=1e-7,rtol=1e-6)


def test_blocked_frequency_is_not_a_gradient_target_but_other_channels_still_are():
    y = torch.randn((2,freq.lag.H,len(freq.lag.CHANNELS)),generator=torch.Generator().manual_seed(11))
    y.requires_grad_()
    blocked = torch.zeros((len(y),freq.lag.H-freq.lag.LAG),dtype=torch.bool)
    d = freq.motion(y,blocked)
    torch.testing.assert_close(d[...,freq.TARGET_F],torch.zeros_like(d[...,freq.TARGET_F]))
    d.square().sum().backward()
    assert not y.grad[...,freq.TARGET_F].any()
    assert y.grad[...,0].abs().sum()>0


def test_joint_treatment_blocks_both_coupled_fields_but_not_n1():
    y = torch.randn((2,freq.lag.H,len(freq.lag.CHANNELS)),generator=torch.Generator().manual_seed(13))
    blocked = torch.zeros((len(y),freq.lag.H-freq.lag.LAG),dtype=torch.bool)
    cols = (freq.TARGET_F,freq.TARGET_A)
    kept = freq.motion(y,blocked,cols)
    torch.testing.assert_close(kept[...,list(cols)],torch.zeros_like(kept[...,list(cols)]))
    n1 = freq.lag.CHANNELS.index(freq.io23.RAW23.index('n=1 amplitude'))
    torch.testing.assert_close(kept[...,n1],freq.lag.motion(y)[...,n1])


def test_equalizes_realized_phase_change_without_amplifying_small_arm():
    delta = torch.zeros((3,2,4,2))
    delta[0,:,0,0] = 2.
    delta[1,:,1,0] = 1.
    delta[2,0,2,0] = 4.
    changed = freq.equalize(delta)
    norm = changed.flatten(2).norm(dim=-1)
    torch.testing.assert_close(norm,torch.tensor([[1.,0.]]).expand(3,-1))
    torch.testing.assert_close(changed[1,0],delta[1,0])
