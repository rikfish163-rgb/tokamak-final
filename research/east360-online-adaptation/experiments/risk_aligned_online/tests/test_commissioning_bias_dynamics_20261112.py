import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_bias_dynamics_20261112 as trial


def test_dynamic_target_rejects_a_constant_offset_per_channel():
    q = torch.arange(2*trial.lag.H*len(trial.lag.CHANNELS),dtype=torch.float32)
    q = q.reshape(2,trial.lag.H,-1)
    shift = torch.arange(q.shape[-1],dtype=q.dtype)[None,None]
    torch.testing.assert_close(trial.dynamic(q),trial.dynamic(q+shift),atol=0.,rtol=0.)


def test_past_bias_has_no_current_window_or_future_access_and_expires():
    r = np.arange(6)*trial.lag.STRIDE
    e = np.broadcast_to(np.linspace(0.,.45,6,dtype=np.float32)[:,None],(6,10)).copy()
    base, support = trial.past_bias(r,e)
    assert not support[:2].any() and support[2:].all()
    assert not base[:2].any()
    after = e.copy()
    after[3:] = 7.
    np.testing.assert_array_equal(trial.past_bias(r,after)[0][:4],base[:4])
    r[-1] += (trial.HISTORY+1)*trial.lag.STRIDE
    z, ok = trial.past_bias(r,e)
    assert not ok[-1] and not z[-1].any()


def test_bias_translation_preserves_masks_and_differences_and_dose():
    x = torch.zeros((2,trial.lag.W,46))
    j = trial.lag.CHANNELS[0]
    x[...,23:] = 1.
    x[:,20,23+j] = 0.
    x[...,j] = torch.linspace(-1.,1.,trial.lag.W)
    b = torch.ones((2,len(trial.lag.CHANNELS)))
    cap = torch.tensor([.02,0.])
    out = trial.bias_patches(x,b,cap)
    torch.testing.assert_close(out[...,23:],x[...,23:],atol=0.,rtol=0.)
    assert out[0,20,j]==x[0,20,j]
    torch.testing.assert_close(out[0,21:,j].diff(),x[0,21:,j].diff())
    torch.testing.assert_close(out[1],x[1],atol=0.,rtol=0.)
    torch.testing.assert_close((out-x).flatten(1).square().mean(1).sqrt(),cap)


def test_temporal_feedforward_rescale_is_not_a_token_or_alarm_offset():
    a = torch.arange(2*7*4,dtype=torch.float32).reshape(2,7,4)
    use = torch.tensor([[True,True,False,True,False,False,False]]).expand(2,-1)
    w = torch.ones((2,4))
    d = trial.ff_input_change(a,use,w)
    assert d[use].abs().sum()>0.
    assert (d[~use]==0.).all()
    torch.testing.assert_close(d.sum(1),torch.zeros_like(w))
    torch.testing.assert_close(trial.ff_input_change(a+10,use,w),d)


def test_neural_arms_have_same_realized_dose_on_all_backgrounds():
    g = torch.arange(1.,1.+4*2*3*2).reshape(4,2,3,2)
    g[1,1] = 0.
    a, dose = trial.common_dose(g,.2)
    torch.testing.assert_close(a.square().mean((2,3)).sqrt(),dose.expand(4,-1))
    assert dose.tolist()[0]>0. and dose.tolist()[1]==0.


def test_first_feedforward_backward_leaves_frozen_encoder_parameters_unchanged():
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    torch.manual_seed(131)
    cfg = Config()
    base = Simple23(cfg,torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    a = torch.randn((2,cfg.window,23))
    m = torch.ones_like(a,dtype=torch.bool)
    gain = torch.zeros((2,2*cfg.dim),requires_grad=True)
    before = base(a,m)['logit']
    hidden = trial.forward_hidden(base,a,m,gain=gain)
    out = base.risk_head(hidden).squeeze(-1)
    torch.testing.assert_close(out,before,atol=1e-6,rtol=0.)
    grad, = torch.autograd.grad(out.sum(),gain)
    assert grad.abs().max()>0.
    assert all(p.grad is None for p in base.parameters())
