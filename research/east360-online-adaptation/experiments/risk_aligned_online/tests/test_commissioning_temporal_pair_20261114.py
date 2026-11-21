from types import SimpleNamespace

import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_temporal_pair_20261114 as pair


def stream():
    gen = np.random.default_rng(21)
    x = gen.normal(size=(384, 23)).astype(np.float32)
    s = dict(values=x, available=np.ones_like(x, dtype=bool))
    base = SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    return base, s


def test_only_observed_packets_mature_and_earlier_has_same_clock():
    base,s = stream()
    q = np.array([127, 159, 191, 223])
    x,observed = pair.observed_packets(base,s,q)
    assert x.shape == (4, 4, 8, 8)
    np.testing.assert_array_equal(observed,True)
    prev,ready = pair.embed.feedback_support(q,observed)
    np.testing.assert_array_equal(prev,[-1,0,1,2])
    np.testing.assert_array_equal(ready,[0,1,1,1])
    n = q[1]+pair.lag.H+1
    earlier = {k:v[:n] for k,v in s.items()}
    z,good = pair.observed_packets(base,earlier,q[:2])
    torch.testing.assert_close(z,x[:2])
    np.testing.assert_array_equal(good,True)
    s['available'][q[1]+2,pair.VALUES[3]] = False
    _,observed = pair.observed_packets(base,s,q)
    np.testing.assert_array_equal(observed,[1,0,1,1])


def test_relative_shift_scrambles_only_matching_and_not_channel_data():
    torch.manual_seed(1)
    z = torch.randn(5, 4, 8, 8)
    d = torch.zeros(5, 12, 64)
    locked = torch.randn(12,64)*.2
    heads = pair.Pairing(12)
    sim = pair.similarity(heads,z,locked,d)
    wrong = pair.similarity(heads,z,locked,d,view='relative')
    torch.testing.assert_close(wrong,sim.roll(1,dims=2))
    joint = pair.similarity(heads,z,locked,d,view='joint_reverse')
    assert not torch.allclose(sim,joint)


def test_inner_step_lowers_own_loss_and_meta_gradient_reaches_pairing_head():
    torch.manual_seed(4)
    z = torch.randn(4, 4, 8, 8)
    w = torch.randn(12, 64)*.2
    heads = pair.Pairing(12)
    ordinary = z.new_zeros((len(z),*w.shape))
    shift, before = pair.negative_gradient(heads,z,w,create_graph=True)
    after = pair.paired_loss(pair.similarity(heads,z,w,ordinary+.01*shift))[0]
    assert float(after.detach().sum()) < float(before[:,0].sum())
    proxy = (shift*torch.randn_like(shift)).mean()
    proxy.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in heads.parameters())
