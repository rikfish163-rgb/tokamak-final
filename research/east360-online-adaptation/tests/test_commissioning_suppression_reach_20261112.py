import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online import commissioning_suppression_reach_20261112 as reach
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_shared_coefficient_cannot_pick_opposite_directions_for_two_endpoints():
    # Both points could be suppressed independently, but the same one-dimensional
    # direction cannot lower both. Give them the same release and full handoff.
    gram = torch.ones((2,1,1))
    rhs = torch.tensor([[1.],[-1.]])
    dose = torch.ones(2)
    slot = torch.ones(2,dtype=torch.long)
    frac = torch.ones(2)
    for direction in (-1.,1.):
        state = torch.tensor([[0.],[direction]])
        change = reach.linear_change(state,slot,frac,gram,rhs,dose)
        torch.testing.assert_close(change,torch.tensor([direction,-direction]))
        assert not (change<0.).all()


def test_full_frozen_model_matches_dose_and_original_parent(monkeypatch):
    torch.manual_seed(51)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    rng = np.random.default_rng(51)
    values = rng.normal(size=(56,23)).astype(np.float32)
    sig = dict(values=values,available=np.ones_like(values,bool))
    monkeypatch.setattr(group.layer,'patcher',lambda *_:lambda x:x*.7)
    ends = np.array([19,30,40])
    x,m = group.lag._windows(base,sig,ends)
    hidden = group.forward_hidden(base,x,m,lambda x:x*.7,state=torch.zeros((3,3,4)))
    z = torch.tensor([.6,.8,.5])
    beta = torch.randn(8)
    phase = correction = torch.zeros(len(ends))
    dose = torch.tensor([.002,.003,.005])
    pred,gram,rhs,error = reach.pointwise(base,x,m,phase,correction,hidden,z,beta,dose)
    torch.testing.assert_close(pred[0],z,atol=1e-6,rtol=0.)
    assert torch.all(pred[1:]<pred[0])
    assert gram.shape==(3,12,12) and rhs.shape==(3,12)
    assert error<1e-6 and all(w.grad is None for w in base.parameters())
