import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online import commissioning_write_delay_20261112 as delay
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_delay_changes_the_first_possible_write_and_leaves_reference_until_then(monkeypatch):
    torch.manual_seed(74)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    raw = np.random.default_rng(74).normal(size=(119, 23)).astype(np.float32)
    signal = dict(values=raw,available=np.ones_like(raw,bool))
    q = np.array([15,47,79])
    phase = correction = torch.zeros(len(q)+1)
    monkeypatch.setattr(group.layer,'patcher',lambda *_:lambda x: .75*x)
    x,m = group.lag._windows(base,signal,np.arange(len(raw)))
    with torch.no_grad():
        h = group.forward_hidden(base,x,m,group.layer.patcher(None,None),state=torch.zeros(len(x),3,4))
    z, beta = torch.zeros(len(x)), torch.randn(8)
    states = torch.randn(5,len(q)+1,3,4)
    states[:,:2] = 0.
    parent = np.full(len(raw),.5,dtype=np.float32)
    a, b, dose, *_ = delay.render(base,signal,q,h,z,beta,phase,correction,states,
                                 np.array([False,True,True]),parent)
    first = q[1]+group.lag.H+1
    assert np.flatnonzero(dose[0])[0] == first+1
    for j, wait in enumerate(delay.WAITS):
        assert np.flatnonzero(dose[j])[0] == first+wait+1
        np.testing.assert_array_equal(a[j,:first+wait+1],parent[:first+wait+1])
        np.testing.assert_array_equal(b[j,:first+wait+1],parent[:first+wait+1])
    np.testing.assert_allclose(a[0,first+9:first+31],a[2,first+9:first+31],rtol=0.,atol=1e-6)

    states[:,3] = 0.
    a, _, dose, *_ = delay.render(base,signal,q,h,z,beta,phase,correction,states,
                                  np.array([False,True,False]),parent)
    for j, wait in enumerate(delay.WAITS):
        latest = q[2]+group.lag.H+1+wait
        assert dose[j,latest-1]>0.
        np.testing.assert_array_equal(dose[j,latest:],0.)
        np.testing.assert_array_equal(a[j,latest:],parent[latest:])
