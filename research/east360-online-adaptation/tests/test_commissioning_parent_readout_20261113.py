import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_parent_readout_20261113 as audit


def test_omit_each_path_inside_nonlinearity_without_reusing_a_marginal_increment():
    pre = torch.tensor([[.2,-.4],[.9,1.4]])
    early = torch.tensor([[.9,-.2],[.1,-.5]])
    latest = torch.tensor([[-.7,.8],[.3,.2]])
    x = np.array([[2.],[1.]],dtype=np.float32)
    coef = torch.tensor([.5,.4,-.8,-.1])
    level = audit.readout_levels(x,pre,pre+early,latest,coef)
    z = torch.tensor([.9,.4])
    torch.testing.assert_close(level[0],z+pre.tanh()@coef[1:3])
    torch.testing.assert_close(level[1],z+(pre+early).tanh()@coef[1:3])
    torch.testing.assert_close(level[2],z+(pre+latest).tanh()@coef[1:3])
    torch.testing.assert_close(level[3],z+(pre+early+latest).tanh()@coef[1:3])
    assert not torch.allclose(level[1]-level[0],level[3]-level[2])
