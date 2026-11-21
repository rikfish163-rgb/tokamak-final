"""Counterexample prevents confusing historical safety with new-shot safety."""
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.label_cone_20261023.learning import Cone
from experiments.risk_aligned_online.live_history_20261023.learning import backtrack


def test_unchanged_history_can_coexist_with_worse_unseen_safe_risk():
    # The known safe example depends on parameter x; an unseen safe example
    # depends on parameter y. Diagnostic fitting pushes y up, leaving x alone.
    fast={'theta':torch.zeros(2,requires_grad=True)}
    auxiliary=(fast['theta'][1]-1).square()/2
    gradient,=torch.autograd.grad(auxiliary,fast['theta'])
    cone=Cone(torch.tensor([[1.,0.]]),torch.tensor([0]))
    direction,_=cone.project(gradient)
    def history(state):
        return F.softplus(state['theta'][:1].detach().double())
    initial=history(fast)
    updated,r=backtrack(fast,.001*direction,history,initial,initial)
    assert r['fraction']==1.
    torch.testing.assert_close(history(updated),initial,rtol=0,atol=0)
    assert F.softplus(updated['theta'][1]).item()>F.softplus(fast['theta'][1]).item()
