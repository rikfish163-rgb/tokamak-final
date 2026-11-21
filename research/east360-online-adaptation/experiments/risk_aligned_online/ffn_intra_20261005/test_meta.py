import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.ffn_intra_20261005.loss import SignalRule
from experiments.risk_aligned_online.ffn_intra_20261005.source import episode_loss
from experiments.risk_aligned_online.ffn_intra_20261005.stream import predict_shot


def case():
    torch.set_num_threads(2);torch.manual_seed(6)
    model=FfnModel(Simple23(Config(fast_site='tokens'),torch.zeros(23),torch.ones(23)))
    rng=np.random.default_rng(6)
    shot=dict(device='J-TEXT',shot=1,values=rng.normal(size=(160,23)).astype(np.float32),
        available=np.ones((160,23),bool),times=np.arange(160)/1000.,is_disrupt=True,down=.16,
        valid=np.ones(160,bool))
    e=dict(shot=1,label=1,burn_in_origins=[],origins=[0,26,52,78],queries=list(range(104,130)))
    return model,shot,e


@pytest.mark.parametrize('radius',[.015,.5])
def test_four_event_meta_derivative_matches_finite_difference(radius):
    model,shot,e=case();model=model.double()
    class Rule(torch.nn.Module):
        def __init__(self):
            super().__init__();self.coefficient=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
        def forward(self,context):
            pattern=torch.linspace(-1,1,46,dtype=context.dtype).reshape(1,2,23)
            return 1+self.coefficient*pattern
    rule=Rule()
    kwargs=dict(step=.01,radius=radius,rule=rule)
    loss,_=episode_loss(model,shot,e,create_graph=True,**kwargs)
    derivative,=torch.autograd.grad(loss,rule.coefficient)
    def value(coefficient):
        with torch.no_grad():rule.coefficient.fill_(coefficient)
        return float(episode_loss(model,shot,e,create_graph=False,**kwargs)[0].detach())
    numerical=(value(.2001)-value(.1999))/.0002
    assert derivative.abs()>1e-9
    assert float(derivative)==pytest.approx(numerical,rel=.003,abs=2e-7)


def test_initial_all_one_rule_matches_fixed_whole_trace():
    model,shot,_=case();shot['device']='EAST'
    fixed,a=predict_shot(model,shot,intra=True,step=.01,radius=.1)
    learned,b=predict_shot(model,shot,intra=True,step=.01,radius=.1,rule=SignalRule())
    np.testing.assert_array_equal(fixed,learned)
    assert [r['delta_norm'] for r in a['events']]==[r['delta_norm'] for r in b['events']]
    assert a['accepted_updates']==b['accepted_updates']
