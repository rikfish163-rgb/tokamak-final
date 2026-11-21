import numpy as np
import torch

from experiments.risk_aligned_online.matured_negative_20261023.historical_control import (
    fixed_objective, historical_step, state_index,
)
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def test_historical_control_has_identical_component_coefficients():
    negative=torch.tensor([-1.,2.],requires_grad=True)
    positive=torch.tensor([.5,-.5],requires_grad=True)
    loss=fixed_objective(negative,positive)
    expected=.5*(torch.nn.functional.softplus(negative).mean()+torch.nn.functional.softplus(-positive).mean())
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    gn,gp=torch.autograd.grad(loss,(negative,positive))
    assert (gn>0).all() and (gp<0).all()


def test_cached_state_timing_never_updates_previous_predictions():
    assert [state_index(i) for i in [0,127,128,255,256,383,384,511,512]]==[0,0,0,0,1,1,2,2,3]


def test_historical_retraining_is_deterministic_and_parent_unchanged():
    torch.manual_seed(131)
    _,net=model('increment');net.eval()
    rng=np.random.default_rng(131)
    x=torch.as_tensor(rng.normal(size=(8,128,23)).astype('float32'))
    m=torch.ones_like(x,dtype=torch.bool)
    negative=(x,m);positive=(x+.1,m)
    parent={k:v.detach().clone() for k,v in net.state_dict().items()}
    a=initial_encoder(net);b=initial_encoder(net)
    for _ in range(3):
        a,ra=historical_step(net,a,negative,positive)
        b,rb=historical_step(net,b,negative,positive)
        assert ra==rb and ra['parameter_delta_norm']>0
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in parent.items())
