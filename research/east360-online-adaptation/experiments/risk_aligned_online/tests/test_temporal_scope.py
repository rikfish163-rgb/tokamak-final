import copy
import torch

from experiments.risk_aligned_online.redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from experiments.risk_aligned_online.increment_forecast_20261023.learning import IncrementFeedback
from experiments.risk_aligned_online.temporal_scope_20261023.model import BothBlockPreparation,to_deployment


def test_extra_scope_only_second_transformer_and_conversion_exact():
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT)
    first=IncrementFeedback(copy.deepcopy(base),'increment')
    both=BothBlockPreparation(copy.deepcopy(base),'increment');both.load_state_dict(first.state_dict())
    extra=set(both.encoder_names())-set(first.encoder_names())
    assert extra and all(n.startswith('base.encoder.layers.1.') for n in extra)
    assert not any('risk_head' in n or n.startswith('base.norm.') for n in both.encoder_names())
    deployed=to_deployment(both)
    assert set(deployed.encoder_names())==set(first.encoder_names())
    assert all(not p.requires_grad for p in deployed.parameters())
    x=torch.randn(2,128,23);m=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():
        a=both(x,m,mode='risk_state')['risk_state'];b=deployed(x,m,mode='risk_state')['risk_state']
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for k,v in both.state_dict().items():torch.testing.assert_close(v,deployed.state_dict()[k],rtol=0,atol=0)
