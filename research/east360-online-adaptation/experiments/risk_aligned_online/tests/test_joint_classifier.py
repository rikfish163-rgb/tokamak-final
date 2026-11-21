import torch
import pytest

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.increment_forecast_20261023.learning import IncrementFeedback
from experiments.risk_aligned_online.joint_classifier_20261023.learning import parameter_names,configure


def model():
    torch.manual_seed(3)
    return IncrementFeedback(Simple23(Config(dim=64,input_view='source',fast_site='tokens'),torch.zeros(23),torch.ones(23)),'increment')


def test_preparation_scope_does_not_expand_intra_scope():
    m=model(); first=set(m.encoder_names());head={'base.risk_head.weight','base.risk_head.bias'}
    assert set(parameter_names(m,'first_head'))==first|head
    full=set(parameter_names(m,'full_classifier'))
    expected={n for n,_ in m.named_parameters() if n.startswith('base.') and not n.startswith('base.forecast_head.')}
    assert full==expected and first|head<full
    configure(m,'full_classifier')
    assert set(m.encoder_names())==first
    assert {n for n,p in m.named_parameters() if p.requires_grad}==full
    with pytest.raises(ValueError):parameter_names(m,'all')


@pytest.mark.parametrize('scope',['first_head','full_classifier'])
def test_actual_classification_backward_changes_only_allowed_parameters(scope):
    torch.set_num_threads(2);m=model();names=set(configure(m,scope))
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    x=torch.randn(4,128,23);a=torch.ones_like(x,dtype=torch.bool)
    h=m(x,a,mode='risk_state')['risk_state'];z=m.base.risk_head(h).squeeze(-1)
    torch.testing.assert_close(z,m(x,a)['logit'],rtol=0,atol=0)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.,0.,1.])).backward()
    for n,p in m.named_parameters():
        if n in names:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,n
        else:assert p.grad is None,n
    opt.step()
    for n,v in m.state_dict().items():
        if n not in names:torch.testing.assert_close(v,before[n],rtol=0,atol=0)
    assert not torch.equal(before['base.risk_head.weight'],m.base.risk_head.weight)
