import pytest
import torch

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.decoder_intra_20261005.model import AnchoredFfnModel,last_observation


def make():
    torch.set_num_threads(2);torch.manual_seed(133)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized'),torch.zeros(23),torch.ones(23))
    x=torch.randn(2,128,23);a=torch.rand(x.shape)>.2
    return base,x,a


def test_zero_decoder_is_persistence_without_changing_classifier():
    base,x,a=make();reference=FfnModel(base)
    expected=reference(x,a,reference.initial_delta())['logit'].detach()
    model=AnchoredFfnModel(base,initialize=True)
    out=model(x,a,model.initial_delta())
    anchor,seen=last_observation(x,a,model.center,model.scale)
    torch.testing.assert_close(out['forecast'],anchor[:,None].expand(-1,25,-1),rtol=0,atol=0)
    torch.testing.assert_close(out['logit'],expected,rtol=0,atol=0)
    assert all(not p.requires_grad for p in model.parameters())
    assert seen.all()


def test_last_observation_masks_and_unseen_source_centre():
    x=torch.zeros(1,128,23);a=torch.zeros_like(x,dtype=torch.bool)
    x[0,5,2]=3.;a[0,5,2]=True;x[0,70,2]=float('nan');a[0,70,2]=True
    x[0,120,2]=999.;a[0,120,2]=False
    anchor,seen=last_observation(x,a,torch.ones(23),torch.ones(23))
    assert anchor[0,2]==torch.asinh(torch.tensor(2.))
    assert seen.sum()==1 and anchor[~seen].eq(0).all()
    assert not anchor.requires_grad


def test_head_training_preserves_registered_classifier_and_risk():
    base,x,a=make();model=AnchoredFfnModel(base,initialize=True)
    before={k:v.clone() for k,v in base.state_dict().items() if not k.startswith('forecast_head.')}
    first=model(x,a,model.initial_delta())['logit'].detach()
    head=model.base.forecast_head
    head.requires_grad_(True)
    opt=torch.optim.SGD(head.parameters(),lr=.01)
    out=model(x,a,model.initial_delta())
    out['forecast'].square().mean().backward();opt.step()
    after=model(x,a,model.initial_delta())
    torch.testing.assert_close(first,after['logit'],rtol=0,atol=0)
    for k,v in before.items():torch.testing.assert_close(base.state_dict()[k],v,rtol=0,atol=0)
    assert all(p.grad is None for k,p in base.named_parameters() if not k.startswith('forecast_head.'))
    assert head.weight.abs().sum()>0


def test_decoder_nonzero_then_loss_backpropagates_into_ffn():
    base,x,a=make();model=AnchoredFfnModel(base,initialize=True)
    with torch.no_grad():model.base.forecast_head.weight.normal_(std=.01)
    delta=model.initial_delta()
    out=model(x,a,delta)
    grad,=torch.autograd.grad(out['forecast'].square().mean(),delta,create_graph=True)
    assert torch.isfinite(grad).all() and grad.norm()>0
    after=model(x,a,delta-.01*grad)['logit'].mean()
    meta,=torch.autograd.grad(after,delta)
    assert torch.isfinite(meta).all()


def test_load_mode_keeps_trained_decoder_weights():
    base,x,a=make();old=base.forecast_head.weight.detach().clone()
    model=AnchoredFfnModel(base,initialize=False)
    torch.testing.assert_close(model.base.forecast_head.weight,old,rtol=0,atol=0)
