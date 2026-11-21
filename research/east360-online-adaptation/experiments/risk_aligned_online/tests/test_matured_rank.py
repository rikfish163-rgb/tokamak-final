import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.matured_rank_20261023.learning import (
    relative_loss, step, update, predict, ARMS,
)
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def test_relative_loss_matches_explicit_pairs_and_cancels_common_output_derivative():
    n=torch.tensor([-2.,1.],requires_grad=True);p=torch.tensor([0.,2.,3.],requires_grad=True)
    loss=relative_loss(n,p)
    expected=torch.stack([torch.nn.functional.softplus(a-b) for a in n for b in p]).mean()
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(loss,relative_loss(n+5,p+5))
    gn,gp=torch.autograd.grad(loss,(n,p))
    assert (gn>0).all() and (gp<0).all()
    torch.testing.assert_close(gn.sum()+gp.sum(),torch.tensor(0.),atol=1e-7,rtol=0)


def test_invalid_relative_loss_is_rejected():
    for n in (torch.empty(0),torch.tensor([float('nan')]),torch.zeros(1,2)):
        with pytest.raises(ValueError):relative_loss(n,torch.ones(8))


def test_matched_step_keeps_length_and_rejects_missing_direction():
    f={'w':torch.tensor([.1,.2],requires_grad=True)}
    a,length,norm=step(f,(torch.tensor([3.,4.]),),length=None)
    assert abs(length-.001)<1e-7 and norm==5.
    b,actual,_=step(f,(torch.tensor([0.,9.]),),length=length)
    assert abs(actual-length)<2e-8
    assert torch.equal(f['w'],torch.tensor([.1,.2]))
    with pytest.raises(ValueError):step(f,(torch.zeros(2),),length=.001)


def fixtures():
    torch.manual_seed(131);_,net=model('increment');net.eval()
    rng=np.random.default_rng(131)
    signal=dict(values=rng.normal(size=(385,23)).astype('float32'),available=np.ones((385,23),bool),times=np.arange(385)/1000.)
    x=torch.tensor(rng.normal(size=(8,128,23)).astype('float32'));m=torch.ones_like(x,dtype=torch.bool)
    return net,signal,(x,m),(x+.2,m)


def test_update_requires_prefix_and_explicit_status():
    net,s,n,p=fixtures();prefix={k:v[:256] for k,v in s.items()}
    states={a:initial_encoder(net) for a in ARMS}
    with pytest.raises(ValueError):update(net,prefix,states,n,p,event_free_through=None,device='cpu')
    with pytest.raises(ValueError):update(net,dict(prefix,down=1.),states,n,p,event_free_through=.255,device='cpu')
    new,rows=update(net,prefix,states,n,p,event_free_through=.255,device='cpu')
    assert all(r['first_affected_prediction']==256 for r in rows.values())
    assert all(r['parameter_delta_norm']>0 for r in rows.values())
    assert max(r['parameter_delta_norm'] for r in rows.values())-min(r['parameter_delta_norm'] for r in rows.values())<2e-7


def test_prediction_prefix_invariance_reset_and_parent_immutability():
    net,s,n,p=fixtures();before={k:v.detach().clone() for k,v in net.state_dict().items()}
    curves,r=predict(net,s,n,p,device='cpu')
    changed={k:v.copy() for k,v in s.items()};changed['values'][384:]=1e4
    again,rr=predict(net,changed,n,p,device='cpu')
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:384],again[arm][:384])
        np.testing.assert_array_equal(curves[arm][:256],curves['static'][:256])
    assert r==rr
    reset,_=predict(net,s,n,p,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm],reset[arm])
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
