import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.matured_negative_20261023.learning import (
    matured_indices, adapt, combine_losses,
)
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def test_maturity_uses_physical_clock_not_point_count():
    times = np.arange(500) / 2000.
    indices = matured_indices(times, event_free_through=times[-1])
    assert len(indices) == 8
    assert indices[-1] == 249
    assert min(indices) >= 127
    assert np.all((times[-1]-times[indices])*1000 >= 125-1e-9)


def test_current_and_immature_windows_are_never_negative_targets():
    times = np.arange(256) / 1000.
    np.testing.assert_array_equal(matured_indices(times, event_free_through=.255),[127,128,129,130])
    assert len(matured_indices(times[:200], event_free_through=.199)) == 0


def test_explicit_current_event_free_certificate_is_required():
    times = np.arange(256) / 1000.
    for value in (None, float('nan'), .3, .2):
        with pytest.raises(ValueError):
            matured_indices(times, event_free_through=value)
    with pytest.raises(ValueError):
        matured_indices(times[::-1], event_free_through=times[0])


def test_update_is_prefix_only_and_does_not_mutate_parent():
    torch.manual_seed(131)
    _, net = model('increment')
    net.eval()
    values = np.random.default_rng(131).normal(size=(384,23)).astype('float32')
    prefix = dict(values=values[:256].copy(), available=np.ones((256,23),bool), times=np.arange(256)/1000.)
    fast = initial_encoder(net)
    original = {k:v.detach().clone() for k,v in net.state_dict().items()}
    new,r = adapt(net,prefix,fast,mode='matured_negative',replay=None,event_free_through=.255,device='cpu')
    assert r['gradient_steps']==1 and r['first_affected_prediction']==256
    assert r['parameter_delta_norm']>0
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in original.items())
    values[256:] = 1e10
    again,rr = adapt(net,prefix,fast,mode='matured_negative',replay=None,event_free_through=.255,device='cpu')
    assert r==rr
    for k in new: torch.testing.assert_close(new[k],again[k],rtol=0,atol=0)
    with pytest.raises(ValueError):
        adapt(net,dict(prefix,is_disrupt=False),fast,mode='matured_negative',replay=None,event_free_through=.255,device='cpu')


def test_positive_replay_is_not_silently_used_when_missing():
    _, net=model('increment')
    prefix=dict(values=np.zeros((256,23),np.float32),available=np.ones((256,23),bool),times=np.arange(256)/1000.)
    with pytest.raises(ValueError):
        adapt(net,prefix,initial_encoder(net),mode='matured_plus_replay',replay=None,event_free_through=.255,device='cpu')


def test_factor_control_keeps_each_component_coefficient_identical():
    neg=torch.tensor(3.,requires_grad=True);pos=torch.tensor(7.,requires_grad=True)
    assert combine_losses({'negative':neg}).item()==1.5
    assert combine_losses({'replay':pos}).item()==3.5
    mixed=combine_losses({'negative':neg,'replay':pos})
    assert mixed.item()==5.
    assert torch.autograd.grad(mixed,(neg,pos))==(torch.tensor(.5),torch.tensor(.5))
