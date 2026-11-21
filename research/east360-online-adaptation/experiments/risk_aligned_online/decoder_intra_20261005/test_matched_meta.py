import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ffn_intra_20261005.test_meta import case
from experiments.risk_aligned_online.ffn_intra_20261005.source import episode_loss
from experiments.risk_aligned_online.ffn_intra_20261005.stream import predict_shot
from experiments.risk_aligned_online.decoder_intra_20261005.matched_meta import (
    objective, enable_head_only, train_pair,
)


@pytest.mark.parametrize('radius', [.015, .5])
def test_head_meta_direction_matches_finite_difference(radius):
    model, shot, episode = case()
    model = model.double()
    enable_head_only(model)
    head = model.base.forecast_head
    torch.manual_seed(54)
    direction = torch.randn_like(head.weight)
    direction /= direction.norm()
    original = head.weight.detach().clone()
    loss, _ = objective(model, shot, episode, mode='risk_aligned',
                         step=.01, radius=radius, boundary=0., create_graph=True)
    gradient, = torch.autograd.grad(loss, head.weight)
    analytical = float((gradient * direction).sum())
    def value(offset):
        with torch.no_grad():
            head.weight.copy_(original + offset * direction)
        return float(objective(model, shot, episode, mode='risk_aligned',
                               step=.01, radius=radius, boundary=0., create_graph=False)[0].detach())
    numerical = (value(.0001) - value(-.0001)) / .0002
    assert abs(analytical) > 1e-8
    assert analytical == pytest.approx(numerical, rel=.005, abs=3e-7)


def test_head_only_step_and_exact_static_risk():
    model, shot, episode = case()
    enable_head_only(model)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    x = torch.tensor(shot['values'][:128])[None]
    a = torch.ones_like(x, dtype=torch.bool)
    risk = model(x,a,model.initial_delta())['logit'].detach()
    optimizer = torch.optim.Adam(model.base.forecast_head.parameters(), lr=3e-5)
    record = train_pair(model, [(shot,episode)], optimizer, mode='risk_aligned',
                        step=.01, radius=.5, boundary=0.)
    assert record['head_gradient_norm'] > 0
    assert not torch.equal(before['base.forecast_head.weight'], model.base.forecast_head.weight)
    for k,v in model.state_dict().items():
        if not k.startswith('base.forecast_head.'):
            assert torch.equal(v,before[k])
    torch.testing.assert_close(risk, model(x,a,model.initial_delta())['logit'], rtol=0,atol=0)
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('base.forecast_head.'))


def test_label_changes_outer_not_signal_objective():
    model, shot, episode = case()
    enable_head_only(model)
    loss1,row1 = objective(model,shot,episode,mode='signal_only',step=.01,radius=.5,boundary=0.)
    alternate = copy.deepcopy(episode); alternate['label']=0
    loss2,row2 = objective(model,shot,alternate,mode='signal_only',step=.01,radius=.5,boundary=0.)
    torch.testing.assert_close(loss1,loss2,rtol=0,atol=0)
    assert row1['risk_loss'] != row2['risk_loss']
    assert row1['mean_signal_loss'] == row2['mean_signal_loss']


def test_target_input_or_wrong_optimizer_rejected():
    model, shot, episode = case(); enable_head_only(model)
    shot['device']='EAST'
    with pytest.raises(ValueError,match='source'):
        objective(model,shot,episode,mode='risk_aligned',step=.01,radius=.5,boundary=0.)
    shot['device']='J-TEXT'
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with pytest.raises(ValueError,match='only'):
        train_pair(model,[(shot,episode)],optimizer,mode='risk_aligned',step=.01,radius=.5,boundary=0.)


def test_invalid_mode_and_identity_rejected():
    model,shot,episode=case(); enable_head_only(model)
    with pytest.raises(ValueError):
        objective(model,shot,episode,mode='unknown',step=.01,radius=.5,boundary=0.)
    episode['shot']=999
    with pytest.raises(ValueError):
        objective(model,shot,episode,mode='signal_only',step=.01,radius=.5,boundary=0.)


def test_risk_gradient_reaches_head_without_maintenance_term():
    model,shot,episode=case(); enable_head_only(model)
    risk,_=episode_loss(model,shot,episode,step=.01,radius=.5,create_graph=True)
    gradient,=torch.autograd.grad(risk,model.base.forecast_head.weight)
    assert torch.isfinite(gradient).all() and gradient.norm()>1e-6


@pytest.mark.parametrize('parameter', ['weight','bias'])
@pytest.mark.parametrize('radius', [.15,1.])
def test_risk_only_finite_difference_at_actual_step(parameter,radius):
    model,shot,episode=case();model=model.double();enable_head_only(model)
    p=getattr(model.base.forecast_head,parameter)
    torch.manual_seed(57);direction=torch.randn_like(p);direction/=direction.norm()
    original=p.detach().clone()
    risk,_=episode_loss(model,shot,episode,step=.1,radius=radius,create_graph=True)
    derivative,=torch.autograd.grad(risk,p)
    analytical=float((derivative*direction).sum())
    def value(offset):
        with torch.no_grad():p.copy_(original+offset*direction)
        return float(episode_loss(model,shot,episode,step=.1,radius=radius)[0].detach())
    numerical=(value(1e-4)-value(-1e-4))/2e-4
    assert abs(analytical)>1e-9
    assert analytical==pytest.approx(numerical,rel=.005,abs=3e-8)
    static,_=episode_loss(model,shot,episode,step=.1,radius=radius,adapt=False)
    no_path,=torch.autograd.grad(static,p,allow_unused=True)
    assert no_path is None


def test_nonempty_burn_in_matches_actual_from_start_replay():
    model,shot,episode=case();enable_head_only(model)
    rng=np.random.default_rng(4)
    shot['values']=rng.normal(size=(208,23)).astype(np.float32)
    shot['available']=np.ones((208,23),bool)
    shot['times']=np.arange(208)/1000.;shot['down']=.23;shot['valid']=np.ones(208,bool)
    episode.update(burn_in_origins=[0,26],origins=[52,78,104,130],queries=list(range(156,182)))
    _,row=episode_loss(model,shot,episode,step=.1,radius=1.,create_graph=True)
    scores,_=predict_shot(model,shot,intra=True,step=.1,radius=1.,offline_source=True)
    assert row['peak_probability']==pytest.approx(float(scores[156:182].max()),abs=2e-7)
