import numpy as np
import torch

from experiments.risk_aligned_online.tests.test_increment_forecast import model
from experiments.risk_aligned_online.tests.test_continual_episode import shot
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.continual_episode_20261023.learning import episode_indices
from experiments.risk_aligned_online.contextual_feedback_20261023.learning import (
    WeightPolicy, feedback, step, policy_loss, train_policy, predict,
)


def setup():
    torch.set_num_threads(1); torch.manual_seed(131)
    _,m=model('increment')
    with torch.no_grad():m.base.forecast_head.weight.normal_(0,.02)
    return m,WeightPolicy(np.zeros(185),np.ones(185)),shot()


def test_uniform_initialization_preserves_original_observed_loss_and_step():
    from experiments.risk_aligned_online.increment_forecast_20261023.learning import update
    m,p,s=setup();signal={k:s[k][:256] for k in ('values','available')}
    state=initial_encoder(m)
    a,ar=step(m,p,signal,256,state,device='cpu')
    b,br=update(m,signal,256,state,device='cpu')
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert ar['weights']==[1.]*23 and ar['loss_before']==br['loss_before']
    assert ar['forecast_origins']==br['forecast_origins']


def test_meta_gradient_reaches_only_policy_and_matches_finite_difference():
    m,p,s=setup();m=m.double();p=p.double();e=episode_indices(s)[-1]
    loss=policy_loss(m,p,[(s,e)],device='cpu')[0]
    param=p.network[-1].bias;g=torch.autograd.grad(loss,param)[0]
    index=int(g.abs().argmax());assert g[index].abs()>1e-10
    vals=[];eps=1e-4
    for sign in (1,-1):
        with torch.no_grad():param[index]=sign*eps
        vals.append(float(policy_loss(m,p,[(s,e)],device='cpu')[0].detach()))
    with torch.no_grad():param[index]=0
    assert np.isclose(float(g[index]),(vals[0]-vals[1])/(2*eps),rtol=.01,atol=1e-9)
    before={k:v.clone() for k,v in m.state_dict().items()}
    r=train_policy(m,p,torch.optim.Adam(p.parameters(),lr=.01),[(s,e)],device='cpu')
    assert r['parameter_delta_norm']>0
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_weights_bounded_and_missing_truth_not_synthetic_feedback():
    m,p,s=setup()
    with torch.no_grad():p.network[-1].bias.copy_(torch.linspace(-50,50,23))
    w=p(torch.zeros(185));assert (w>=np.exp(-2)).all() and (w<=np.exp(2)+1e-6).all()
    signal={k:s[k][:256].copy() for k in ('values','available')};signal['available'][:]=False
    state=initial_encoder(m);new,r=step(m,p,signal,256,state,device='cpu')
    assert r['gradient_steps']==0 and r['reason']=='no_measured_forecast_target'
    for k in state:assert torch.equal(state[k],new[k])


def test_prediction_prefix_invariance_reset_and_frozen_network():
    m,p,s=setup();signal={k:s[k][:385].copy() for k in ('values','available')}
    a,ar=predict(m,p,p,signal,device='cpu')
    b,br=predict(m,p,p,signal,device='cpu')
    changed={k:v.copy() for k,v in signal.items()};changed['values'][384:]=100
    c,cr=predict(m,p,p,changed,device='cpu')
    assert ar==br==cr
    for k in a:
        np.testing.assert_array_equal(a[k],b[k]);np.testing.assert_array_equal(a[k][:384],c[k][:384])
