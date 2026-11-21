import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.qk_matched_head_20261029.learning import (
    adapt,assemble,train_step,query_pools,
)


def fixture():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    torch.set_num_threads(1);torch.manual_seed(17);torch.backends.mha.set_fastpath_enabled(False)
    net=MatrixAttention(Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)))
    rng=np.random.default_rng(17)
    s=dict(shot=123,device='J-TEXT',values=rng.normal(size=(500,23)).astype('float32'),
        available=np.ones((500,23),bool),times=np.arange(500)*.001,valid=np.ones(500,bool),
        y=np.zeros(500,'float32'),is_disrupt=True,down=.499)
    s['y'][475:]=1
    return net,s


def test_feedback_is_causal_and_uses_deployed_clock():
    net,s=fixture();e=assemble([s],[(0,300)],'cpu')
    assert e['receipts'][0]['count']==256 and e['feedback']['truth'].shape==(1,4,25,23)
    changed={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in s.items()}
    changed['values'][256:]+=20
    e2=assemble([changed],[(0,300)],'cpu')
    for k in e['feedback']:torch.testing.assert_close(e['feedback'][k],e2['feedback'][k],atol=0,rtol=0)
    a=adapt(net,e['feedback'])[0];b=adapt(net,e2['feedback'])[0]
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert 0<float(a.norm())<=.050001
    with pytest.raises(ValueError):adapt(net,dict(e['feedback'],y=e['y']))


def test_zero_mask_means_zero_step_not_an_arbitrary_direction():
    net,s=fixture();e=assemble([s],[(0,300)],'cpu')
    e['feedback']['mask'].zero_()
    for mode in ('true','reverse','hold'):
        delta,loss,_,_=adapt(net,e['feedback'],mode=mode)
        assert torch.equal(delta,torch.zeros_like(delta)) and float(loss.sum().detach())==0


def test_double_backward_changes_only_forecast_head_and_no_update_risk_is_identical():
    net,s=fixture();e=assemble([s],[(0,300),(0,480)],'cpu')
    before={k:v.clone() for k,v in net.state_dict().items()}
    net.base.forecast_head.requires_grad_(True)
    opt=torch.optim.AdamW(net.base.forecast_head.parameters(),lr=1e-4,weight_decay=0)
    q=lambda:net(e['qx'],e['qa'],torch.zeros(2,8192),None,'qk',forecast=False)['logit'].detach()
    old=q();r=train_step(net,opt,e,'matched')
    assert np.isfinite(list(r.values())).all() and r['head_gradient_norm']>0
    torch.testing.assert_close(old,q(),atol=0,rtol=0)
    state=net.state_dict()
    assert any(not torch.equal(state[k],v) for k,v in before.items() if 'forecast_head.' in k)
    assert all(torch.equal(state[k],v) for k,v in before.items() if 'forecast_head.' not in k)


def test_pools_exclude_exact10ms_and_query_before_first_feedback():
    _,s=fixture();p,excluded=query_pools([s])
    danger=p['danger'][0][1]
    assert danger.min()==475 and danger.max()==488
    assert all(q>=256 for group in p.values() for _,qs in group for q in qs)
    assert not p['safe'] and excluded==[]


def test_finite_step_meta_gradient_matches_central_difference():
    from experiments.risk_aligned_online.qk_matched_head_20261029.learning import paired_risk
    net,s=fixture();net.double();net.base.forecast_head.requires_grad_(True)
    e=assemble([s],[(0,300),(0,480)],'cpu')
    e['feedback']={k:v.double() if v.is_floating_point() else v for k,v in e['feedback'].items()}
    e['qx']=e['qx'].double();e['y']=e['y'].double()
    def objective():
        r=paired_risk(net,e,create_graph=True)
        # H is detached in the actual objective. Finite-difference validation
        # here targets the genuine through-update risk term plus signal term.
        return .5*r['after'].mean()+.2*r['signal'].mean()
    weight=net.base.forecast_head.weight
    g,=torch.autograd.grad(objective(),weight);direction=g/g.norm();original=weight.detach().clone()
    values=[];eps=1e-5
    for sign in (1,-1):
        with torch.no_grad():weight.copy_(original+sign*eps*direction)
        values.append(float(objective().detach()))
    with torch.no_grad():weight.copy_(original)
    assert (values[0]-values[1])/(2*eps)==pytest.approx(float(g.norm()),rel=2e-4,abs=1e-7)
