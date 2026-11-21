import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.latent_completion_20261028.model import tokens,readout
from experiments.risk_aligned_online.attention_lowrank_20261029.model import AttentionAdapter
from experiments.risk_aligned_online.cold_capacity_20261029.probe import optimize


def setup():
    torch.set_num_threads(1);torch.manual_seed(41);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    return base,AttentionAdapter(base)


def test_zero_delta_matches_original_risk_forecast_and_cold_parent():
    base,net=setup();x=torch.randn(3,128,23);a=torch.rand_like(x)>.2
    a[0,:64]=False;a[1]=False
    for mode in ('qk','v'):
        z=net(x,a,torch.zeros(3,512),None,mode)
        original=base(x,a)
        torch.testing.assert_close(z['logit'],original['logit'],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(z['forecast'],original['forecast'],atol=1e-6,rtol=1e-5)
        slow=torch.randn(64,64)*.01
        h,v=tokens(base,x,a,torch.zeros(3,128))
        torch.testing.assert_close(net(x,a,torch.zeros(3,512),slow,mode)['logit'],readout(base,h,v,slow),atol=1e-6,rtol=1e-5)


def test_projection_scope_is_real_and_states_are_independent():
    base,net=setup();x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    state=torch.randn(2,512)*.1
    _,p0,v0=net.first(x,a,torch.zeros_like(state),'qk')
    _,pq,vq=net.first(x,a,state,'qk')
    _,pv,vv=net.first(x,a,state,'v')
    assert torch.equal(p0['value'],pq['value'])
    assert torch.equal(p0['query'],pv['query']) and torch.equal(p0['key'],pv['key'])
    assert not torch.equal(p0['query'],pq['query']) and not torch.equal(p0['value'],pv['value'])
    assert torch.equal(v0,vq) and torch.equal(v0,vv)
    all_result=net(x,a,state,None,'qk')['logit']
    for i in range(2):
        one=net(x[i:i+1],a[i:i+1],state[i:i+1],None,'qk')['logit']
        torch.testing.assert_close(all_result[i:i+1],one,atol=1e-6,rtol=1e-5)


def test_oracle_gradients_exist_and_frozen_parameters_stay_identical():
    base,net=setup();saved={k:v.clone() for k,v in net.state_dict().items()}
    x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool);signs=torch.tensor([1.,-1.])
    for mode in ('qk','v'):
        r=optimize(lambda s:net(x,a,s,None,mode)['logit'],torch.zeros(2,512),signs,radius=1,steps=3,step=.1)
        assert (r['gradient_norm']>0).all()
        for i in range(2):
            one=optimize(lambda s:net(x[i:i+1],a[i:i+1],s,None,mode)['logit'],torch.zeros(1,512),signs[i:i+1],radius=1,steps=3,step=.1)
            torch.testing.assert_close(r['state'][i:i+1],one['state'],atol=2e-6,rtol=1e-4)
    assert all(torch.equal(v,saved[k]) for k,v in net.state_dict().items())
    assert all(p.grad is None for p in net.parameters())


def test_arrived_update_has_only_mature_targets_and_disposable_state():
    from experiments.risk_aligned_online.attention_lowrank_20261029.online import predict
    base,net=setup();rng=np.random.default_rng(4)
    signal=dict(values=rng.normal(size=(288,23)).astype('float32'),available=np.ones((288,23),bool))
    first,info=predict(net,signal,None);repeat,ri=predict(net,signal,None)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][256:]*=30
    other,oi=predict(net,changed,None)
    for k in first:
        assert np.array_equal(first[k],repeat[k])
        assert np.array_equal(first[k][:256],other[k][:256])
        np.testing.assert_allclose(first[k][:256],first['cold'][:256],atol=1e-7,rtol=1e-5)
    assert info==ri==oi
    assert all(r['updates']==1 and r['last_observations']==[255] and r['max_state_norm']<=1.000001 for r in info.values())


def test_missing_targets_do_not_invent_feedback():
    from experiments.risk_aligned_online.attention_lowrank_20261029.online import update
    import pytest
    base,net=setup()
    prefix=dict(values=np.ones((256,23),np.float32),available=np.zeros((256,23),bool))
    s=torch.randn(512)*.001
    new,r=update(net,prefix,s,None,'qk_arrived')
    assert torch.equal(new,s) and r['updates']==0 and r['observed_targets']==0
    with pytest.raises(ValueError):
        update(net,dict(prefix,y=1),s,None,'qk_arrived')
