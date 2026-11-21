import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention,project_rows,proposal,feedback_gradient


def test_projection_rank_norm_and_empty_feedback():
    torch.manual_seed(51)
    a=project_rows(torch.randn(2,2,64,dtype=torch.float64))
    g=torch.randn(3,2,64,64,dtype=torch.float64)
    d=proposal(g,a,torch.tensor(.3,dtype=torch.float64))
    torch.testing.assert_close(a@a.transpose(-1,-2),torch.eye(2,dtype=a.dtype).expand(2,2,2))
    torch.testing.assert_close(d.flatten(1).norm(dim=1),torch.full((3,),.3,dtype=a.dtype))
    assert (torch.linalg.svdvals(d)[:,:,2:]<1e-12).all()
    assert torch.count_nonzero(proposal(torch.zeros_like(g),a,torch.tensor(.3)))==0
    with pytest.raises(ValueError):project_rows(torch.zeros_like(a))


def test_basis_outer_gradient_includes_both_paths():
    torch.manual_seed(52)
    a=project_rows(torch.randn(2,2,64,dtype=torch.float64)).requires_grad_()
    g=torch.randn(2,2,64,64,dtype=torch.float64);t=torch.randn_like(g);eta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    f=lambda x,y:(proposal(g,x,y).sin()*t).sum()
    ga,ge=torch.autograd.grad(f(a,eta),(a,eta));eps=1e-5
    for index in ((0,0,1),(1,1,17)):
        e=torch.zeros_like(a);e[index]=eps
        torch.testing.assert_close(ga[index],(f(a+e,eta)-f(a-e,eta))/(2*eps),atol=1e-8,rtol=1e-5)
    torch.testing.assert_close(ge,(f(a,eta+eps)-f(a,eta-eps))/(2*eps),atol=1e-8,rtol=1e-5)


def test_full_matrix_gradient_equals_direct_lora_zero_step():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_lowrank_20261029.model import AttentionAdapter
    from experiments.risk_aligned_online.attention_weights_20261029.learning import feedback_terms
    torch.set_num_threads(1);torch.manual_seed(53);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=MatrixAttention(base);low=AttentionAdapter(base);rng=np.random.default_rng(53)
    signal=dict(values=rng.normal(size=(256,23)).astype('float32'),available=np.ones((256,23),bool))
    g,info=feedback_gradient(net,signal,None)
    terms,state,_=feedback_terms(low,signal,None);sg,=torch.autograd.grad(terms.sum(),state)
    ds=low.deltas(-.05*sg/sg.norm(),'qk');expected=torch.stack((ds['q'][0],ds['k'][0]))
    a=torch.stack((low.q_basis,low.k_basis))
    torch.testing.assert_close(proposal(g[None],a,torch.tensor(.05))[0],expected,atol=2e-8,rtol=1e-4)
    assert info['count']==256 and info['observed']>0
    x=torch.from_numpy(signal['values'][:128])[None];mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():
        got=net(x,mask,torch.zeros(1,8192),None,'qk')['logit']
        old=low(x,mask,torch.zeros(1,512),None,'qk')['logit']
    torch.testing.assert_close(got,old,atol=0,rtol=0)
    signal['available'][:]=False
    g,info=feedback_gradient(net,signal,None)
    assert torch.count_nonzero(g)==0 and info['observed']==0
    with pytest.raises(ValueError):feedback_gradient(net,dict(signal,label=1),None)


def test_stream_resets_state_and_cannot_rewrite_earlier_predictions():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_basis_20261029.learning import predict
    torch.set_num_threads(1);torch.manual_seed(54);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=MatrixAttention(base);rng=np.random.default_rng(54)
    signal=dict(values=rng.normal(size=(416,23)).astype('float32'),available=np.ones((416,23),bool))
    policy=dict(basis=torch.stack((net.q_basis,net.k_basis)).tolist(),step=.3,mean_gradient=rng.normal(size=(2,64,64)).tolist())
    first,info=predict(net,signal,None,policy);again,info2=predict(net,signal,None,policy)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][288:]+=3
    later,_=predict(net,changed,None,policy)
    for k in first:
        assert np.array_equal(first[k],again[k])
        assert np.array_equal(first[k][:288],later[k][:288])
        assert np.array_equal(first[k][:256],first['cold'][:256])
    assert info==info2 and all(r['updates']==2 and r['max_matrix_norm']<=.300001 for r in info.values())
    np.testing.assert_array_equal(first['random_basis'],first['learned_basis'])
