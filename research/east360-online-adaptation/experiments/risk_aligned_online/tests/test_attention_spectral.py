import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_spectral_20261029.learning import (
    cross_moment, positive_projector, proposal, predict,
)


def test_trace_identity_optimum_and_descent():
    torch.manual_seed(61)
    g=torch.randn(9,2,64,64,dtype=torch.float64)
    h=torch.randn_like(g);w=torch.arange(1,10,dtype=g.dtype);w=w/w.sum()
    s=cross_moment(g,h,w);p,ev=positive_projector(s)
    ng=g/g.flatten(1).norm(dim=1)[:,None,None,None]
    dot=((h*(ng@p)).flatten(1).sum(1)*w).sum()
    torch.testing.assert_close(dot,(p*s.transpose(-1,-2)).sum())
    torch.testing.assert_close(dot,ev[:,-2:].clamp_min(0).sum())
    torch.testing.assert_close(p@p,p,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(p,p.transpose(-1,-2))
    d=proposal(g,p,.4)
    assert (d.flatten(1).norm(dim=1)<=.4000001).all()
    assert ((g*d).flatten(1).sum(1)<=0).all()
    assert (torch.linalg.svdvals(d)[:,:,2:]<1e-12).all()
    for _ in range(10):
        q,_=torch.linalg.qr(torch.randn(2,64,2,dtype=g.dtype))
        assert ((q@q.transpose(-1,-2))*s.transpose(-1,-2)).sum()<=dot+1e-10


def test_zero_negative_and_input_validation():
    g=torch.zeros(2,2,64,64,dtype=torch.float64);h=torch.ones_like(g);w=torch.ones(2,dtype=g.dtype)/2
    s=cross_moment(g,h,w);p,_=positive_projector(s)
    assert torch.count_nonzero(p)==0 and torch.count_nonzero(proposal(g,p,1.))==0
    p,_=positive_projector(-torch.eye(64,dtype=g.dtype).expand(2,64,64))
    assert torch.count_nonzero(p)==0
    with pytest.raises(ValueError):cross_moment(g,h,-w)
    with pytest.raises(ValueError):positive_projector(torch.ones(2,64,63))
    with pytest.raises(ValueError):proposal(g,p,-1.)


def test_full_gradient_normalization_is_not_projected_normalization():
    g=torch.zeros(1,2,64,64,dtype=torch.float64);g[0,0,0,0]=3;g[0,0,0,1]=4
    p=torch.zeros(2,64,64,dtype=g.dtype);p[:,0,0]=1
    d=proposal(g,p,1.)
    assert float(d[0,0,0,0])==-.6
    assert float(d.norm())==.6


def test_actual_network_future_cannot_change_past_and_shots_reset():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    torch.set_num_threads(1);torch.manual_seed(62);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=MatrixAttention(base);a=torch.stack((net.q_basis,net.k_basis));p=a.transpose(-1,-2)@a
    rng=np.random.default_rng(62)
    signal=dict(values=rng.normal(size=(416,23)).astype('float32'),available=np.ones((416,23),bool))
    policy=dict(projector=p.tolist(),step=.3,mean_gradient=rng.normal(size=(2,64,64)).tolist())
    first,info=predict(net,signal,None,policy);again,info2=predict(net,signal,None,policy)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][288:]+=3
    later,_=predict(net,changed,None,policy)
    for k in first:
        np.testing.assert_array_equal(first[k],again[k])
        np.testing.assert_array_equal(first[k][:288],later[k][:288])
        np.testing.assert_array_equal(first[k][:256],first['cold'][:256])
    assert info==info2 and all(r['updates']==2 and r['max_matrix_norm']<=.300001 for r in info.values())
    np.testing.assert_array_equal(first['random_projector'],first['spectral'])
    with pytest.raises(ValueError):predict(net,dict(signal,label=1),None,policy)
