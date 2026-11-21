import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.attention_protected_20261029.learning import (
    validate_matrix, matched_random, accept_history, predict,
)
from experiments.risk_aligned_online.attention_spectral_20261029.learning import proposal


def matrix():
    x=torch.zeros(2,64,64,dtype=torch.float64)
    x[0].diagonal()[:3]=torch.tensor([1.,.6,.4])
    x[1].diagonal()[:5]=.4
    return x


def test_random_control_matches_full_spectrum_and_update_bound():
    p=matrix();validate_matrix(p);r=matched_random(p)
    torch.testing.assert_close(torch.linalg.eigvalsh(p),torch.linalg.eigvalsh(r),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(r,matched_random(p),atol=0,rtol=0)
    assert not torch.allclose(r,p)
    g=torch.randn(4,2,64,64,dtype=p.dtype);d=proposal(g,p,.25)
    assert (d.flatten(1).norm(dim=1)<=.25+1e-12).all()
    assert ((g*d).flatten(1).sum(1)<=0).all()
    with pytest.raises(ValueError):validate_matrix(-p)
    with pytest.raises(ValueError):validate_matrix(torch.eye(64).expand(2,64,64))


def test_step_selection_ignores_held_shots_but_never_compensates_fitted_harm():
    f=np.array([True,False,True,False]);take=np.array([0,1])
    assert accept_history(np.array([-.01,0,9,9]),f,take)
    assert accept_history(np.array([-.01,0,-9,-9]),f,take)
    assert not accept_history(np.array([-.1,.001,0,0]),f,take)
    assert not accept_history(np.zeros(4),f,take)
    with pytest.raises(ValueError):accept_history(np.zeros(4),f,np.array([0,2]))


def test_actual_network_causality_and_resets_with_fractional_psd():
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    torch.set_num_threads(1);torch.manual_seed(65);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=MatrixAttention(base);p=matrix();rng=np.random.default_rng(65)
    signal=dict(values=rng.normal(size=(416,23)).astype('float32'),available=np.ones((416,23),bool))
    policy=dict(matrix=p.tolist(),random_matrix=p.tolist(),step=.25,mean_gradient=rng.normal(size=(2,64,64)).tolist())
    first,info=predict(net,signal,None,policy);again,info2=predict(net,signal,None,policy)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][288:]+=3
    later,_=predict(net,changed,None,policy)
    for k in first:
        np.testing.assert_array_equal(first[k],again[k])
        np.testing.assert_array_equal(first[k][:288],later[k][:288])
        np.testing.assert_array_equal(first[k][:256],first['cold'][:256])
    assert info==info2 and all(r['updates']==2 and r['max_matrix_norm']<=.250001 for r in info.values())
    np.testing.assert_array_equal(first['random_contraction'],first['protected'])
    with pytest.raises(ValueError):predict(net,dict(signal,label=1),None,policy)
