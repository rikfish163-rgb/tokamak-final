import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.forecast_token_scope_20261023.learning import (
    PatchForecast, temporary, predict, ARMS, adapt,
)
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.preparation_factorial_20261023.learning import predict as legacy_predict


def models():
    a=small_model();b=copy.deepcopy(a)
    b.base.forecast_head=PatchForecast(a.base.forecast_head,a.cfg.dim)
    b.set_stage('frozen')
    return {'full':a,'patch':b}


def test_patch_forecast_has_zero_direct_cls_gradient_but_not_zero_patch_gradient():
    linear=torch.nn.Linear(12,4);head=PatchForecast(linear,3)
    x=torch.randn(2,12,requires_grad=True)
    y=head(x);expected=linear(torch.cat((x[:,:9],torch.zeros_like(x[:,9:])),1))
    torch.testing.assert_close(y,expected,rtol=0,atol=0)
    g,=torch.autograd.grad(y.sum(),x)
    assert not g[:,-3:].any() and g[:,:9].abs().sum()>0
    torch.testing.assert_close(head.weight,linear.weight,rtol=0,atol=0)
    with pytest.raises(ValueError):PatchForecast(linear,13)


def test_parameter_subset_really_leaves_cls_and_position_fixed():
    m=models()['patch'];f=temporary(m,'observation')
    assert not f['base.cls'].requires_grad and not f['base.position'].requires_grad
    s=dict(values=np.random.default_rng(2).normal(size=(256,23)).astype('float32'),available=np.ones((256,23),bool))
    out,r=adapt(m,s,256,f,device='cpu')
    for k in ('base.cls','base.position'):torch.testing.assert_close(out[k],f[k],rtol=0,atol=0)
    assert r['gradient_steps']==1 and r['parameter_delta_norm']>0
    assert r['group_delta_norm']['summary']==0 and r['group_delta_norm']['observation']>0
    with pytest.raises(ValueError):temporary(m,'invented')


def test_original_reproduced_prefix_causal_reset_and_classifier_unchanged():
    m=models();before={n:{k:v.clone() for k,v in a.state_dict().items()} for n,a in m.items()}
    s=dict(values=np.random.default_rng(2).normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    result,receipts=predict(m,s,device='cpu');reference,rr=legacy_predict(m['full'],s,device='cpu')
    np.testing.assert_array_equal(result['static'],reference['static'])
    np.testing.assert_array_equal(result['full_all'],reference['intra'])
    for arm in ARMS:
        np.testing.assert_array_equal(result['static'][:256],result[arm][:256])
        assert len(receipts[arm])==2
    s['values'][256:]=999
    second,sr=predict(m,s,device='cpu');third,tr=predict(m,s,device='cpu')
    for arm in ['static']+list(ARMS):
        np.testing.assert_array_equal(result[arm][:256],second[arm][:256])
        np.testing.assert_array_equal(second[arm],third[arm])
    assert sr==tr
    for arm in ARMS:assert sr[arm][0]==receipts[arm][0]
    for n,a in m.items():
        for k,v in a.state_dict().items():torch.testing.assert_close(v,before[n][k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(m,dict(s,y=1),device='cpu')
