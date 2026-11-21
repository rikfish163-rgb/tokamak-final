import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.preparation_factorial_20261023.learning import (
    CONDITIONS, mapping_for, objective, predict,
)
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import fit_statistics,transform
from experiments.risk_aligned_online.io23 import RAW23
from experiments.risk_aligned_online.tests.test_task_geometry import small_model


def test_mapping_changes_only_density_and_keeps_availability():
    rng=np.random.default_rng(10)
    source=[dict(device='J-TEXT',shot=100,values=rng.uniform(1,2,(256,23)).astype('float32'),available=np.ones((256,23),bool))]
    target=[dict(device='EAST',shot=i,values=rng.uniform(100,200,(256,23)).astype('float32'),available=np.ones((256,23),bool)) for i in range(15)]
    stats=fit_statistics(source,device='J-TEXT',allowed_ids=[100])
    x={k:target[0][k].copy() for k in ('values','available')}
    x['available'][0,0]=False
    for mode in CONDITIONS:
        m=mapping_for(stats,target,list(range(15)),mode)
        out=transform(x,m)
        np.testing.assert_array_equal(out['available'],x['available'])
        changed=np.flatnonzero(np.any(out['values']!=x['values'],axis=0)).tolist()
        assert changed==([RAW23.index('ne_nG')] if mode.startswith('mapped') else [])
        assert not m['physical_calibration_certified']
    with pytest.raises(ValueError):mapping_for(stats,target,list(range(15)),'invented')
    with pytest.raises(ValueError):mapping_for(stats,target,list(range(14)),'raw_bce')


def test_loss_switch_matches_exact_bce_or_existing_conditional_objective():
    from experiments.risk_aligned_online.labeled_alignment_20261023.learning import conditional_moment_loss
    torch.manual_seed(131)
    hs=torch.randn(8,4,requires_grad=True);ht=torch.randn(8,4,requires_grad=True)
    ys=yt=torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.]);head=torch.nn.Linear(4,1)
    bce=.5*(F.binary_cross_entropy_with_logits(head(hs).squeeze(-1),ys)+F.binary_cross_entropy_with_logits(head(ht).squeeze(-1),yt))
    cm=conditional_moment_loss(hs,ht,ys,yt)
    for mode in CONDITIONS:
        loss,components=objective(head,hs,ht,ys,yt,mode)
        expected=bce+cm if mode.endswith('moments') else bce
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        actual=torch.autograd.grad(loss,(hs,ht),retain_graph=True)
        desired=torch.autograd.grad(expected,(hs,ht),retain_graph=True)
        for a,b in zip(actual,desired):torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert components['alignment']==float(cm.detach())


def test_prediction_uses_arrived_signal_resets_and_never_changes_parent():
    model=small_model();rng=np.random.default_rng(19)
    s=dict(values=rng.normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    origin={k:v.clone() for k,v in model.state_dict().items()}
    scores,updates=predict(model,s,device='cpu')
    repeat,reup=predict(model,s,device='cpu')
    altered={k:v.copy() for k,v in s.items()};altered['values'][256:]=100
    other,otherup=predict(model,altered,device='cpu')
    for arm in scores:
        np.testing.assert_array_equal(scores[arm],repeat[arm])
        np.testing.assert_array_equal(scores[arm][:256],other[arm][:256])
        np.testing.assert_array_equal(scores[arm][:256],scores['static'][:256])
    assert updates==reup and updates[0]==otherup[0]
    assert updates[0]['last_observation']==255 and updates[0]['first_affected_prediction']==256
    assert updates[0]['parameter_delta_norm']>0
    for k,v in model.state_dict().items():torch.testing.assert_close(v,origin[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(s,outcome=1),device='cpu')
