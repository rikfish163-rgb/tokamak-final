import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.learned_update_loss_20261023.learning import (
    UpdateLoss, context_features, step, folds,
)
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.tests.test_arrived_teacher import signal
from experiments.risk_aligned_online.arrived_teacher_20261023.learning import encode
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_context_means_change_masks_and_missing_values():
    x=torch.zeros(1,128,23);m=torch.ones_like(x,dtype=torch.bool)
    x[:,-25:]=3.; x[:,-50:-25]=1.
    m[:,:,0]=False;x[:,:,0]=float('nan')
    c=context_features(x,m,torch.zeros(23),torch.ones(23))
    assert c.shape==(1,69) and torch.isfinite(c).all()
    assert c[0,1].item()==pytest.approx(np.arcsinh(3),abs=1e-6)
    assert c[0,24].item()==pytest.approx(np.arcsinh(3)-np.arcsinh(1),abs=1e-6)
    assert c[0,47]==1 and c[0,[0,23,46]].eq(0).all()


def test_zero_start_is_identity_but_outer_gradient_reaches_loss_network():
    model=small_model();rule=UpdateLoss(model.cfg.dim);x=signal()
    original={k:v.clone() for k,v in model.state_dict().items()}
    fast,r=step(model,rule,x,228,create_graph=True,device='cpu')
    assert r['delta_norm']==0
    h=encode(model,x,[252],fast,'cpu')
    risk=F.binary_cross_entropy_with_logits(model.base.risk_head(h),torch.ones(1,1))
    grads=torch.autograd.grad(risk,tuple(rule.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0
    with torch.no_grad():
        for p,g in zip(rule.parameters(),grads):p.add_(g,alpha=-1.)
    fast,r=step(model,rule,x,228,device='cpu')
    assert 0<r['delta_norm']<=.00101
    assert set(fast)==set(model.encoder_names())
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)


def test_future_cannot_change_update_and_labels_are_rejected():
    model=small_model();rule=UpdateLoss(model.cfg.dim)
    torch.nn.init.normal_(rule.net[-1].weight,std=.2)
    x=signal();a,ra=step(model,rule,x,228,device='cpu')
    x['values'][228:]=1e6
    b,rb=step(model,rule,x,228,device='cpu')
    assert ra==rb and ra['first_affected_prediction']==228
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    with pytest.raises(ValueError):step(model,rule,dict(x,y=1),228,device='cpu')
    with pytest.raises(ValueError):step(model,rule,x,127,device='cpu')
    x['available'][:]=False
    fast,r=step(model,rule,x,228,device='cpu')
    assert r['gradient_steps']==0
    for k,v in initial_encoder(model).items():torch.testing.assert_close(fast[k],v,rtol=0,atol=0)


def test_folds_partition_identities_not_windows():
    rows=[dict(shot=i,is_disrupt=i<4) for i in range(20)]
    out=folds(rows)
    assert len(out)==4
    assert sorted(i for r in out for i in r['held'])==list(range(20))
    for r in out:
        assert len(r['fit'])==15 and len(r['held'])==5
        assert not set(r['fit'])&set(r['held'])
        assert sum(i<4 for i in r['held'])==1
    with pytest.raises(ValueError):folds(rows[:-1])
