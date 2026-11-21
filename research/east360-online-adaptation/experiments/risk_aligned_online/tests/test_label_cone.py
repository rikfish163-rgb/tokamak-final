import itertools

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.label_cone_20261023.learning import Cone, predict, ARMS
from experiments.risk_aligned_online.history_tangent_20261023.learning import flat, logit
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.tests.test_arrived_teacher import signal


def exhaustive_projection(h,g):
    candidates=[]
    for active in itertools.product((False,True),repeat=len(h)):
        a=h[np.asarray(active)]
        multipliers=np.linalg.solve(a@a.T,-a@g) if len(a) else np.empty(0)
        u=g+a.T@multipliers
        if (multipliers>=-1e-10).all() and (h@u>=-1e-10).all():
            candidates.append(u)
    return min(candidates,key=lambda u:np.linalg.norm(u-g))


def test_signed_constraints_allow_helpful_change_and_block_harm():
    cone=Cone(torch.eye(2,dtype=torch.float64),torch.tensor([0,1]))
    good=torch.tensor([2.,-3.],dtype=torch.float64)
    u,r=cone.project(good)
    torch.testing.assert_close(u,good,rtol=0,atol=0)
    assert r['active_constraints']==0
    bad=torch.tensor([-2.,3.],dtype=torch.float64)
    u,r=cone.project(bad)
    torch.testing.assert_close(u,torch.zeros(2,dtype=torch.float64),atol=1e-12,rtol=0)
    assert r['active_constraints']==2
    # Equalities would discard good directions as well; cone deliberately does not.


def test_nnls_projection_matches_independent_active_set_enumeration():
    rng=np.random.default_rng(8)
    for _ in range(25):
        rows=rng.normal(size=(4,7));labels=np.array([0,1,0,1])
        cone=Cone(torch.tensor(rows),torch.tensor(labels))
        g=rng.normal(size=7);u,r=cone.project(torch.tensor(g))
        h=rows/np.linalg.norm(rows,axis=1,keepdims=True)*(1-2*labels)[:,None]
        np.testing.assert_allclose(u.numpy(),exhaustive_projection(h,g),rtol=1e-9,atol=1e-9)
        assert r['min_margin_after']>=-1e-9 and r['kkt_complementarity']<1e-8
        assert u.norm()<=np.linalg.norm(g)+1e-9
        assert abs(float(u@torch.tensor(g)-u@u))<1e-8


def test_bad_geometry_fails_instead_of_unprotected_fallback():
    for rows,labels in ((torch.zeros(2,3),torch.tensor([0,1])),
                        (torch.ones(2,3),torch.tensor([0,1])),
                        (torch.eye(2),torch.tensor([0.,.5]))):
        with pytest.raises(ValueError):Cone(rows,labels)
    c=Cone(torch.eye(2),torch.tensor([0,1]))
    with pytest.raises(ValueError):c.project(torch.tensor([float('nan'),1.]))


def setup_case():
    model=small_model();s=signal(420);fast=initial_encoder(model)
    rng=np.random.default_rng(17)
    x=torch.tensor(rng.normal(size=(2,128,23)).astype('float32'));m=torch.ones_like(x,dtype=torch.bool)
    z=logit(model,x,m,fast)
    rows=torch.stack([flat(torch.autograd.grad(z[j],tuple(fast.values()),retain_graph=True)).detach() for j in range(2)])
    labels=torch.tensor([0,1]);cones={'signed':Cone(rows,labels),'wrong_labels':Cone(rows,labels.flip(0))}
    return model,s,cones,dict(x=x,mask=m,logits=z.detach(),labels=labels)


def test_causal_reset_frozen_parent_and_matched_step_lengths():
    model,s,cones,anchors=setup_case();old={k:v.clone() for k,v in model.state_dict().items()}
    curves,updates,drift=predict(model,s,cones,anchors,device='cpu')
    again,uu,dd=predict(model,s,cones,anchors,device='cpu')
    assert updates==uu and drift==dd
    s['values'][300:]=1e3
    changed,_,_=predict(model,s,cones,anchors,device='cpu')
    for a in ARMS:
        np.testing.assert_array_equal(curves[a],again[a])
        np.testing.assert_array_equal(curves[a][:300],changed[a][:300])
        assert [r['first_affected_prediction'] for r in updates[a]]==[256,384]
        for r,ref in zip(updates[a],updates['signed']):
            assert r['requested_step_norm']==ref['requested_step_norm']
            assert abs(r['parameter_delta_norm']-ref['parameter_delta_norm'])<2e-7
            assert len(r['history_bce_change'])==2
    for k,v in model.state_dict().items():torch.testing.assert_close(v,old[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(s,y=1),cones,anchors,device='cpu')


def test_all_missing_gives_no_update_not_zero_targets():
    model,s,cones,anchors=setup_case();s['available'][:]=False
    _,u,_=predict(model,s,cones,anchors,device='cpu')
    for a in ARMS:
        assert all(r['gradient_steps']==0 and r['parameter_delta_norm']==0 for r in u[a])
