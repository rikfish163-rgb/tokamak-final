import copy

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.learned_update_loss_20261023.learning import UpdateLoss,step,summarize
from experiments.risk_aligned_online.learned_update_loss_20261023.run import schedule
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.tests.test_arrived_teacher import signal
from experiments.risk_aligned_online.arrived_teacher_20261023.learning import encode


def test_meta_gradient_agrees_with_finite_difference():
    model=small_model();rule=UpdateLoss(model.cfg.dim);s=signal()
    def outer(candidate):
        fast,_=step(model,candidate,s,228,create_graph=True,device='cpu')
        h=encode(model,s,[252],fast,'cpu')
        return F.binary_cross_entropy_with_logits(model.base.risk_head(h),torch.ones(1,1))
    risk=outer(rule)
    grad,=torch.autograd.grad(risk,rule.net[-1].weight)
    direction=grad/torch.linalg.vector_norm(grad)
    # A .1 perturbation crosses the gradient-clipping kink (factor about .45),
    # so its secant is not the local derivative at the zero-initialized head.
    # .01 and .001 stay in the unclipped neighbourhood; verify that explicitly.
    losses=[];eps=.001
    for sign in (-1,1):
        candidate=copy.deepcopy(rule)
        with torch.no_grad():candidate.net[-1].weight.add_(direction,alpha=sign*eps)
        _,receipt=step(model,candidate,s,228,create_graph=True,device='cpu')
        assert receipt['clip_factor']==1.
        losses.append(float(outer(candidate).detach()))
    numerical=(losses[1]-losses[0])/(2*eps)
    np.testing.assert_allclose(numerical,float((grad*direction).sum()),rtol=.01,atol=2e-6)


def test_shuffled_supervision_keeps_budget_and_identity_isolation():
    items=[dict(shot=s,group='safe' if s>=4 else 'disruptive_timely') for s in range(20) for _ in range(3)]
    fit=list(range(1,4))+list(range(8,20))
    steps=schedule(items,fit,0)
    assert steps==schedule(items,fit,0) and len(steps)==192
    assert sum(s['swap'] for s in steps)==96
    for row in steps:
        assert len(row['indices'])==2
        assert [items[j]['group'] for j in row['indices']]==['safe','disruptive_timely']
        assert all(items[j]['shot'] in fit for j in row['indices'])


def test_gate_not_satisfied_by_just_reducing_safe_loss():
    rows=[dict(shot=s,group='disruptive_timely' if s<4 else 'safe',before=1.,after=1.01 if s<4 else .5)
          for s in range(20)]
    assert not summarize(rows)['passes_screen']
    for r in rows:
        if r['shot']<3:r['after']=.9
    assert summarize(rows)['passes_screen']
