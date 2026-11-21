import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.preparation_factors_20261023.design import (
    class_aware_selection, risk_objective, gradient_relation,
)


def test_class_quota_is_exact_deterministic_and_never_uses_held_identity():
    ids = list(range(30)); x = np.random.default_rng(7).normal(size=(30,5))
    labels = {i:i<6 for i in ids}
    a = class_aware_selection(ids,x,labels)
    assert len(a)==len(set(a))==20 and sum(labels[s] for s in a)==4
    order = np.random.default_rng(3).permutation(30)
    b = class_aware_selection([ids[i] for i in order],x[order],labels)
    assert a==b
    with pytest.raises(ValueError):
        class_aware_selection(ids,x,{**labels,999:True})
    with pytest.raises(ValueError):
        class_aware_selection(ids,x,{i:i<3 for i in ids})


def test_source_gradient_is_removed_without_changing_target_coefficient():
    s=torch.tensor(2.,requires_grad=True);t=torch.tensor(3.,requires_grad=True)
    assert torch.autograd.grad(risk_objective(s,t,True),(s,t))==(torch.tensor(.5),torch.tensor(.5))
    gs,gt=torch.autograd.grad(risk_objective(s,t,False),(s,t),allow_unused=True)
    assert gs is None and gt==.5


def test_gradient_conflict_is_measured_not_treated_as_a_physical_cause():
    p=torch.tensor([1.,2.],requires_grad=True)
    result=gradient_relation(p.sum(),-p.sum(),[p])
    assert result['cosine']==pytest.approx(-1.)
    assert result['source_norm']==pytest.approx(2**.5)
    result=gradient_relation((p*0).sum(),p.sum(),[p])
    assert result['cosine'] is None and result['zero_gradient']
