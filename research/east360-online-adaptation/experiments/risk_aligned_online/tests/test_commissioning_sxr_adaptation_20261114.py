import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_sxr_adaptation_20261114 as model


def test_auxiliary_fit_does_not_overweight_long_shots():
    rng = np.random.default_rng(93)
    x, y = rng.normal(size=(7, 6)), rng.normal(size=(7, 5, 4))
    a = model.fit_head([(x, y), (x[:4]+1., y[:4]-1.)])
    b = model.fit_head([(np.tile(x,(3,1)), np.tile(y, (3,1,1))), (x[:4]+1., y[:4]-1.)])
    for u,v in zip(a,b,strict=True):
        np.testing.assert_allclose(u,v,atol=5e-13,rtol=0.)


def test_separated_feedback_only_changes_requested_coordinates():
    prediction = torch.zeros((2,5,4))
    observed = torch.arange(40, dtype=torch.float32).view(2,5,4)/100.
    force = model.force(observed,prediction)
    g,s = model.separate(force)
    torch.testing.assert_close(g+s,force)
    assert not g[...,1:].any() and not s[...,0].any()
    assert not torch.allclose(s,model.force(observed.roll(2,1),prediction))


def test_added_direction_is_absent_off_support_and_capped_by_unit_gradient():
    base = torch.tensor([[[0.,3.],[4.,0.]],[[0.,3.],[4.,0.]]])
    aux = torch.tensor([[[[3.,0.],[0.,4.]],[[0.,0.],[0.,1.]]]])
    values = model.compose(base,aux,torch.tensor([True,False]))
    assert values.shape == (2,2,2,2)
    torch.testing.assert_close(values[0],base)
    torch.testing.assert_close(values[1,1],base[1])
    torch.testing.assert_close(values[1,0],(base[0]+aux[0,0])/5.)
    assert not torch.allclose(values[0,0],values[1,0])
    cos = model.cosine_rows(values)
    torch.testing.assert_close(cos,torch.tensor([[2.**(-.5)],[1.]]))
