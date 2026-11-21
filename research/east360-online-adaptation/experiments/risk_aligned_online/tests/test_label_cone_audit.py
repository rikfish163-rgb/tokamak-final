import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.label_cone_20261023.audit import binary_loss


def test_independent_bce_is_stable_and_matches_framework():
    z=np.array([-1000.,-10.,-.2,0.,.2,10.,1000.])
    for y in (np.zeros_like(z),np.ones_like(z)):
        result=binary_loss(z,y)
        assert np.isfinite(result).all() and (result>=0).all()
        expected=F.binary_cross_entropy_with_logits(torch.tensor(z),torch.tensor(y),reduction='none')
        np.testing.assert_allclose(result,expected.numpy(),rtol=1e-10,atol=1e-14)


def test_bce_telescope_detects_omitted_or_changed_update():
    y=np.array([0.,1.]);zs=np.array([[1.,-1.],[.8,-.8],[.2,-.2]])
    losses=binary_loss(zs,y)
    differences=np.diff(losses,axis=0)
    np.testing.assert_allclose(differences.sum(0),losses[-1]-losses[0])
    assert not np.allclose(differences[0],losses[-1]-losses[0])
