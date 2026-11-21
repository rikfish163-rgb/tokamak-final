import torch
import pytest

from experiments.risk_aligned_online.gradient_forensics_20261022.run import channel_losses, gradient_geometry


def test_channel_loss_decomposition_preserves_missingness_and_total():
    p=torch.randn(4,25,23,requires_grad=True);truth=torch.randn_like(p)
    m=torch.ones_like(p,dtype=torch.bool);m[:,:,2]=False;truth[:,:,2]=float('nan')
    center=torch.zeros(23);scale=torch.ones(23)
    losses=channel_losses(p,truth,m,center,scale)
    clean=torch.where(m,truth,center)
    total=torch.nn.functional.smooth_l1_loss(p,torch.asinh(clean),reduction='none').masked_fill(~m,0).sum()/m.sum()
    torch.testing.assert_close(losses.sum(),total)
    assert losses[2].item()==0
    summed=torch.autograd.grad(losses.sum(),p,retain_graph=True)[0]
    expected=torch.autograd.grad(total,p)[0]
    torch.testing.assert_close(summed,expected)
    assert torch.isfinite(summed).all() and not summed[:,:,2].any()


def test_gradient_geometry_has_correct_update_sign_and_zero_handling():
    r=gradient_geometry([torch.tensor([2.,0.])],[torch.tensor([1.,0.])],.1)
    assert r['cosine']==1 and r['predicted_risk_loss_change']==pytest.approx(-.4/2)
    r=gradient_geometry([torch.tensor([-2.,0.])],[torch.tensor([1.,0.])],.1)
    assert r['cosine']==-1 and r['predicted_risk_loss_change']==pytest.approx(.2)
    r=gradient_geometry([torch.zeros(2)],[torch.ones(2)],.1)
    assert r['cosine'] is None and r['predicted_risk_loss_change']==0
