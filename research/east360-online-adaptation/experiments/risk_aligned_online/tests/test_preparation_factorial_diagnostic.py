import torch
from experiments.risk_aligned_online.preparation_factorial_20261023.diagnostic import channel_losses
from experiments.risk_aligned_online.tests.test_task_geometry import small_model


def test_channel_contributions_reconstruct_original_loss_and_gradient():
    model=small_model();torch.manual_seed(11)
    prediction=torch.randn(4,25,23,requires_grad=True)
    truth=torch.randn_like(prediction);mask=torch.rand_like(prediction)>.2
    mask[:,:,3]=False
    parts=channel_losses(model,prediction,truth,mask)
    original=model.base.forecast_loss(prediction,truth,mask)
    torch.testing.assert_close(parts.sum(),original,rtol=1e-6,atol=1e-7)
    a=torch.autograd.grad(parts.sum(),prediction,retain_graph=True)[0]
    b=torch.autograd.grad(original,prediction)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert parts[3]==0
