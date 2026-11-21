import torch

from experiments.risk_aligned_online.attention_basis_20261029.train_staged import optimizer_step


def test_step_parameter_has_no_optimizer_history_until_second_stage():
    torch.manual_seed(66)
    a=torch.linalg.qr(torch.randn(2,64,2)).Q.transpose(-1,-2).detach().requires_grad_()
    eta=torch.tensor(.05,requires_grad=True);opt=torch.optim.Adam([a,eta],lr=.03)
    for epoch in range(20):
        optimizer_step(a,eta,opt,torch.ones_like(a),torch.tensor(1.),epoch)
        assert float(eta.detach())==float(torch.tensor(.05)) and eta not in opt.state
    optimizer_step(a,eta,opt,torch.ones_like(a),torch.tensor(1.),20)
    assert float(eta.detach())<.05 and eta in opt.state
