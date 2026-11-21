import torch

from experiments.risk_aligned_online.commissioning_persistence_dose_20261111 import dose_views


def test_dose_changes_only_signed_hidden_perturbation():
    pre = torch.tensor([[.2, -.1], [1., -1.]])
    change = torch.tensor([[.3, .7], [0., 0.]])
    arms = dose_views(pre, change)
    torch.testing.assert_close(arms['inherited'], pre.tanh(), rtol=0., atol=0.)
    torch.testing.assert_close(arms['sustained'], (pre+change).tanh(), rtol=0., atol=0.)
    torch.testing.assert_close(arms['real2'], (pre+2*change).tanh(), rtol=0., atol=0.)
    torch.testing.assert_close(arms['opposite4'], (pre-4*change).tanh(), rtol=0., atol=0.)
    for v in arms.values():
        torch.testing.assert_close(v[1], pre[1].tanh(), rtol=0., atol=0.)
