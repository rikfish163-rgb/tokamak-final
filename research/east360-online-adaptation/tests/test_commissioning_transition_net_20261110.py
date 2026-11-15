import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.commissioning_transition_net_20261110 import (
    TransitionNet, move, replay,
)


def test_feedback_gradient_matches_autograd_and_missing_feedback_does_not_decay():
    torch.manual_seed(11)
    net = TransitionNet(2)
    x = torch.randn(2, 24)
    pre, key = net.design(x)
    a = (torch.randn(2, 24, 4)*.01).requires_grad_()
    y = torch.randint(0, 3, (2, 5, 2))
    m = torch.tensor([True, False])[:, None, None].expand_as(y)
    h = net.project(pre, key, a)
    loss = F.cross_entropy(net.logits(h)[m], y[m], reduction='mean')
    g = torch.autograd.grad(loss, a)[0]
    computed, _ = net.update(pre, key, a, y, m)
    expected = move(a, g, m.any(dim=(1, 2)))
    torch.testing.assert_close(computed, expected)
    torch.testing.assert_close(computed[1], a[1], rtol=0, atol=0)


def test_replay_releases_after_observation_and_matches_torch_updates():
    torch.manual_seed(12)
    net = TransitionNet(2)
    x = torch.randn(240, 24)
    y = torch.randint(0, 3, (3, 5, 2))
    m = torch.ones_like(y, dtype=torch.bool)
    origins = np.array([127, 159, 191])
    evidence, audit = replay(net, x.numpy(), origins, y.numpy(), m.numpy())
    n = 185
    short, _ = replay(net, x[:n].numpy(), origins[:2], y[:2].numpy(), m[:2].numpy())
    for k in evidence:
        np.testing.assert_allclose(evidence[k][:n], short[k], atol=1e-6)
    np.testing.assert_array_equal(evidence['real'][:153], evidence['fixed'][:153])
    assert not np.allclose(evidence['real'][153:185], evidence['fixed'][153:185])
    a, k = net.design(x)
    s, _ = net.update(a[127:128], k[127:128], net.initial(1), y[:1], m[:1])
    expected = net.project(a[153:185], k[153:185], s.expand(32, -1, -1))
    np.testing.assert_allclose(evidence['real'][153:185], expected.detach().numpy(), atol=1e-6)
    assert audit['nonzero_steps']['real'] == 3
