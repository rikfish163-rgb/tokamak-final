import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.diagnostic_adapter import DiagnosticAdapter, bounded_step


def setup():
    torch.manual_seed(131)
    parent = Simple23(Config(window=32, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    parent.eval().requires_grad_(False)
    adapter = DiagnosticAdapter(64, 25)
    x = torch.randn(2, 32, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    return parent, adapter, x, mask


@pytest.mark.parametrize('site', ['cls', 'last_patch'])
def test_zero_state_parent_identical_and_projection_has_named_coordinates(site):
    parent, adapter, x, mask = setup()
    adapter.site = site
    base = parent(x, mask)['logit']
    out = adapter(parent, x, mask)
    assert torch.equal(base, out['logit'])
    fast = torch.randn(2, 12) * .1
    after = adapter(parent, x, mask, fast)
    torch.testing.assert_close(after['coefficients']-out['coefficients'], fast)
    torch.testing.assert_close(adapter.decode(after['internal_after'])-adapter.decode(after['internal_before']), fast)
    assert not torch.equal(base, after['logit'])
    assert after['trajectory'].shape == (2, 25, 6)


def test_signal_update_has_double_backward_and_no_parent_gradient():
    parent, adapter, x, mask = setup()
    fast = adapter.initial_fast(2).requires_grad_(True)
    loss = adapter(parent, x, mask, fast)['trajectory'].square().mean()
    gradient, = torch.autograd.grad(loss, fast, create_graph=True)
    adapted = bounded_step(fast, gradient)
    outer = adapter(parent, x, mask, adapted)['logit'].square().mean()
    outer.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in parent.parameters())
    assert not parent.encoder.layers[0]._forward_hooks


def test_hook_removed_on_forward_error_and_each_episode_bounded_separately():
    parent, adapter, x, mask = setup()
    with pytest.raises(ValueError):
        adapter(parent, x[:, :10], mask[:, :10])
    assert not parent.encoder.layers[0]._forward_hooks
    current = torch.zeros(2, 12)
    gradient = torch.zeros_like(current)
    gradient[0, 0], gradient[1, 0] = 100, 1
    batch = bounded_step(current, gradient)
    for i in range(2):
        torch.testing.assert_close(batch[i:i+1], bounded_step(current[i:i+1], gradient[i:i+1]))
    assert bounded_step(torch.full((2, 12), 5.), gradient).norm(dim=-1).max() <= 2.00001
