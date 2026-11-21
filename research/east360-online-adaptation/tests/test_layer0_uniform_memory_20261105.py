import pytest
import torch

from layer0_local_memory_20261105 import Episode
from run_layer0_uniform_memory_20261105 import uniform_memory_actions


def test_uniform_control_uses_same_parameter_at_any_causal_state():
    keys = torch.tensor([0.0, 200.0])[:, None].expand(2, 192)
    x = torch.zeros(128)
    x[0] = 0.3
    y = torch.zeros(128)
    y[1] = 0.4
    ep = (Episode(keys[:1], x), Episode(keys[1:], y))
    torch.testing.assert_close(
        uniform_memory_actions(keys, ep, 1.0), ((x + y) / 2).expand(2, -1)
    )
    torch.testing.assert_close(
        uniform_memory_actions(keys, (), 1.0), torch.zeros(2, 128)
    )


def test_uniform_control_keeps_differentiable_proposal_weight():
    keys = torch.zeros(3, 192)
    a = torch.zeros(128)
    b = torch.zeros(128, requires_grad=True)
    ep = (Episode(keys[:1], a), Episode(keys[:1], b))
    uniform_memory_actions(keys, ep, 1.0).sum().backward()
    torch.testing.assert_close(b.grad, torch.full_like(b, 1.5))
    with pytest.raises(ValueError):
        uniform_memory_actions(keys, ep, -1.0)
