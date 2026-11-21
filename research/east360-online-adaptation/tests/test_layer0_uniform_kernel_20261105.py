import pytest
import torch

import layer0_local_memory_20261105 as memory
from run_layer0_uniform_kernel_20261105 import uniform_kernel


def test_uniform_fitting_does_not_update_previously_committed_parameters():
    p = torch.ones(1, 192)
    old = torch.full((128,), 0.01, requires_grad=True)
    new = torch.zeros(128, requires_grad=True)
    episodes = (memory.Episode(p, old), memory.Episode(p * 10, new))
    result = uniform_kernel(torch.stack((p[0], -p[0])), episodes, 1.0, fixed_prefix=1)
    torch.testing.assert_close(result[0], result[1])
    result.sum().backward()
    assert old.grad is None
    torch.testing.assert_close(new.grad, torch.ones_like(new))


def test_uniform_hook_rejects_impossible_fixed_prefix():
    with pytest.raises(ValueError):
        uniform_kernel(torch.zeros(1, 192), (), 0.5, fixed_prefix=1)
