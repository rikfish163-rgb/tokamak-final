import pytest
import torch

from layer0_local_memory_20261105 import Episode, memory_actions
from run_layer0_memory_polarity_20261105 import isolate_origin


def test_ablation_keeps_the_same_kernel_denominator():
    k = torch.zeros(1, 192)
    a, b = torch.full((128,), 0.01), torch.full((128,), -0.02)
    full = (Episode(k, a), Episode(k, b))
    unsafe = isolate_origin(full, [True, False], True)
    safe = isolate_origin(full, [True, False], False)
    torch.testing.assert_close(unsafe[1].keys, full[1].keys)
    torch.testing.assert_close(safe[0].keys, full[0].keys)
    for part in (unsafe, safe):
        assert len(part) == 2
    torch.testing.assert_close(memory_actions(k, unsafe, 1.0), a[None] / 2)
    torch.testing.assert_close(memory_actions(k, safe, 1.0), b[None] / 2)
    torch.testing.assert_close(
        memory_actions(k, unsafe, 1.0) + memory_actions(k, safe, 1.0),
        memory_actions(k, full, 1.0),
    )
    with pytest.raises(ValueError):
        isolate_origin(full, [True], True)
