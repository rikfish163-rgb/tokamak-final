import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cold20_compare_20261104.training import (
    batch,
    fit_model,
    make_readout,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def base():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(31)
    return (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("mode,count", [("bias", 1), ("head", 65)])
def test_readout_clone_has_only_declared_parameters_and_preserves_source(
    base, mode, count
):
    original = {k: v.clone() for k, v in base.state_dict().items()}
    clone = make_readout(base, mode)
    assert sum(p.numel() for p in clone.parameters() if p.requires_grad) == count
    x = torch.randn(2, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    a[0, :92] = False
    with torch.no_grad():
        assert torch.equal(base(x, a)["logit"], clone(x, a)["logit"])
    for name, p in clone.named_parameters():
        assert p.data_ptr() != dict(base.named_parameters())[name].data_ptr()
    assert all(torch.equal(v, original[k]) for k, v in base.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())


def test_unknown_readout_mode_rejected(base):
    with pytest.raises(ValueError):
        make_readout(base, "threshold_scan")


def example():
    x = np.linspace(-0.3, 0.5, 8 * 23, dtype=np.float32).reshape(8, 23)
    return {1: dict(values=x, available=np.ones_like(x, dtype=bool))}, [
        dict(shot=1, query=2, label=1),
        dict(shot=1, query=5, label=0),
    ]


def test_batch_only_uses_the_selected_causal_window_and_keeps_target_separate():
    shots, records = example()
    x, a, y = batch(shots, records, [0], device="cpu")
    assert x.shape == (1, 128, 23) and a.shape == x.shape and y.tolist() == [1.0]
    assert not a[:, :125].any()
    shots[1]["values"][3:] += 100
    after, _, _ = batch(shots, records, [0], device="cpu")
    assert torch.equal(x, after)
    with pytest.raises(ValueError):
        batch(shots, [dict(shot=99, query=0, label=0)], [0], device="cpu")


def test_fit_changes_only_target_clone_and_leaves_no_grad_afterward(base):
    shots, records = example()
    clone = make_readout(base, "head")
    original = {k: v.clone() for k, v in clone.state_dict().items()}
    source = {k: v.clone() for k, v in base.state_dict().items()}
    schedule = [dict(step=i + 1, indices=[0, 1]) for i in range(3)]
    history = fit_model(clone, shots, records, schedule, device="cpu", mode="head")
    changed = {
        k for k, v in clone.state_dict().items() if not torch.equal(v, original[k])
    }
    assert changed and changed <= {"risk_head.weight", "risk_head.bias"}
    assert len(history) == 3 and all(np.isfinite(r["loss"]) for r in history)
    assert all(not p.requires_grad and p.grad is None for p in clone.parameters())
    assert all(torch.equal(source[k], v) for k, v in base.state_dict().items())
