import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import layer0_local_memory_20261105 as memory
from layer0_direct_transfer_fit_20261112 import fit_direct


class WindowBase(nn.Module):
    """Two real norm hooks; the first layer's CLS bias shifts the logit by 4b."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(dim=64, layers=2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for block in self.encoder.layers:
            block.norm1 = nn.LayerNorm(64)
        self.eval().requires_grad_(False)

    def forward(self, x, available):
        tokens = x.new_zeros((len(x), 17, 64))
        first, second = (block.norm1(tokens) for block in self.encoder.layers)
        return {"logit": x[:, 0, 0] + 4 * first[:, -1, 0] + second[:, -1, 0]}


def case_at(start):
    x = torch.linspace(start, start + 1, 170).reshape(-1, 1, 1).expand(-1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    ctx = (x.requires_grad_(), a, None, None, torch.zeros(2, 66), None, None)
    fast = torch.zeros(2, 128)
    fast[:, 64] = torch.tensor([0.12, -0.16])
    return ctx, fast.requires_grad_(), torch.zeros(2, 192, requires_grad=True)


def analytic_loss(bias, regularizer):
    threshold = math.log(memory.THRESHOLD / (1 - memory.THRESHOLD))
    # One positive endpoint at anchor 1; three negatives straddle the anchor boundary.
    positive = -2 + 168 / 169 + 4 * bias - threshold
    negative = [-6 + end / 169 + 4 * bias - threshold for end in (135, 166, 169)]
    p = (math.log1p(math.exp(-positive)) + math.log1p(math.exp(-positive + 0.64))) / 2
    n = sum(math.log1p(math.exp(z)) + math.log1p(math.exp(z + 4 * fast))
            for z, fast in zip(negative, (0.12, 0.12, -0.16))) / 6
    return (p + n) / 2 + regularizer


def test_joint_fit_uses_equal_shot_and_arm_weights_and_only_learns_delta():
    base = WindowBase()
    positive, negative = case_at(-2), case_at(-6)
    q_pos, q_neg = np.array([168]), np.array([135, 166, 169])
    history = [(positive, q_pos, np.bool_(True)), (negative, q_neg, False)]
    initial = torch.zeros(128)
    initial[64] = 0.05
    initial.requires_grad_()
    inputs = [initial, *(t for ctx, fast, keys in (positive, negative)
                         for t in (ctx[0], ctx[1], fast, keys))]
    snapshots = [t.detach().clone() for t in inputs]
    source = {k: v.clone() for k, v in base.state_dict().items()}

    with torch.no_grad():
        delta, stats = fit_direct(base, history, initial)
    again, again_stats = fit_direct(base, history, initial)

    assert delta.shape == initial.shape and delta.dtype == initial.dtype
    assert delta.device == initial.device and not delta.requires_grad
    torch.testing.assert_close(again, delta, rtol=0, atol=0)
    assert again_stats == stats
    assert stats == {
        "steps": 20, "teaching_shots": 2, "teaching_points": 4,
        "norm": pytest.approx(float(delta.norm())),
        "initial_loss": pytest.approx(analytic_loss(0.05, 0.10 * 0.05**2), abs=1e-6),
        "final_loss": pytest.approx(
            analytic_loss(float(delta[64]), 0.10 * float(delta.square().sum())), abs=1e-6
        ),
    }
    assert stats["final_loss"] < stats["initial_loss"]
    assert 0.59 < float(delta.norm()) <= memory.EPISODE_RADIUS
    assert torch.count_nonzero(delta).item() == 1
    assert all(t.grad is None and torch.equal(t, before) for t, before in zip(inputs, snapshots))
    assert np.array_equal(q_pos, [168]) and np.array_equal(q_neg, [135, 166, 169])
    assert not base.training and all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(torch.equal(v, source[k]) for k, v in base.state_dict().items())


def test_no_teaching_windows_preserve_accepted_vector():
    initial = torch.full((128,), 0.01)
    history = [((None, None, None), np.array([], dtype=np.int64), True)]
    delta, stats = fit_direct(WindowBase(), history, initial)
    torch.testing.assert_close(delta, initial)
    assert delta.data_ptr() != initial.data_ptr()
    assert stats == dict(steps=0, initial_loss=None, final_loss=None,
                         norm=float(initial.norm()), teaching_shots=0, teaching_points=0)


@pytest.mark.parametrize("indices", [[134], [170], [135, 135], [168, 135], [135.0]])
def test_rejects_invalid_absolute_endpoints(indices):
    with pytest.raises(ValueError):
        fit_direct(WindowBase(), [(case_at(-2), np.array(indices), True)], torch.zeros(128))


def test_rejects_invalid_initial_vector_outcome_and_unfrozen_base():
    base = WindowBase()
    history = [(case_at(-2), np.array([135]), True)]
    for initial in (torch.zeros(127), torch.ones(128), torch.full((128,), float("nan"))):
        with pytest.raises(ValueError):
            fit_direct(base, history, initial)
    with pytest.raises(ValueError):
        fit_direct(base, [(history[0][0], history[0][1], 1)], torch.zeros(128))
    for unfrozen in (base.train(), WindowBase().requires_grad_(True)):
        with pytest.raises(ValueError):
            fit_direct(unfrozen, history, torch.zeros(128))
