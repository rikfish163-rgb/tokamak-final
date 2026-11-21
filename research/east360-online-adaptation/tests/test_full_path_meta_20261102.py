from types import SimpleNamespace

import pytest
import torch

from experiments.risk_aligned_online.full_path_meta_20261102 import core
from experiments.risk_aligned_online.innovation_metric_20261102 import core as truncated


def fixture(monkeypatch):
    model = SimpleNamespace(
        base=SimpleNamespace(
            center=torch.zeros(23, dtype=torch.float64),
            scale=torch.ones(23, dtype=torch.float64),
        )
    )
    pack = dict(
        phi=torch.eye(4, dtype=torch.float64).repeat(4, 1)[None],
        valid=torch.ones(1, 17, dtype=torch.bool),
    )
    t = torch.linspace(0, 6, 25, dtype=torch.float64).reshape(1, 25, 1)
    sine, cosine = t.sin().expand(1, 25, 23), t.cos().expand(1, 25, 23)

    def auxiliary(_model, _pack, state, b):
        return sine * (state[0, 0] + b[0, 0]) + cosine * (state[0, 1] + b[0, 1])

    monkeypatch.setattr(core, "auxiliary", auxiliary)
    monkeypatch.setattr(truncated, "auxiliary", auxiliary)
    truth = [(1.5 * sine + 0.4 * cosine).sinh(), (1.7 * sine + 0.2 * cosine).sinh()]
    return model, pack, truth, torch.ones_like(truth[0], dtype=torch.bool)


def rollout(data, b, *, graph, full=True):
    model, pack, truths, mask = data
    state = torch.zeros(4, 64, dtype=torch.float64)
    receipts = []
    for truth in truths:
        if full:
            state, receipt = core.advance(
                model, pack, state, truth, mask, b, create_graph=graph
            )
        else:
            state, receipt = truncated.propose(
                model,
                pack,
                state,
                truth,
                truth,
                mask,
                b,
                geometry="metric",
                create_graph=graph,
            )
        receipts.append(receipt)
    return state, receipts


def test_full_path_is_exact_same_deployed_numeric_updates(monkeypatch):
    data = fixture(monkeypatch)
    b = torch.full((25, 23), 0.1, dtype=torch.float64, requires_grad=True)
    full, receipts = rollout(data, b, graph=True)
    old, original = rollout(data, b, graph=True, full=False)
    assert torch.equal(full, old)
    assert receipts == original
    assert all(r["accepted"] for r in receipts)


def test_full_path_gradient_matches_finite_difference_and_differs_from_tbptt1(
    monkeypatch,
):
    data = fixture(monkeypatch)
    b = torch.full((25, 23), 0.1, dtype=torch.float64, requires_grad=True)
    full, _ = rollout(data, b, graph=True)
    old, _ = rollout(data, b, graph=True, full=False)
    (gf,) = torch.autograd.grad(full[0, 0] + full[0, 1] * 0.3, b)
    (gt,) = torch.autograd.grad(old[0, 0] + old[0, 1] * 0.3, b)
    direction = torch.zeros_like(b)
    direction[0, 0], direction[0, 1] = 0.7, -0.2
    values = []
    for sign in (-1, 1):
        state, receipts = rollout(
            data, b.detach() + sign * 1e-5 * direction, graph=False
        )
        assert all(r["accepted"] for r in receipts)
        values.append(float(state[0, 0] + 0.3 * state[0, 1]))
    finite_difference = (values[1] - values[0]) / 2e-5
    assert float((gf * direction).sum()) == pytest.approx(
        finite_difference, rel=1e-6, abs=1e-8
    )
    assert abs(float(((gf - gt) * direction).sum())) > 1e-4


def test_non_graph_mode_drops_history_and_empty_target_preserves_numeric_state(
    monkeypatch,
):
    model, pack, truths, mask = fixture(monkeypatch)
    b = torch.full((25, 23), 0.1, dtype=torch.float64, requires_grad=True)
    state, _ = rollout((model, pack, truths, mask), b, graph=False)
    assert not state.requires_grad
    after, receipt = core.advance(
        model, pack, state, truths[0], torch.zeros_like(mask), b, create_graph=True
    )
    assert torch.equal(after, state)
    assert not receipt["accepted"]
    assert receipt["reason"] == "no_target"


def test_empty_target_does_not_erase_earlier_meta_path():
    b = torch.ones(25, 23, requires_grad=True)
    state = torch.ones(4, 64) * b[0, 0]
    truth = torch.zeros(1, 25, 23)
    after, _ = core.advance(
        None,
        {},
        state,
        truth,
        torch.zeros_like(truth, dtype=torch.bool),
        b,
        create_graph=True,
    )
    (gradient,) = torch.autograd.grad(after.sum(), b)
    assert gradient[0, 0] == 256
