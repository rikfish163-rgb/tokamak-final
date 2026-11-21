import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.dual_history_ttt_20261101.model import (
    encode_views,
    latent_forward,
)
from experiments.risk_aligned_online.centered_gate_stream_20261101.model import (
    loss,
    step,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23

torch.set_num_threads(2)


@pytest.fixture
def setup():
    torch.manual_seed(744)
    torch.backends.mha.set_fastpath_enabled(False)
    model = (
        ContextAdapter(
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                np.zeros(23),
                np.ones(23),
            )
        )
        .eval()
        .requires_grad_(False)
    )
    rng = np.random.default_rng(44)
    signal = dict(
        values=rng.normal(size=(416, 23)).astype(np.float32),
        available=np.ones((416, 23), bool),
    )
    return model, signal


def test_loss_removes_physical_file_level_in_source_asinh_coordinates(setup):
    model, _ = setup
    p = torch.randn(1, 25, 23, requires_grad=True)
    y = torch.randn_like(p)
    mask = torch.ones_like(p, dtype=torch.bool)
    original = loss(model, p, y.sinh(), mask)
    delta = torch.randn(1, 1, 23)
    shifted = loss(model, p + delta, (y + delta * 2).sinh(), mask)
    assert torch.allclose(original, shifted, atol=1e-6, rtol=0)
    g = torch.autograd.grad(original, p)[0]
    assert torch.allclose(g.sum(1), torch.zeros(1, 23), atol=1e-7, rtol=0)


def test_step_acceptance_matches_centered_objective_and_changes_only_gates(setup):
    model, signal = setup
    pack = encode_views(model, signal, [300])
    beta = torch.full((16, 2), 0.2)
    truth = torch.as_tensor(signal["values"][301:326][None])
    mask = torch.ones_like(truth, dtype=torch.bool)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    after, r = step(model, pack, beta, truth, mask, step_size=0.01)
    assert r["gradient_norm"] > 0 and r["loss_retained"] <= r["loss_before"]
    assert r["accepted"] == (r["loss_proposal"] < r["loss_before"])
    assert (after >= 0).all() and (after.sum(-1) <= 1).all()
    with torch.no_grad():
        predicted = latent_forward(model, pack, after, forecast=True)["forecast"]
        objective = loss(model, predicted, truth, mask) + 0.01 * after.square().mean()
    assert float(objective) == r["loss_retained"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_no_observation_and_equal_candidate_cannot_create_update(setup):
    model, signal = setup
    pack = encode_views(model, signal, [300])
    beta = torch.zeros(16, 2)
    truth = torch.zeros(1, 25, 23)
    mask = torch.zeros_like(truth, dtype=torch.bool)
    after, receipt = step(model, pack, beta, truth, mask)
    assert torch.equal(after, beta) and not receipt["accepted"]
    assert receipt["observed"] == 0 and receipt["loss_before"] is None
    after, receipt = step(model, pack, beta, truth, ~mask, step_size=0)
    assert torch.equal(after, beta) and not receipt["accepted"]
    truth[:] = float("nan")
    with pytest.raises(ValueError):
        step(model, pack, beta, truth, ~mask)


def test_centered_stream_clock_reset_and_future_prefix_invariance(setup):
    from experiments.risk_aligned_online.centered_gate_stream_20261101.run import (
        predict,
    )

    model, signal = setup
    full, receipt = predict(model, signal)
    prefix, prefix_receipt = predict(model, {k: v[:320] for k, v in signal.items()})
    altered = {k: v.copy() for k, v in signal.items()}
    altered["values"][160:] += 1000
    altered["available"][160:] = ~altered["available"][160:]
    changed, changed_receipt = predict(model, altered)
    for a in full:
        assert np.array_equal(full[a][:320], prefix[a])
        assert np.array_equal(full[a][:160], changed[a][:160])
    for a in receipt["updates"]:
        events = receipt["updates"][a]
        assert [r["target_last"] for r in events] == [159, 287, 415]
        assert [r["first_affected"] for r in events] == [160, 288, 416]
        assert events[:2] == prefix_receipt["updates"][a]
        assert events[0] == changed_receipt["updates"][a][0]
        assert np.array_equal(full[a][:160], full["cold"][:160])
    repeat, repeat_receipt = predict(model, signal)
    assert all(np.array_equal(repeat[a], full[a]) for a in full)
    assert repeat_receipt == receipt
