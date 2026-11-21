import json

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.causal_span_20261101.core import span_windows
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import (
    batch,
)
from experiments.risk_aligned_online.dual_history_ttt_20261101.model import (
    encode_views,
    latent_forward,
    project_gates,
    step,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.window_history_diagnostic_20261101.core import (
    history_view,
)

torch.set_num_threads(2)


@pytest.fixture
def model():
    torch.manual_seed(431)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(
        Config(fast_site="tokens", fast_rule="normalized"),
        np.zeros(23),
        np.ones(23),
    )
    adapter = ContextAdapter(base).eval().requires_grad_(False)
    with torch.no_grad():
        adapter.slow.weight.copy_(torch.randn_like(adapter.slow.weight) * 0.02)
        adapter.base.fast_origin.copy_(torch.randn_like(adapter.base.fast_origin) * 0.2)
        adapter.memory.output.weight.copy_(
            torch.randn_like(adapter.memory.output.weight) * 0.02
        )
    return adapter


@pytest.fixture
def signal():
    rng = np.random.default_rng(1234)
    values = rng.normal(size=(360, 23)).astype(np.float32)
    values += np.linspace(-1, 1, 360, dtype=np.float32)[:, None]
    available = rng.random(values.shape) > 0.1
    return {"values": values, "available": available}


def test_projection_exact_corners_and_euclidean_boundary():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [-2.0, -3.0],
            [0.2, 0.3],
            [1.5, 0.1],
            [0.8, 0.7],
            [-2.0, 2.0],
            [0.0, 1.0],
        ]
    )
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.3],
            [1.0, 0.0],
            [0.55, 0.45],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    actual = project_gates(points)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert torch.equal(actual[[0, 1, 3, 5, 6]], expected[[0, 1, 3, 5, 6]])
    assert torch.equal(project_gates(actual), actual)


def test_projection_random_feasible_idempotent_and_nonfinite_rejected():
    torch.manual_seed(7)
    points = torch.randn(1000, 2) * 5
    projected = project_gates(points)
    assert (projected >= 0).all() and (projected.sum(-1) <= 1).all()
    assert torch.equal(project_gates(projected), projected)
    multiplier = (points - projected).amax(-1, keepdim=True).clamp_min(0)
    residual = points - projected - multiplier
    assert torch.allclose(
        residual[projected > 0], torch.zeros_like(residual[projected > 0]), atol=1e-6
    )
    assert (residual[projected == 0] <= 1e-6).all()
    assert torch.allclose(
        multiplier.squeeze(-1) * (1 - projected.sum(-1)), torch.zeros(1000), atol=1e-6
    )
    with pytest.raises((ValueError, FloatingPointError)):
        project_gates(torch.tensor([[float("nan"), 0.0]]))


def test_encode_views_uses_exact_existing_views_and_validity(model, signal):
    ends = [0, 15, 127, 253, 254, 300]
    pack = encode_views(model, signal, ends)
    assert set(pack) == {"h", "valid", "long", "snapshot"}
    assert pack["h"].shape == (6, 17, 64)
    assert pack["valid"].shape == (6, 17)
    assert pack["long"].shape == pack["snapshot"].shape == (6, 16, 64)
    assert all(not value.requires_grad for value in pack.values())
    with torch.no_grad():
        x, available = batch(signal, ends, model.base.center.device)
        short, valid = model.tokens(x, available)
        long_x, long_a, eligible = span_windows(signal, ends, 2)
        long_h, long_valid = model.tokens(
            torch.as_tensor(long_x), torch.as_tensor(long_a)
        )
        snap_x, snap_a = history_view(x, available, ends, "last_observed")
        snap_h, snap_valid = model.tokens(snap_x, snap_a)
        expected_long = torch.where(
            (valid[:, :16] & long_valid[:, :16])[..., None],
            long_h[:, :16] - short[:, :16],
            0.0,
        )
        expected_snap = torch.where(
            (valid[:, :16] & snap_valid[:, :16])[..., None],
            snap_h[:, :16] - short[:, :16],
            0.0,
        )
    assert eligible.tolist() == [False, False, False, False, True, True]
    assert torch.equal(pack["h"], short) and torch.equal(pack["valid"], valid)
    assert torch.equal(pack["long"], expected_long)
    assert torch.equal(pack["snapshot"], expected_snap)
    assert not pack["long"][:4].any()
    assert pack["long"][4:].abs().sum() > 0


def test_zero_gate_is_exact_cold_and_forecast_matches_source(model, signal):
    ends = [0, 23, 127, 253, 254, 300]
    pack = encode_views(model, signal, ends)
    beta = torch.zeros(16, 2)
    with torch.no_grad():
        out = latent_forward(model, pack, beta, forecast=True)
        cold = model.risk(pack["h"], pack["valid"], torch.zeros(len(ends), 92))
        assert torch.equal(out["logit"], cold)
        assert not out["shift"].any()
        assert latent_forward(model, pack, beta)["forecast"] is None
        model.slow.weight.zero_()
        x, available = batch(signal, ends, model.base.center.device)
        with sdpa_kernel(SDPBackend.MATH):
            source = model.base(x, available)
        out = latent_forward(model, pack, beta, forecast=True)
        assert torch.equal(out["forecast"], source["forecast"])
        assert torch.equal(out["logit"], source["logit"])


def test_gate_gradients_are_nonzero_and_frozen_weights_are_unchanged(model, signal):
    pack = encode_views(model, signal, [270, 300])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    beta = torch.zeros(16, 2, requires_grad=True)
    output = latent_forward(model, pack, beta, forecast=True)
    (output["forecast"].square().sum() + output["logit"].square().sum()).backward()
    assert beta.grad is not None and torch.isfinite(beta.grad).all()
    assert beta.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


def test_shifts_vary_by_token_and_slow_acts_after_the_shift(model, signal):
    pack = encode_views(model, signal, [300])
    beta = torch.zeros(16, 2)
    beta[:, 0] = torch.linspace(0.05, 0.8, 16)
    beta[:, 1] = 0.1
    out = latent_forward(model, pack, beta, forecast=True)
    expected_shift = (
        beta[None, :, :1] * pack["long"] + beta[None, :, 1:] * pack["snapshot"]
    )
    assert torch.equal(out["shift"], expected_shift)
    assert not torch.allclose(out["shift"][:, 0], out["shift"][:, -1])
    with torch.no_grad():
        data = pack["h"][:, :16] + expected_shift
        last = data[:, -1] + model.slow(data[:, -1]) * pack["valid"][:, -2, None]
        hidden = torch.cat((data[:, :-1], last[:, None], pack["h"][:, -1:]), 1)
        with sdpa_kernel(SDPBackend.MATH):
            hidden = model.base.encoder.layers[1](
                hidden, src_key_padding_mask=~pack["valid"]
            )
        gain, bias = model.base.initial_fast(len(hidden)).chunk(2, -1)
        hidden = model.base.norm(hidden) * (1 + 0.1 * gain[:, None].tanh())
        hidden = hidden + 0.1 * bias[:, None].tanh()
        risk = model.base.risk_head(hidden[:, -1]).squeeze(-1)
        hidden = hidden.masked_fill(~pack["valid"][:, :, None], 0.0)
        forecast = model.base.forecast_head(hidden.flatten(1)).reshape(1, 25, 23)
    assert torch.equal(out["logit"], risk)
    assert torch.equal(out["forecast"], forecast)


def test_absent_long_support_cannot_write(model, signal):
    signal["available"][46:61:2] = False
    pack = encode_views(model, signal, [300])
    assert pack["valid"][0, 0]
    assert not pack["long"][0, 0].any()
    beta = torch.zeros(16, 2, requires_grad=True)
    output = latent_forward(model, pack, beta, forecast=True)
    gradient = torch.autograd.grad(output["forecast"].sum(), beta)[0]
    assert gradient[0, 0] == 0
    with torch.no_grad():
        beta[0, 0] = 1
    assert not latent_forward(model, pack, beta)["shift"].any()


@pytest.mark.parametrize("end", [0, 15, 127, 253, 254, 300])
def test_all_views_are_exactly_prefix_invariant_for_same_end_batch(model, signal, end):
    ends = [max(0, end - 1), end]
    altered = {key: value.copy() for key, value in signal.items()}
    altered["values"][end + 1 :] += 10000
    altered["available"][end + 1 :] = ~altered["available"][end + 1 :]
    truncated = {key: value[: end + 1].copy() for key, value in signal.items()}
    baseline = encode_views(model, signal, ends)
    beta = torch.full((16, 2), 0.2)
    expected = latent_forward(model, baseline, beta, forecast=True)
    for changed in (altered, truncated):
        actual = encode_views(model, changed, ends)
        assert all(torch.equal(actual[key], baseline[key]) for key in baseline)
        output = latent_forward(model, actual, beta, forecast=True)
        assert all(torch.equal(output[key], expected[key]) for key in expected)


def test_step_uses_only_observed_targets_and_accepts_only_improvement(model, signal):
    pack = encode_views(model, signal, [270, 300])
    beta = torch.full((16, 2), 0.2)
    torch.manual_seed(44)
    truth = torch.randn(2, 25, 23)
    mask = torch.rand(2, 25, 23) > 0.3
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    updated, receipt = step(model, pack, beta, truth, mask, step_size=0.01)
    assert receipt["observed"] == int(mask.sum())
    assert receipt["gradient_norm"] > 0
    assert len(receipt["gradient32"]) == len(receipt["beta32_after"]) == 32
    assert not updated.requires_grad and (updated >= 0).all()
    assert (updated.sum(-1) <= 1).all()
    assert receipt["accepted"]
    assert receipt["loss_proposal"] < receipt["loss_before"]
    assert receipt["loss_retained"] == receipt["loss_proposal"]
    with torch.no_grad():
        prediction = latent_forward(model, pack, updated, forecast=True)["forecast"]
        objective = (
            model.base.forecast_loss(prediction, truth, mask)
            + 0.01 * updated.square().mean()
        )
    assert receipt["loss_retained"] == float(objective)
    assert all(
        torch.equal(weights[name], p) and p.grad is None
        for name, p in model.named_parameters()
    )
    changed = truth.clone()
    changed[~mask] = 12345
    other, other_receipt = step(model, pack, beta, changed, mask, step_size=0.01)
    assert torch.equal(other, updated) and other_receipt == receipt
    json.dumps(receipt, allow_nan=False)


def test_step_rejects_equal_candidate_and_abstains_without_targets(model, signal):
    pack = encode_views(model, signal, [300])
    beta = torch.full((16, 2), 0.2)
    truth = torch.zeros(1, 25, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    retained, receipt = step(model, pack, beta, truth, mask, step_size=0)
    assert not receipt["accepted"] and torch.equal(retained, beta)
    assert (
        receipt["loss_retained"] == receipt["loss_before"] == receipt["loss_proposal"]
    )
    truth.fill_(float("nan"))  # Missing targets are never interpreted as zero labels.
    retained, receipt = step(model, pack, beta, truth, torch.zeros_like(mask))
    assert not receipt["accepted"] and receipt["observed"] == 0
    assert receipt["reason"] == "no_target" and torch.equal(retained, beta)
    assert (
        receipt["loss_before"]
        is receipt["loss_proposal"]
        is receipt["loss_retained"]
        is None
    )
    assert receipt["gradient_norm"] == receipt["step_norm"] == 0
    assert len(receipt["gradient32"]) == len(receipt["beta32_after"]) == 32
    json.dumps(receipt, allow_nan=False)


def test_step_rejects_a_strictly_worse_proposal(model, signal):
    pack = encode_views(model, signal, [300])
    beta = torch.full((16, 2), 0.2)
    desired = beta.clone()
    desired[5, 0] += 0.05
    with torch.no_grad():
        target = latent_forward(model, pack, desired, forecast=True)["forecast"].sinh()
    retained, receipt = step(
        model,
        pack,
        beta,
        target,
        torch.ones_like(target, dtype=torch.bool),
        step_size=10,
        anchor=0,
    )
    assert receipt["gradient_norm"] > 0
    assert receipt["loss_proposal"] > receipt["loss_before"]
    assert receipt["loss_retained"] == receipt["loss_before"]
    assert not receipt["accepted"] and torch.equal(retained, beta)


def test_zero_gradient_retains_beta_even_inside_no_grad(model, signal):
    pack = encode_views(model, signal, [0])
    assert not pack["long"].any() and not pack["snapshot"].any()
    beta = torch.zeros(16, 2)
    truth = torch.zeros(1, 25, 23)
    with torch.no_grad():
        retained, receipt = step(
            model, pack, beta, truth, torch.ones_like(truth, dtype=torch.bool)
        )
    assert receipt["gradient_computed"] and receipt["gradient_norm"] == 0
    assert receipt["gradient32"] == [0.0] * 32
    assert not receipt["accepted"] and torch.equal(retained, beta)


def test_unfrozen_model_or_fused_attention_fails_visibly(model, signal):
    model.slow.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        encode_views(model, signal, [300])
    model.slow.weight.requires_grad_(False)
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        with pytest.raises((RuntimeError, ValueError), match="fastpath"):
            encode_views(model, signal, [300])
    finally:
        torch.backends.mha.set_fastpath_enabled(False)


def test_step_fails_visibly_for_nonfinite_observed_data(model, signal):
    pack = encode_views(model, signal, [300])
    beta = torch.zeros(16, 2)
    truth = torch.zeros(1, 25, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    truth[0, 0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        step(model, pack, beta, truth, mask)


def test_cpu_runner_keeps_batched_prefix_and_feedback_receipts(model, signal):
    from experiments.risk_aligned_online.dual_history_ttt_20261101.run import (
        ARMS,
        LEARNERS,
        predict,
    )

    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    curves, info = predict(model, signal)
    prefix_signal = {key: value[:288].copy() for key, value in signal.items()}
    prefix_curves, prefix_info = predict(model, prefix_signal)
    for arm in ARMS:
        assert len(curves[arm]) == 360
        assert np.array_equal(curves[arm][:288], prefix_curves[arm])
    for arm in LEARNERS:
        assert np.array_equal(curves[arm][:160], curves["cold"][:160])
        receipts = info["updates"][arm]
        assert [r["origin"] for r in receipts] == [134, 262]
        assert [r["target_last"] for r in receipts] == [159, 287]
        assert [r["first_affected"] for r in receipts] == [160, 288]
        assert receipts == prefix_info["updates"][arm]
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in state.items()
    )
