from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.timeline import run_shot
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_backend():
    old_mha = torch.backends.mha.get_fastpath_enabled()
    old_threads = torch.get_num_threads()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(1)
    yield
    torch.backends.mha.set_fastpath_enabled(old_mha)
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def trained_shape_model():
    base, _ = load_parent(DEFAULT_PARENT)
    generator = torch.Generator().manual_seed(61104)
    w0 = torch.randn(168, 70, generator=generator) * 0.02
    model = JointLoRA(base, w0)
    with torch.no_grad():
        model.residual_decoder.weight.copy_(
            torch.randn(168, 192, generator=generator) * 0.03
        )
        model.residual_decoder.bias.copy_(torch.randn(168, generator=generator) * 0.01)
    # Match the finished ended-shot training API without performing a fit.
    return model.eval().requires_grad_(False)


def shot_for(model, n=175):
    rng = np.random.default_rng(103)
    center = model.clone.center.detach().numpy()
    scale = model.clone.scale.detach().numpy()
    values = (center + scale * rng.normal(0, 0.3, (n, 23))).astype(np.float32)
    available = np.ones((n, 23), dtype=bool)
    available[:, 2] = False
    available[::4, 1] = False
    return {"values": values, "available": available}


def snapshot(model):
    return {
        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "flags": {k: p.requires_grad for k, p in model.named_parameters()},
        "grad": {
            k: None if p.grad is None else p.grad.detach().clone()
            for k, p in model.named_parameters()
        },
        "training": {k: m.training for k, m in model.named_modules()},
    }


def assert_unchanged(model, old):
    for name, value in model.state_dict().items():
        assert torch.equal(value, old["state"][name])
    for name, p in model.named_parameters():
        assert p.requires_grad == old["flags"][name]
        assert (
            (p.grad is None)
            if old["grad"][name] is None
            else torch.equal(p.grad, old["grad"][name])
        )
    assert {k: m.training for k, m in model.named_modules()} == old["training"]


def test_all_arms_publish_same_targets_and_masks_and_only_none_has_no_updates(
    trained_shape_model,
):
    shot = shot_for(trained_shape_model)
    result = {
        mode: run_shot(trained_shape_model, shot, mode, device="cpu")
        for mode in ("none", "real", "hold", "reverse")
    }
    for mode, (probabilities, history, forecasts) in result.items():
        assert probabilities.shape == (175,) and probabilities.dtype == np.float32
        assert np.isfinite(probabilities).all()
        np.testing.assert_array_equal(forecasts["origin"], [127, 159])
        np.testing.assert_array_equal(forecasts["released_at"], [135, 167])
        assert forecasts["prediction"].shape == forecasts["target"].shape == (2, 168)
        assert (
            forecasts["mask"].shape == (2, 168) and forecasts["mask"].dtype == np.bool_
        )
        assert [row["arrival"] for row in history] == [135, 167]
        assert [row["origin"] for row in history] == [127, 159]
        assert all(row["supported"] > 0 for row in history)
        assert all(
            set(
                (
                    "loss_before_actual",
                    "postloss_actual",
                    "hold_mse",
                    "gradient_norm",
                    "parameter_displacement",
                    "updated",
                    "supported",
                )
            )
            <= set(row)
            for row in history
        )
        assert (
            all(not row["updated"] for row in history)
            if mode == "none"
            else any(row["updated"] for row in history)
        )
        np.testing.assert_array_equal(forecasts["mask"], result["none"][2]["mask"])
        np.testing.assert_array_equal(forecasts["target"], result["none"][2]["target"])
        np.testing.assert_array_equal(
            forecasts["prediction"][0], result["none"][2]["prediction"][0]
        )
        np.testing.assert_array_equal(probabilities[:135], result["none"][0][:135])
    assert not np.array_equal(result["real"][0][135:], result["none"][0][135:])


def test_prefix_future_causality_and_fixed_batch_padding(trained_shape_model):
    long_shot = shot_for(trained_shape_model, 175)
    cutoff = 150
    short_shot = {k: v[:cutoff].copy() for k, v in long_shot.items()}
    altered = deepcopy(long_shot)
    altered["values"][cutoff:] *= np.float32(1.7)
    for mode in ("none", "real", "hold", "reverse"):
        short = run_shot(trained_shape_model, short_shot, mode, device="cpu")
        full = run_shot(trained_shape_model, long_shot, mode, device="cpu")
        changed = run_shot(trained_shape_model, altered, mode, device="cpu")
        np.testing.assert_array_equal(short[0], full[0][:cutoff])
        np.testing.assert_array_equal(short[0], changed[0][:cutoff])
        np.testing.assert_array_equal(short[2]["prediction"], full[2]["prediction"][:1])
        assert short[1] == full[1][:1] == changed[1][:1]


def test_arrival_update_precedes_risk_at_that_index(trained_shape_model):
    shot = shot_for(trained_shape_model, 137)
    no_update = run_shot(trained_shape_model, shot, "none", device="cpu", lr=0.05)
    real = run_shot(trained_shape_model, shot, "real", device="cpu", lr=0.05)
    np.testing.assert_array_equal(no_update[0][:135], real[0][:135])
    assert real[1][0]["arrival"] == 135 and real[1][0]["updated"]
    assert real[0][135] != no_update[0][135]


@pytest.mark.parametrize(
    "n,origins,released",
    [
        (0, [], []),
        (127, [], []),
        (128, [127], [-1]),
        (135, [127], [-1]),
        (136, [127], [135]),
        (160, [127, 159], [135, -1]),
    ],
)
def test_short_shots_and_pending_forecasts(trained_shape_model, n, origins, released):
    probabilities, history, forecasts = run_shot(
        trained_shape_model, shot_for(trained_shape_model, n), "real", device="cpu"
    )
    assert probabilities.shape == (n,)
    np.testing.assert_array_equal(forecasts["origin"], origins)
    np.testing.assert_array_equal(forecasts["released_at"], released)
    assert len(history) == sum(i >= 0 for i in released)
    for index, arrival in enumerate(released):
        if arrival < 0:
            assert np.isnan(forecasts["target"][index]).all()
            assert not forecasts["mask"][index].any()


def test_unsupported_target_never_updates_or_reports_fake_loss(trained_shape_model):
    shot = shot_for(trained_shape_model, 170)
    shot["available"][[127, 159], :] = False
    none = run_shot(trained_shape_model, shot, "none", device="cpu")
    for mode in ("real", "hold", "reverse"):
        probabilities, history, forecasts = run_shot(
            trained_shape_model, shot, mode, device="cpu"
        )
        np.testing.assert_array_equal(probabilities, none[0])
        assert all(row["supported"] == 0 and not row["updated"] for row in history)
        assert all(
            row["loss_before_actual"] is None
            and row["postloss_actual"] is None
            and row["hold_mse"] is None
            for row in history
        )
        assert not forecasts["mask"].any()


def test_local_copy_reset_input_and_shot_immutability(trained_shape_model):
    model = deepcopy(trained_shape_model)
    model.clone.risk_head.bias.grad = torch.full_like(model.clone.risk_head.bias, 0.125)
    old_model = snapshot(model)
    shot = shot_for(model)
    old_shot = deepcopy(shot)
    first = run_shot(model, shot, "real", device="cpu")
    second = run_shot(model, shot, "real", device="cpu")
    assert_unchanged(model, old_model)
    for key in shot:
        np.testing.assert_array_equal(shot[key], old_shot[key])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    for name in first[2]:
        np.testing.assert_array_equal(first[2][name], second[2][name])


def test_decoder_frozen_risk_head_frozen_and_lora_only_optimizer(
    trained_shape_model, monkeypatch
):
    seen = []
    real_forward = JointLoRA.forward
    original_step = torch.optim.Adam.step

    def checked_forward(self, *args, **kwargs):
        assert self is not trained_shape_model
        assert len(args[0]) in (1, 32)
        assert all(not p.requires_grad for p in self.decoder_parameters())
        assert not self.W0.requires_grad
        assert all(not p.requires_grad for p in self.clone.risk_head.parameters())
        assert all(not p.requires_grad for p in self.clone.forecast_head.parameters())
        seen.append(True)
        return real_forward(self, *args, **kwargs)

    def checked_step(self, *args, **kwargs):
        params = [p for group in self.param_groups for p in group["params"]]
        assert sum(p.numel() for p in params) == 512
        assert all(p.requires_grad for p in params)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(JointLoRA, "forward", checked_forward)
    monkeypatch.setattr(torch.optim.Adam, "step", checked_step)
    _, history, _ = run_shot(
        trained_shape_model, shot_for(trained_shape_model), "real", device="cpu"
    )
    assert seen and any(row["updated"] for row in history)


def test_recorded_prediction_is_issued_value_not_later_recomputation(
    trained_shape_model, monkeypatch
):
    original_forward = JointLoRA.forward
    issued = []

    def varied_single_forward(self, x, a):
        result = original_forward(self, x, a)
        if x.shape[0] == 1:
            result["diagnostic"] = result["diagnostic"] + 0.01 * len(issued)
            issued.append(result["diagnostic"].detach().cpu().numpy().copy()[0])
        return result

    monkeypatch.setattr(JointLoRA, "forward", varied_single_forward)
    _, history, forecast = run_shot(
        trained_shape_model, shot_for(trained_shape_model, 136), "real", device="cpu"
    )
    assert len(issued) >= 2 and history[0]["updated"]
    np.testing.assert_array_equal(forecast["prediction"][0], issued[0])
    assert not np.array_equal(forecast["prediction"][0], issued[1])


def test_history_scores_use_sealed_prediction_and_real_target(trained_shape_model):
    for mode in ("none", "real", "hold", "reverse"):
        _, history, forecast = run_shot(
            trained_shape_model, shot_for(trained_shape_model, 136), mode, device="cpu"
        )
        mask = forecast["mask"][0]
        y = forecast["target"][0, mask].astype(np.float64)
        p = forecast["prediction"][0, mask].astype(np.float64)
        assert history[0]["loss_before_actual"] == pytest.approx(
            np.mean((p - y) ** 2), abs=1e-6
        )
        assert history[0]["hold_mse"] == pytest.approx(np.mean(y**2), abs=1e-6)
        assert history[0]["supported"] == int(mask.sum())
        if mode == "none":
            assert history[0]["postloss_actual"] == history[0]["loss_before_actual"]
            assert history[0]["gradient_norm"] is None


def test_zero_decoder_does_not_report_a_pseudo_update(trained_shape_model):
    model = deepcopy(trained_shape_model)
    with torch.no_grad():
        model.residual_decoder.weight.zero_()
        model.residual_decoder.bias.zero_()
    shot = shot_for(model, 136)
    none = run_shot(model, shot, "none", device="cpu")
    real = run_shot(model, shot, "real", device="cpu")
    assert real[1][0]["supported"] > 0
    assert real[1][0]["gradient_norm"] == 0.0
    assert not real[1][0]["updated"] and real[1][0]["parameter_displacement"] == 0
    np.testing.assert_array_equal(real[0], none[0])


def test_unsupported_event_does_not_advance_an_existing_adam_state(
    trained_shape_model, monkeypatch
):
    shot = shot_for(trained_shape_model, 170)
    shot["available"][159, :] = False
    calls = []
    original_step = torch.optim.Adam.step

    def counted_step(self, *args, **kwargs):
        calls.append(True)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    _, history, _ = run_shot(trained_shape_model, shot, "real", device="cpu")
    assert len(calls) == 1 and history[0]["updated"]
    assert history[1]["supported"] == 0 and not history[1]["updated"]
    assert history[1]["parameter_displacement"] == history[0]["parameter_displacement"]


def test_global_gradient_clipping_before_adam(trained_shape_model, monkeypatch):
    model = deepcopy(trained_shape_model)
    with torch.no_grad():
        model.residual_decoder.weight.mul_(1000.0)
    seen = []
    original_step = torch.optim.Adam.step

    def checked_step(self, *args, **kwargs):
        params = [p for group in self.param_groups for p in group["params"]]
        grad_norm = torch.linalg.vector_norm(
            torch.cat([p.grad.reshape(-1) for p in params])
        ).item()
        seen.append(grad_norm)
        assert grad_norm <= 1.0 + 1e-6
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", checked_step)
    _, history, _ = run_shot(model, shot_for(model, 136), "real", device="cpu")
    assert seen and history[0]["gradient_norm"] > 1.0


def test_nonzero_start_lora_remains_shot_local(trained_shape_model):
    model = deepcopy(trained_shape_model)
    with torch.no_grad():
        lora = model.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[
            0
        ]
        lora.lora_B_q.fill_(0.03)
        lora.lora_B_v.fill_(-0.02)
    before = snapshot(model)
    first = run_shot(model, shot_for(model, 136), "real", device="cpu", radius=0.001)
    second = run_shot(model, shot_for(model, 136), "real", device="cpu", radius=0.001)
    assert first[1][0]["updated"] and first[1][0]["parameter_displacement"] <= 0.001
    np.testing.assert_array_equal(first[0], second[0])
    assert_unchanged(model, before)


def test_radius_is_global_from_shot_start_and_controls_differ(trained_shape_model):
    shot = shot_for(trained_shape_model, 200)
    runs = {
        m: run_shot(trained_shape_model, shot, m, device="cpu", lr=0.1, radius=0.001)
        for m in ("real", "hold", "reverse")
    }
    for _, history, _ in runs.values():
        assert any(row["updated"] for row in history)
        assert all(row["parameter_displacement"] <= 0.001 for row in history)
        assert all(
            row["gradient_norm"] is not None and np.isfinite(row["gradient_norm"])
            for row in history
        )
    assert not np.array_equal(runs["real"][0][135:], runs["reverse"][0][135:])


@pytest.mark.parametrize("mode", ["", "rotated", "REAL", None])
def test_invalid_mode_rejected(trained_shape_model, mode):
    with pytest.raises(ValueError):
        run_shot(
            trained_shape_model, shot_for(trained_shape_model, 1), mode, device="cpu"
        )


@pytest.mark.parametrize(
    "kwargs", [{"lr": 0}, {"lr": float("nan")}, {"radius": 0}, {"radius": float("inf")}]
)
def test_invalid_hyperparameter_rejected(trained_shape_model, kwargs):
    with pytest.raises(ValueError):
        run_shot(
            trained_shape_model,
            shot_for(trained_shape_model, 1),
            "real",
            device="cpu",
            **kwargs,
        )
