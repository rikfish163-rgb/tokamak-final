"""Synthetic contracts for the sealed-shot measurement decomposition."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import joint_lora_measurement_probe_20261104 as probe
from experiments.risk_aligned_online.joint_lora_20261104.timeline import _targets


def test_candidates_use_deployment_origins_and_uniform_indices():
    times = np.arange(400, dtype=np.float64) / 1000
    rows = probe.select_cases(times, False, None)
    assert [r["origin"] for r in rows] == [127, 223, 351]
    assert [r["future"] for r in rows] == [159, 255, 383]
    assert all(r["arrival"] == r["origin"] + 8 for r in rows)
    assert all(r["group"] == "safe" and r["label"] == 0 for r in rows)
    assert probe.select_cases(times[:159], False, None) == []


def test_positive_lead_is_strictly_above_30ms_and_future_preterminal():
    times = np.arange(192, dtype=np.float64) / 1000
    assert probe.select_cases(times, True, times[159] + 0.030) == []
    rows = probe.select_cases(times, True, times[159] + 0.031)
    assert len(rows) == 1 and rows[0]["origin"] == 127
    assert rows[0]["group"] == "positive" and rows[0]["lead_ms"] == pytest.approx(31)
    rows = probe.select_cases(times, True, times[159] + 0.126)
    assert rows[0]["group"] == "earlier" and rows[0]["label"] == 0
    with pytest.raises(ValueError, match="clock"):
        probe.select_cases(np.array([0.0, 0.2, 0.1]), False, None)


def test_real_hold_motion_and_reverse_gradient_identities():
    prediction = torch.tensor([1.0, -2.0, 3.0, -7.0], requires_grad=True)
    target = torch.tensor([0.2, 0.4, -0.5, 100.0])
    mask = torch.tensor([True, True, True, False])
    losses = probe.measurement_losses(prediction, target, mask)
    grads = {
        name: torch.autograd.grad(value, prediction, retain_graph=True)[0]
        for name, value in losses.items()
    }
    torch.testing.assert_close(grads["real"], grads["hold"] + grads["motion"])
    torch.testing.assert_close(grads["reverse"], grads["hold"] - grads["motion"])
    torch.testing.assert_close(grads["anti_motion"], -grads["motion"])
    assert grads["real"][-1] == 0
    assert probe.measurement_losses(prediction, target, torch.zeros_like(mask)) is None


def test_mature_target_is_real_increment_and_cannot_see_later_samples():
    values = np.arange(170 * 23, dtype=np.float32).reshape(170, 23) / 1000
    availability = np.ones_like(values, dtype=bool)
    availability[130, 3] = False
    shot = dict(values=values, available=availability)
    model = SimpleNamespace(clone=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23)))
    _, _, target, observed = _targets(model, shot, 127, device="cpu")
    channels = [c for c in range(23) if c not in (2, 10)]
    expected = np.arcsinh(values[128:136, channels]) - np.arcsinh(values[127, channels])
    mask = availability[128:136, channels] & availability[127, channels]
    np.testing.assert_allclose(target.numpy().reshape(8, 21), np.where(mask, expected, 0), rtol=2e-6, atol=3e-7)
    np.testing.assert_array_equal(observed.numpy().reshape(8, 21), mask)
    values[136:] = -123456
    _, _, after, after_mask = _targets(model, shot, 127, device="cpu")
    torch.testing.assert_close(after, target, rtol=0, atol=0)
    assert torch.equal(observed, after_mask)


def test_actual_adam_step_clips_and_has_fresh_zero_moments():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.4, 0.1]))
    initial = parameter.detach().clone()
    gradient = torch.tensor([4.0, -3.0, 0.5])
    norm, displacement = probe.step_from_gradients((parameter,), (initial,), (gradient,))
    assert norm == pytest.approx(float(gradient.norm()))
    expected = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([expected], lr=1e-4)
    expected.grad = gradient.clone()
    torch.nn.utils.clip_grad_norm_([expected], 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert 0 < displacement < 0.5
    with torch.no_grad():
        parameter.copy_(initial)
    probe.step_from_gradients((parameter,), (initial,), (gradient,))
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_seal_validation_rejects_wrong_curve_forecast_or_parent():
    seal = dict(shot=123, curve_sha256="curve", forecast_sha256="forecast",
                current_outcome_used_in_prediction=False,
                shot_start_parents={"static": "warm"}, same_slow_for_inter_and_both=True)
    probe.validate_seal(seal, 123, "curve", "forecast", "warm")
    for curve, forecast, warm in (("bad", "forecast", "warm"), ("curve", "bad", "warm"), ("curve", "forecast", "bad")):
        with pytest.raises(ValueError, match="seal"):
            probe.validate_seal(seal, 123, curve, forecast, warm)
    seal["current_outcome_used_in_prediction"] = True
    with pytest.raises(ValueError, match="seal"):
        probe.validate_seal(seal, 123, "curve", "forecast", "warm")


def test_group_summary_uses_shot_equal_weight_and_keeps_noops():
    def row(sid, real, hold, support):
        return dict(shot=sid, group="safe", supported=support,
                    arms={name: {"bce": val, "bce_centered": val, "prob": 0.5}
                          for name, val in {"none": 1, "real": real, "hold": hold,
                                            "reverse": 1, "motion": 1, "anti_motion": 1}.items()})
    rows = [row(1, 0.8, 1.0, 1), row(1, 0.8, 1.0, 1), row(2, 1.0, 1.0, 0)]
    summary = probe.summarize(rows, threshold=0.5)["groups"]["safe"]
    assert summary["shots"] == 2 and summary["cases"] == 3 and summary["no_support_cases"] == 1
    pair = summary["unshifted_bce"]["real_vs_none"]
    assert pair["mean_delta"] == pytest.approx(-0.1)
    assert (pair["improved_shots"], pair["worse_shots"], pair["tied_shots"]) == (1, 0, 1)


def test_unsupported_case_is_retained_with_identical_noop_arms(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(512))

        def neural_parameters(self):
            return (self.p,)

        def forward(self, x, available):
            return {"diagnostic": self.p[0].expand(1, 168), "logit": self.p[0].reshape(1)}

    x = torch.zeros(1, 128, 23)
    a = torch.zeros_like(x, dtype=torch.bool)
    y = torch.zeros(1, 168)
    mask = torch.zeros_like(y, dtype=torch.bool)
    monkeypatch.setattr(probe, "_targets", lambda *args, **kwargs: (x, a, y, mask))
    monkeypatch.setattr(probe, "window_batch", lambda *args, **kwargs: (x, a))
    forecast = {"origin": np.array([127]), "released_at": np.array([135]),
                "prediction": np.zeros((1, 168), np.float32),
                "target": np.zeros((1, 168), np.float32), "mask": np.zeros((1, 168), bool)}
    case = dict(origin=127, future=159, label=0, group="safe", arrival=135, lead_ms=None)
    result = probe.evaluate_case(Model(), {}, case, forecast, np.full(160, 0.5), 0.5)
    assert result["no_op"] and result["supported"] == 0
    assert result["gradient_decomposition"] is None
    assert all(result["arms"][name] == result["arms"]["none"] for name in probe.ARMS)
