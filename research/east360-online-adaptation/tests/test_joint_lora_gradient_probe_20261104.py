import numpy as np
import pytest
import torch
from torch.nn import functional as F

from joint_lora_gradient_probe_20261104 import (
    centered_risk_loss,
    control_losses,
    gradient_metrics,
    select_candidates,
    summarize,
)


def test_safe_coordinates_are_uniform_and_bounded_without_prediction_ranking():
    times = np.arange(400, dtype=np.float64) / 1000.0
    rows = select_candidates(times, False, None)
    candidates = list(range(135, len(times) - 32, 32))
    assert len(rows) == 3
    assert [r["origin"] for r in rows] == [candidates[i] for i in (0, 3, 7)]
    assert all(r["future_risk_point"] == r["origin"] + 32 for r in rows)
    assert all(r["arrival"] == r["origin"] + 8 for r in rows)
    assert all(
        r["group"] == "safe" and r["label"] == 0 and r["actuallead"] is None
        for r in rows
    )


def test_nominal_thirty_ms_is_excluded_even_if_raw_float_is_above_thirty():
    times = np.arange(200, dtype=np.float64) / 1000.0
    down = float(times[167] + 0.03 + 2.5e-16)
    assert (down - times[167]) * 1000 > 30
    assert select_candidates(times, True, down) == []
    rows = select_candidates(times, True, float(times[167] + 0.031))
    assert len(rows) == 1 and rows[0]["origin"] == 135
    assert rows[0]["actuallead"] == pytest.approx(31.0)
    assert rows[0]["label"] == 1 and rows[0]["group"] == "positive"


def test_future_risk_labels_are_125ms_and_preterminal():
    times = np.arange(400, dtype=np.float64) / 1000.0
    rows = select_candidates(times, True, 0.290)
    assert [r["origin"] for r in rows] == [135, 167, 199]
    assert [r["future_risk_point"] for r in rows] == [167, 199, 231]
    assert [r["group"] for r in rows] == ["positive", "positive", "positive"]
    assert all(r["actuallead"] > 30 for r in rows)
    earlier = select_candidates(times, True, 0.390)
    assert earlier[0]["group"] == "earlier" and earlier[0]["label"] == 0
    assert all(r["future_risk_point"] < len(times) for r in earlier)


def test_centered_bce_uses_threshold_logit_not_unshifted_logit():
    logits = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    threshold = 0.7258946895599365
    loss = centered_risk_loss(logits, 1, threshold)
    tau = np.log(threshold / (1 - threshold))
    torch.testing.assert_close(
        loss, F.binary_cross_entropy_with_logits(logits - tau, torch.ones_like(logits))
    )
    assert not torch.equal(
        loss, F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    )


def test_controls_use_one_mask_and_expected_gradients():
    prediction = torch.tensor([1.0, 2.0, 7.0], requires_grad=True)
    actual = torch.tensor([0.5, -1.0, 20.0])
    mask = torch.tensor([True, True, False])
    losses = control_losses(prediction, actual, mask)
    expected_targets = {
        "real": actual,
        "hold": torch.zeros_like(actual),
        "reverse": -actual,
    }
    for i, (mode, value) in enumerate(losses.items()):
        (gradient,) = torch.autograd.grad(value, prediction, retain_graph=i < 2)
        expected = torch.where(mask, prediction.detach() - expected_targets[mode], 0.0)
        torch.testing.assert_close(gradient, expected)
        assert gradient[2] == 0
    assert (
        control_losses(prediction.detach(), actual, torch.zeros(3, dtype=torch.bool))
        is None
    )


def test_alignment_sign_norm_and_zero_are_distinct():
    risk = torch.tensor([3.0, 4.0])
    same = gradient_metrics(torch.tensor([3.0, 4.0]), risk)
    assert same["inner_product"] == pytest.approx(25.0)
    assert same["cosine"] == pytest.approx(1.0)
    assert same["first_order_delta_per_unit_step"] == pytest.approx(-25.0)
    reverse = gradient_metrics(-risk, risk)
    assert reverse["cosine"] == pytest.approx(-1.0)
    assert reverse["first_order_delta_per_unit_step"] == pytest.approx(25.0)
    zero = gradient_metrics(torch.zeros(2), risk)
    assert zero["diag_norm"] == 0 and zero["risk_norm"] == pytest.approx(5.0)
    assert zero["cosine"] is None and zero["first_order_delta_per_unit_step"] == 0


def test_summary_separates_unsupported_zero_risk_and_zero_diag():
    risk = torch.tensor([1.0, 0.0])
    good = gradient_metrics(torch.tensor([1.0, 0.0]), risk)
    wrong = gradient_metrics(torch.tensor([-1.0, 0.0]), risk)
    no_diag = gradient_metrics(torch.zeros(2), risk)
    no_risk = gradient_metrics(torch.tensor([1.0, 0.0]), torch.zeros(2))
    rows = [
        {"group": "safe", "supported": 2, "controls": {"real": good}},
        {"group": "safe", "supported": 2, "controls": {"real": wrong}},
        {"group": "safe", "supported": 2, "controls": {"real": no_diag}},
        {"group": "safe", "supported": 2, "controls": {"real": no_risk}},
        {"group": "safe", "supported": 0, "controls": {"real": None}},
    ]
    summary = summarize(rows, modes=("real",))["safe"]["real"]
    assert summary["unsupported"] == 1
    assert summary["zero_diag_norm"] == 1 and summary["zero_risk_norm"] == 1
    assert (
        summary["comparable"] == 2
        and summary["aligned"] == 1
        and summary["opposed"] == 1
    )
    assert summary["aligned_fraction"] == 0.5
