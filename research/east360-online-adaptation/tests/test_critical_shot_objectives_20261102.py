"""Synthetic-only checks for critical-shot selection and task objectives."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.risk_aligned_online.critical_shot_meta_20261102.objectives import (
    last_update,
    margin_loss,
    select_critical,
    task_label,
)


@pytest.mark.parametrize("disruptive", [False, True])
def test_global_early_maximum_is_not_replaced_by_an_adaptable_point(disruptive):
    times = np.arange(320, dtype=np.float64) / 1000
    logits = np.full(times.shape, -10.0)
    logits[159], logits[160] = 8.0, 7.0

    critical = select_critical(times, logits, disruptive, 0.320 if disruptive else None)

    assert isinstance(critical, int)
    assert critical == 159
    assert last_update(critical) is None
    assert last_update(160) == 159


def test_disruptive_selection_rounds_lead_before_strict_30ms_comparison():
    times = np.array([0.900, 0.9699999999994, 0.9699999999996, 0.970, 0.980])
    logits = np.array([1.0, 2.0, 30.0, 40.0, 50.0])
    leads_ms = np.round((1.0 - times) * 1000, 9)
    assert leads_ms.tolist() == [100.0, 30.000000001, 30.0, 30.0, 20.0]

    assert select_critical(times, logits, True, 1.0) == 1


def test_disruptive_selection_has_no_3000ms_upper_lead_limit():
    times = np.array([0.0, 3.9, 4.8, 4.97, 4.99])
    logits = np.array([8.0, 7.0, 6.0, 100.0, 200.0])

    critical = select_critical(times, logits, True, 5.0)

    assert critical == 0
    assert (5.0 - times[critical]) * 1000 > 3000


def test_safe_selection_considers_all_points_and_breaks_ties_first():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    logits = np.array([1.0, 3.0, 8.0, 8.0])

    assert select_critical(times, logits, False, None) == 2


def test_disruptive_selection_breaks_eligible_ties_first():
    times = np.array([0.0, 0.2, 0.97, 0.99])
    logits = np.array([8.0, 8.0, 100.0, 200.0])

    assert select_critical(times, logits, True, 1.0) == 0


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (0, None),
        (158, None),
        (159, None),
        (160, 159),
        (287, 159),
        (288, 287),
        (415, 287),
        (416, 415),
        (10000, 9887),
    ],
)
def test_last_update_is_latest_clock_tick_strictly_before_query(query, expected):
    assert last_update(query) == expected


@pytest.mark.parametrize(
    ("index", "point_label"),
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
)
def test_disruptive_operational_and_original_point_labels_are_distinct(index, point_label):
    # The existing EAST point label is positive from 125 ms up to disruption;
    # the shot-level label is positive even at an early warning point.
    times = np.array([0.874, 0.875, 0.900, 0.969, 0.970, 1.0])

    assert task_label(times, index, True, 1.0, "point") == point_label
    assert task_label(times, index, True, 1.0, "operational") == 1


@pytest.mark.parametrize("kind", ["point", "operational"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_safe_shot_task_labels_are_zero_at_every_point(kind, index):
    times = np.array([0.0, 0.5, 1.0])

    assert task_label(times, index, False, None, kind) == 0


@pytest.mark.parametrize("threshold", [0.01, 0.5, 0.7258946895599365, 0.99])
@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("value", [-1000.0, -1.25, 0.0, 2.0, 1000.0])
def test_margin_loss_matches_stable_shifted_bce_and_has_correct_gradient(
    threshold, label, value
):
    logit = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    shifted = logit - math.log(threshold / (1 - threshold))
    expected = F.binary_cross_entropy_with_logits(shifted, torch.full_like(logit, label))

    loss = margin_loss(logit, threshold, label)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)
    (gradient,) = torch.autograd.grad(loss, logit)
    torch.testing.assert_close(gradient, torch.sigmoid(shifted).detach() - label)


def test_margin_loss_reduces_tensor_logits_to_mean_with_per_element_gradients():
    threshold = 0.7258946895599365
    logits = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64, requires_grad=True)
    shifted = logits - math.log(threshold / (1 - threshold))
    expected = F.binary_cross_entropy_with_logits(shifted, torch.ones_like(logits))

    loss = margin_loss(logits, threshold, 1)

    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)
    (gradient,) = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, (torch.sigmoid(shifted).detach() - 1) / 3)


@pytest.mark.parametrize("label", [0, 1])
def test_margin_loss_supports_second_order_meta_gradients(label):
    threshold = 0.7258946895599365
    logit = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    loss = margin_loss(logit, threshold, label)
    (gradient,) = torch.autograd.grad(loss, logit, create_graph=True)
    (curvature,) = torch.autograd.grad(gradient, logit)
    probability = torch.sigmoid(logit.detach() - math.log(threshold / (1 - threshold)))

    torch.testing.assert_close(curvature, probability * (1 - probability))


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_margin_loss_rejects_invalid_probability_thresholds(threshold):
    with pytest.raises(ValueError):
        margin_loss(torch.tensor(0.0), threshold, 1)


@pytest.mark.parametrize(
    ("times", "logits", "disruptive", "down"),
    [
        ([], [], False, None),
        ([0.0, 0.1], [0.0], False, None),
        ([0.0, float("nan")], [0.0, 1.0], False, None),
        ([0.0, 0.1], [0.0, float("nan")], False, None),
        ([0.0, 0.1], [0.0, float("inf")], False, None),
        ([0.0, 0.1], [0.0, 1.0], True, None),
        ([0.0, 0.1], [0.0, 1.0], True, float("nan")),
        ([0.97, 0.98], [0.0, 1.0], True, 1.0),
    ],
)
def test_select_critical_rejects_invalid_or_empty_eligible_inputs(
    times, logits, disruptive, down
):
    with pytest.raises(ValueError):
        select_critical(np.array(times), np.array(logits), disruptive, down)
