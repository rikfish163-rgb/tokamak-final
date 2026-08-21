from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts/train_source_padfix.py"
)


def module():
    spec = importlib.util.spec_from_file_location("train_source_padfix", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_context_crop_deletes_only_older_history() -> None:
    padfix = module()
    windows = np.arange(2 * 128 * 3, dtype=np.float32).reshape(2, 128, 3)
    cropped = padfix.crop_older_context(
        windows, np.asarray([128, 96]), np.asarray([64, 32]),
    )

    np.testing.assert_array_equal(cropped[0, :64], 0.0)
    np.testing.assert_array_equal(cropped[0, 64:], windows[0, 64:])
    np.testing.assert_array_equal(cropped[1, :96], 0.0)
    np.testing.assert_array_equal(cropped[1, 96:], windows[1, 96:])
    np.testing.assert_array_equal(cropped[:, -1], windows[:, -1])


def test_context_crop_rejects_future_or_unobserved_samples() -> None:
    padfix = module()
    windows = np.zeros((1, 128, 2), np.float32)

    with pytest.raises(ValueError, match="no longer than observed"):
        padfix.crop_older_context(windows, np.asarray([64]), np.asarray([96]))


def test_gradient_reversal_changes_only_backward_sign() -> None:
    padfix = module()
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    reversed_value = padfix.reverse_gradient(value)

    torch.testing.assert_close(reversed_value, value)
    reversed_value.sum().backward()
    torch.testing.assert_close(value.grad, -torch.ones_like(value))

    scaled = torch.tensor([1.0, -2.0], requires_grad=True)
    padfix.reverse_gradient(scaled, 0.125).sum().backward()
    torch.testing.assert_close(scaled.grad, -0.125 * torch.ones_like(scaled))


def test_context_buckets_are_registered_and_ordered() -> None:
    padfix = module()
    np.testing.assert_array_equal(
        padfix.context_bucket(np.asarray([32, 64, 96, 128])),
        np.asarray([0, 1, 2, 3]),
    )
    with pytest.raises(ValueError, match="unregistered"):
        padfix.context_bucket(np.asarray([48]))


def test_all_context_views_share_the_same_endpoint() -> None:
    padfix = module()
    windows = np.arange(4 * 128 * 2, dtype=np.float32).reshape(4, 128, 2)
    labels = np.asarray([0, 1, 0, 1], np.float32)
    views, lengths, outcomes, buckets = padfix._all_context_views(
        windows, np.full(4, 128), labels, maximum_examples=4,
    )

    assert views.shape == (16, 128, 2)
    np.testing.assert_array_equal(lengths, np.repeat([32, 64, 96, 128], 4))
    np.testing.assert_array_equal(buckets, np.repeat([0, 1, 2, 3], 4))
    np.testing.assert_array_equal(outcomes, np.tile(labels, 4))
    for group in range(4):
        np.testing.assert_array_equal(views[group * 4:(group + 1) * 4, -1], windows[:, -1])


def test_paired_representation_consistency_is_zero_only_for_equal_views() -> None:
    padfix = module()
    full = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    equal = padfix.paired_representation_consistency(full, full.clone())
    shifted = padfix.paired_representation_consistency(full, full + 1.0)

    assert float(equal) == pytest.approx(0.0)
    assert float(shifted) > 0.0
    with pytest.raises(ValueError, match="identical shapes"):
        padfix.paired_representation_consistency(full, full[:, :1])


def test_checkpoint_selection_prefers_a_model_that_passes_every_validation_gate() -> None:
    padfix = module()
    strong_auc_but_shortcut = {
        "timely_tpr": 0.90,
        "fpr": 0.10,
        "auc": 0.99,
        "paired_context_probe_macro_auc": 0.80,
        "paired_context_mean_abs_risk_difference": 0.01,
        "paired_context_alarm_flip_rate": 0.01,
    }
    admitted = {
        "timely_tpr": 0.90,
        "fpr": 0.15,
        "auc": 0.90,
        "paired_context_probe_macro_auc": 0.64,
        "paired_context_mean_abs_risk_difference": 0.049,
        "paired_context_alarm_flip_rate": 0.049,
    }

    assert padfix.validation_selection_key(admitted) > padfix.validation_selection_key(
        strong_auc_but_shortcut
    )


def test_checkpoint_selection_minimizes_worst_gate_violation_before_fpr() -> None:
    padfix = module()
    near_gate = {
        "timely_tpr": 0.90,
        "fpr": 0.16,
        "auc": 0.90,
        "paired_context_probe_macro_auc": 0.66,
        "paired_context_mean_abs_risk_difference": 0.01,
        "paired_context_alarm_flip_rate": 0.01,
    }
    low_fpr_but_large_shortcut = {
        "timely_tpr": 0.90,
        "fpr": 0.10,
        "auc": 0.99,
        "paired_context_probe_macro_auc": 0.80,
        "paired_context_mean_abs_risk_difference": 0.01,
        "paired_context_alarm_flip_rate": 0.01,
    }

    assert padfix.validation_selection_key(near_gate) > padfix.validation_selection_key(
        low_fpr_but_large_shortcut
    )
