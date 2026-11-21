import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.group_protected_meta_20261102.objectives import (
    paired_risk,
)


def test_mean_can_hide_a_worse_group_but_worst_cannot():
    before = torch.tensor([2.0, 2.0], requires_grad=True)
    after = torch.tensor([0.0, 0.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0], requires_grad=True)
    mean = paired_risk(after, before, labels, mode="mean")
    worst = paired_risk(after, before, labels, mode="worst")
    assert mean["risk"] < 0 < worst["risk"]
    expected = F.binary_cross_entropy_with_logits(
        after, labels, reduction="none"
    ) - F.binary_cross_entropy_with_logits(before, labels, reduction="none")
    torch.testing.assert_close(worst["delta"], expected)
    worst["risk"].backward()
    torch.testing.assert_close(after.grad, torch.tensor([0.0, -0.5]))
    assert before.grad is None and labels.grad is None


def test_exact_tie_shares_gradient_symmetrically():
    after = torch.tensor([0.0, 0.0], dtype=torch.double, requires_grad=True)
    before = after.detach().clone()
    paired_risk(
        after, before, torch.tensor([0.0, 1.0], dtype=torch.double), mode="worst"
    )["risk"].backward()
    torch.testing.assert_close(
        after.grad, torch.tensor([0.25, -0.25], dtype=torch.double)
    )


@pytest.mark.parametrize(
    "labels", [[1.0, 0.0], [0.0, 0.0], [0.0, 0.5], [0.0, float("nan")]]
)
def test_wrong_group_labels_are_not_silently_reordered(labels):
    with pytest.raises(ValueError):
        paired_risk(torch.zeros(2), torch.zeros(2), torch.tensor(labels), mode="mean")


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1),
        torch.zeros(2, 1),
        torch.ones(2, dtype=torch.long),
        torch.tensor([float("inf"), 0.0]),
    ],
)
def test_bad_logits_fail(bad):
    with pytest.raises(ValueError):
        paired_risk(bad, torch.zeros(2), torch.tensor([0.0, 1.0]), mode="mean")


def test_mismatched_dtype_and_unknown_mode_fail():
    with pytest.raises(ValueError):
        paired_risk(
            torch.zeros(2),
            torch.zeros(2, dtype=torch.double),
            torch.tensor([0.0, 1.0]),
            mode="mean",
        )
    with pytest.raises(ValueError):
        paired_risk(
            torch.zeros(2), torch.zeros(2), torch.tensor([0.0, 1.0]), mode="risk"
        )
