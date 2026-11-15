"""Two-sample risk-change aggregation, not a full group-DRO implementation.

The max prevents compensation only within this sampled pair's objective.
It does not guarantee either population improves or that future risks are safe.
"""

import torch
from torch.nn import functional as F


def paired_risk(after_logits, before_logits, labels, *, mode):
    values = (after_logits, before_logits, labels)
    if mode not in ("mean", "worst") or any(
        not isinstance(t, torch.Tensor) or t.shape != (2,) or not t.is_floating_point()
        for t in values
    ):
        raise ValueError(
            "two floating logits/labels in safe,danger order and known mode required"
        )
    if any(
        t.dtype != after_logits.dtype or t.device != after_logits.device for t in values
    ):
        raise ValueError("matching floating dtype/device required")
    if any(not torch.isfinite(t).all() for t in values) or not torch.equal(
        labels, labels.new_tensor([0.0, 1.0])
    ):
        raise ValueError("finite logits and labels exactly [0,1] required")
    target = labels.detach()
    delta = F.binary_cross_entropy_with_logits(
        after_logits, target, reduction="none"
    ) - F.binary_cross_entropy_with_logits(
        before_logits.detach(), target, reduction="none"
    )
    return dict(delta=delta, risk=delta.mean() if mode == "mean" else delta.amax())
