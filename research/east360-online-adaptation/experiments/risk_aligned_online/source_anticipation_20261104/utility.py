"""Offline actual-update risk utility for a separate diagnostic head.

Labels and the probability threshold are supplied by the caller. This module
does not read data, change the source, or change deployment risk scores.
"""

import math
from numbers import Real

import torch
from torch.nn import functional as F

from ..simple_online import FAST_STEP
from . import meta


def _margin_delta(before, after, labels, threshold_logit):
    """Per-row float64 BCE difference with a detached static baseline."""
    target = labels.detach().to(torch.float64)
    baseline = F.binary_cross_entropy_with_logits(
        before.detach().to(torch.float64) - threshold_logit, target, reduction="none"
    )
    updated = F.binary_cross_entropy_with_logits(
        after.to(torch.float64) - threshold_logit, target, reduction="none"
    )
    return updated - baseline


def objective(
    base,
    head,
    origin_x,
    origin_mask,
    query_x,
    query_mask,
    truth25,
    target_mask,
    labels,
    *,
    initial_head_state,
    threshold,
):
    """Return actual-update utility plus the existing proxy and anchor terms.

    The threshold is a fixed probability in (0, 1). Only rows with observed
    diagnostic targets and nonzero raw query-logit sensitivity are averaged.
    All supervision is caller-supplied and offline.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a finite probability in (0, 1)")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be a finite probability in (0, 1)")
    threshold_logit = math.log(threshold / (1 - threshold))

    with torch.enable_grad():
        _, info = meta.objective(
            base,
            head,
            origin_x,
            origin_mask,
            query_x,
            query_mask,
            truth25,
            target_mask,
            labels,
            initial_head_state=initial_head_state,
        )
        updated_state = meta.inner_state(
            base, info["delta"], info["proxy_gradient"], create_graph=True
        )
        # Use the same grad-enabled source execution path as the updated state,
        # then detach its value. A no-grad baseline can select CPU MHA fastpath
        # while the after-forward cannot, producing nonzero utility at zero
        # update (regression test covers both fastpath settings).
        baseline_logit = meta.features(
            base, query_x, query_mask, info["delta"]
        )["logit"].detach()
        updated_logit = meta.features(base, query_x, query_mask, updated_state)["logit"]
        perrow_delta = _margin_delta(
            baseline_logit, updated_logit, labels, threshold_logit
        )

        desired = info["desired_gradient"].detach()
        supported = target_mask.flatten(1).any(1) & desired.ne(0).any(1)
        utility_rows = int(supported.sum())
        if not utility_rows:
            raise ValueError(
                "no_utility_support: no observed rows with nonzero sensitivity"
            )
        sensitivity = (desired.double().square().sum(1) + 1e-12).sqrt().detach()
        scale = (FAST_STEP * sensitivity).detach()
        if not torch.isfinite(perrow_delta).all() or not torch.isfinite(scale).all():
            raise FloatingPointError(
                "finite risk deltas and sensitivity scale required"
            )
        risk_utility = (perrow_delta[supported] / scale[supported]).mean()
        total = risk_utility + 0.20 * info["absolute_proxy"] + 1e-4 * info["anchor"]
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite actual-update utility objective")

    return total, dict(
        info,
        risk_utility=risk_utility,
        perrowrisk_delta=perrow_delta,
        sensitivity=sensitivity,
        normalization_scale=scale,
        updated_state=updated_state,
        baseline_logit=baseline_logit,
        risk_utility_rows=utility_rows,
    )
