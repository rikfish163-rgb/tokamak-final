"""Offline teacher for a separate native diagnostic head; no data access.

The caller supplies labels and an independent snapshot of the head's initial
state. The frozen source risk path is used only to construct a detached teacher
direction; no source weights, native heads, or scores are modified.
"""

from collections.abc import Mapping

import torch
from torch import nn

from ..fast_update import normalized_step
from ..native_token_probe_20261104.core import _check_inputs, _check_source
from ..simple_online import FAST_RADIUS, FAST_STEP


def features(base, x, a, delta):
    """Capture native forecast-head inputs without detaching their delta path."""
    _check_inputs(base, x, a, delta, None)
    captured = []
    hook = base.forecast_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0])
    )
    try:
        result = base(x, a, fast=base.initial_fast(len(x)) + delta)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape != (len(x), 1088):
        raise RuntimeError("exactly one native Bx1088 forecast-head input required")
    if (
        not torch.isfinite(captured[0]).all()
        or not torch.isfinite(result["logit"]).all()
    ):
        raise FloatingPointError("finite native features and risk logits required")
    return {"logit": result["logit"], "features": captured[0]}


def _check_head(base, head, initial):
    if (
        not isinstance(head, nn.Linear)
        or (head.in_features, head.out_features) != (1088, 575)
        or head.bias is None
        or not all(p.requires_grad for p in head.parameters())
    ):
        raise ValueError("independent trainable Linear1088->575 head required")
    if any(
        p.device != base.center.device or p.dtype != base.center.dtype
        for p in head.parameters()
    ):
        raise ValueError("head must match source dtype and device")
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError("finite trainable head required")
    source_storage = {
        value.untyped_storage().data_ptr()
        for value in (*base.parameters(), *base.buffers())
    }
    head_storage = {p.untyped_storage().data_ptr() for p in head.parameters()}
    if source_storage & head_storage:
        raise ValueError("trainable head must not share source storage")
    if not isinstance(initial, Mapping) or set(initial) != {"weight", "bias"}:
        raise ValueError("initial head snapshot must contain weight and bias")
    for name, parameter in head.named_parameters():
        value = initial[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or value.device != parameter.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "initial head snapshot must match finite parameter tensors"
            )
        if value.untyped_storage().data_ptr() in head_storage:
            raise ValueError(
                "initial head snapshot must not share trainable head storage"
            )


def _check_targets(base, truth, mask, labels, batch):
    if (
        not isinstance(truth, torch.Tensor)
        or truth.shape != (batch, 25, 23)
        or truth.device != base.center.device
        or truth.dtype != base.center.dtype
        or not isinstance(mask, torch.Tensor)
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
        or mask.device != truth.device
    ):
        raise ValueError("matching Bx25x23 targets and boolean observed mask required")
    if not torch.isfinite(truth[mask]).all():
        raise ValueError("observed target values must be finite")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.shape != (batch,)
        or labels.device != truth.device
        or labels.is_complex()
        or not torch.isfinite(labels).all()
        or not ((labels == 0) | (labels == 1)).all()
    ):
        raise ValueError(
            "binary finite labels with shape B on the source device required"
        )


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
):
    """Return (differentiable teacher loss, components and support counts).

    Both label types are caller-supplied offline; this module never reads HDF5.
    The teacher direction is detached. All-unobserved or all-zero-teacher
    batches raise instead of silently performing proxy/anchor-only training.
    """
    _check_source(base)
    _check_head(base, head, initial_head_state)
    if (
        not isinstance(origin_x, torch.Tensor)
        or origin_x.ndim != 3
        or not len(origin_x)
    ):
        raise ValueError("nonempty Bx128x23 origin input required")
    batch = len(origin_x)
    _check_targets(base, truth25, target_mask, labels, batch)
    observed = target_mask.flatten(1).any(1)
    observed_rows = int(observed.sum())
    if not observed_rows:
        raise ValueError(f"no_target: observed_rows=0 unobserved_rows={batch}")

    with torch.enable_grad():
        delta = base.center.new_zeros(batch, 128).requires_grad_(True)
        origin = features(base, origin_x, origin_mask, delta)
        prediction = head(origin["features"]).reshape(batch, 25, 23)
        proxy = base.forecast_loss(prediction, truth25, target_mask)
        if not torch.isfinite(proxy):
            raise FloatingPointError("nonfinite absolute diagnostic proxy")
        (proxy_gradient,) = torch.autograd.grad(proxy, delta, create_graph=True)

        query = features(base, query_x, query_mask, delta)
        (logit_gradient,) = torch.autograd.grad(query["logit"].sum(), delta)
        desired = (
            (1 - 2 * labels.detach().to(delta.dtype))[:, None] * logit_gradient
        ).detach()
        if (
            not torch.isfinite(proxy_gradient).all()
            or not torch.isfinite(desired).all()
        ):
            raise FloatingPointError("finite proxy and teacher gradients required")
        desired_nonzero = desired.ne(0).any(1)
        proxy_nonzero = proxy_gradient.detach().ne(0).any(1)
        supported = observed & desired_nonzero
        alignment_rows = int(supported.sum())
        zero_desired_rows = int((observed & ~desired_nonzero).sum())
        zero_proxy_rows = int((observed & ~proxy_nonzero).sum())
        if not alignment_rows:
            raise ValueError(
                f"no_teacher_direction: observed_rows={observed_rows} "
                f"zero_desired_rows={zero_desired_rows} unobserved_rows={batch - observed_rows}"
            )
        g, d = proxy_gradient[supported], desired[supported]
        cosine = (g * d).sum(1) / (
            (g.square().sum(1) + 1e-12).sqrt() * (d.square().sum(1) + 1e-12).sqrt()
        )
        alignment = (1 - cosine).mean()
        anchor = sum(
            (parameter - initial_head_state[name].detach()).square().sum()
            for name, parameter in head.named_parameters()
        ) / sum(parameter.numel() for parameter in head.parameters())
        total = alignment + 0.20 * proxy + 1e-4 * anchor
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite teacher objective")
    return total, dict(
        alignment=alignment,
        absolute_proxy=proxy,
        anchor=anchor,
        observed_rows=observed_rows,
        unobserved_rows=batch - observed_rows,
        zero_desired_rows=zero_desired_rows,
        zero_proxy_rows=zero_proxy_rows,
        alignment_rows=alignment_rows,
        delta=delta,
        proxy_gradient=proxy_gradient,
        desired_gradient=desired,
    )


def inner_state(base, delta, proxy_gradient, *, create_graph=False):
    """Apply the native normalized step; retain the outer graph only on request."""
    _check_source(base)
    if type(create_graph) is not bool:
        raise TypeError("create_graph must be boolean")
    if (
        not isinstance(delta, torch.Tensor)
        or delta.ndim != 2
        or delta.shape[1] != 128
        or not len(delta)
        or delta.dtype != base.center.dtype
        or delta.device != base.center.device
        or not isinstance(proxy_gradient, torch.Tensor)
        or proxy_gradient.shape != delta.shape
        or proxy_gradient.dtype != delta.dtype
        or proxy_gradient.device != delta.device
        or not torch.isfinite(delta).all()
        or not torch.isfinite(proxy_gradient).all()
    ):
        raise ValueError(
            "matching finite Bx128 source-coordinate offsets/gradients required"
        )
    origin = base.initial_fast(len(delta)).detach()
    with torch.set_grad_enabled(create_graph):
        return (
            normalized_step(
                origin + delta,
                origin,
                proxy_gradient,
                step=FAST_STEP,
                radius=FAST_RADIUS,
            )
            - origin
        )
