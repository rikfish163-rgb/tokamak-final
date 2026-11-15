"""Condition diagnostic learning, never add event scores to risk outputs."""

import torch

from ..full_path_meta_20261102.core import advance
from ..risk_bridge_forecaster_20261102.core import auxiliary


def effective_table(tables, context):
    if (
        not isinstance(context, torch.Tensor)
        or context.shape != (6,)
        or not torch.isfinite(context).all()
        or (context < 0).any()
        or not torch.isclose(context.sum(), context.new_tensor(1.0), atol=1e-6, rtol=0)
    ):
        raise ValueError("six finite nonnegative probabilities summing to one required")
    if (
        not isinstance(tables, torch.Tensor)
        or tables.shape not in ((25, 23), (6, 25, 23))
        or not torch.isfinite(tables).all()
    ):
        raise ValueError("finite shared25x23 or conditional6x25x23 tables required")
    if tables.dtype != context.dtype or tables.device != context.device:
        raise ValueError("matching context/table device and dtype required")
    if tables.ndim == 2:
        return tables
    return (context.detach() @ tables.reshape(6, -1)).reshape(25, 23)


def prediction(model, pack, state, tables, context):
    return auxiliary(model, pack, state, effective_table(tables, context))


def update(model, pack, state, truth, mask, tables, context, *, create_graph=False):
    return advance(
        model,
        pack,
        state,
        truth,
        mask,
        effective_table(tables, context),
        create_graph=create_graph,
    )
