"""Centered diagnostic normal equations and causal mature-window indexing."""

import numpy as np
import torch


def block_system(jacobian, residual):
    """Return Gram and real/shifted right-hand sides for (..., 5, C, D) blocks.

    Both inputs are centered independently over the five times for each channel;
    only the centered residual is rolled by two times for the shifted control.
    """
    if (
        jacobian.ndim < 3
        or jacobian.shape[-3] != 5
        or residual.shape != jacobian.shape[:-1]
        or not jacobian.is_floating_point()
        or jacobian.dtype != residual.dtype
        or jacobian.device != residual.device
    ):
        raise ValueError("expected same-dtype/device float tensors (..., 5, C, D) and (..., 5, C)")

    centered_j = jacobian - jacobian.mean(dim=-3, keepdim=True)
    centered_r = residual - residual.mean(dim=-2, keepdim=True)
    flat_j = centered_j.flatten(-3, -2)
    flat_real = centered_r.flatten(-2, -1)
    flat_shifted = centered_r.roll(2, dims=-2).flatten(-2, -1)
    transpose = flat_j.transpose(-2, -1)
    gram = transpose @ flat_j
    real = (transpose @ flat_real.unsqueeze(-1)).squeeze(-1)
    shifted = (transpose @ flat_shifted.unsqueeze(-1)).squeeze(-1)
    return gram, real, shifted


def solve_step(gram, rhs, *, damping=0.1):
    """Solve batched trace-scaled ridge systems; an all-zero Gram gives zero."""
    if (
        gram.ndim < 2
        or gram.shape[-1] == 0
        or gram.shape[-2] != gram.shape[-1]
        or rhs.shape != gram.shape[:-1]
        or not gram.is_floating_point()
        or gram.dtype != rhs.dtype
        or gram.device != rhs.device
    ):
        raise ValueError("expected same-dtype/device float tensors (..., D, D) and (..., D)")
    if damping <= 0:
        raise ValueError("damping must be positive")

    matrix = gram.to(torch.float64)
    vector = rhs.to(torch.float64)
    empty = (matrix == 0).all(dim=(-2, -1))
    ridge = damping * matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / matrix.shape[-1]
    ridge = torch.where(empty, torch.ones_like(ridge), ridge)
    vector = torch.where(empty.unsqueeze(-1), torch.zeros_like(vector), vector)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    result = torch.linalg.solve(matrix + ridge[..., None, None] * identity, vector.unsqueeze(-1))
    return result.squeeze(-1).to(gram.dtype)


def mature_history(q, eligible, *, horizon, stride, count):
    """Index recent released windows and append the current eligible window.

    History is selected from physical times within the inclusive lookback and
    released no later than the current q.  The last column is only a pointer to
    the current window: its observations may be used at q + horizon + 1.
    """
    if (
        not isinstance(q, np.ndarray)
        or q.ndim != 1
        or not np.issubdtype(q.dtype, np.integer)
        or not isinstance(eligible, np.ndarray)
        or eligible.shape != q.shape
        or eligible.dtype != np.bool_
    ):
        raise ValueError("expected a one-dimensional integer q and matching boolean eligible")
    if horizon < 0 or stride <= 0 or count < 1:
        raise ValueError("expected horizon >= 0, stride > 0 and count >= 1")

    result = np.full((len(q), count), -1, dtype=np.intp)
    for j in range(len(q)):
        if j and q[j] < q[j - 1]:
            raise ValueError("q must be sorted")
        if eligible[j]:
            result[j, -1] = j
        if not j or count == 1:
            continue
        current = int(q[j])
        previous = q[:j]
        first = np.searchsorted(previous, current - (count - 1) * stride, side="left")
        end = np.searchsorted(previous, current - horizon - 1, side="right")
        indices = np.flatnonzero(eligible[first:end]) + first
        indices = indices[-(count - 1):]
        if len(indices):
            result[j, count - 1 - len(indices):count - 1] = indices
    return result
