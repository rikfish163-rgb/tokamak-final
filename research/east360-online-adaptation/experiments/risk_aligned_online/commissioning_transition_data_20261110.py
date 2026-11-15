"""Causal input windows and masked transition targets for commissioning."""

import numpy as np


def _series_and_mask(values, mask):
    values = np.asarray(values)
    mask = np.asarray(mask)
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError("values and boolean mask must have the same two-dimensional shape")
    return values, mask


def _indices(indices, lower, upper):
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if indices.size == 0:
        return indices.astype(np.intp)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must be integers")
    if np.any(indices < lower) or np.any(indices >= upper):
        raise ValueError("indices are outside the available range")
    return indices.astype(np.intp)


def _window_mean(values, mask, endpoints, start, length):
    indices = endpoints[:, None] + np.arange(start, start + length)
    available = indices >= 0
    indices = np.maximum(indices, 0)
    selected_mask = mask[indices] & available[:, :, None]
    counts = np.sum(selected_mask, axis=1)
    window = values[indices]
    first_valid = np.argmax(selected_mask, axis=1)
    anchor = np.take_along_axis(window, first_valid[:, None, :], axis=1)[:, 0, :]
    anchor = np.where(counts > 0, anchor, 0).astype(np.float64)
    centered = np.zeros(window.shape, dtype=np.float64)
    np.subtract(window, anchor[:, None, :], out=centered, where=selected_mask)
    totals = np.sum(centered, axis=1, dtype=np.float64)
    means = anchor + np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return means, counts


def inputs(z, mask, context, context_mask, endpoints):
    """Build the five-bin feature matrix using no rows after each endpoint."""
    z, mask = _series_and_mask(z, mask)
    context, context_mask = _series_and_mask(context, context_mask)
    if context.shape != (z.shape[0], 2):
        raise ValueError("context must have shape (N, 2)")
    endpoints = _indices(endpoints, 0, z.shape[0])

    batch, channels = endpoints.size, z.shape[1]
    means = np.empty((batch, 5, channels), dtype=np.float64)
    counts = np.empty((batch, 5, channels), dtype=np.int64)
    for bin_index in range(5):
        means[:, bin_index], counts[:, bin_index] = _window_mean(
            z, mask, endpoints, -79 + 16 * bin_index, 16
        )
    context_means, context_counts = _window_mean(context, context_mask, endpoints, -15, 16)
    context_means = np.where(context_counts >= 8, context_means, 0)

    bin_valid = counts >= 8
    differences = np.where(
        bin_valid & bin_valid[:, -1:, :], means - means[:, -1:, :], 0
    )
    recent = np.where(bin_valid[:, -1, :], means[:, -1, :], 0)
    return np.concatenate(
        (
            np.clip(context_means, -4, 4),
            np.clip(differences, -4, 4).reshape(batch, 5 * channels),
            (counts / 16).reshape(batch, 5 * channels),
            np.clip(recent, -4, 4),
        ),
        axis=1,
    ).astype(np.float32)


def targets(z, mask, origins):
    """Label five successive future groups relative to the recent trend."""
    z, mask = _series_and_mask(z, mask)
    origins = _indices(origins, 31, z.shape[0] - 25)
    batch, channels = origins.size, z.shape[1]

    previous, previous_counts = _window_mean(z, mask, origins, -31, 16)
    recent, recent_counts = _window_mean(z, mask, origins, -15, 16)
    change = recent - previous
    history_valid = (previous_counts >= 8) & (recent_counts >= 8) & (np.abs(change) >= 0.05)

    future_means = np.empty((batch, 5, channels), dtype=np.float64)
    future_counts = np.empty((batch, 5, channels), dtype=np.int64)
    for group in range(5):
        future_means[:, group], future_counts[:, group] = _window_mean(
            z, mask, origins, 1 + 5 * group, 5
        )
    valid = history_valid[:, None, :] & (future_counts >= 3)
    displacement = (
        np.sign(change)[:, None, :] * (future_means - recent[:, None, :])
        / np.maximum(np.abs(change)[:, None, :], 0.05)
    )
    labels = np.ones((batch, 5, channels), dtype=np.int64)
    labels[valid & (displacement <= -0.25)] = 0
    labels[valid & (displacement >= 0.25)] = 2
    return labels, valid


def reverse_valid(labels, valid):
    """Reverse valid time positions independently per sample and channel.

    Label release remains the caller's responsibility.
    """
    labels = np.asarray(labels)
    valid = np.asarray(valid)
    if labels.ndim != 3 or labels.shape[1] != 5 or valid.shape != labels.shape:
        raise ValueError("labels and valid must have the same shape (B, 5, C)")
    if valid.dtype != np.bool_:
        raise TypeError("valid must be boolean")

    result = labels.copy()
    ranks = np.cumsum(valid, axis=1)
    counts = np.sum(valid, axis=1, keepdims=True)
    for source in range(5):
        source_valid = valid[:, source:source + 1, :]
        source_rank = ranks[:, source:source + 1, :]
        destination = valid & source_valid & (ranks == counts + 1 - source_rank)
        np.copyto(result, labels[:, source:source + 1, :], where=destination)
    return result
