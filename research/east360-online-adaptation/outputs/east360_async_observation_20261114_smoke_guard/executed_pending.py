"""Causal, array-only input views for a delayed observation window."""

import numpy as np


def pending_views(values, arrived, ends, columns, delay, center):
    """Return carry values, their encoder mask, and a center-value control.

    ``values`` and boolean ``arrived`` are already-masked [B, W, 23] windows;
    ``ends`` gives each row's zero-based physical endpoint. For each selected
    channel, only the observation at endpoint minus ``delay`` can be carried
    into unobserved slots in the final ``delay`` positions. The anchor must be
    in the window, in the shot, observed, and finite. A missing anchor never
    causes a search further back. ``center`` is the 23-channel source center
    in the same physical units as ``values``.

    Both value views use the returned mask. Its added true entries denote
    expectations for a risk-prediction forward pass, not real observations.
    Use the original ``arrived`` mask for adaptation and supervision. All
    outputs are independent copies; neither input array is modified.
    """
    values = np.asarray(values)
    arrived = np.asarray(arrived)
    ends = np.asarray(ends)
    columns = np.asarray(columns)
    center = np.asarray(center)
    if values.ndim != 3 or values.shape[2] != 23 or values.dtype.kind != "f":
        raise ValueError("values must be a floating-point [B, W, 23] array")
    if arrived.shape != values.shape or arrived.dtype != np.bool_:
        raise ValueError("arrived must be a matching boolean array")
    if ends.shape != (values.shape[0],) or ends.dtype.kind not in "iu" or (ends < 0).any():
        raise ValueError("ends must contain one nonnegative physical index per window")
    if columns.ndim != 1 or (columns.size and (columns.dtype.kind not in "iu"
                                               or (columns < 0).any()
                                               or (columns >= 23).any())):
        raise ValueError("columns must be channel indices between 0 and 22")
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    if center.shape != (23,) or center.dtype.kind not in "fiu" or not np.isfinite(center).all():
        raise ValueError("center must contain 23 finite source centers")

    carry = values.copy()
    expectation_mask = arrived.copy()
    center_only = values.copy()
    width = values.shape[1]
    if delay == 0 or width <= delay:
        return carry, expectation_mask, center_only

    anchor = width - 1 - delay
    for column in columns:
        ready = (ends >= delay) & arrived[:, anchor, column] & np.isfinite(values[:, anchor, column])
        fill = ready[:, None] & ~arrived[:, -delay:, column]
        np.copyto(carry[:, -delay:, column], values[:, anchor, column, None], where=fill)
        np.copyto(center_only[:, -delay:, column], center[column], where=fill)
        expectation_mask[:, -delay:, column] |= fill
    return carry, expectation_mask, center_only
