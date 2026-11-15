"""Deterministic, real-valued subspace geometry using only NumPy.

All row arrays have shape ``(n_rows, D)``; empty row sets must retain their
second dimension, for example ``np.empty((0, D))``. Bases have shape ``(D, k)``
and orthonormal columns. Invalid shapes, complex values, and non-finite inputs
raise ``ValueError`` rather than being silently altered.
"""

import numpy as np


_RELATIVE_CUTOFF = 1e-6
_ABSOLUTE_FLOOR = 1e-10


def _real_array(value, name, ndim):
    """Convert a finite real array, retaining and validating its dimensions."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _svd_rank(singular_values, input_scale):
    """Rank for an SVD computed after dividing its input by input_scale."""
    if singular_values.size == 0 or input_scale == 0:
        return 0
    with np.errstate(over="ignore", divide="ignore"):
        absolute = np.divide(_ABSOLUTE_FLOOR, input_scale)
    cutoff = max(_RELATIVE_CUTOFF * singular_values[0], absolute)
    return int(np.count_nonzero(singular_values > cutoff))


def _orient_columns(q):
    """Choose a reproducible sign for each column without changing its span."""
    if q.shape[1] == 0:
        return q
    pivots = np.argmax(np.abs(q), axis=0)
    signs = np.where(q[pivots, np.arange(q.shape[1])] < 0, -1.0, 1.0)
    return q * signs


def _basis(rows):
    """Basis implementation for an already validated two-dimensional array."""
    dimension = rows.shape[1]
    if rows.shape[0] == 0 or dimension == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    nonzero = np.any(rows != 0, axis=1)
    if not np.any(nonzero):
        return np.empty((dimension, 0), dtype=np.float64)
    nonzero_rows = rows[nonzero]
    scale = np.max(np.abs(nonzero_rows))
    _, singular_values, vh = np.linalg.svd(nonzero_rows / scale, full_matrices=False)
    rank = _svd_rank(singular_values, scale)
    return _orient_columns(vh[:rank].T)


def basis(rows):
    """Return a ``(D, k)`` orthonormal basis for the row space of ``rows``.

    Exactly zero rows are ignored. A singular value is retained only when it
    exceeds both 1e-6 times the largest singular value and the 1e-10 absolute
    floor. Thus uniform scaling preserves the result only while the floor is
    inactive. Signs are canonicalized; repeated singular values can still admit
    different equally valid bases, so compare their projectors, not columns.
    """
    return _basis(_real_array(rows, "rows", 2))


def _scaled_product(*values):
    """Multiply nonnegative finite numbers without avoidable intermediate overflow."""
    mantissa = 1.0
    exponent = 0
    for value in values:
        if value == 0:
            return 0.0
        part, power = np.frexp(value)
        mantissa *= part
        exponent += int(power)
    with np.errstate(over="ignore", under="ignore"):
        result = float(np.ldexp(mantissa, exponent))
    if not np.isfinite(result):
        raise OverflowError("a result exceeds the finite float64 range")
    return result


def _norm(values):
    """Euclidean norm without avoidable squaring underflow or overflow."""
    scale = np.max(np.abs(values), initial=0.0)
    if scale == 0:
        return 0.0
    return _scaled_product(float(np.linalg.norm(values / scale)), scale)


def capture(q, past, future):
    """Project ``future`` into ``span(q)`` intersected with ``null(past)``.

    ``q`` must have orthonormal columns with shape ``(D, k)``, ``past`` has
    shape ``(n, D)``, and ``future`` is a length-D vector. The constraint is
    obtained from the SVD of ``past @ q`` with the same rank cutoffs as
    :func:`basis` (scaling the SVD input for numerical stability).

    The returned dictionary contains exactly:

    * ``rank``: the number of columns of q;
    * ``constrained_rank``: dimension left after imposing the past constraints;
    * ``absolute_norm``: Euclidean norm of the returned vector;
    * ``fraction``: that norm divided by the norm of future's projection into
      the full null space of past, not merely into span(q);
    * ``residual_past``: Euclidean norm of ``past @ vector``;
    * ``vector``: the resulting length-D vector in the original coordinates.

    If both full-null and captured norms are zero up to projection roundoff
    (16 * float64 epsilon * max(1, sqrt(D)) * norm(future)), the vector and
    fraction are defined as zero (including 0/0). If numerical rank cutoffs
    instead produce a nonzero capture when the full-null norm is numerically
    zero, ``ValueError`` is raised rather than reporting a false finite fraction.
    Overflow that cannot fit float64 raises ``OverflowError``. Numerical rank
    truncation can otherwise leave a nonzero residual or a fraction above one;
    these are exposed rather than hidden or clipped. Approximately orthonormal
    q is re-orthonormalized after validation so its span is projected correctly.
    """
    q = _real_array(q, "q", 2)
    past = _real_array(past, "past", 2)
    future = _real_array(future, "future", 1)
    dimension, rank = q.shape
    if past.shape[1] != dimension or future.shape[0] != dimension:
        raise ValueError("q, past, and future must have the same ambient dimension")
    if (
        rank > dimension
        or np.max(np.abs(q), initial=0.0) > 1.0 + 1e-7
        or not np.allclose(q.T @ q, np.eye(rank), rtol=1e-7, atol=1e-8)
    ):
        raise ValueError("q must have orthonormal columns")
    if rank:
        q = np.linalg.qr(q, mode="reduced")[0]

    past_scale = np.max(np.abs(past), initial=0.0)
    if past_scale == 0 or rank == 0:
        constraint_rank = 0
        null_coordinates = np.eye(rank)
    else:
        scaled_past = past / past_scale
        # Wide matrices need all right singular vectors; tall ones do not need
        # the potentially enormous full left-singular-vector matrix.
        _, singular_values, vh = np.linalg.svd(
            scaled_past @ q, full_matrices=past.shape[0] < rank
        )
        constraint_rank = _svd_rank(singular_values, past_scale)
        null_coordinates = vh[constraint_rank:].T
    constrained_rank = rank - constraint_rank

    future_scale = np.max(np.abs(future), initial=0.0)
    vector_scaled = np.zeros(dimension, dtype=np.float64)
    fraction = 0.0
    if future_scale != 0:
        scaled_future = future / future_scale
        if constrained_rank:
            constrained_basis = q @ null_coordinates
            vector_scaled = constrained_basis @ (constrained_basis.T @ scaled_future)
        past_basis = _basis(past)
        null_future = scaled_future - past_basis @ (past_basis.T @ scaled_future)
        denominator = _norm(null_future)
        numerator = _norm(vector_scaled)
        roundoff = (
            16
            * np.finfo(np.float64).eps
            * max(1.0, np.sqrt(dimension))
            * _norm(scaled_future)
        )
        if denominator <= roundoff:
            if numerator > roundoff:
                raise ValueError(
                    "numerical rank cutoffs yield an undefined capture fraction"
                )
            vector_scaled.fill(0.0)
        else:
            fraction = float(numerator / denominator)
            if not np.isfinite(fraction):
                raise OverflowError("capture fraction exceeds the finite float64 range")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        vector = vector_scaled * future_scale
    if not np.isfinite(vector).all():
        raise OverflowError("capture vector exceeds the finite float64 range")
    absolute_norm = _scaled_product(_norm(vector_scaled), future_scale)
    if past_scale == 0 or future_scale == 0:
        residual_past = 0.0
    else:
        scaled_residual = (past / past_scale) @ vector_scaled
        residual_past = _scaled_product(
            _norm(scaled_residual), past_scale, future_scale
        )
    return {
        "rank": rank,
        "constrained_rank": constrained_rank,
        "absolute_norm": absolute_norm,
        "fraction": fraction,
        "residual_past": residual_past,
        "vector": vector,
    }


def _auxiliary_basis(rows, old):
    """SVD of normalized auxiliary rows after removing the old subspace."""
    dimension = rows.shape[1]
    if rows.shape[0] == 0 or dimension == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    scales = np.max(np.abs(rows), axis=1)
    nonzero = scales != 0
    if not np.any(nonzero):
        return np.empty((dimension, 0), dtype=np.float64)
    normalized = rows[nonzero] / scales[nonzero, None]
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    residual = normalized - (normalized @ old) @ old.T
    # Remove cancellation error a second time before the residual SVD.
    residual -= (residual @ old) @ old.T
    return _basis(residual)


def matched_augments(old_rows, true_rows, reverse_rows):
    """Return old, true-joined, reverse-joined bases and common extra rank.

    Preserve every retained old direction. Each nonzero auxiliary input row is
    normalized before its old-space component is removed; its residual row
    space is then ordered by SVD strength. Retain the smaller of the two extra
    ranks on both sides, appending those dominant residual directions to the
    unchanged old basis. Per-row nonzero rescaling, including sign reversal,
    therefore cannot grant an auxiliary set extra rank or singular weight.
    If truncation splits a singular-value tie, the chosen extra subspace is
    not unique and may depend on the numerical SVD's choice of basis.
    """
    old_rows = _real_array(old_rows, "old_rows", 2)
    true_rows = _real_array(true_rows, "true_rows", 2)
    reverse_rows = _real_array(reverse_rows, "reverse_rows", 2)
    if (
        true_rows.shape[1] != old_rows.shape[1]
        or reverse_rows.shape[1] != old_rows.shape[1]
    ):
        raise ValueError("all row sets must have the same ambient dimension")
    old = _basis(old_rows)
    true_extra = _auxiliary_basis(true_rows, old)
    reverse_extra = _auxiliary_basis(reverse_rows, old)
    common = min(true_extra.shape[1], reverse_extra.shape[1])
    return (
        old,
        np.concatenate((old, true_extra[:, :common]), axis=1),
        np.concatenate((old, reverse_extra[:, :common]), axis=1),
        common,
    )
