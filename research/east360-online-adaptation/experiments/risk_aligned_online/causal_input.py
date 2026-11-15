"""Causal input-contract and prefix-alignment candidate for EAST flat23 data.

This module is deliberately independent of the source model.  It changes only
the input arrays supplied by a caller and does not inspect labels, outcomes, or
future samples.  A new :class:`PrefixAlignment` instance is intended for every
shot.

The CIII rule is a candidate contract for this processed dataset, informed by
the pointwise masking and input-quality evidence in
``error_input_audit.py`` and the accompanying EAST error-audit documents.  Its
default value and tolerance are copied from the verified local contract at
``src/atlas23/contract.py`` (``CIII_MISSING_VALUE=-10.0`` and
``CIII_MISSING_ATOL=1e-2``).  They are not a universal CIII missing value, unit
convention, or physical interpretation; callers must not reuse this rule for
another dataset without re-establishing its data contract.  Other channels
are left alone because the existing audits do not justify turning their zeros,
constant records, or otherwise uncertain values into missingness.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


__all__ = [
    "CIII_MISSING_ATOL",
    "CIII_MISSING_VALUE",
    "PrefixAlignment",
    "apply_input_contract",
]

N_CHANNELS = 23
CIII_INDEX = 1

# Local copy of src/atlas23/contract.py.  Keep the source path next to the
# constants so this independent candidate remains auditable without importing
# the broader data-contract module.
CIII_MISSING_VALUE = -10.0
CIII_MISSING_ATOL = 1e-2

# The names are only an index convenience.  There is intentionally no default
# selection of channels: callers must pass an explicit ``channels`` sequence.
_CHANNEL_NAMES = (
    "Bt",
    "CIII",
    "MNM",
    "Mir_VV",
    "Mir_avg_amp",
    "Mir_avg_fre",
    "Z_proxy",
    "dx_a",
    "dy_a",
    "ip",
    "mode_number_n",
    "n=1 amplitude",
    "ne_nG",
    "qa_proxy",
    "sxr_kurt",
    "sxr_mean",
    "sxr_skew",
    "sxr_var",
    "v_loop",
    "xuv_kurt",
    "xuv_ratio",
    "xuv_skew",
    "xuv_var",
)


def _flat23(values, available, *, name: str):
    """Validate one flat23 value/mask pair and return NumPy views."""
    values = np.asarray(values)
    available = np.asarray(available)
    if values.ndim != 2 or values.shape[1:] != (N_CHANNELS,):
        raise ValueError(f"{name} values must have shape N x 23")
    if available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError(f"{name} available must be a matching boolean N x 23 array")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} values must be real numeric data")
    return values, available


def apply_input_contract(
    shot,
    *,
    ciii_sentinel=CIII_MISSING_VALUE,
    ciii_atol=CIII_MISSING_ATOL,
):
    """Return a copied shot with only pointwise CIII sentinels unavailable.

    ``shot`` must contain numeric ``values`` and boolean ``available`` arrays
    of shape ``N x 23`` in the canonical flat23 order, where CIII is column 1.
    Values within ``ciii_atol`` of ``ciii_sentinel`` (with zero relative
    tolerance) are marked unavailable at those points only.  The defaults are
    the local ``src/atlas23/contract.py`` CIII=-10, absolute-tolerance=0.01
    contract.  Values are copied unchanged, including the sentinel itself,
    because the availability mask is the missingness channel and preserving
    the raw value keeps this candidate reversible for audit purposes.

    Real zero observations remain available.  No inference is made for any
    other channel, including all-zero or constant channels, and the caller's
    shot and its arrays are never mutated.
    """
    if not isinstance(shot, Mapping):
        raise TypeError("shot must be a mapping with values and available")
    if "values" not in shot or "available" not in shot:
        raise ValueError("shot must contain values and available")
    values, available = _flat23(shot["values"], shot["available"], name="shot")
    try:
        sentinel = float(ciii_sentinel)
    except (TypeError, ValueError) as exc:
        raise ValueError("ciii_sentinel must be a finite scalar") from exc
    if not np.isfinite(sentinel):
        raise ValueError("ciii_sentinel must be a finite scalar")
    try:
        atol = float(ciii_atol)
    except (TypeError, ValueError) as exc:
        raise ValueError("ciii_atol must be a finite non-negative scalar") from exc
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("ciii_atol must be a finite non-negative scalar")

    changed = dict(shot)
    changed["values"] = np.array(values, copy=True)
    changed["available"] = np.array(available, copy=True)
    sentinel_points = np.isclose(
        values[:, CIII_INDEX], sentinel, rtol=0.0, atol=atol
    )
    changed["available"][:, CIII_INDEX] &= ~sentinel_points
    return changed


def _normalise_channels(channels):
    if isinstance(channels, (str, bytes)):
        raise TypeError("channels must be an explicit sequence of indices or names")
    try:
        channels = list(channels)
    except TypeError as exc:
        raise TypeError("channels must be an explicit sequence of indices or names") from exc
    if not channels:
        raise ValueError("channels must contain at least one explicit channel")

    normalised = []
    for channel in channels:
        if isinstance(channel, str):
            try:
                channel = _CHANNEL_NAMES.index(channel)
            except ValueError as exc:
                raise ValueError(f"unknown flat23 channel: {channel}") from exc
        if isinstance(channel, (bool, np.bool_)) or not isinstance(
            channel, (int, np.integer)
        ):
            raise TypeError("channels must contain integer indices or flat23 names")
        channel = int(channel)
        if not 0 <= channel < N_CHANNELS:
            raise ValueError(f"channel index out of range: {channel}")
        if channel in normalised:
            raise ValueError(f"duplicate channel: {channel}")
        normalised.append(channel)
    return tuple(normalised)


class PrefixAlignment:
    """Freeze causal prefix statistics and map selected channels numerically.

    ``source_center`` and ``source_scale`` are the source model's 23-channel
    numerical coordinates.  ``channels`` is required and must be an explicit
    list/tuple of flat23 indices (canonical names are also accepted for
    convenience).  No channel is selected implicitly.

    The caller supplies the cumulative, causal prefix explicitly through
    :meth:`fit_prefix`.  This method may be called again with a longer prefix
    while a channel is still waiting for ``min_points`` observations.  Once a
    channel reaches the threshold, only its first ``min_points`` finite,
    available observations are used and its statistics are frozen.  Repeated
    calls cannot make a fitted prefix grow or be replaced by a rolling window.

    :meth:`transform` accepts only one current past window with shape ``N x
    23``.  It returns ``(mapped_values, audit)``; availability is never altered
    by alignment.  The affine map preserves within-channel relative changes,
    but it is only a numerical distribution alignment and must not be called
    physical calibration.  A zero-IQR prefix is not identifiable for scale
    alignment, so that channel is returned unchanged and recorded as
    ``constant_unalignable``.
    """

    def __init__(self, source_center, source_scale, *, channels, min_points=100):
        self.source_center = self._source_vector(source_center, "source_center")
        self.source_scale = self._source_vector(source_scale, "source_scale")
        if np.any(self.source_scale <= 0):
            raise ValueError("source_scale must be finite and positive")
        self.channels = _normalise_channels(channels)
        if isinstance(min_points, (bool, np.bool_)) or not isinstance(
            min_points, (int, np.integer)
        ):
            raise TypeError("min_points must be a positive integer")
        if min_points < 1:
            raise ValueError("min_points must be a positive integer")
        self.min_points = int(min_points)
        self._stats = {channel: None for channel in self.channels}
        self._last_prefix_counts = {channel: 0 for channel in self.channels}

    @staticmethod
    def _source_vector(values, name):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (N_CHANNELS,):
            raise ValueError(f"{name} must have shape (23,)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        return values.copy()

    def fit_prefix(self, prefix_values, prefix_available):
        """Consume an explicitly supplied causal prefix and freeze new stats.

        The arrays must contain the complete prefix seen by the caller at that
        time; passing only the newly arrived suffix would not represent a
        cumulative prefix and is therefore outside this contract.  This method
        never reads a label or future sample and returns an audit snapshot for
        the prefix-fitting state.
        """
        values, available = _flat23(
            prefix_values, prefix_available, name="prefix"
        )
        for channel in self.channels:
            if self._stats[channel] is not None:
                continue
            observed = values[:, channel][
                available[:, channel] & np.isfinite(values[:, channel])
            ].astype(np.float64, copy=False)
            count = int(observed.size)
            self._last_prefix_counts[channel] = count
            if count < self.min_points:
                continue

            sample = observed[: self.min_points]
            q25, median, q75 = np.quantile(sample, [0.25, 0.5, 0.75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr < 0:
                raise ValueError(f"channel {channel} produced a non-finite IQR")
            status = "aligned" if iqr > 0 else "constant_unalignable"
            self._stats[channel] = {
                "status": status,
                "frozen": True,
                "median": float(median),
                "iqr": iqr,
                "available_prefix_points": count,
                "used_prefix_points": self.min_points,
                "scale_aligned": bool(iqr > 0),
            }

        return self._audit(window_shape=values.shape, transformed_points={})

    def transform(self, values, available):
        """Map one current past ``N x 23`` window and return ``(window, audit)``.

        ``values`` is not used to fit or update prefix statistics.  The caller
        must call :meth:`fit_prefix` with the causal cumulative prefix when it
        wants alignment to start.  This explicit separation prevents a rolling
        current window from silently changing the frozen reference.
        """
        values, available = _flat23(values, available, name="window")
        output_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
        mapped = np.array(values, dtype=output_dtype, copy=True)
        transformed_points = {}
        for channel in self.channels:
            stats = self._stats[channel]
            transformed_points[channel] = 0
            if stats is None or stats["status"] != "aligned":
                continue
            usable = available[:, channel] & np.isfinite(values[:, channel])
            if not np.any(usable):
                continue
            transformed = self.source_center[channel] + (
                values[usable, channel].astype(np.float64, copy=False) - stats["median"]
            ) * self.source_scale[channel] / stats["iqr"]
            mapped[usable, channel] = transformed
            transformed_points[channel] = int(np.sum(usable))
        return mapped, self._audit(
            window_shape=values.shape, transformed_points=transformed_points
        )

    def _audit(self, *, window_shape, transformed_points):
        channel_audit = {}
        for channel in self.channels:
            stats = self._stats[channel]
            if stats is None:
                entry = {
                    "status": "insufficient_prefix",
                    "frozen": False,
                    "median": None,
                    "iqr": None,
                    "available_prefix_points": self._last_prefix_counts[channel],
                    "used_prefix_points": 0,
                    "scale_aligned": False,
                }
            else:
                entry = dict(stats)
            entry["transformed_points"] = int(transformed_points.get(channel, 0))
            channel_audit[channel] = entry

        statuses = [entry["status"] for entry in channel_audit.values()]
        if all(status == "aligned" for status in statuses):
            overall_status = "ready"
        elif any(status != "insufficient_prefix" for status in statuses):
            overall_status = "partial"
        else:
            overall_status = "waiting"
        return {
            "status": overall_status,
            "ready": bool(overall_status == "ready"),
            "channels": channel_audit,
            "window_shape": tuple(int(size) for size in window_shape),
            "min_points": self.min_points,
            "prefix_explicit": True,
            "uses_labels": False,
            "uses_future": False,
            "physical_calibration": False,
            "transformed_points": int(
                sum(entry["transformed_points"] for entry in channel_audit.values())
            ),
        }
