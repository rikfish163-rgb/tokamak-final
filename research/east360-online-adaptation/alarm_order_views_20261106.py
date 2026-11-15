"""Mask-preserving, whole-patch alarm-order interventions.

Windows contain 128 samples in chronological order, oldest first, with the
current endpoint in patch 15. A permutation maps each destination patch to its
source patch; samples and channels within a patch are never rearranged. The
caller owns window construction, timestamps, and source normalization.
"""

from numbers import Integral

import numpy as np


ARMS = ("recent_reverse", "old_reverse", "recent_inner", "old_inner")

_PATCHES = 16
_PATCH_SIZE = 8
_CHANNELS = 23
_WINDOW_SHAPE = (_PATCHES * _PATCH_SIZE, _CHANNELS)
_IDENTITY = np.arange(_PATCHES, dtype=np.int64)


def _mask_array(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[1:] != _WINDOW_SHAPE or mask.dtype != bool:
        raise ValueError("mask must be boolean with shape [B, 128, 23]")
    return mask


def _permutation_array(perm):
    perm = np.asarray(perm)
    if perm.ndim < 1 or perm.shape[-1] != _PATCHES or perm.dtype.kind not in "iu":
        raise ValueError("permutation must be an integer array of shape [..., 16]")
    if np.any(perm < 0) or np.any(perm >= _PATCHES):
        raise ValueError("permutation patch indices must be in [0, 16)")
    if not np.all(np.sort(perm, axis=-1) == _IDENTITY):
        raise ValueError("each row must be a true permutation")
    if np.any(perm[..., -1] != _PATCHES - 1):
        raise ValueError("the terminal patch must remain fixed")
    return perm.astype(np.int64, copy=False)


def _validated_windows(values, mask, perm):
    values = np.asarray(values)
    mask = _mask_array(mask)
    perm = _permutation_array(perm)
    if values.shape != mask.shape:
        raise ValueError("values and mask must have shape [B, 128, 23]")
    if perm.shape != (values.shape[0], _PATCHES):
        raise ValueError("window permutations must have shape [B, 16]")
    patches = mask.reshape(-1, _PATCHES, _PATCH_SIZE, _CHANNELS)
    moved_mask = np.take_along_axis(patches, perm[:, :, None, None], axis=1)
    if not np.array_equal(moved_mask, patches):
        raise ValueError("permutation must preserve every mask position exactly")
    return values, mask, perm


def plans(mask):
    """Return destination-to-source plans and exact-mask support for four arms.

    Unsupported arms return identity for the entire window. Empty observation
    masks are supported if the proposed permutation preserves them exactly.
    """
    mask = _mask_array(mask)
    batch = mask.shape[0]
    patches = mask.reshape(batch, _PATCHES, _PATCH_SIZE, _CHANNELS)
    perm = np.broadcast_to(_IDENTITY, (batch, len(ARMS), _PATCHES)).copy()
    supported = np.empty((batch, len(ARMS)), dtype=bool)
    arm_specs = ((11, True), (7, True), (11, False), (7, False))
    for arm, (start, reverse) in enumerate(arm_specs):
        intended = _IDENTITY.copy()
        if reverse:
            intended[start : start + 4] = np.arange(start + 3, start - 1, -1)
        else:
            intended[start + 1 : start + 3] = [start + 2, start + 1]
        supported[:, arm] = np.all(patches[:, intended] == patches, axis=(1, 2, 3))
        perm[:, arm] = np.where(supported[:, arm, None], intended, _IDENTITY)
    return perm, supported


def transport(perm, advance):
    """Move the same absolute donor/destination samples to a later endpoint.

    ``advance`` is a nonnegative whole-patch sample count. Every moved patch
    must remain inside the later window; no patch is wrapped or clipped. The
    caller must ensure the later window endpoint really advanced by this amount.
    """
    perm = _permutation_array(perm)
    if isinstance(advance, (bool, np.bool_)) or not isinstance(advance, Integral):
        raise ValueError("advance must be an integer sample count")
    advance = int(advance)
    if advance < 0 or advance % _PATCH_SIZE:
        raise ValueError("advance must be nonnegative and divisible by 8")
    shift = advance // _PATCH_SIZE
    result = np.broadcast_to(_IDENTITY, perm.shape).copy()
    rows = zip(perm.reshape(-1, _PATCHES), result.reshape(-1, _PATCHES))
    for source_row, result_row in rows:
        destinations = np.flatnonzero(source_row != _IDENTITY)
        if not destinations.size:
            continue
        donors = source_row[destinations]
        if shift > int(destinations.min()) or shift > int(donors.min()):
            raise ValueError("transport would move a patch outside the later window")
        result_row[destinations - shift] = donors - shift
    return _permutation_array(result)


def apply(values, mask, perm):
    """Reorder whole patches while leaving the mask at its original positions.

    All source values move verbatim, including payloads at missing positions.
    """
    values, _, perm = _validated_windows(values, mask, perm)
    patches = values.reshape(-1, _PATCHES, _PATCH_SIZE, _CHANNELS)
    reordered = np.take_along_axis(patches, perm[:, :, None, None], axis=1)
    return reordered.reshape(values.shape)


def motion_rms(z, mask, perm):
    """RMS normalized-asinh difference over observed, moved coordinates only.

    ``z`` must already use the caller's source normalization. Missing payloads
    and unchanged patches do not enter either the numerator or denominator.
    """
    z, mask, perm = _validated_windows(z, mask, perm)
    if z.dtype.kind not in "iuf":
        raise ValueError("normalized values must have a real numeric dtype")
    patches = z.astype(np.float64, copy=False).reshape(
        -1, _PATCHES, _PATCH_SIZE, _CHANNELS
    )
    donors = np.take_along_axis(patches, perm[:, :, None, None], axis=1)
    active = mask.reshape(patches.shape) & (perm != _IDENTITY)[:, :, None, None]
    if not np.isfinite(patches[active]).all() or not np.isfinite(donors[active]).all():
        raise ValueError("observed moved values must be finite")
    delta = np.zeros_like(patches, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(donors, patches, out=delta, where=active)
    if not np.isfinite(delta[active]).all():
        raise ValueError("observed motion difference must be finite")
    # Scale before squaring to avoid overflowing on large, finite differences.
    scale = np.max(np.abs(delta), axis=(1, 2, 3))
    normalized = np.zeros_like(delta)
    np.divide(
        delta,
        scale[:, None, None, None],
        out=normalized,
        where=scale[:, None, None, None] != 0,
    )
    count = active.sum(axis=(1, 2, 3))
    mean_square = np.zeros(z.shape[0], dtype=np.float64)
    np.divide(
        np.sum(normalized * normalized, axis=(1, 2, 3)),
        count,
        out=mean_square,
        where=count != 0,
    )
    return scale * np.sqrt(mean_square)
