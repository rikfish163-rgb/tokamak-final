"""Independent, offline recomputation from the three sealed TRAIN tapes.

The returned future length is an offline censoring/label quantity, not a
feature available to a decision rule. No outcomes or policy replay are used.
"""

import numpy as np

import run_patch_utility_multiview_20261106 as m


def verify_truth(tapes, ids, tau):
    """Return (first, length, allowed) in ``m.VIEWS`` order.

    ``first[v,b,a]`` is the 1-based first threshold crossing among actual
    recorded points m+9 through m+32, or zero if none was recorded. This is
    computed for all 13 original curves, independently of eligibility and
    outcome. ``length[b]`` is the number of actual points in that interval.
    ``allowed`` keeps abstention and independently checks the observed static
    prefix and the symmetric m+8 guard, in addition to the original menu.
    """
    ids = np.asarray(ids)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
        raise ValueError("shot ids must be a one-dimensional integer array")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("shot ids must be unique")
    if isinstance(tau, (bool, np.bool_)) or np.ndim(tau) != 0:
        raise ValueError("tau must be a finite real scalar")
    try:
        tau = float(tau)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tau must be a finite real scalar") from exc
    if not np.isfinite(tau):
        raise ValueError("tau must be a finite real scalar")

    reference = tapes[m.VIEWS[0]]
    nblocks = len(reference.owner)
    first = np.zeros((len(m.VIEWS), nblocks, 13), dtype=np.int64)
    length = np.zeros(nblocks, dtype=np.int64)
    allowed = np.zeros((len(m.VIEWS), nblocks, 13), dtype=bool)
    allowed[:, :, 0] = True
    ref_clock = np.asarray(reference.data["clock"])
    ref_owner = np.asarray(reference.owner)
    ref_lengths = None
    horizon_offsets = np.arange(24)
    bound = m.sel.bank.pair.LIMIT - m.sel.bank.pair.GUARD_HEADROOM

    for vi, name in enumerate(m.VIEWS):
        tape = tapes[name]
        clocks = np.asarray(tape.data["clock"])
        owner = np.asarray(tape.owner)
        curves = np.asarray(tape.data["curves"])
        original = np.asarray(tape.allowed)
        if (len(tape.shots) != len(ids) or clocks.shape != (nblocks,)
                or clocks.dtype.kind not in "iu" or owner.shape != (nblocks,)
                or owner.dtype.kind not in "iu"):
            raise ValueError("shot, owner or clock alignment is invalid")
        if (curves.shape != (nblocks, 13, 32) or curves.dtype.kind not in "iuf"
                or original.shape != (nblocks, 13) or original.dtype != np.bool_):
            raise ValueError("expected real [blocks,13,32] curves and boolean menu")
        if not np.array_equal(clocks, ref_clock) or not np.array_equal(owner, ref_owner):
            raise ValueError("view clock or owner alignment changed")
        if ref_lengths is None:
            ref_lengths = [len(q.native) for q in tape.shots]
        elif [len(q.native) for q in tape.shots] != ref_lengths:
            raise ValueError("view shot lengths are not aligned")

        next_block = 0
        for shot_index, (q, sid) in enumerate(zip(tape.shots, ids, strict=True)):
            sl = q.blocks
            if (q.sid != sid or not isinstance(sl, slice) or sl.step not in (None, 1)
                    or sl.start != next_block or not isinstance(sl.stop, int)
                    or not sl.start <= sl.stop <= nblocks):
                raise ValueError("shot id or block alignment changed")
            next_block = sl.stop
            if not np.all(owner[sl] == shot_index):
                raise ValueError("owner does not match shot blocks")
            z = np.asarray(q.native)
            release = clocks[sl].astype(np.int64, copy=False)
            if z.ndim != 1 or z.dtype.kind not in "iuf":
                raise ValueError("native shot scores must be a real vector")
            if np.any(release < 0) or np.any(release >= len(z)):
                raise ValueError("clock is outside the recorded shot")
            nfuture = np.clip(len(z) - release - 9, 0, 24)
            if vi == 0:
                length[sl] = nfuture
            elif not np.array_equal(length[sl], nfuture):
                raise ValueError("view future lengths are not aligned")

            c = curves[sl]
            # Verify the static curve against actual records, never its padding.
            for j, start in enumerate(release):
                count = min(32, len(z) - int(start) - 1)
                if not np.array_equal(c[j, 0, :count], z[start + 1:start + 1 + count]):
                    raise ValueError("static curve does not match its native shot")

            future = c[:, :, 8:32].astype(np.float64, copy=False)
            recorded = horizon_offsets[None, None, :] < nfuture[:, None, None]
            if not np.all(np.isfinite(future) | ~recorded):
                raise ValueError("non-finite recorded future score")
            crossing = (future >= tau) & recorded
            first[vi, sl] = np.where(crossing.any(axis=2), crossing.argmax(axis=2) + 1, 0)

            # The guard is evaluated only where m+8 and a real successor exist.
            ready = nfuture > 0
            if ready.any():
                last = release[ready] + 8
                prefix_end = int(last.max()) + 1
                prefix = z[:prefix_end].astype(np.float64, copy=False)
                if not np.isfinite(prefix).all():
                    raise ValueError("non-finite observed native prefix")
                no_prior_crossing = ~np.maximum.accumulate(prefix >= tau)[last]
                at_commit = c[ready, :, 7].astype(np.float64, copy=False)
                if not np.isfinite(at_commit).all():
                    raise ValueError("non-finite observed candidate score")
                delta = np.abs(at_commit[:, 1:] - at_commit[:, :1])
                safe_pair = (delta.reshape(-1, 6, 2).max(axis=2) <= bound).repeat(2, axis=1)
                eligible = original[sl][ready, 1:] & safe_pair & no_prior_crossing[:, None]
                rows = np.flatnonzero(ready) + sl.start
                allowed[vi, rows, 1:] = eligible

        if next_block != nblocks:
            raise ValueError("shot blocks do not cover all owners")

    return first, length, allowed
