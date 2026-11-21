"""Label-free block observer for conditional, reversible token calibration."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


GROUPS = (("mir", (3, 4, 5)), ("sxr", (14, 15, 16, 17)), ("xuv", (19, 20, 21, 22)))
C, BLOCK, MIN_POINTS = 23, 32, 24
WARM, REF, CONFIRM, MAX_HOLD = 8, 3, 2, 12
MAD, NOISE_FLOOR, LIMIT, THRESHOLD = 1.4826, 0.075, 0.75, 3.0


@dataclass
class Hypothesis:
    history: deque = field(default_factory=lambda: deque(maxlen=WARM))
    recent: deque = field(default_factory=lambda: deque(maxlen=CONFIRM))
    reference: np.ndarray | None = None
    scale: np.ndarray | None = None
    age: int = 0

    def release(self, level, *, rebase=False):
        self.recent.clear()
        self.age = 0
        self.reference = self.scale = None
        if rebase:
            self.history.clear()
        self.history.append(level)

    def advance(self, level):
        if len(self.history) < WARM:
            self.history.append(level)
            return False, np.zeros_like(level)
        if self.age == MAX_HOLD:
            self.release(level, rebase=True)
            return False, np.zeros_like(level)
        if self.reference is None:
            x = np.stack(self.history)
            ref = np.median(x[-REF:], axis=0)
            scale = np.maximum(
                MAD * np.median(np.abs(np.diff(x, axis=0)), axis=0), NOISE_FLOOR
            )
        else:
            ref, scale = self.reference, self.scale
        d = level - ref
        if np.median(np.abs(d) / scale) < THRESHOLD:
            self.release(level)
            return False, np.zeros_like(level)
        self.reference, self.scale = ref, scale
        if self.recent and float((self.recent[-1] * d / scale**2).sum()) <= 0.0:
            # A reversed innovation starts a new confirmation, not an accumulated vote.
            # It does not reset the total hold clock of the same frozen reference.
            self.recent.clear()
        self.age += 1
        self.recent.append(d)
        correction = (
            np.clip(np.median(np.stack(self.recent), axis=0), -LIMIT, LIMIT)
            if len(self.recent) == CONFIRM
            else np.zeros_like(level)
        )
        return True, correction


def relocate(adjustment):
    """Move a single group's correction to a fixed wrong family at equal L2 norm."""
    a = np.asarray(adjustment)
    if (
        a.ndim != 2
        or a.shape[1] != C
        or a.dtype.kind != "f"
        or not np.isfinite(a).all()
    ):
        raise ValueError("finite [time,23] correction required")
    columns = {j for _, g in GROUPS for j in g}
    if a[:, sorted(set(range(C)) - columns)].any():
        raise ValueError("correction outside declared diagnostic groups")
    use = np.stack([np.any(a[:, g] != 0, axis=1) for _, g in GROUPS], axis=1)
    if (use.sum(1) > 1).any():
        raise ValueError(
            "wrong-family control requires at most one active source group"
        )
    out = np.zeros_like(a)
    for i, (_, g) in enumerate(GROUPS):
        dst = GROUPS[(i + 1) % len(GROUPS)][1]
        v = a[:, g]
        lead = v[np.arange(len(a)), np.argmax(np.abs(v), axis=1)]
        magnitude = np.linalg.norm(v.astype(np.float64), axis=1)
        out[:, dst] = (np.sign(lead) * magnitude / np.sqrt(len(dst)))[:, None]
    return out


def observer(z, available):
    """Write only after a complete observation block; reset the state per call/shot."""
    v, a = np.asarray(z), np.asarray(available)
    if (
        v.ndim != 2
        or v.shape[1] != C
        or v.dtype.kind != "f"
        or a.shape != v.shape
        or a.dtype != bool
    ):
        raise ValueError("paired [time,23] source-coordinate signal and mask required")
    nb = len(v) // BLOCK
    single = np.zeros_like(v, dtype=np.float32)
    coherent, stale = single.copy(), single.copy()
    flags = np.zeros((nb, len(GROUPS)), dtype=bool)
    support = np.zeros(nb, dtype=bool)
    states = [Hypothesis() for _ in GROUPS]
    for i in range(nb):
        lo, hi = i * BLOCK, (i + 1) * BLOCK
        mask = a[lo:hi]
        # Never count a missing family as evidence for isolation.
        if any((mask[:, g].sum(0) < MIN_POINTS).any() for _, g in GROUPS):
            states = [Hypothesis() for _ in GROUPS]
            continue
        support[i] = True
        current = v[lo:hi]
        region = slice(hi, hi + BLOCK)
        proposed = []
        for f, ((_, group), state) in enumerate(zip(GROUPS, states, strict=True)):
            q, m = current[:, group], mask[:, group]
            if not np.isfinite(q[m]).all():
                raise ValueError("observed block must be finite")
            level = np.array(
                [np.median(col[valid]) for col, valid in zip(q.T, m.T, strict=True)],
                dtype=np.float64,
            )
            flags[i, f], d = state.advance(level)
            single[region, group] = d
            proposed.append(d)
        n = flags[i].sum()
        previous = flags[i - 2] if i >= 2 else np.zeros(len(GROUPS), dtype=bool)
        for f, (_, group) in enumerate(GROUPS):
            if n == 1 and flags[i, f]:
                coherent[region, group] = proposed[f]
            if i >= 2 and support[i - 2] and previous.sum() - int(previous[f]) == 0:
                stale[region, group] = proposed[f]
    return dict(
        single=single,
        coherent=coherent,
        stale=stale,
        wrong_family=relocate(coherent),
        completed_supported=support,
        candidate_flags=flags,
    )
