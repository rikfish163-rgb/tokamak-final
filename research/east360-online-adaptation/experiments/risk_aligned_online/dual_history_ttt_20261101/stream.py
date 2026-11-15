"""Fixed delayed-feedback clock and label-free, common-support controls."""

import numpy as np


def feedback_origin(end):
    if type(end) is not int or end < 0:
        raise ValueError("nonnegative integer endpoint required")
    return end - 25 if end >= 159 and (end - 159) % 128 == 0 else None


def targets(signal, origin, arrived):
    if (
        set(signal) != {"values", "available"}
        or type(origin) is not int
        or origin < 127
        or arrived != origin + 25
        or arrived >= len(signal["values"])
    ):
        raise ValueError(
            "full past window and exactly matured25 signal targets required"
        )
    y = signal["values"][origin + 1 : arrived + 1].copy()
    mask = signal["available"][origin + 1 : arrived + 1].copy()
    mask &= signal["available"][origin]
    mask[:, 10] = False  # Discrete mode identity is not a continuous forecast target.
    mask &= mask[::-1].copy()  # All feedback arms have identical observed support.
    return {
        "real": y[None],
        "reverse": y[::-1].copy()[None],
        "hold": np.repeat(signal["values"][origin : origin + 1], 25, axis=0)[None],
    }, mask[None]
