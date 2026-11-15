"""Fixed cold-coverage sensitivity ordering over the current-order roster.

This is an offline ordering: the existing current-order scheduler reads each
shot's initial Ip prefix, including prefixes of shots later in the replay.
No labels or full shots are read by this module.
"""

from . import order


_ROSTER_SIZE = 360
_COLD_BUDGET = 20
_LOW_CURRENT_POOL = 120


def order_stream(ids, reader=order.read_prefix):
    """Select 20 evenly spaced ranks among 120, then retain all other rows.

    The original order and validation are supplied by ``order.order_stream``.
    Roster size is checked first, before the delegated reader can be called.
    Every returned row is a copy annotated with its original one-based rank
    and its commissioning phase.
    """
    try:
        ids = list(ids)
    except TypeError as error:
        raise ValueError("a roster of exactly 360 shot IDs is required") from error
    if len(ids) != _ROSTER_SIZE:
        raise ValueError("a roster of exactly 360 shot IDs is required")

    rows = order.order_stream(ids, reader=reader)
    selected = [
        k * (_LOW_CURRENT_POOL - 1) // (_COLD_BUDGET - 1)
        for k in range(_COLD_BUDGET)
    ]
    selected_set = set(selected)
    remaining = [index for index in range(_ROSTER_SIZE) if index not in selected_set]

    result = []
    for index in selected + remaining:
        row = dict(rows[index])
        row["initial_current_rank1based"] = index + 1
        row["commissioning_phase"] = (
            "cold_coverage20" if index in selected_set else "remainder340"
        )
        result.append(row)
    return result
