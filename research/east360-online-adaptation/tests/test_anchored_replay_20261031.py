import copy

import numpy as np
import pytest

from ANCHORED_FULLSTREAM_REPLAY_20261031 import compare_exact, select_shots
from experiments.risk_aligned_online.anchored_stream_20261031.core import NAMES


def test_selection_includes_every_changed_shot_and_fixed_reference():
    order = list(range(1, 21)) + [70323] + list(range(22, 361))
    comparisons = {n: dict(rescued=[], harmed=[]) for n in NAMES}
    comparisons['origin__real']['harmed'] = [37]
    comparisons['mature__donor']['rescued'] = [360]
    assert select_shots(order, comparisons) == [70323, 37, 360]
    comparisons['origin__real']['rescued'] = [3]
    with pytest.raises(ValueError):
        select_shots(order, comparisons)


def test_replay_is_exact_not_just_close_and_receipts_must_match():
    times = np.array([0., .001], dtype=np.float64)
    curves = {n: np.array([.2, .3], dtype=np.float32) for n in NAMES}
    saved = dict(times=times, **curves)
    receipt = [dict(origin=127, maturity=152, state_norm=.1)]
    compare_exact(times, curves, receipt, saved, copy.deepcopy(receipt))
    changed = {k:v.copy() for k,v in saved.items()}
    changed['origin__real'][1] = np.nextafter(changed['origin__real'][1], np.float32(1))
    with pytest.raises(ValueError):
        compare_exact(times, curves, receipt, changed, receipt)
    with pytest.raises(ValueError):
        compare_exact(times, curves, receipt, saved, [])
