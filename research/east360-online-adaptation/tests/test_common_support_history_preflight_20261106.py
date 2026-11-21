import numpy as np
import pytest

from common_support_history_preflight_20261106 import (
    BLOCKS,
    COUNTS,
    candidate_rows,
    choose_block,
    geometry_checks,
    independent_support,
)


def test_independent_support_on_exact_edge_and_missing_block():
    mask = np.ones((260, 23), dtype=bool)
    anchors = np.array([127, 254, 255, 259])
    q = independent_support(mask, anchors, 32)
    np.testing.assert_array_equal(q.sum(1), [0, 3, 4, 4])
    mask[:32] = False
    mask[:32, [1, 2, 10]] = True  # noncontributing channels do not mark support
    q = independent_support(mask, anchors, 32)
    assert not q[2, -1] and q[3].all()


def test_source_group_counts_do_not_relabel_east_like_negative_endpoints():
    t = np.arange(501) / 1000
    mask = np.ones((len(t), 23), dtype=bool)
    counts = candidate_rows(t, mask, True, 0.500, 25.0)[32]
    assert counts["positive"] > 0
    assert counts["timely_positive"] > 0
    assert counts["east_like_positive"] == 0
    assert counts["east_like_negative"] > 0
    long = candidate_rows(t, mask, True, 0.500, 80.0)[32]
    assert long["east_like_positive"] > 0
    safe = candidate_rows(t, mask, False, None, 25.0)[32]
    assert safe["safe"] > 0 and safe["positive"] == safe["outside"] == 0


def test_dense_and_clock_only_diagnostics_separate_grid_from_missingness():
    t = np.arange(281) / 1000
    mask = np.ones((len(t), 23), dtype=bool)
    args = (t, mask, True, 0.280, 35.0)
    assert candidate_rows(*args)[24]["outside"] == 0
    assert candidate_rows(*args, dense=True)[24]["outside"] == 22
    mask[:] = False
    assert candidate_rows(*args, dense=True)[24]["outside"] == 0
    assert candidate_rows(*args, dense=True, clock_only=True)[24]["outside"] == 22


def test_largest_block_requires_all_declared_source_gates():
    total_disrupt, total_safe = 100, 100
    candidates = {b: dict.fromkeys(COUNTS, 0) for b in BLOCKS}
    candidates[24].update(
        timely_positive=81, safe=100, outside=75, east_like_positive=17
    )
    candidates[32].update(
        timely_positive=80, safe=60, outside=60, east_like_positive=15
    )
    candidates[40].update(
        timely_positive=80, safe=60, outside=59, east_like_positive=15
    )
    assert choose_block(candidates, total_disrupt, total_safe) == 32
    candidates[32]["east_like_positive"] = 14
    assert choose_block(candidates, total_disrupt, total_safe) == 24
    candidates[24]["timely_positive"] = 79
    assert choose_block(candidates, total_disrupt, total_safe) is None
    assert geometry_checks(candidates, total_disrupt, total_safe)[24] == {
        "timely_positive": False,
        "outside": True,
        "safe": True,
        "east_like_positive": True,
    }
    with pytest.raises(ValueError, match="both source classes"):
        choose_block(candidates, 0, total_safe)
