import numpy as np
import pytest

from source_supported_internal_update_20261104 import select_records, family_weights


def test_selection_excludes_held_roles_and_keeps_one_median_per_shot_group():
    rows = [dict(shot=1, group="safe", query=q) for q in (153, 159, 184, 209)] + [
        dict(shot=2, group="danger", query=200)
    ]
    selected = select_records(rows, {1, 2})
    assert [(r["shot"], r["query"]) for r in selected] == [(1, 184), (2, 200)]
    with pytest.raises(ValueError, match="TRAIN"):
        select_records(rows, {1})


def test_family_weighting_is_nominal_and_unknown_is_not_safe():
    rows = [
        dict(shot=1, group="safe", family=-1),
        dict(shot=2, group="safe", family=-1),
        dict(shot=3, group="safe", family=5),
        dict(shot=4, group="early", family=0),
        dict(shot=4, group="danger", family=3),
    ]
    weights = family_weights(rows, use_family=True)
    np.testing.assert_allclose(weights, [1 / 12, 1 / 12, 1 / 6, 1 / 3, 1 / 3])
    uniform = family_weights(rows, use_family=False)
    np.testing.assert_allclose(uniform, [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3])
    assert weights[-1] == weights[-2]
