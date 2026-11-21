import numpy as np
import pytest

from within_shot_coherence_20261106 import GROUPS, observer, relocate


def trace(n=640):
    return np.zeros((n, 23), dtype=np.float32), np.ones((n, 23), dtype=bool)


def test_isolated_group_writes_only_after_two_completed_blocks_and_then_releases():
    x, mask = trace()
    first = GROUPS[0][1]
    x[256:512, first] = 0.60
    result = observer(x, mask)
    assert not result["coherent"][:320].any()
    np.testing.assert_allclose(result["coherent"][320:544, :][:, first], 0.60)
    assert not result["coherent"][544:].any()
    assert not result["coherent"][:, 9].any()
    assert np.array_equal(result["coherent"], result["single"])


@pytest.mark.parametrize("stagger", [False, True])
def test_current_witnesses_protect_shared_change_and_lag_can_miss_it(stagger):
    x, mask = trace()
    for k, (_, members) in enumerate(GROUPS):
        x[256 + (32 * k if stagger else 0) : 512, members] = 0.6
    result = observer(x, mask)
    assert not result["coherent"][320:512].any()
    assert result["single"][320:512].any()
    assert result["stale"][320:352].any()
    both = (result["stale"] != 0) & (result["single"] != 0)
    np.testing.assert_array_equal(result["stale"][both], result["single"][both])


def test_missing_peer_is_not_silence_and_incomplete_blocks_never_write():
    x, mask = trace()
    x[256:512, GROUPS[0][1]] = 0.6
    mask[288:512, GROUPS[1][1][0]] = False
    x[~mask] = np.nan
    result = observer(x, mask)
    assert not result["coherent"].any()
    assert result["completed_supported"].sum() < 20


def test_arbitrary_future_edits_and_reset_do_not_change_past_state():
    x, mask = trace()
    x[256:512, GROUPS[2][1]] = -0.6
    before = observer(x, mask)
    cut = 362
    x[cut:] = 1000.0
    mask[cut + 3 :] = False
    after = observer(x, mask)
    for k in ("single", "coherent", "stale", "wrong_family"):
        np.testing.assert_array_equal(before[k][:cut], after[k][:cut])
        assert not observer(*trace())[k].any()


def test_relocate_keeps_per_time_norm_and_never_reuses_actual_group():
    a = np.zeros((4, 23), dtype=np.float32)
    a[1, GROUPS[0][1]] = [-0.6, 0.3, -0.1]
    a[2, GROUPS[2][1]] = [0.6, 0.6, 0.6, 0.6]
    b = relocate(a)
    np.testing.assert_allclose(
        np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), rtol=1e-6
    )
    assert not (a * b).any()
    assert not b[[0, 3]].any()
