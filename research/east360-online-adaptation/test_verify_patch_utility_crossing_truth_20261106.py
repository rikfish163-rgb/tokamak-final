"""Synthetic, CPU-only checks of the independent crossing-truth audit."""

from types import SimpleNamespace

import numpy as np
import pytest

import run_patch_utility_multiview_20261106 as m
from verify_patch_utility_crossing_truth_20261106 import verify_truth


def make_tapes(specs, fill=-1.0, dtype=np.float64):
    """Build synthetic shot records without any outcome or HDF5 fields."""
    ids = np.arange(101, 101 + len(specs), dtype=np.int64)
    tapes = {}
    for name in m.VIEWS:
        shots, clocks, owner = [], [], []
        for j, ((n, release), sid) in enumerate(zip(specs, ids, strict=True)):
            start = len(clocks)
            clocks.extend(release)
            owner.extend([j] * len(release))
            shots.append(SimpleNamespace(
                sid=int(sid), native=np.full(n, fill, dtype=dtype),
                blocks=slice(start, len(clocks)),
            ))
        count = len(clocks)
        curves = np.full((count, 13, 32), fill, dtype=dtype)
        for q in shots:
            for i in range(q.blocks.start, q.blocks.stop):
                actual = min(32, max(0, len(q.native) - int(clocks[i]) - 1))
                # With tau=0, this tempting padding must not become a crossing.
                curves[i, :, actual:] = 0
        tapes[name] = SimpleNamespace(
            shots=shots,
            owner=np.asarray(owner, dtype=np.int64),
            allowed=np.ones((count, 13), dtype=bool),
            data={"clock": np.asarray(clocks, dtype=np.int64), "curves": curves},
        )
    return tapes, ids


def test_first_crossing_uses_view_order_offsets_and_unfiltered_actions():
    tapes, ids = make_tapes([(50, [4])])
    native, masked, reduced = (tapes[v] for v in m.VIEWS)
    native.shots[0].native[16] = 0  # m+8+4, not an observed-prefix hit
    native.data["curves"][0, 0, 11] = 0
    native.data["curves"][0, 1, 8] = 0
    native.data["curves"][0, 2, 31] = 0
    native.data["curves"][0, 3, 10] = 0
    native.allowed[0, 3:5] = False
    masked.data["curves"][0, 1, 12] = 0
    reduced.data["curves"][0, 1, 15] = 0

    before = [(t.data["curves"].copy(), t.allowed.copy(), t.shots[0].native.copy())
              for t in tapes.values()]
    first, length, allowed = verify_truth(tapes, ids, 0.0)

    assert first.shape == (3, 1, 13) and first.dtype.kind in "iu"
    assert length.shape == (1,) and length.dtype.kind in "iu"
    assert allowed.shape == (3, 1, 13) and allowed.dtype == np.bool_
    assert length.tolist() == [24]
    assert first[0, 0, :4].tolist() == [4, 1, 24, 3]
    assert first[:, 0, 1].tolist() == [1, 5, 8]
    assert not allowed[0, 0, 3:5].any()
    for t, (curves, old_allowed, z) in zip(tapes.values(), before, strict=True):
        np.testing.assert_array_equal(t.data["curves"], curves)
        np.testing.assert_array_equal(t.allowed, old_allowed)
        np.testing.assert_array_equal(t.shots[0].native, z)


def test_partial_and_missing_future_are_not_padding_crossings():
    tapes, ids = make_tapes([(8, [0]), (9, [0]), (10, [0]), (11, [0]),
                              (14, [0]), (33, [0]), (40, [0])])
    for t in tapes.values():
        t.data["curves"][3, 1, 9] = 0  # second and final actual future point
        t.data["curves"][4, 2, 12] = 0  # fifth and final actual future point
        t.data["curves"][5, 3, 31] = 0  # 24th actual future point

    first, length, allowed = verify_truth(tapes, ids, 0.0)

    assert length.tolist() == [0, 0, 1, 2, 5, 24, 24]
    assert not first[:, :3].any()
    assert not first[:, :, 0].any()
    assert (first[:, 3, 1] == 2).all()
    assert (first[:, 4, 2] == 5).all()
    assert (first[:, 5, 3] == 24).all()
    assert allowed[:, :, 0].all()
    assert not allowed[:, :2, 1:].any()
    assert allowed[:, 2:, 1:].all()


def test_guard_uses_actual_native_prefix_and_both_members_of_pair():
    tapes, ids = make_tapes([(40, [0])], fill=0.0)
    limit = m.sel.bank.pair.LIMIT - m.sel.bank.pair.GUARD_HEADROOM
    normal, prior_hit, last_hit = (tapes[v] for v in m.VIEWS)
    c = normal.data["curves"]
    c[0, 1:3, 7] = [limit, -limit]  # equality is allowed
    c[0, 3:5, 7] = [limit + 1e-6, 0]  # one unsafe sign rejects both
    c[0, 5:7, 7] = [0, -limit - 1e-6]  # likewise for the other sign
    normal.allowed[0, 7:9] = False
    normal.allowed[0, 0] = False  # abstention remains available unconditionally
    prior_hit.shots[0].native[0] = 1.0  # outside the eight-frame c prefix
    last_hit.shots[0].native[8] = 1.0  # threshold equality at m+8
    last_hit.data["curves"][0, :, 7] = 1.0

    _, length, allowed = verify_truth(tapes, ids, 1.0)

    assert length.tolist() == [24]
    assert allowed[:, :, 0].all()
    assert allowed[0, 0, 1:3].all()
    assert not allowed[0, 0, 3:9].any()
    assert allowed[0, 0, 9:].all()
    assert not allowed[1:, 0, 1:].any()


def test_float32_scores_are_compared_to_tau_in_float64():
    tapes, ids = make_tapes([(40, [0])], fill=1.0, dtype=np.float32)
    tau = 1.00000003  # distinguishable in float64 but rounds to 1 in float32
    for t in tapes.values():
        t.data["curves"][0, 1, 8] = np.nextafter(np.float32(1), np.float32(2))

    first, length, allowed = verify_truth(tapes, ids, tau)

    assert length.tolist() == [24]
    assert (first[:, 0, 0] == 0).all()
    assert (first[:, 0, 1] == 1).all()
    assert allowed[:, 0, 1:].all()


def test_multiple_blocks_are_aligned_by_owner_and_shot_id():
    tapes, ids = make_tapes([(80, [0, 32]), (15, [0])])
    first, length, _ = verify_truth(tapes, ids, 0.0)
    assert first.shape == (3, 3, 13)
    assert length.tolist() == [24, 24, 6]

    with pytest.raises(ValueError, match="(?i)(id|shot|align)"):
        verify_truth(tapes, ids[::-1], 0.0)

    tapes[m.VIEWS[2]].owner[2] = 0
    with pytest.raises(ValueError, match="(?i)(owner|shot|align)"):
        verify_truth(tapes, ids, 0.0)


def test_view_schedule_mismatch_is_rejected():
    tapes, ids = make_tapes([(40, [0])])
    tapes[m.VIEWS[1]].data["clock"][0] = 1
    with pytest.raises(ValueError, match="(?i)(clock|schedule|align)"):
        verify_truth(tapes, ids, 0.0)


def test_missing_padding_can_be_nan_but_actual_missing_score_is_rejected():
    tapes, ids = make_tapes([(11, [0])])
    for tape in tapes.values():
        tape.data["curves"][0, :, 10:] = np.nan
    first, length, _ = verify_truth(tapes, ids, 0.0)
    assert length.tolist() == [2]
    assert not first.any()

    tapes[m.VIEWS[0]].data["curves"][0, 1, 9] = np.nan
    with pytest.raises(ValueError, match="(?i)non-finite.*future"):
        verify_truth(tapes, ids, 0.0)


def test_misaligned_static_curve_and_nonfinite_observed_scores_are_rejected():
    tapes, ids = make_tapes([(40, [0])])
    native = tapes[m.VIEWS[0]]
    native.data["curves"][0, 0, 2] = -0.5
    with pytest.raises(ValueError, match="(?i)static curve"):
        verify_truth(tapes, ids, 0.0)
    native.data["curves"][0, 0, 2] = -1.0

    native.shots[0].native[0] = np.nan
    with pytest.raises(ValueError, match="(?i)non-finite.*prefix"):
        verify_truth(tapes, ids, 0.0)
    native.shots[0].native[0] = -1.0

    native.data["curves"][0, 1, 7] = np.nan
    with pytest.raises(ValueError, match="(?i)non-finite.*candidate"):
        verify_truth(tapes, ids, 0.0)


@pytest.mark.parametrize("tau", [np.nan, np.inf, True, [0.0]])
def test_invalid_threshold_is_rejected(tau):
    tapes, ids = make_tapes([(40, [0])])
    with pytest.raises(ValueError, match="(?i)tau"):
        verify_truth(tapes, ids, tau)


def test_random_tapes_match_a_direct_scalar_oracle():
    rng = np.random.default_rng(20261106)
    sizes = rng.integers(7, 120, size=20)
    specs = [(int(n), list(range(0, int(n), 32))) for n in sizes]
    tapes, ids = make_tapes(specs)
    tau = 0.0
    for t in tapes.values():
        for q in t.shots:
            q.native[:] = rng.normal(-0.18, 0.10, size=len(q.native))
            for b in range(q.blocks.start, q.blocks.stop):
                release = int(t.data["clock"][b])
                count = min(32, len(q.native) - release - 1)
                curves = t.data["curves"][b]
                curves[:] = 5.0  # padding must not turn into a crossing
                curves[0, :count] = q.native[release + 1:release + 1 + count]
                curves[1:, :count] = (
                    curves[0, :count][None, :] + rng.normal(0, 0.10, (12, count))
                )
                t.allowed[b, 1:] = np.repeat(rng.integers(0, 2, 6).astype(bool), 2)

    # Deliberately reverse dict insertion order; output follows m.VIEWS.
    tapes = dict(reversed(list(tapes.items())))
    first, length, allowed = verify_truth(tapes, ids, tau)
    bound = m.sel.bank.pair.LIMIT - m.sel.bank.pair.GUARD_HEADROOM
    for vi, view in enumerate(m.VIEWS):
        t = tapes[view]
        for q in t.shots:
            for b in range(q.blocks.start, q.blocks.stop):
                release = int(t.data["clock"][b])
                nfuture = min(24, max(0, len(q.native) - release - 9))
                assert length[b] == nfuture
                c = t.data["curves"][b]
                assert allowed[vi, b, 0]
                for action in range(13):
                    crossing = 0
                    for offset in range(1, nfuture + 1):
                        if float(c[action, 7 + offset]) >= tau:
                            crossing = offset
                            break
                    assert first[vi, b, action] == crossing
                    if action:
                        partner = action + 1 if action % 2 else action - 1
                        prefix_clear = not any(float(z) >= tau for z in q.native[:release + 9])
                        pair_safe = all(abs(float(c[a, 7]) - float(c[0, 7])) <= bound
                                        for a in (action, partner))
                        expected = bool(nfuture and prefix_clear and pair_safe
                                        and t.allowed[b, action])
                        assert allowed[vi, b, action] == expected
