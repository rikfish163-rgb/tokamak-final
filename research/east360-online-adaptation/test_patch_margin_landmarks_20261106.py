from dataclasses import is_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import patch_margin_landmarks_20261106 as landmarks


def synthetic_tape(specs):
    """Build only synthetic shots; unused curve positions contain sentinels."""
    clock = np.array([m for m, _ in specs], dtype=np.int64)
    curves = np.full((len(specs), 13, 32), -77.0, dtype=np.float32)
    shots = []
    for row, (m, n) in enumerate(specs):
        native = (0.125 + np.arange(n) / 4096).astype(np.float32)
        for k in range(min(32, max(0, n - 1 - m))):
            curves[row, :, k] = native[m + k + 1] + np.arange(13) / 4096
        shots.append(SimpleNamespace(native=native, blocks=slice(row, row + 1)))
    return SimpleNamespace(
        data={"clock": clock, "curves": curves},
        allowed=np.ones((len(specs), 13), dtype=bool),
        shots=shots,
    )


def test_sample_late_known_hashes_and_original_keys_across_views():
    ids = np.array([5, 13, 987654321], dtype=np.int64)
    owner = np.array([0, 1, 0, 1, 2, 0, 0], dtype=np.int64)
    clock = np.array([63, 63, 95, 127, 0, 63, 63], dtype=np.int64)
    got = landmarks.sample_late(ids, owner, clock)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [17, 28, 13, 17, 26, 17, 17])
    # Reordered shot inventories and repeated views preserve physical keys.
    other = landmarks.sample_late(ids[[2, 0, 1]], np.array([1, 2, 1, 2, 0, 1, 1]), clock)
    np.testing.assert_array_equal(other, got)
    assert np.all((got >= 9) & (got <= 31))


def test_sample_late_empty_and_validation():
    empty = np.array([], dtype=np.int64)
    assert landmarks.sample_late(empty, empty, empty).shape == (0,)
    with pytest.raises(ValueError):
        landmarks.sample_late([5], [0, 0], [63])
    with pytest.raises(ValueError):
        landmarks.sample_late([[5]], [0], [63])
    with pytest.raises(ValueError):
        landmarks.sample_late([5], [-1], [63])
    with pytest.raises(ValueError):
        landmarks.sample_late([5], [1], [63])
    with pytest.raises(ValueError):
        landmarks.sample_late([5], [0.0], [63])
    with pytest.raises(ValueError):
        landmarks.sample_late([5], [0], [63.0])


@pytest.mark.parametrize("delay", [8, 9, 17, 31])
def test_full_window_uses_exact_reached_and_zero_padded_tail(delay):
    tape = synthetic_tape([(3, 50)])
    win = landmarks.select(tape, delay, tau=1.0, bound=1.0)
    c = tape.data["curves"]
    assert is_dataclass(win) and isinstance(win, landmarks.Window)
    assert win.reached.shape == (1, 13, 8) and win.reached.dtype == np.float32
    assert win.tail.shape == (1, 13, 24) and win.tail.dtype == np.float32
    assert win.allowed.shape == (1, 13) and win.allowed.dtype == np.bool_
    assert win.length.dtype == np.int64 and win.delay.dtype == np.int64
    np.testing.assert_array_equal(win.delay, [delay])
    np.testing.assert_array_equal(win.length, [32 - delay])
    np.testing.assert_array_equal(win.reached, c[:, :, delay - 8:delay])
    np.testing.assert_array_equal(win.tail[:, :, :32 - delay], c[:, :, delay:])
    assert not win.tail[:, :, 32 - delay:].any()
    assert win.allowed.all()


def test_mixed_partial_last_and_unobserved_landmarks_never_expose_padding():
    specs = [(0, 45), (3, 12), (5, 13), (0, 11), (0, 10), (0, 32), (0, 33)]
    tape = synthetic_tape(specs)
    delays = np.array([8, 8, 8, 8, 8, 31, 31], dtype=np.int64)
    win = landmarks.select(tape, delays, tau=1.0, bound=1.0)
    np.testing.assert_array_equal(win.length, [24, 0, 0, 2, 1, 0, 1])
    np.testing.assert_array_equal(win.delay, delays)
    for row, (m, n) in enumerate(specs):
        d = delays[row]
        if m + d < n:
            np.testing.assert_array_equal(win.reached[row], tape.data["curves"][row, :, d - 8:d])
        else:
            assert not win.reached[row].any()
        count = win.length[row]
        np.testing.assert_array_equal(win.tail[row, :, :count], tape.data["curves"][row, :, d:d + count])
        assert not win.tail[row, :, count:].any()
        assert win.allowed[row, 0]
        assert bool(win.allowed[row, 1:].all()) == bool(count)
    assert not (win.reached == -77).any()
    assert not (win.tail == -77).any()


def test_pair_guard_checks_both_signs_at_landmark_and_original_pair():
    tape = synthetic_tape([(0, 40)])
    c = tape.data["curves"]
    c[0, :, 7] = 0.25
    c[0, 1, 7], c[0, 2, 7] = 0.375, 0.125  # equality to bound is safe
    c[0, 3, 7], c[0, 4, 7] = 0.25, 0.5  # one unsafe sign blocks both
    c[0, 5, 0] = 20.0  # only the current response magnitude is checked
    tape.allowed[0, 7:9] = False
    tape.allowed[0, 10] = False  # an asymmetric original pair is not allowed
    tape.allowed[0, 0] = False  # identity is always available
    win = landmarks.select(tape, 8, tau=1.0, bound=0.125)
    np.testing.assert_array_equal(
        win.allowed[0],
        [True, True, True, False, False, True, True,
         False, False, False, False, True, True],
    )


@pytest.mark.parametrize("cross_at", [2, 7, 12])
def test_any_early_or_current_static_crossing_blocks_all_writes(cross_at):
    tape = synthetic_tape([(4, 40)])
    tape.shots[0].native[cross_at] = 0.5
    # An early crossing is no longer visible in the last eight values.
    if cross_at == 12:
        tape.data["curves"][0, 0, 7] = 0.5
        tape.data["curves"][0, 1:, 7] = 0.5
    win = landmarks.select(tape, 8, tau=0.5, bound=1.0)
    assert win.allowed[0, 0]
    assert not win.allowed[0, 1:].any()


def test_hidden_suffix_mutation_cannot_change_reached_or_allowed():
    tape = synthetic_tape([(2, 50), (4, 30)])
    delays = np.array([9, 17])
    before = landmarks.select(tape, delays, tau=0.5, bound=0.125)
    for row, (q, d) in enumerate(zip(tape.shots, delays, strict=True)):
        m = tape.data["clock"][row]
        q.native[m + d + 1:] = 5.0  # future crossing must not close the guard
        tape.data["curves"][row, :, d:] = -12.0
    after = landmarks.select(tape, delays, tau=0.5, bound=0.125)
    np.testing.assert_array_equal(after.reached, before.reached)
    np.testing.assert_array_equal(after.allowed, before.allowed)
    np.testing.assert_array_equal(after.length, before.length)
    assert not np.array_equal(after.tail, before.tail)  # offline target may change


def test_unobserved_padding_mutation_cannot_change_any_output():
    tape = synthetic_tape([(0, 8), (0, 10)])
    before = landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape.data["curves"][0, :, 7:] = 999.0
    tape.data["curves"][1, :, 9:] = 999.0
    after = landmarks.select(tape, 8, tau=1.0, bound=1.0)
    for field in ("reached", "tail", "length", "allowed", "delay"):
        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))


def test_zero_length_rows_and_output_do_not_mutate_inputs():
    tape = synthetic_tape([(0, 9), (0, 8)])
    old_curves = tape.data["curves"].copy()
    old_allowed = tape.allowed.copy()
    old_native = [q.native.copy() for q in tape.shots]
    delays = np.array([8, 8])
    win = landmarks.select(tape, delays, tau=1.0, bound=1.0)
    assert not win.length.any()
    assert win.reached[0].any()  # d exists but is the final native point
    assert not win.reached[1].any()  # d itself does not exist
    assert not win.allowed[:, 1:].any()
    win.reached[:] = 0
    win.tail[:] = 0
    win.allowed[:] = False
    win.delay[:] = 31
    np.testing.assert_array_equal(tape.data["curves"], old_curves)
    np.testing.assert_array_equal(tape.allowed, old_allowed)
    np.testing.assert_array_equal(delays, [8, 8])
    for q, old in zip(tape.shots, old_native, strict=True):
        np.testing.assert_array_equal(q.native, old)


@pytest.mark.parametrize("bad_delay", [7, 32, 8.0, [8, 9], [[8]]])
def test_invalid_delay_rejected(bad_delay):
    with pytest.raises(ValueError):
        landmarks.select(synthetic_tape([(0, 40)]), bad_delay, tau=1.0, bound=1.0)


def test_invalid_tape_shapes_and_parameters_rejected():
    tape = synthetic_tape([(0, 40)])
    tape.data["clock"] = np.array([[0]])
    with pytest.raises(ValueError):
        landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape = synthetic_tape([(0, 40)])
    tape.data["curves"] = np.zeros((1, 13, 31), dtype=np.float32)
    with pytest.raises(ValueError):
        landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape = synthetic_tape([(0, 40)])
    tape.allowed = np.ones((1, 12), dtype=bool)
    with pytest.raises(ValueError):
        landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape = synthetic_tape([(0, 40)])
    tape.shots[0].native = np.zeros((2, 2))
    with pytest.raises(ValueError):
        landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape = synthetic_tape([(0, 40), (0, 40)])
    tape.shots[1].blocks = slice(0, 1)
    with pytest.raises(ValueError):
        landmarks.select(tape, 8, tau=1.0, bound=1.0)
    tape = synthetic_tape([(0, 40)])
    for tau, bound in [(np.nan, 1.0), (1.0, np.inf), (1.0, -1.0)]:
        with pytest.raises(ValueError):
            landmarks.select(tape, 8, tau=tau, bound=bound)
