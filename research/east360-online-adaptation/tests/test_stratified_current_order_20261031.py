"""Cold-coverage ordering tests using only synthetic in-memory prefixes."""

import inspect
import random

import numpy as np
import pytest

from experiments.risk_aligned_online.current_order_20261031 import order
from experiments.risk_aligned_online.current_order_20261031 import stratified


EXPECTED_RANKS = [1, 7, 13, 19, 26, 32, 38, 44, 51, 57,
                  63, 69, 76, 82, 88, 94, 101, 107, 113, 120]


def synthetic_prefix(shot, current=None):
    if current is None:
        current = float(shot * 1000)
    return {
        "planning_ip": np.full(110, current),
        "rate": 1000.0,
        "start": -0.25 + shot / 100000.0,
    }


def test_ranks_remainder_retention_metadata_and_existing_order_delegation(monkeypatch):
    ids = list(range(360, 0, -1))
    original_order = order.order_stream
    delegated_calls = []
    original_rows = []
    snapshots = []

    def delegated(roster, reader):
        delegated_calls.append(list(roster))
        rows = original_order(roster, reader=reader)
        original_rows.extend(rows)
        snapshots.extend(row.copy() for row in rows)
        return rows

    monkeypatch.setattr(order, "order_stream", delegated)
    result = stratified.order_stream(ids, reader=synthetic_prefix)

    assert delegated_calls == [ids]
    assert [row["initial_current_rank1based"] for row in result[:20]] == EXPECTED_RANKS
    expected_ranks = EXPECTED_RANKS + [rank for rank in range(1, 361) if rank not in EXPECTED_RANKS]
    assert [row["initial_current_rank1based"] for row in result] == expected_ranks
    assert [row["shot"] for row in result] == expected_ranks
    assert len(result) == len({row["shot"] for row in result}) == 360
    assert [row["commissioning_phase"] for row in result] == ["cold_coverage20"] * 20 + ["remainder340"] * 340
    assert original_rows == snapshots  # The delegated records were not modified.
    for row in result:
        source = original_rows[row["initial_current_rank1based"] - 1]
        assert row is not source
        assert {k: v for k, v in row.items() if k not in {"initial_current_rank1based", "commissioning_phase"}} == source
        assert row["signal_start_time"] == -0.25 + row["shot"] / 100000.0
        assert row["actual_rate_hz"] == 1000.0
        assert row["prefix_samples"] == 100


def test_ties_and_permuted_inputs_have_identical_output():
    ids = list(range(1, 361))
    shuffled = ids.copy()
    random.Random(20261031).shuffle(shuffled)

    def tied_reader(shot):
        return synthetic_prefix(shot, current=100_000.0 + (shot % 7) * 1000.0)

    expected_current_order = sorted(ids, key=lambda shot: (shot % 7, shot))
    expected_ranks = EXPECTED_RANKS + [rank for rank in range(1, 361) if rank not in EXPECTED_RANKS]
    expected_shots = [expected_current_order[rank - 1] for rank in expected_ranks]
    baseline = stratified.order_stream(ids, reader=tied_reader)
    assert [row["shot"] for row in baseline] == expected_shots
    assert stratified.order_stream(reversed(ids), reader=tied_reader) == baseline
    assert stratified.order_stream(shuffled, reader=tied_reader) == baseline


def test_reads_each_synthetic_prefix_exactly_once_and_nothing_else():
    ids = list(range(360, 0, -1))
    reads = []
    slices = []

    class CurrentOnly(dict):
        def __getitem__(self, key):
            assert key in {"planning_ip", "rate", "start"}, f"unexpected field: {key}"
            return super().__getitem__(key)

    class PrefixOnly:
        def __init__(self, shot, count):
            self.shot, self.count = shot, count

        def __array__(self, *args, **kwargs):
            pytest.fail("the complete shot must never be converted or read")

        def __getitem__(self, selection):
            assert selection == slice(None, self.count)
            slices.append((self.shot, selection))
            return np.full(self.count, float(self.shot * 1000))

    def reader(shot):
        reads.append(shot)
        rate, count = (1000.25, 101) if shot % 2 == 0 else (1000.0, 100)
        return CurrentOnly(planning_ip=PrefixOnly(shot, count), rate=rate, start=0.125)

    rows = stratified.order_stream(ids, reader=reader)
    assert reads == ids
    assert slices == [(shot, slice(None, 101 if shot % 2 == 0 else 100)) for shot in ids]
    for row in rows:
        assert row["prefix_samples"] == (101 if row["shot"] % 2 == 0 else 100)
        assert row["actual_rate_hz"] == (1000.25 if row["shot"] % 2 == 0 else 1000.0)
        assert row["signal_start_time"] == 0.125


def test_reader_default_is_the_existing_audited_reader():
    assert inspect.signature(stratified.order_stream).parameters["reader"].default is order.read_prefix


def test_roster_and_reader_payloads_are_not_mutated():
    ids = list(range(360, 0, -1))
    original_ids = ids.copy()
    payloads = {shot: synthetic_prefix(shot) for shot in ids}
    snapshots = {
        shot: (payload["planning_ip"].copy(), payload["rate"], payload["start"])
        for shot, payload in payloads.items()
    }
    stratified.order_stream(ids, reader=payloads.__getitem__)
    assert ids == original_ids
    for shot, payload in payloads.items():
        ip, rate, start = snapshots[shot]
        np.testing.assert_array_equal(payload["planning_ip"], ip)
        assert payload["rate"] == rate
        assert payload["start"] == start


@pytest.mark.parametrize("size", [0, 1, 20, 119, 120, 359, 361, 400])
def test_wrong_roster_size_fails_before_any_reader_call(size):
    def never_read(shot):
        pytest.fail(f"invalid roster must not read shot {shot}")

    with pytest.raises(ValueError):
        stratified.order_stream(range(1, size + 1), reader=never_read)


def test_noniterable_roster_is_rejected_before_reading():
    def never_read(shot):
        pytest.fail(f"invalid roster must not read shot {shot}")

    with pytest.raises(ValueError):
        stratified.order_stream(None, reader=never_read)


@pytest.mark.parametrize("bad_shot", [0, -1, True, 1.0, "1", [], 2])
def test_360_roster_still_uses_existing_id_validation_before_reading(bad_shot):
    ids = list(range(1, 361))
    ids[0] = bad_shot  # 2 also duplicates the second ID.

    def never_read(shot):
        pytest.fail(f"invalid roster must not read shot {shot}")

    with pytest.raises(ValueError):
        stratified.order_stream(ids, reader=never_read)


@pytest.mark.parametrize("bad_current", [np.nan, np.inf, -np.inf, 0.0])
def test_invalid_current_fails_without_skipping_any_shot(bad_current):
    def reader(shot):
        return synthetic_prefix(shot, current=bad_current if shot == 5 else None)

    with pytest.raises(ValueError):
        stratified.order_stream(range(1, 361), reader=reader)


def test_reader_error_propagates_without_fallback():
    def broken(shot):
        raise OSError("prefix unavailable")

    with pytest.raises(OSError, match="prefix unavailable"):
        stratified.order_stream(range(1, 361), reader=broken)
