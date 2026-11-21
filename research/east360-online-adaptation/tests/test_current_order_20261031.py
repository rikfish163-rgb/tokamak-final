"""Outcome-blind, current-only, first-100-ms offline ordering contract."""

import math

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.current_order_20261031 import order as current_order


def signals(current, *, rate=1000.0, start=0.375, count=160):
    return {
        "planning_ip": np.full(count, current, dtype=np.float64),
        "rate": rate,
        "start": start,
    }


def test_order_is_current_only_with_exact_row_fields_and_prefix():
    class CurrentOnly(dict):
        def __getitem__(self, key):
            assert key in {"planning_ip", "rate", "start"}
            return super().__getitem__(key)

    raw = {
        1: CurrentOnly(signals(-300_000.0)),
        2: CurrentOnly(signals(100_000.0)),
        3: CurrentOnly(signals(-200_000.0)),
    }
    raw[2]["planning_ip"][100:] = np.nan
    rows = current_order.order_stream([1, 3, 2], reader=raw.__getitem__)

    assert [row["shot"] for row in rows] == [2, 3, 1]
    assert rows[0] == {
        "shot": 2,
        "initial_ip_A": 100_000.0,
        "current_energy_proxy": 1.0,
        "proxy": 1.0,
        "prefix_samples": 100,
        "actual_rate_hz": 1000.0,
        "signal_start_time": 0.375,
    }
    assert [row["proxy"] for row in rows] == [1.0, 4.0, 9.0]
    assert all(set(row) == set(rows[0]) for row in rows)


def test_all_360_input_ids_are_retained_and_each_read_once():
    seen = []

    def reader(shot):
        seen.append(shot)
        return signals(float(shot) * 1000.0)

    ids = list(range(360, 0, -1))
    rows = current_order.order_stream(ids, reader=reader)
    assert seen == ids
    assert [row["shot"] for row in rows] == list(range(1, 361))
    assert len(rows) == len(ids) == len({row["shot"] for row in rows})


def test_ties_are_broken_by_shot_id_independent_of_input_order():
    raw = {9: signals(-100_000.0), 3: signals(100_000.0), 7: signals(100_000.0)}
    for ids in ([9, 3, 7], [7, 9, 3]):
        assert [r["shot"] for r in current_order.order_stream(ids, raw.__getitem__)] == [3, 7, 9]


@pytest.mark.parametrize("factor", [0.01, 7.5, -2.0])
def test_common_nonzero_current_scaling_preserves_order(factor):
    raw = {4: signals(-450_000.0), 1: signals(125_000.0), 2: signals(-250_000.0)}
    scaled = {
        shot: {**record, "planning_ip": record["planning_ip"] * factor}
        for shot, record in raw.items()
    }
    before = current_order.order_stream([4, 1, 2], raw.__getitem__)
    after = current_order.order_stream([4, 1, 2], scaled.__getitem__)
    assert [r["shot"] for r in before] == [r["shot"] for r in after] == [1, 2, 4]
    np.testing.assert_allclose(
        [r["proxy"] for r in after], np.array([r["proxy"] for r in before]) * factor**2
    )


def test_proxy_uses_median_of_absolute_prefix_not_the_mean_or_future():
    ip = np.r_[np.full(49, -1_000.0), np.full(51, 200_000.0), np.full(50, 1e9)]
    raw = {**signals(0.0), "planning_ip": ip}
    row = current_order.order_stream([5], lambda shot: raw)[0]
    assert row["initial_ip_A"] == 200_000.0
    assert row["proxy"] == row["current_energy_proxy"] == 4.0


@pytest.mark.parametrize("ids", [[], None, [0], [-1], [True], [1.0], ["1"], [1, 1], [1, []]])
def test_invalid_roster_fails_before_any_read(ids):
    def never_read(shot):
        pytest.fail(f"invalid roster must not read shot {shot}")

    with pytest.raises(ValueError):
        current_order.order_stream(ids, reader=never_read)


@pytest.mark.parametrize(
    "rate, n",
    [
        (999.5, 100),
        (999.5004995004996, 100),
        (1000.0, 100),
        (1000.0000000005, 100),
        (1000.000000002, 101),
        (1000.25, 101),
        (1000.5, 101),
    ],
)
def test_nominal_clock_and_exact_prefix_sample_count(rate, n):
    start = -0.123456789
    row = current_order.order_stream([1], lambda shot: signals(200_000.0, rate=rate, start=start))[0]
    assert n == math.ceil(0.1 * rate - 1e-10) == row["prefix_samples"]
    assert row["actual_rate_hz"] == rate
    assert row["signal_start_time"] == start
    assert row["signal_start_time"] + (n - 1) / row["actual_rate_hz"] == start + (n - 1) / rate


@pytest.mark.parametrize(
    "rate", [0.0, -1000.0, 999.499, 1000.501, np.nan, np.inf, -np.inf,
             np.complex128(1000 + 1j), None, "1000.0", "bad", [], [1000, 1000]]
)
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        current_order.order_stream([1], lambda shot: signals(100_000.0, rate=rate))


@pytest.mark.parametrize("start", [np.nan, np.inf, -np.inf, np.complex128(0.5 + 1j), None,
                                    "0.375", "bad", [], [0, 1]])
def test_invalid_start_time_rejected(start):
    with pytest.raises(ValueError):
        current_order.order_stream([1], lambda shot: signals(100_000.0, start=start))


@pytest.mark.parametrize(
    "bad_ip",
    [
        np.full(99, 100_000.0),
        np.full(100, np.nan),
        np.r_[np.full(99, 100_000.0), np.nan],
        np.r_[np.full(99, 100_000.0), np.inf],
        np.zeros(100),
        np.r_[np.zeros(51), np.full(49, 100_000.0)],
        np.zeros((100, 1)),
        np.full(100, 100_000.0 + 1j),
        np.array([np.complex128(100_000.0 + 1j)] * 100, dtype=object),
        ["100000"] * 100,
        ["bad"] * 100,
    ],
)
def test_bad_prefix_rejected_without_skipping(bad_ip):
    raw = {1: signals(100_000.0), 2: {**signals(100_000.0), "planning_ip": bad_ip}}
    with pytest.raises(ValueError):
        current_order.order_stream([1, 2], raw.__getitem__)


def test_nonfinite_energy_is_rejected():
    with pytest.raises(ValueError):
        current_order.order_stream([1], lambda shot: signals(1e200))


def test_io_errors_propagate_instead_of_triggering_exclusion_or_fallback():
    def broken(shot):
        raise OSError("read failed")

    with pytest.raises(OSError, match="read failed"):
        current_order.order_stream([1], reader=broken)


@pytest.mark.parametrize("rate, n", [(1000.0, 100), (1000.25, 101)])
def test_injected_reader_is_sliced_before_any_array_conversion(rate, n):
    class PrefixOnly:
        def __array__(self, *args, **kwargs):
            pytest.fail("converting the whole signal would touch the future")

        def __getitem__(self, selection):
            assert selection == slice(None, n)
            return np.full(n, 100_000.0)

    raw = {**signals(100_000.0, rate=rate), "planning_ip": PrefixOnly()}
    row = current_order.order_stream([1], reader=lambda shot: raw)[0]
    assert row["initial_ip_A"] == 100_000.0
    assert row["prefix_samples"] == n


def test_real_hdf_reader_touches_only_ip_and_exact_prefix(tmp_path, monkeypatch):
    paths = {}
    specs = {11: (999.5004995004996, 0.375, -200_000.0, 100),
             22: (1000.25, -0.125, 100_000.0, 101)}
    for shot, (rate, start, current, n) in specs.items():
        path = paths[shot] = tmp_path / f"{shot}.hdf5"
        with h5py.File(path, "w") as handle:
            ip = handle.create_dataset("data/ip", data=np.r_[np.full(n, current), np.full(20, np.nan)])
            ip.attrs["SampleRate"] = rate
            ip.attrs["StartTime"] = start
            handle.create_dataset("data/Bt", data=[123.0])
            handle.create_dataset("meta/IsDisrupt", data=[1])

    monkeypatch.setattr(io23, "path_for", lambda device, shot: paths[shot] if device == "EAST" else None)
    file_getitem = h5py.File.__getitem__
    dataset_getitem = h5py.Dataset.__getitem__
    file_reads = []
    data_reads = []

    def checked_file_getitem(handle, key):
        file_reads.append(key)
        assert key == "data/ip", f"forbidden HDF access: {key}"
        return file_getitem(handle, key)

    def checked_dataset_getitem(dataset, selection):
        shot = int(dataset.file.filename.rsplit("/", 1)[-1].split(".")[0])
        assert dataset.name == "/data/ip"
        assert selection == slice(None, specs[shot][3])
        data_reads.append((shot, selection))
        return dataset_getitem(dataset, selection)

    monkeypatch.setattr(h5py.File, "__getitem__", checked_file_getitem)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_dataset_getitem)
    rows = current_order.order_stream([11, 22])

    assert file_reads == ["data/ip", "data/ip"]
    assert data_reads == [(11, slice(None, 100)), (22, slice(None, 101))]
    assert [row["shot"] for row in rows] == [22, 11]
    for row in rows:
        rate, start, current, n = specs[row["shot"]]
        assert row["actual_rate_hz"] == rate
        assert row["signal_start_time"] == start
        assert row["initial_ip_A"] == abs(current)
        assert row["prefix_samples"] == n


def test_ip_soft_link_cannot_redirect_reader_to_another_signal(tmp_path, monkeypatch):
    path = tmp_path / "1.hdf5"
    with h5py.File(path, "w") as handle:
        bt = handle.create_dataset("data/Bt", data=np.ones(110))
        bt.attrs["SampleRate"] = 1000.0
        bt.attrs["StartTime"] = 0.0
        handle["data/ip"] = h5py.SoftLink("/data/Bt")
    monkeypatch.setattr(io23, "path_for", lambda device, shot: path)
    with pytest.raises(ValueError):
        current_order.order_stream([1])


@pytest.mark.parametrize("storage", ["external", "virtual"])
def test_ip_storage_cannot_redirect_to_another_file(tmp_path, monkeypatch, storage):
    path = tmp_path / "1.hdf5"
    with h5py.File(path, "w") as handle:
        if storage == "external":
            ip = handle.create_dataset(
                "data/ip", (110,), dtype="f8",
                external=[(str(tmp_path / "other.raw"), 0, h5py.h5f.UNLIMITED)],
            )
        else:
            layout = h5py.VirtualLayout(shape=(110,), dtype="f8")
            layout[:] = h5py.VirtualSource(str(tmp_path / "other.hdf5"), "data/Bt", shape=(110,))
            ip = handle.create_virtual_dataset("data/ip", layout)
        ip.attrs["SampleRate"] = 1000.0
        ip.attrs["StartTime"] = 0.0
    monkeypatch.setattr(io23, "path_for", lambda device, shot: path)
    with pytest.raises(ValueError):
        current_order.order_stream([1])


@pytest.mark.parametrize(
    "rate, start, shape",
    [(np.nan, 0.0, (110,)), (1000.0, np.nan, (110,)),
     (1000.0, None, (110,)), (1000.0, 0.0, (99,)),
     (1000.25, 0.0, (100,)), (1000.0, 0.0, (100, 1))],
)
def test_real_hdf_invalid_clock_or_prefix_fails_closed(tmp_path, monkeypatch, rate, start, shape):
    path = tmp_path / "1.hdf5"
    with h5py.File(path, "w") as handle:
        ip = handle.create_dataset("data/ip", data=np.full(shape, 100_000.0))
        ip.attrs["SampleRate"] = rate
        if start is not None:
            ip.attrs["StartTime"] = start
    monkeypatch.setattr(io23, "path_for", lambda device, shot: path)
    with pytest.raises(ValueError):
        current_order.order_stream([1])
