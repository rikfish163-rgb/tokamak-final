"""Local contracts for the frozen, origin-only gate feature producer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import joint_lora_gate_features_20261104 as producer
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)


def _cases():
    ids = list(range(100, 120))
    rows = [{"shot": sid, "origin": origin} for sid in ids for origin in (135, 327, 519)]
    return ids, rows


def test_case_order_requires_exact_span_probe_order_and_no_duplicates():
    ids, rows = _cases()
    assert producer.case_order(rows, rows, ids) == [(r["shot"], r["origin"]) for r in rows]
    reordered = rows.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="order"):
        producer.case_order(reordered, rows, ids)
    duplicated = rows.copy()
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="duplicate"):
        producer.case_order(duplicated, duplicated, ids)
    with pytest.raises(ValueError, match="20"):
        producer.case_order(rows, rows, ids[:-1])


def test_case_order_rejects_invalid_origin_and_wrong_grouping():
    ids, rows = _cases()
    invalid = rows.copy()
    invalid[0] = {"shot": ids[0], "origin": 126}
    with pytest.raises(ValueError, match="origin"):
        producer.case_order(invalid, invalid, ids)
    invalid[0] = {"shot": ids[0], "origin": 136}
    with pytest.raises(ValueError, match="origin"):
        producer.case_order(invalid, invalid, ids)
    invalid = rows.copy()
    invalid[0], invalid[3] = invalid[3], invalid[0]
    with pytest.raises(ValueError, match="group"):
        producer.case_order(invalid, invalid, ids)


class FakeDataset:
    def __init__(self, values):
        self.values = values
        self.attrs = {"SampleRate": 1000.0, "StartTime": 0.0}
        self.accesses = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, selection):
        self.accesses.append(selection)
        return self.values[selection]


class SignalOnlyFile:
    def __init__(self, values):
        self.channels = {name: FakeDataset(values[:, i]) for i, name in enumerate(io23.RAW23)}
        self.paths = []

    def __getitem__(self, key):
        self.paths.append(key)
        assert key == "data", "opening metadata/outcomes is forbidden"
        return self.channels


def test_window_reads_only_origin_slice_and_matches_prior_preprocessing():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(300, 23)).astype(np.float32)
    raw[100, 0] = np.nan
    raw[101, 1] = -10.0
    raw[102, 2] = 7.0
    mapping = {"gain": [1.0] * 23, "offset": [0.0] * 23}
    mapping["gain"][12] = 0.01
    handle = SignalOnlyFile(raw)

    values, available = producer.read_causal_window(handle, 123, 191, mapping)

    old_available = np.isfinite(raw)
    old_values = np.where(old_available, raw, 0).astype(np.float32)
    old_available[:, 2] = False
    old_values[~old_available] = 0
    old = prepare_actual(dict(device="EAST", shot=123, values=old_values, available=old_available), 21, mapping)
    expected_values, expected_available = simple_data.window(old, 191, 128)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(available, expected_available)
    assert handle.paths == ["data"]
    assert all(ds.accesses == [slice(64, 192)] for ds in handle.channels.values())
    assert values.shape == available.shape == (128, 23)
    assert not available[37, 1]  # CIII missing-value sentinel
    assert not available[:, 2].any()  # EAST MNM is always unavailable


def test_window_rejects_out_of_bounds_and_unaligned_metadata_before_data_read():
    handle = SignalOnlyFile(np.ones((200, 23), dtype=np.float32))
    mapping = {"gain": [1.0] * 23, "offset": [0.0] * 23}
    with pytest.raises(ValueError, match="origin"):
        producer.read_causal_window(handle, 123, 200, mapping)
    assert all(not ds.accesses for ds in handle.channels.values())
    handle.channels[io23.RAW23[-1]].attrs["SampleRate"] = 900.0
    with pytest.raises(ValueError, match="unaligned"):
        producer.read_causal_window(handle, 123, 135, mapping)
    assert all(not ds.accesses for ds in handle.channels.values())


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.calls = 0

    def forward(self, x, available):
        assert x.shape == available.shape == (1, 128, 23)
        assert x.device.type == "cpu" and not torch.is_grad_enabled()
        self.calls += 1
        return {"latent": (x.sum() * self.weight).expand(1, 192)}


def test_feature_extraction_requires_frozen_eval_cpu_and_finite_192():
    model = FakeModel().eval()
    values = np.ones((128, 23), np.float32)
    available = np.ones((128, 23), bool)
    latent = producer.extract_latent(model, values, available)
    assert latent.shape == (192,) and latent.dtype == np.float32
    assert model.calls == 1
    model.train()
    with pytest.raises(ValueError, match="frozen"):
        producer.extract_latent(model, values, available)
    model.eval()
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        producer.extract_latent(model, values, available)


def test_hdf_size_and_nanosecond_mtime_are_both_bound():
    stat = SimpleNamespace(st_size=50, st_mtime_ns=123456)
    producer.require_hdf_stat(123, stat, {"shot": 123, "size": 50, "mtime_ns": 123456})
    with pytest.raises(ValueError, match="stat"):
        producer.require_hdf_stat(123, stat, {"shot": 123, "size": 50, "mtime_ns": 123457})
    with pytest.raises(ValueError, match="stat"):
        producer.require_hdf_stat(123, stat, {"shot": 124, "size": 50, "mtime_ns": 123456})
