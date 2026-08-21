from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _modules():
    source = importlib.import_module("train_jtext_source_transformer")
    grid = importlib.import_module("grid_search")
    return source, grid


def _source_checkpoint(path: Path, grid, *, seed: int = 131) -> dict:
    torch.manual_seed(seed)
    encoder = grid.Enc(17)
    head = grid.Head()
    payload = {
        "schema_version": grid.SOURCE_CHECKPOINT_SCHEMA,
        "architecture_version": grid.SOURCE_ARCHITECTURE_VERSION,
        "feature_names": list(grid.RH.REBUILD),
        "model_config": grid.source_model_config(17),
        "seed": seed,
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "scaler_center": torch.zeros(17),
        "scaler_scale": torch.ones(17),
        "split_counts": {"train": 1353, "val": 160, "test": 220},
        "split_sha256": "a" * 64,
        "test_shards_read": 0,
    }
    torch.save(payload, path)
    return payload


def test_canonical_jtext_roles_are_1353_160_220_and_disjoint() -> None:
    source, _grid = _modules()

    roles = source.load_canonical_jtext_roles(ROOT)

    assert {name: len(ids) for name, ids in roles.items()} == {
        "train": 1353,
        "val": 160,
        "test": 220,
    }
    assert set(roles["train"]).isdisjoint(roles["val"])
    assert set(roles["train"]).isdisjoint(roles["test"])
    assert set(roles["val"]).isdisjoint(roles["test"])


def test_causal_window_left_pads_short_history_without_reading_future() -> None:
    source, _grid = _modules()
    values = np.arange(12, dtype=np.float32).reshape(6, 2)

    window, valid_length = source.causal_window(values, endpoint=2, window=8)

    assert valid_length == 3
    np.testing.assert_array_equal(window[:5], np.zeros((5, 2), np.float32))
    np.testing.assert_array_equal(window[5:], values[:3])
    assert 6 not in window[:, 0]


def test_source_checkpoint_loader_binds_exact_architecture_and_features(
    tmp_path: Path,
) -> None:
    _source, grid = _modules()
    path = tmp_path / "source_checkpoint.pt"
    payload = _source_checkpoint(path, grid)

    encoder, head, loaded = grid.load_source_checkpoint(
        path, channels=17, expected_seed=131, device="cpu"
    )

    assert loaded["schema_version"] == grid.SOURCE_CHECKPOINT_SCHEMA
    for name, expected in payload["encoder"].items():
        torch.testing.assert_close(encoder.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in payload["head"].items():
        torch.testing.assert_close(head.state_dict()[name], expected, rtol=0, atol=0)

    payload["feature_names"] = list(reversed(payload["feature_names"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="feature contract"):
        grid.load_source_checkpoint(path, channels=17, expected_seed=131, device="cpu")


def test_east_adaptation_with_checkpoint_never_retrains_source(
    tmp_path: Path,
) -> None:
    _source, grid = _modules()
    path = tmp_path / "source_checkpoint.pt"
    payload = _source_checkpoint(path, grid)

    def forbidden_source_sampler(_positive_probability: float):
        raise AssertionError("source sampler must not be called after checkpoint freeze")

    rng = np.random.default_rng(9)

    def target_sampler(_positive_probability: float):
        return rng.normal(size=(128, 17)).astype(np.float32), np.float32(0.0)

    encoder, _head = grid.train_adapt(
        forbidden_source_sampler,
        target_sampler,
        17,
        epochs=0,
        bs=4,
        seed=131,
        source_checkpoint=path,
        device="cpu",
    )

    for name, expected in payload["encoder"].items():
        torch.testing.assert_close(encoder.state_dict()[name], expected, rtol=0, atol=0)


def test_east_support_threshold_keeps_the_existing_tpr_first_rule() -> None:
    import shot_alarm

    scores = np.asarray([0.9, 0.7, 0.2, 0.8, 0.6, 0.1], dtype=np.float64)
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)

    selected = shot_alarm.select_tpr_first_threshold(
        scores, labels, target_tpr=2 / 3
    )

    assert selected["threshold"] == pytest.approx(0.7)
    assert selected["tpr"] == pytest.approx(2 / 3)
    assert selected["fpr"] == pytest.approx(1 / 3)


def test_online_runner_accepts_lineage_specific_parent_directories(
    tmp_path: Path,
) -> None:
    import run_unified_online_active as unified

    a1 = tmp_path / "a1"
    adapter = tmp_path / "adapter"
    resolved = unified.resolve_parent_directories(
        seed=131, a1_dir=a1, adapter_dir=adapter
    )

    assert resolved == (a1.resolve(), adapter.resolve())
