"""Synthetic CPU tests only: never fit/evaluate the real feature cache."""

import copy
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_conditional_feedback_20261102 import (
    context_fit as c,
)
from experiments.risk_aligned_online.event_prefix_probe import (
    balanced_event_weights,
    shuffle_event_targets,
)


def toy_cache():
    rng = np.random.default_rng(17)
    event = np.repeat(np.arange(36), np.tile([1, 2, 3], 12))
    role = np.where(event < 24, "train", np.where(event < 30, "val", "test"))
    return dict(
        raw=rng.normal(size=(len(event), 115)).astype(np.float32),
        target=event % 6,
        shot=100 + event,
        endpoint=np.arange(len(event)),
        event_id=event,
        outcome=(event // 6) % 2,
        role=role,
    )


def test_only_variation_blocks_and_no_mnm_are_used_without_mutation():
    x = np.arange(230, dtype=np.float32).reshape(2, 115)
    old = x.copy()
    actual = c.select_features(x)
    expected = x.copy()
    expected[:, :46] = 0
    expected[:, [23 * b + c.RAW23.index("MNM") for b in range(5)]] = 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, old)


def test_selection_delegates_to_shared_context_without_reimplementing(monkeypatch):
    module = ModuleType(c.__package__ + ".context")
    raw = np.ones((2, 115), dtype=np.float32)
    selected = np.zeros_like(raw)
    calls = []

    def selector(value):
        calls.append(value)
        return selected

    module.select_variation_features = selector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert c.select_features(raw) is selected
    assert len(calls) == 1 and calls[0] is raw


@pytest.mark.parametrize("bad", [np.zeros((2, 114)), np.full((2, 115), np.nan)])
def test_bad_feature_arrays_rejected(bad):
    with pytest.raises(ValueError):
        c.select_features(bad)


def test_splits_never_transform_or_validate_test_values():
    cache = toy_cache()
    test = cache["role"] == "test"
    cache["raw"][test] = np.nan
    cache["target"][test] = 99
    splits = c.split_cache(cache)
    assert set(splits) == {"train", "val"}
    for role, data in splits.items():
        assert c.counts(data)["rows"] == int((cache["role"] == role).sum())
        assert np.isfinite(data["raw"]).all()
        assert (data["target"] < 6).all()


@pytest.mark.parametrize("defect", ["class", "outcome", "event", "shot_role", "role"])
def test_invalid_training_metadata_rejected(defect):
    cache = toy_cache()
    if defect == "class":
        cache["target"][0] = 6
    elif defect == "outcome":
        cache["outcome"][0] = 3
    elif defect == "event":
        cache["target"][2] = 5  # A two-row event now has two classes.
    elif defect == "shot_role":
        cache["shot"][cache["role"] == "val"] = cache["shot"][0]
    else:
        cache["role"][0] = "other"
    with pytest.raises(ValueError):
        c.split_cache(cache)


def test_train_equal_event_moments_and_standard_deviation_floor():
    train = c.split_cache(toy_cache())["train"]
    train["raw"][:, 46] = train["event_id"]
    train["raw"][:, 47] = 7
    x, mean, scale = c.standardize_train(train)
    w = balanced_event_weights(train["event_id"])
    assert mean[46] == pytest.approx(np.average(train["event_id"], weights=w))
    assert mean[46] != pytest.approx(train["event_id"].mean())
    assert scale[47] == pytest.approx(1e-6)  # Matches the audited source probe.
    assert np.all(x[:, :46] == 0)
    assert np.all(mean[:46] == 0)
    np.testing.assert_allclose(np.average(x[:, 46], weights=w), 0, atol=1e-7)


@pytest.mark.parametrize("shuffled", [False, True])
def test_sampling_and_shuffling_exactly_match_old_train_only_formula(shuffled):
    train = c.split_cache(toy_cache())["train"]
    boundaries = np.quantile(np.unique(train["shot"]), [0.25, 0.5, 0.75])
    strata = np.searchsorted(boundaries, train["shot"]) + 4 * train["outcome"]
    expected_y = (
        shuffle_event_targets(train["target"], train["event_id"], strata, seed=131)
        if shuffled
        else train["target"]
    )
    w = balanced_event_weights(train["event_id"])
    expected_p = w / np.bincount(expected_y, weights=w, minlength=6)[expected_y]
    expected_p /= expected_p.sum()
    y, p = c.training_targets(train, shuffled=shuffled)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(p, expected_p)
    np.testing.assert_allclose(np.bincount(y, weights=p, minlength=6), np.ones(6) / 6)


def test_missing_training_family_is_rejected():
    train = c.split_cache(toy_cache())["train"]
    train["target"][train["target"] == 5] = 4
    with pytest.raises(ValueError, match="six"):
        c.training_targets(train, shuffled=False)


def test_weighted_validation_confusion_recall_and_macro_f1():
    val = dict(
        target=np.array([0, 0, 1]),
        event_id=np.array([10, 10, 11]),
        shot=np.array([1, 1, 2]),
    )
    result = c.evaluate_validation(val, np.array([0, 1, 1]))
    matrix = np.asarray(result["confusion_event_weighted"])
    np.testing.assert_array_equal(matrix[:2, :2], [[0.5, 0.5], [0, 1]])
    assert matrix.sum() == 2
    assert result["per_class_recall"] == dict(A=0.5, B=1.0, C=0.0, D=0.0, E=0.0, F=0.0)
    assert result["event_weighted_macro_f1"] == pytest.approx((2 / 3 + 0.8) / 6)


@pytest.mark.parametrize("shuffled", [False, True])
def test_toy_fit_is_cpu_fixed_budget_repeatable_and_holdout_does_not_fit(
    monkeypatch, shuffled
):
    monkeypatch.setattr(c, "STEPS", 3)
    monkeypatch.setattr(c, "BATCH_SIZE", 8)
    splits = c.split_cache(toy_cache())
    first, result = c.fit_head(splits["train"], splits["val"], shuffled=shuffled)
    changed = copy.deepcopy(splits["val"])
    changed["raw"][:] = 1e8
    changed["target"] = (changed["target"] + 1) % 6
    second, other = c.fit_head(splits["train"], changed, shuffled=shuffled)
    assert first == second
    assert result["optimizer_steps"] == other["optimizer_steps"] == 3
    assert result["sampler_seed"] == result["initialization_seed"] == 131
    assert np.asarray(first["weight"]).shape == (6, 115)
    assert np.asarray(first["bias"]).shape == (6,)
    assert set(first) == {"mean", "scale", "weight", "bias"}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(131)
        old = torch.nn.Linear(115, 6, device="cpu")
    x, mean, scale = c.standardize_train(splits["train"])
    y, p = c.training_targets(splits["train"], shuffled=shuffled)
    rng = np.random.default_rng(131)
    optimizer = torch.optim.AdamW(old.parameters(), lr=0.003, weight_decay=0.001)
    for _ in range(3):
        idx = rng.choice(len(y), 8, replace=True, p=p)
        loss = torch.nn.functional.cross_entropy(
            old(torch.from_numpy(x)[idx]), torch.from_numpy(y)[idx]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    np.testing.assert_array_equal(first["mean"], mean)
    np.testing.assert_array_equal(first["scale"], scale)
    np.testing.assert_array_equal(first["weight"], old.weight.detach().numpy())


def test_cache_location_is_main_only_and_loader_binds_metadata_without_model_load(
    monkeypatch,
):
    assert c.CACHE == Path(
        "/media/hetaisheng/044A81D94A81C83E/Fuison/outputs/event_prefix_probe_20261021"
    )
    cache = toy_cache()
    metadata = {
        "completed.json": dict(
            status="COMPLETE_PROBE_ONLY",
            feature_cache_sha256=c.CACHE_SHA,
            source_file_and_parameters_unchanged=True,
        ),
        "protocol.json": dict(
            source_parent_sha256=c.SOURCE_SHA, labels_sha256=c.LABELS_SHA
        ),
    }

    class Archive(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(c, "read_json", lambda p: metadata[p.name])
    monkeypatch.setattr(c, "sha256", lambda p: c.CACHE_SHA)
    monkeypatch.setattr(c.np, "load", lambda p, **kw: Archive(cache))
    monkeypatch.setattr(
        c, "EXPECTED_COUNTS", {r: c.counts(d) for r, d in c.split_cache(cache).items()}
    )
    monkeypatch.setattr(c, "CACHE_ROWS", len(cache["role"]))
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("no source model load")
    )
    hdf_calls = []

    def hdf_outcome(device, sid):
        assert device == "J-TEXT" and 100 <= sid < 130
        hdf_calls.append(sid)
        return bool(sid % 2), 3.0 if sid % 2 else None

    monkeypatch.setattr(c.io23, "read_outcome", hdf_outcome)
    splits, bound = c.load_cache()
    assert set(splits) == {"train", "val"}
    assert str(c.CACHE / "features.npz") in bound
    assert hdf_calls == list(range(130 - 30, 130))
    for data in splits.values():
        np.testing.assert_array_equal(data["outcome"], data["shot"] % 2)
    metadata["protocol.json"]["source_parent_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source"):
        c.load_cache()
    metadata["protocol.json"]["source_parent_sha256"] = c.SOURCE_SHA
    monkeypatch.setattr(c, "sha256", lambda p: "wrong")
    with pytest.raises(ValueError, match="hash"):
        c.load_cache()


def test_existing_output_rejected_before_loading_or_fitting(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda p: True)
    monkeypatch.setattr(c, "load_cache", lambda: pytest.fail("must reject first"))
    with pytest.raises(FileExistsError):
        c.main(Path("/not-a-real-output"))


def test_main_writes_four_new_jsons_and_no_test_or_source_claims(monkeypatch):
    output = Path("/fake-in-memory-output")
    splits = c.split_cache(toy_cache())
    written, calls = {}, []
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(Path, "mkdir", lambda p, **kw: calls.append(p))
    monkeypatch.setattr(c, "load_cache", lambda: (splits, {"/fake-cache": "a" * 64}))
    monkeypatch.setattr(c, "sha256", lambda p: "a" * 64)
    monkeypatch.setattr(c, "write_json", lambda p, obj: written.update({p.name: obj}))

    def fit(train, val, *, shuffled):
        assert train is splits["train"] and val is splits["val"]
        return {"test_fake": shuffled}, {"optimizer_steps": 1024}

    monkeypatch.setattr(c, "fit_head", fit)
    c.main(output)
    assert calls == [output]
    assert set(written) == {
        "protocol.json",
        "context_heads.json",
        "validation.json",
        "completed.json",
    }
    assert set(written["context_heads.json"]) == {"true", "shuffled"}
    done = written["completed.json"]
    assert done["status"] == "COMPLETED"
    for key in (
        "source_parameter_updates",
        "east_parameter_updates",
        "east_data_reads",
        "test_fit_rows",
        "test_evaluated_rows",
    ):
        assert done[key] == 0
    assert done["source_model_loaded"] is False
    assert done["training_counts"] == c.counts(splits["train"])
    assert done["validation_counts"] == c.counts(splits["val"])
    assert len(done["artifact_sha256"]) == 3
    protocol = written["protocol.json"]
    assert protocol["cache_contains_test"] is True
    assert "decompression includes TEST bytes" in protocol["test_usage"]
    assert protocol["standard_deviation_floor"] == 1e-6
    assert protocol["east_gain_measured"] is False
