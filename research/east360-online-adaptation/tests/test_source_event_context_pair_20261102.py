"""Synthetic CPU-only pairing tests; no actual cache fit or output writes."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.source_event_tasks_20261102 import (
    context_pair as c,
)
from experiments.risk_aligned_online.event_conditional_feedback_20261102 import (
    context_fit as old,
)
from experiments.risk_aligned_online.event_conditional_feedback_20261102.context import (
    probabilities,
)


def toy_data():
    rng = np.random.default_rng(17)
    family = rng.permutation(np.repeat(np.arange(6), [8, 6, 4, 3, 2, 1]))
    event = np.repeat(np.arange(24), np.tile([1, 2, 4], 8))
    train = dict(
        raw=rng.normal(size=(len(event), 115)).astype(np.float32),
        target=family[event],
        event_id=event,
        shot=1000 + event,
        endpoint=np.arange(len(event)),
        outcome=event % 2,
    )
    val = dict(
        raw=rng.normal(size=(12, 115)).astype(np.float32),
        target=np.repeat(np.arange(6), 2),
        event_id=np.repeat(np.arange(30, 36), 2),
        shot=np.repeat(np.arange(2000, 2006), 2),
        endpoint=np.arange(12),
        outcome=np.repeat(np.arange(6) % 2, 2),
    )
    return train, val


def test_design_is_event_equal_not_family_dependent_and_uses_old_shuffle():
    train, _ = toy_data()
    sampling, targets, class_counts, class_weights = c.training_design(train)
    weights = old.balanced_event_weights(train["event_id"])
    np.testing.assert_array_equal(sampling, weights / weights.sum())
    old_sampling = {}
    for name in ("true", "shuffled"):
        y, old_sampling[name] = old.training_targets(train, shuffled=name == "shuffled")
        np.testing.assert_array_equal(targets[name], y)
        counts = np.bincount(y, weights=weights, minlength=6)
        np.testing.assert_allclose(class_counts[name], counts)
        np.testing.assert_allclose(class_weights[name], 24 / (6 * counts))
        np.testing.assert_allclose(
            np.bincount(y, weights=sampling * class_weights[name][y], minlength=6),
            np.full(6, 1 / 6),
        )
    assert np.any(targets["true"] != targets["shuffled"])
    assert not np.array_equal(old_sampling["true"], old_sampling["shuffled"])


def test_actual_forward_batches_are_identical_and_schedule_hash_is_replayable(
    monkeypatch,
):
    train, val = toy_data()
    seen, inits = [], []
    forward, initialize = torch.nn.Linear.forward, torch.nn.Linear.__init__

    def init(self, *args, **kwargs):
        inits.append(kwargs.get("device"))
        initialize(self, *args, **kwargs)

    def capture(self, x):
        assert x.device.type == "cpu"
        seen.append(x.detach().numpy().copy())
        return forward(self, x)

    monkeypatch.setattr(torch.nn.Linear, "__init__", init)
    monkeypatch.setattr(torch.nn.Linear, "forward", capture)
    _, _, evidence = c.fit_pair(train, val, steps=3)
    assert inits == ["cpu"]  # One initialization, two deep copies.
    assert len(seen) == 6
    rng = np.random.default_rng(131)
    x, _, _ = old.standardize_train(train)
    sampling, _, _, _ = c.training_design(train)
    digest = hashlib.sha256()
    for step in range(3):
        idx = rng.choice(len(x), 512, replace=True, p=sampling)
        digest.update(np.asarray(idx, dtype="<i8").tobytes())
        np.testing.assert_array_equal(seen[2 * step], x[idx])
        np.testing.assert_array_equal(seen[2 * step + 1], x[idx])
    assert evidence["schedule_sha256"] == digest.hexdigest()
    assert evidence["paired_minibatches"] is True
    for arm in evidence["arms"].values():
        assert arm["same_initial"] is True
        assert arm["initial_parameters_sha256"] == evidence["initial_parameters_sha256"]
        assert arm["schedule_sha256"] == digest.hexdigest()
        assert arm["optimizer_steps"] == 3


def test_parameter_hash_is_canonical_float32_weight_then_bias():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(131)
        head = torch.nn.Linear(115, 6, device="cpu")
    expected = hashlib.sha256(
        head.weight.detach().numpy().astype("<f4").tobytes()
        + head.bias.detach().numpy().astype("<f4").tobytes()
    ).hexdigest()
    assert c.parameter_sha256(head) == expected


def test_pair_matches_explicit_ce_times_class_weight_mean_not_torch_weighted_mean():
    train, val = toy_data()
    heads, metrics, evidence = c.fit_pair(train, val, steps=2)
    x, mean, scale = old.standardize_train(train)
    sampling, targets, _, class_weights = c.training_design(train)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(131)
        initial = torch.nn.Linear(115, 6, device="cpu")
    assert evidence["initial_parameters_sha256"] == c.parameter_sha256(initial)
    models = {name: copy.deepcopy(initial) for name in heads}
    optimizers = {
        name: torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.001)
        for name, m in models.items()
    }
    rng = np.random.default_rng(131)
    differs_from_weighted_ce = False
    for _ in range(2):
        idx = rng.choice(len(x), 512, replace=True, p=sampling)
        for name, model in models.items():
            logits = model(torch.from_numpy(x)[idx])
            y = torch.from_numpy(targets[name])[idx]
            cw = torch.as_tensor(class_weights[name], dtype=torch.float32)
            loss = (
                torch.nn.functional.cross_entropy(logits, y, reduction="none") * cw[y]
            ).mean()
            weighted_mean = torch.nn.functional.cross_entropy(logits, y, weight=cw)
            differs_from_weighted_ce |= not torch.isclose(loss, weighted_mean).item()
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[name].step()
    assert differs_from_weighted_ce
    for name, model in models.items():
        assert set(heads[name]) == {"mean", "scale", "weight", "bias"}
        np.testing.assert_array_equal(
            heads[name]["weight"], model.weight.detach().numpy()
        )
        np.testing.assert_array_equal(heads[name]["bias"], model.bias.detach().numpy())
        np.testing.assert_array_equal(heads[name]["mean"], mean)
        np.testing.assert_array_equal(heads[name]["scale"], scale)
        expected = old.evaluate_validation(
            val, probabilities(val["raw"], heads[name]).argmax(1)
        )
        assert metrics[name] == expected


def test_validation_changes_do_not_affect_fit_or_schedule_and_std_floor_is_legacy():
    train, val = toy_data()
    train["raw"][:, 47] = np.where(train["event_id"] % 2, 1e-5, -1e-5)
    first, _, receipt = c.fit_pair(train, val, steps=2)
    changed = copy.deepcopy(val)
    changed["raw"] *= 1e5
    changed["target"] = (changed["target"] + 1) % 6
    second, _, repeated = c.fit_pair(train, changed, steps=2)
    assert first == second
    assert receipt == repeated
    assert first["true"]["scale"][47] == pytest.approx(1e-5)
    np.testing.assert_array_equal(first["true"]["mean"], first["shuffled"]["mean"])


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_bad_steps_rejected(steps):
    train, val = toy_data()
    with pytest.raises(ValueError, match="steps"):
        c.fit_pair(train, val, steps=steps)


def test_missing_training_family_rejected():
    train, _ = toy_data()
    train["target"][train["target"] == 5] = 4
    with pytest.raises(ValueError, match="six"):
        c.training_design(train)


def test_source_helpers_bound_as_imported_code_not_old_plan_dependency():
    bound = c.imported_code_hashes()
    names = {Path(p).name for p in bound}
    assert {
        "context_pair.py",
        "context_fit.py",
        "context.py",
        "event_prefix_probe.py",
        "probe.py",
        "io23.py",
    } <= names
    assert all(Path(p).suffix == ".py" for p in bound)
    assert str(old.PLAN) not in bound
    assert all(len(h) == 64 for h in bound.values())


def test_existing_output_rejected_before_cache_access(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda p: True)
    monkeypatch.setattr(c, "load_cache", lambda: pytest.fail("cache must not be read"))
    with pytest.raises(FileExistsError):
        c.main(Path("/fake-existing-output"))


def test_main_writes_only_four_new_files_with_pairing_and_no_test_results(monkeypatch):
    train, val = toy_data()
    output = Path("/in-memory-output")
    written, calls = {}, []
    cache_binding = {"/fake/features.npz": "a" * 64}
    code_binding = {"/fake/context_pair.py": "a" * 64}
    receipt = dict(
        paired_minibatches=True,
        schedule_sha256="b" * 64,
        initial_parameters_sha256="c" * 64,
        arms={},
    )
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(Path, "mkdir", lambda p, **k: calls.append(p))
    monkeypatch.setattr(
        c, "load_cache", lambda: (dict(train=train, val=val), cache_binding)
    )
    monkeypatch.setattr(c, "imported_code_hashes", lambda: code_binding)
    monkeypatch.setattr(c, "sha256", lambda p: "a" * 64)
    monkeypatch.setattr(c, "write_json", lambda p, obj: written.update({p.name: obj}))

    def fit(a, b, steps=1024):
        assert a is train and b is val and steps == 1024
        return dict(true={}, shuffled={}), dict(true={}, shuffled={}), receipt

    monkeypatch.setattr(c, "fit_pair", fit)
    c.main(output)
    assert calls == [output]
    assert list(written) == [
        "protocol.json",
        "context_heads.json",
        "validation.json",
        "completed.json",
    ]
    protocol, done = written["protocol.json"], written["completed.json"]
    assert done["status"] == "COMPLETED" and done["paired_minibatches"] is True
    assert done["pairing"] == receipt
    assert done["training_counts"] == old.counts(train)
    assert done["validation_counts"] == old.counts(val)
    assert done["bound_files_sha256"] == {**cache_binding, **code_binding}
    assert len(done["artifact_sha256"]) == 3
    for key in (
        "source_parameter_updates",
        "base_model_parameter_updates",
        "east_parameter_updates",
        "east_data_reads",
        "test_fit_rows",
        "test_evaluated_rows",
    ):
        assert done[key] == 0
    assert done["source_model_loaded"] is False
    assert protocol["standard_deviation_floor"] == 1e-6
    assert protocol["cache_contains_test"] is True
    assert "includes TEST bytes" in protocol["test_usage"]
    assert protocol["east_gain_measured"] is False
    assert set(written["validation.json"]["heads"]) == {"true", "shuffled"}


def test_input_drift_prevents_completed_marker(monkeypatch):
    train, val = toy_data()
    written = []
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(Path, "mkdir", lambda p, **k: None)
    monkeypatch.setattr(
        c, "load_cache", lambda: (dict(train=train, val=val), {"/cache": "a" * 64})
    )
    monkeypatch.setattr(c, "imported_code_hashes", lambda: {})
    monkeypatch.setattr(c, "sha256", lambda p: "drift")
    monkeypatch.setattr(c, "write_json", lambda p, obj: written.append(p.name))
    monkeypatch.setattr(c, "fit_pair", lambda *a, **k: ({}, {}, {}))
    with pytest.raises(ValueError, match="changed"):
        c.main(Path("/in-memory-output"))
    assert "completed.json" not in written
