import io
import json

import numpy as np
import pytest

import deep_miss_risk_probe_20261106 as probe


def synthetic_data(n=53):
    rng = np.random.default_rng(47)
    x = rng.normal(size=(n, 3))
    x[:, 2] = 3.25
    target = (x[:, 0] - 0.3 * x[:, 1] > 0).astype(np.float64)
    native = rng.normal(scale=0.2, size=n)
    weight = np.linspace(0.2, 2.0, n)
    return x, native, target, weight


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_fixed_training_contract():
    assert probe.HIDDEN_DIM == 48
    assert probe.EPOCHS == 15
    assert probe.BATCH_SIZE == 512
    assert probe.LEARNING_RATE == 0.003
    assert probe.WEIGHT_DECAY == 0.001
    assert probe.RESIDUAL_CAP == 4.0
    assert probe.MIN_STD == 1e-3
    assert probe.FEATURE_CLIP == 8.0


def test_zero_readout_initially_preserves_native_and_same_initialization(monkeypatch):
    monkeypatch.setattr(probe, "EPOCHS", 0)
    x, native, target, weight = synthetic_data()
    states, _ = probe.train_heads({"a": x, "b": x + 5}, native, target, weight, 131, "cpu")
    assert_states_equal({k: v for k, v in states["a"].items() if k not in {"mean", "scale"}},
                        {k: v for k, v in states["b"].items() if k not in {"mean", "scale"}})
    np.testing.assert_array_equal(states["a"]["2.weight"], np.zeros((1, 48)))
    np.testing.assert_array_equal(states["a"]["2.bias"], np.zeros(1))
    np.testing.assert_array_equal(probe.predict(states["a"], x, native, "cpu"), native)


def test_weighted_statistics_and_serialization_are_train_only(monkeypatch):
    monkeypatch.setattr(probe, "EPOCHS", 1)
    x, native, target, weight = synthetic_data()
    other = x * np.array([2.0, -3.0, 1.0]) + 4.0
    states, diagnostics = probe.train_heads({"a": x, "b": other}, native, target, weight, 317, "cpu")
    for name, data in (("a", x), ("b", other)):
        state = states[name]
        expected_mean = np.average(data, axis=0, weights=weight)
        expected_scale = np.maximum(np.sqrt(np.average((data - expected_mean) ** 2, axis=0, weights=weight)), 1e-3)
        np.testing.assert_allclose(state["mean"], expected_mean, atol=1e-12)
        np.testing.assert_allclose(state["scale"], expected_scale, atol=1e-12)
        assert state["scale"][2] == pytest.approx(1e-3)
        assert all(isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                   and np.isfinite(v).all() for v in state.values())
        json.dumps({k: v.tolist() for k, v in state.items()}, allow_nan=False)
        buffer = io.BytesIO()
        np.savez(buffer, **state)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            assert_states_equal(state, {k: saved[k] for k in saved.files})
        before = {k: v.copy() for k, v in state.items()}
        scores = probe.predict(state, data * 1000, native, "cpu")
        assert scores.shape == native.shape
        assert np.isfinite(scores).all()
        assert_states_equal(state, before)
    json.dumps(diagnostics, allow_nan=False)
    assert set(diagnostics) == {"a", "b"}


@pytest.mark.parametrize("seed", [131, 317, 563])
def test_same_seed_same_shuffle_arm_order_and_isolation(monkeypatch, seed):
    monkeypatch.setattr(probe, "EPOCHS", 3)
    monkeypatch.setattr(probe, "BATCH_SIZE", 8)
    x, native, target, weight = synthetic_data()
    original = {"a": x, "duplicate": x.copy(), "different": -x}
    states, diagnostics = probe.train_heads(original, native, target, weight, seed, "cpu")
    repeat, repeat_diag = probe.train_heads(dict(reversed(list(original.items()))), native, target, weight, seed, "cpu")
    assert_states_equal(states["a"], states["duplicate"])
    for name in original:
        assert_states_equal(states[name], repeat[name])
        assert diagnostics[name] == repeat_diag[name]
    changed, _ = probe.train_heads({"a": x, "different": x + 100}, native, target, weight, seed, "cpu")
    assert_states_equal(states["a"], changed["a"])


def test_epoch_shuffles_are_seeded_and_identical_across_arms(monkeypatch):
    import torch

    monkeypatch.setattr(probe, "EPOCHS", 3)
    monkeypatch.setattr(probe, "BATCH_SIZE", 8)
    original_randperm = torch.randperm
    seen = []

    def record_randperm(*args, **kwargs):
        permutation = original_randperm(*args, **kwargs)
        seen.append(permutation.numpy().copy())
        return permutation

    monkeypatch.setattr(probe.torch, "randperm", record_randperm)
    x, native, target, weight = synthetic_data()
    probe.train_heads({"a": x, "b": -x}, native, target, weight, 317, "cpu")
    generator = torch.Generator(device="cpu").manual_seed(317)
    expected = [original_randperm(len(x), generator=generator, device="cpu").numpy() for _ in range(3)]
    assert len(seen) == 6
    for actual, wanted in zip(seen, expected + expected):
        np.testing.assert_array_equal(actual, wanted)


def test_seed_changes_initial_hidden_weights_without_disturbing_global_torch_rng(monkeypatch):
    import torch

    monkeypatch.setattr(probe, "EPOCHS", 1)
    x, native, target, weight = synthetic_data()
    torch.manual_seed(2026)
    before = torch.random.get_rng_state().clone()
    first, _ = probe.train_heads({"a": x}, native, target, weight, 131, "cpu")
    assert torch.equal(before, torch.random.get_rng_state())
    second, _ = probe.train_heads({"a": x}, native, target, weight, 317, "cpu")
    assert not np.array_equal(first["a"]["0.weight"], second["a"]["0.weight"])


def test_training_is_independent_of_torch_default_dtype(monkeypatch):
    import torch

    monkeypatch.setattr(probe, "EPOCHS", 1)
    x, native, target, weight = synthetic_data()
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        state, _ = probe.train_heads({"a": x}, native, target, weight, 131, "cpu")
        assert np.isfinite(probe.predict(state["a"], x, native, "cpu")).all()
    finally:
        torch.set_default_dtype(default_dtype)


def test_prediction_uses_saved_scaling_silu_clipping_and_bounded_logit_residual(monkeypatch):
    monkeypatch.setattr(probe, "EPOCHS", 0)
    x, native, target, weight = synthetic_data()
    states, _ = probe.train_heads({"a": x}, native, target, weight, 563, "cpu")
    state = states["a"]
    state["0.weight"][:] = 0
    state["0.bias"][:] = 0
    state["2.weight"][:] = 0
    state["2.bias"][:] = 0
    state["0.weight"][0, 0] = 1
    state["2.weight"][0, 0] = 20
    standardized = np.array([-1000, -8, -1, 0, 1, 8, 1000], dtype=float)
    data = np.tile(state["mean"], (len(standardized), 1))
    data[:, 0] += standardized * state["scale"][0]
    offset = np.linspace(-2, 2, len(data))
    scores = probe.predict(state, data, offset, "cpu")
    z = np.clip(standardized, -8, 8)
    raw = 20 * z / (1 + np.exp(-z))
    expected = offset + 4 * np.tanh(raw / 4)
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    assert np.all(np.abs(scores - offset) <= 4.0)
    np.testing.assert_array_equal(scores, probe.predict(state, data, offset, "cpu"))
    assert probe.predict(state, data[:0], offset[:0], "cpu").shape == (0,)


def test_scaling_handles_large_finite_values_and_constant_columns(monkeypatch):
    monkeypatch.setattr(probe, "EPOCHS", 0)
    constant, _ = probe.train_heads({"a": np.full((3, 1), 1e200)}, np.zeros(3),
                                    np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]), 131, "cpu")
    assert constant["a"]["scale"][0] == 1e-3
    states, _ = probe.train_heads({"a": np.array([[1e308], [-1e308]])}, np.zeros(2),
                                   np.array([0, 1]), np.array([9, 1]), 131, "cpu")
    state = states["a"]
    state["0.weight"][:] = 0
    state["0.bias"][:] = 0
    state["2.weight"][:] = 0
    state["0.weight"][0, 0] = 1
    state["2.weight"][0, 0] = 10
    z = -3.0
    expected = 4 * np.tanh((10 * z / (1 + np.exp(-z))) / 4)
    actual = probe.predict(state, np.array([[-1e308]]), np.zeros(1), "cpu")
    np.testing.assert_allclose(actual, [expected], atol=1e-6)


def test_training_improves_weighted_bce_and_does_not_change_inputs(monkeypatch):
    monkeypatch.setattr(probe, "BATCH_SIZE", 32)
    x, native, target, weight = synthetic_data(192)
    before = [a.copy() for a in (x, native, target, weight)]
    states, diagnostics = probe.train_heads({"risk": x}, native, target, weight, 131, "cpu")
    prediction = probe.predict(states["risk"], x, native, "cpu")
    def weighted_bce(logits):
        return np.average(np.logaddexp(0, np.where(target == 1, -logits, logits)), weights=weight)
    assert weighted_bce(prediction) < weighted_bce(native) - 0.1
    assert diagnostics["risk"]["initial_loss"] == pytest.approx(weighted_bce(native))
    assert diagnostics["risk"]["loss_curve"][-1] == pytest.approx(weighted_bce(prediction))
    assert len(diagnostics["risk"]["loss_curve"]) == 15
    assert np.isfinite(diagnostics["risk"]["loss_curve"]).all()
    for actual, expected in zip((x, native, target, weight), before):
        np.testing.assert_array_equal(actual, expected)


def test_training_matches_independent_weighted_adamw_reference(monkeypatch):
    import torch
    from torch.nn import functional as F

    monkeypatch.setattr(probe, "EPOCHS", 2)
    monkeypatch.setattr(probe, "BATCH_SIZE", 5)
    x, native, target, weight = synthetic_data(17)
    states, _ = probe.train_heads({"a": x}, native, target, weight, 563, "cpu")
    mean = np.average(x, axis=0, weights=weight)
    std = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weight)), 1e-3)
    standardized = torch.tensor(np.clip((x - mean) / std, -8, 8), dtype=torch.float32)
    offset = torch.tensor(native, dtype=torch.float64)
    labels = torch.tensor(target, dtype=torch.float64)
    weights = torch.tensor(weight / weight.sum(), dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(563)
        model = torch.nn.Sequential(torch.nn.Linear(3, 48, dtype=torch.float32), torch.nn.SiLU(),
                                    torch.nn.Linear(48, 1, dtype=torch.float32))
        torch.nn.init.zeros_(model[2].weight)
        torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    generator = torch.Generator(device="cpu").manual_seed(563)
    for _ in range(2):
        permutation = torch.randperm(len(x), generator=generator, device="cpu")
        for index in permutation.split(5):
            optimizer.zero_grad(set_to_none=True)
            logits = offset[index] + 4 * torch.tanh(model(standardized[index]).squeeze(-1) / 4)
            per_row = F.binary_cross_entropy_with_logits(logits, labels[index], reduction="none")
            (torch.sum(per_row * weights[index]) / (len(index) / len(x))).backward()
            optimizer.step()
    for name, value in model.state_dict().items():
        np.testing.assert_allclose(states["a"][name], value.detach().numpy(), rtol=1e-6, atol=1e-7)


def test_full_default_schedule_includes_short_last_batch(monkeypatch):
    x, native, target, weight = synthetic_data(521)
    original_loss = probe._weighted_loss
    batch_sizes = []

    def record_loss(model, batch_x, *args, **kwargs):
        batch_sizes.append(len(batch_x))
        return original_loss(model, batch_x, *args, **kwargs)

    monkeypatch.setattr(probe, "_weighted_loss", record_loss)
    states, diagnostics = probe.train_heads({"a": x}, native, target, weight, 317, "cpu")
    assert batch_sizes == [512, 9] * 15
    assert len(diagnostics["a"]["loss_curve"]) == 15
    assert diagnostics["a"]["loss_curve"][-1] < diagnostics["a"]["initial_loss"]
    assert np.isfinite(probe.predict(states["a"], x, native, "cpu")).all()


def test_minibatches_use_global_weights_and_actual_final_batch_size(monkeypatch):
    import torch
    from torch.nn import functional as F

    monkeypatch.setattr(probe, "EPOCHS", 1)
    monkeypatch.setattr(probe, "BATCH_SIZE", 4)
    x, native, target, weight = synthetic_data(11)
    weight = np.geomspace(0.001, 1000, len(x))
    original_loss = probe._weighted_loss
    batch_sizes = []

    def check_loss(model, batch_x, batch_native, batch_target, batch_weights, *args, **kwargs):
        loss = original_loss(model, batch_x, batch_native, batch_target, batch_weights, *args, **kwargs)
        with torch.no_grad():
            logits = batch_native + 4 * torch.tanh(model(batch_x).squeeze(-1) / 4)
            per_row = F.binary_cross_entropy_with_logits(logits, batch_target, reduction="none")
            expected = torch.sum(per_row * batch_weights) / (len(batch_x) / len(x))
        assert loss.item() == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)
        batch_sizes.append(len(batch_x))
        return loss

    monkeypatch.setattr(probe, "_weighted_loss", check_loss)
    states, diagnostics = probe.train_heads({"a": x}, native, target, weight, 131, "cpu")
    assert batch_sizes == [4, 4, 3]
    predicted = probe.predict(states["a"], x, native, "cpu")
    def weighted_bce(z):
        return np.average(np.logaddexp(0, np.where(target == 1, -z, z)), weights=weight)
    assert diagnostics["a"]["initial_loss"] == pytest.approx(weighted_bce(native))
    assert diagnostics["a"]["loss_curve"][-1] == pytest.approx(weighted_bce(predicted))


def test_bce_respects_weights_and_handles_extreme_native_logits(monkeypatch):
    monkeypatch.setattr(probe, "EPOCHS", 1)
    x = np.zeros((3, 1))
    y = np.array([0.0, 0.0, 1.0])
    native = np.zeros(3)
    positive, _ = probe.train_heads({"x": x}, native, y, np.array([1.0, 1.0, 8.0]), 131, "cpu")
    negative, _ = probe.train_heads({"x": x}, native, y, np.array([8.0, 8.0, 1.0]), 131, "cpu")
    assert probe.predict(positive["x"], x, native, "cpu")[0] > 0
    assert probe.predict(negative["x"], x, native, "cpu")[0] < 0
    _, diagnostics = probe.train_heads({"x": x}, np.array([1000., -1000., -1000.]), y,
                                       np.ones(3), 131, "cpu")
    json.dumps(diagnostics, allow_nan=False)
    assert np.isfinite(diagnostics["x"]["loss_curve"]).all()


@pytest.mark.parametrize("case", [
    "empty_arms", "bad_arm_name", "shape", "mismatched_arms", "empty_rows", "empty_columns",
    "native_shape", "target_shape", "weight_shape", "bad_target", "zero_weight", "negative_weight",
    "nonfinite_x", "nonfinite_native", "nonfinite_target", "nonfinite_weight", "bad_seed",
])
def test_train_rejects_invalid_inputs(case):
    x, native, target, weight = synthetic_data()
    arms = {"a": x}
    seed = 131
    if case == "empty_arms":
        arms = {}
    elif case == "bad_arm_name":
        arms = {1: x}
    elif case == "shape":
        arms = {"a": x[:, 0]}
    elif case == "mismatched_arms":
        arms["b"] = x[:, :1]
    elif case == "empty_rows":
        arms, native, target, weight = {"a": x[:0]}, native[:0], target[:0], weight[:0]
    elif case == "empty_columns":
        arms = {"a": x[:, :0]}
    elif case == "native_shape":
        native = native[:, None]
    elif case == "target_shape":
        target = target[:-1]
    elif case == "weight_shape":
        weight = weight[:, None]
    elif case == "bad_target":
        target[0] = 0.5
    elif case == "zero_weight":
        weight[0] = 0
    elif case == "negative_weight":
        weight[0] = -1
    elif case == "nonfinite_x":
        x[0, 0] = np.nan
    elif case == "nonfinite_native":
        native[0] = np.inf
    elif case == "nonfinite_target":
        target[0] = np.nan
    elif case == "nonfinite_weight":
        weight[0] = np.nan
    else:
        seed = 1.5
    with pytest.raises((ValueError, TypeError)):
        probe.train_heads(arms, native, target, weight, seed, "cpu")


@pytest.mark.parametrize("case", ["feature_shape", "feature_nonfinite", "native_shape", "native_nonfinite",
                                  "missing_parameter", "bad_parameter_shape", "bad_parameter_value", "bad_scale",
                                  "below_min_scale"])
def test_predict_rejects_invalid_inputs(monkeypatch, case):
    monkeypatch.setattr(probe, "EPOCHS", 0)
    x, native, target, weight = synthetic_data()
    states, _ = probe.train_heads({"a": x}, native, target, weight, 131, "cpu")
    state = states["a"]
    if case == "feature_shape":
        x = x[:, :1]
    elif case == "feature_nonfinite":
        x[0, 0] = np.nan
    elif case == "native_shape":
        native = native[:, None]
    elif case == "native_nonfinite":
        native[0] = np.inf
    elif case == "missing_parameter":
        del state["0.weight"]
    elif case == "bad_parameter_shape":
        state["2.weight"] = np.zeros((2, 48))
    elif case == "bad_parameter_value":
        state["0.bias"][0] = np.nan
    elif case == "below_min_scale":
        state["scale"][0] = 1e-4
    else:
        state["scale"][0] = 0
    with pytest.raises((ValueError, TypeError)):
        probe.predict(state, x, native, "cpu")
