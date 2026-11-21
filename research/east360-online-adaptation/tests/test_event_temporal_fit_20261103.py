"""Synthetic, CPU-only regression tests for the event temporal teacher."""

import copy
import json
from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from experiments.risk_aligned_online.event_temporal_teacher_20261103.fit import fit


ARMS = ("true_order", "shuffled_order", "matched_other_order")


def _records():
    events = [
        (11, 1, 0, False, 6),
        (11, 2, 0, False, 1),
        (11, 3, 1, False, 1),
        (22, 1, 1, False, 2),
        (22, 2, 2, False, 1),
        (22, 3, 3, False, 1),
        (33, 1, 4, True, 3),
    ]
    return [
        {"shot": shot, "event_id": event_id, "family": family, "disruptive": flag}
        for shot, event_id, family, flag, count in events
        for _ in range(count)
    ]


def _differences(n):
    rng = np.random.default_rng(812)
    true = rng.normal(size=(n, 128)).astype(np.float32)
    return {
        "true_order": true,
        "shuffled_order": rng.normal(size=(n, 128)).astype(np.float32),
        "matched_other_order": true.copy(),
    }


@pytest.fixture(scope="module")
def trained():
    records = _records()
    differences = _differences(len(records))
    return records, differences, fit(differences, records)


def test_default_budget_serialization_and_finite_final_heads(trained):
    records, _, result = trained
    json.dumps(result, allow_nan=False)
    assert result["seed"] == 131
    assert len(result["initial_weight128"]) == 128
    assert result["initial_weight128"] == result["initial_weights"]
    assert result["initial_weight128"] is not result["initial_weights"]
    assert result["actual_budget"]["steps"] == 512
    assert result["actual_budget"]["batch_size"] == 64
    assert result["actual_budget"]["examples_per_arm"] == 512 * 64
    assert len(result["schedule"]) == 512
    assert all(len(batch) == 64 for batch in result["schedule"])
    for arm in ARMS:
        assert len(result["heads"][arm]) == 128
        assert np.isfinite(result["heads"][arm]).all()
        assert len(result["signs"][arm]) == len(records)
        assert len(result["training_losses"][arm]) == 512
        assert np.isfinite(result["training_losses"][arm]).all()


def test_schedule_samples_shots_then_events_then_rows_uniformly(trained):
    records, _, result = trained
    sampled = [index for batch in result["schedule"] for index in batch]
    assert all(type(index) is int and 0 <= index < len(records) for index in sampled)
    shot_counts = Counter(records[index]["shot"] for index in sampled)
    for shot in (11, 22, 33):
        assert shot_counts[shot] / len(sampled) == pytest.approx(1 / 3, abs=0.02)
    for shot in (11, 22):
        event_counts = Counter(
            records[index]["event_id"]
            for index in sampled
            if records[index]["shot"] == shot
        )
        for event in (1, 2, 3):
            assert event_counts[event] / shot_counts[shot] == pytest.approx(
                1 / 3, abs=0.03
            )
    row_counts = Counter(
        index
        for index in sampled
        if (records[index]["shot"], records[index]["event_id"]) == (11, 1)
    )
    assert set(row_counts) == set(range(6))
    for count in row_counts.values():
        assert count / sum(row_counts.values()) == pytest.approx(1 / 6, abs=0.03)


def test_whole_event_balancing_and_singleton_disclosure_are_deterministic(trained):
    records, differences, result = trained
    repeat = fit(differences, records, steps=3)
    assert repeat["signs"] == result["signs"]
    assert repeat["shuffle_sign_control"] == result["shuffle_sign_control"]
    assert set(result["signs"]["true_order"]) == {1}
    assert set(result["signs"]["matched_other_order"]) == {1}

    event_signs = defaultdict(set)
    for record, sign in zip(records, result["signs"]["shuffled_order"], strict=True):
        assert sign in (-1, 1)
        event_signs[(record["shot"], record["event_id"])].add(sign)
    assert all(len(signs) == 1 for signs in event_signs.values())
    signs = {event: next(iter(values)) for event, values in event_signs.items()}
    assert {signs[(11, 1)], signs[(11, 2)]} == {-1, 1}
    assert {signs[(11, 3)], signs[(22, 1)]} == {-1, 1}
    assert {signs[(22, 2)], signs[(22, 3)]} == {-1, 1}
    assert signs[(33, 1)] == 1
    assert result["shuffle_sign_control"]["forced_positive_singletons"] == [
        {"shot": 33, "event_id": 1, "family": 4, "disruptive": True}
    ]


def test_odd_singleton_pool_discloses_the_unpaired_positive_event():
    records = [
        {"shot": shot, "event_id": 1, "family": family, "disruptive": False}
        for shot, family in ((10, 0), (20, 1), (30, 2))
    ] + [{"shot": 40, "event_id": 1, "family": 3, "disruptive": True}]
    result = fit(_differences(len(records)), records, steps=1)
    shuffled = result["signs"]["shuffled_order"]
    assert sorted(shuffled[:3]) == [-1, 1, 1]
    assert shuffled[3] == 1
    forced = result["shuffle_sign_control"]["forced_positive_singletons"]
    assert len(forced) == 2
    assert {item["disruptive"] for item in forced} == {False, True}
    for item in forced:
        index = next(i for i, record in enumerate(records) if record == item)
        assert shuffled[index] == 1


def test_paired_initialization_common_schedule_and_fixed_final_step(trained):
    records, differences, result = trained
    repeat = fit(differences, records, steps=3)
    assert repeat["initial_weights"] == result["initial_weights"]
    assert repeat["schedule"] == result["schedule"][:3]
    generator = torch.Generator(device="cpu").manual_seed(131)
    bound = 1 / np.sqrt(128)
    expected_initial = torch.empty(128, dtype=torch.float32, device="cpu").uniform_(
        -bound, bound, generator=generator
    )
    np.testing.assert_allclose(
        result["initial_weights"], expected_initial.numpy(), atol=1e-8
    )
    assert result["heads"]["true_order"] == result["heads"]["matched_other_order"]
    assert (
        result["training_losses"]["true_order"]
        == result["training_losses"]["matched_other_order"]
    )
    for arm in ARMS:
        assert repeat["training_losses"][arm] == result["training_losses"][arm][:3]
        weight = torch.nn.Parameter(
            torch.tensor(repeat["initial_weight128"], dtype=torch.float32)
        )
        optimizer = torch.optim.AdamW([weight], lr=0.001, weight_decay=0.001)
        for indices in repeat["schedule"]:
            optimizer.zero_grad(set_to_none=True)
            values = torch.tensor(differences[arm][indices], dtype=torch.float32)
            signs = torch.tensor(
                np.asarray(repeat["signs"][arm])[indices], dtype=torch.float32
            )
            functional.softplus(-signs * (values @ weight)).mean().backward()
            optimizer.step()
        np.testing.assert_allclose(
            repeat["heads"][arm], weight.detach().numpy(), atol=1e-7
        )

    one_step = fit(differences, records, steps=1)
    indices = one_step["schedule"][0]
    for arm in ARMS:
        weight = torch.nn.Parameter(
            torch.tensor(one_step["initial_weights"], dtype=torch.float32)
        )
        optimizer = torch.optim.AdamW([weight], lr=0.001, weight_decay=0.001)
        values = torch.tensor(differences[arm][indices], dtype=torch.float32)
        signs = torch.tensor(
            np.asarray(one_step["signs"][arm])[indices], dtype=torch.float32
        )
        loss = functional.softplus(-signs * (values @ weight)).mean()
        loss.backward()
        optimizer.step()
        assert one_step["training_losses"][arm][0] == pytest.approx(
            float(loss.detach()), abs=1e-7
        )
        np.testing.assert_allclose(
            one_step["heads"][arm], weight.detach().numpy(), atol=1e-7
        )


def test_model_receives_only_row_aligned_features_and_no_metadata(monkeypatch):
    records = _records()
    differences = _differences(len(records))
    differences["matched_other_order"] = differences["true_order"] + 2
    captured = []
    captured_weights = []
    original = torch.nn.Linear.forward

    def capture(layer, values):
        assert layer.bias is None
        assert layer.weight.device.type == values.device.type == "cpu"
        assert values.shape == (64, 128)
        captured.append(values.detach().numpy().copy())
        captured_weights.append(layer.weight.detach().numpy().copy().reshape(-1))
        return original(layer, values)

    monkeypatch.setattr(torch.nn.Linear, "forward", capture)
    result = fit(differences, records, steps=2)
    assert len(captured) == 2 * len(ARMS)
    for weight in captured_weights[: len(ARMS)]:
        np.testing.assert_array_equal(weight, result["initial_weight128"])
    for step, indices in enumerate(result["schedule"]):
        for offset, arm in enumerate(ARMS):
            np.testing.assert_array_equal(
                captured[step * len(ARMS) + offset], differences[arm][indices]
            )


def test_inputs_and_global_random_states_are_not_mutated():
    records = _records()
    differences = _differences(len(records))
    originals = {arm: values.copy() for arm, values in differences.items()}
    records_before = copy.deepcopy(records)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    fit(differences, records, steps=2)
    assert records == records_before
    for arm in ARMS:
        np.testing.assert_array_equal(differences[arm], originals[arm])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_schedule_does_not_depend_on_control_metadata():
    records = _records()
    differences = _differences(len(records))
    changed = [
        dict(record, family=(record["family"] + 1) % 6, disruptive=False)
        for record in records
    ]
    first = fit(differences, records, steps=3)
    second = fit(differences, changed, steps=3)
    assert first["schedule"] == second["schedule"]
    assert first["initial_weights"] == second["initial_weights"]
    assert first["heads"]["true_order"] == second["heads"]["true_order"]
    assert (
        first["heads"]["matched_other_order"] == second["heads"]["matched_other_order"]
    )


def test_public_adamw_stays_on_cpu_without_cuda_execution_or_torch_load(monkeypatch):
    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"{name} is forbidden")

        return fail

    for name in (
        "init",
        "current_device",
        "set_device",
        "manual_seed",
        "manual_seed_all",
        "synchronize",
        "_lazy_init",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden(f"torch.cuda.{name}"))
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden("Tensor.cuda"))
    monkeypatch.setattr(torch.nn.Module, "cuda", forbidden("Module.cuda"))
    monkeypatch.setattr(torch, "load", forbidden("torch.load"))

    def check_target(target):
        if isinstance(target, torch.Tensor):
            assert target.device.type != "cuda", "CUDA tensor target is forbidden"
        elif isinstance(target, torch.device):
            assert target.type != "cuda", "CUDA device target is forbidden"
        elif isinstance(target, str):
            assert not target.startswith("cuda"), "CUDA device target is forbidden"
        elif isinstance(target, int) and not isinstance(target, bool):
            raise AssertionError("integer accelerator device target is forbidden")

    def guard_factory(name):
        original = getattr(torch, name)

        def guarded(*args, **kwargs):
            check_target(kwargs.get("device"))
            result = original(*args, **kwargs)
            assert result.device.type != "cuda", "CUDA tensor allocation is forbidden"
            return result

        return guarded

    for name in ("empty", "zeros", "ones", "tensor", "as_tensor"):
        monkeypatch.setattr(torch, name, guard_factory(name))

    original_to = torch.Tensor.to

    def guarded_to(self, *args, **kwargs):
        check_target(kwargs.get("device"))
        if args:
            check_target(args[0])
        result = original_to(self, *args, **kwargs)
        assert result.device.type != "cuda", "CUDA tensor transfer is forbidden"
        return result

    monkeypatch.setattr(torch.Tensor, "to", guarded_to)
    original_module_to = torch.nn.Module.to

    def guarded_module_to(self, *args, **kwargs):
        check_target(kwargs.get("device"))
        if args:
            check_target(args[0])
        return original_module_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", guarded_module_to)

    constructed = []
    original_init = torch.optim.AdamW.__init__

    def capture_optimizer(self, *args, **kwargs):
        assert type(self) is torch.optim.AdamW
        original_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(torch.optim.AdamW, "__init__", capture_optimizer)

    with pytest.raises(AssertionError, match="CUDA"):
        torch.empty(1, device="cuda")
    with pytest.raises(AssertionError, match="CUDA"):
        torch.empty(1).to("cuda")
    with pytest.raises(AssertionError, match="cuda"):
        torch.empty(1).cuda()
    with pytest.raises(AssertionError, match="CUDA"):
        torch.nn.Linear(128, 1).to("cuda")
    with pytest.raises(AssertionError, match="torch.cuda.init"):
        torch.cuda.init()
    with pytest.raises(AssertionError, match="torch.load"):
        torch.load("no-file-access")

    records = _records()
    result = fit(_differences(len(records)), records, steps=2)
    assert len(constructed) == len(ARMS)
    for optimizer in constructed:
        for group in optimizer.param_groups:
            assert all(parameter.device.type == "cpu" for parameter in group["params"])
        assert optimizer.state
        for state in optimizer.state.values():
            assert all(
                value.device.type == "cpu"
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
    assert all(np.isfinite(result["heads"][arm]).all() for arm in ARMS)


def test_cpu_training_allows_cuda_availability_and_capture_metadata(monkeypatch):
    queries = []

    def available():
        queries.append("available")
        return True

    def capturing():
        queries.append("capturing")
        return False

    def forbid_initialization(*args, **kwargs):
        raise AssertionError("CUDA initialization is forbidden")

    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    monkeypatch.setattr(torch.cuda, "init", forbid_initialization)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_initialization)
    records = _records()
    result = fit(_differences(len(records)), records, steps=1)
    assert {"available", "capturing"} <= set(queries)
    assert result["actual_budget"]["device"] == "cpu"
    assert all(np.isfinite(result["heads"][arm]).all() for arm in ARMS)


def test_training_ignores_a_non_cpu_default_device():
    records = _records()
    with torch.device("meta"):
        result = fit(_differences(len(records)), records, steps=1)
        assert torch.empty(0).device.type == "meta"
    assert result["actual_budget"]["device"] == "cpu"
    assert all(np.isfinite(result["heads"][arm]).all() for arm in ARMS)


@pytest.mark.parametrize("steps", [0, -1, 513, 1.5, True, None])
def test_invalid_budgets_are_rejected(steps):
    records = _records()
    with pytest.raises((TypeError, ValueError)):
        fit(_differences(len(records)), records, steps=steps)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "not_array",
        "width",
        "rank",
        "length",
        "nan",
        "inf",
        "complex",
        "bool",
        "timedelta",
        "unrepresentable",
        "empty",
    ],
)
def test_invalid_differences_are_rejected(kind):
    records = _records()
    differences = _differences(len(records))
    if kind == "missing":
        del differences["true_order"]
    elif kind == "extra":
        differences["other"] = differences["true_order"]
    elif kind == "not_array":
        differences["true_order"] = differences["true_order"].tolist()
    elif kind == "width":
        differences["true_order"] = np.zeros((len(records), 127))
    elif kind == "rank":
        differences["true_order"] = np.zeros((len(records), 128, 1))
    elif kind == "length":
        differences["true_order"] = np.zeros((len(records) - 1, 128))
    elif kind == "nan":
        differences["true_order"][0, 0] = np.nan
    elif kind == "inf":
        differences["true_order"][0, 0] = np.inf
    elif kind == "complex":
        differences["true_order"] = differences["true_order"].astype(complex)
    elif kind == "bool":
        differences["true_order"] = np.zeros((len(records), 128), dtype=bool)
    elif kind == "timedelta":
        differences["true_order"] = np.zeros(
            (len(records), 128), dtype="timedelta64[ns]"
        )
    elif kind == "unrepresentable":
        differences["true_order"] = np.full(
            (len(records), 128), np.finfo(np.float64).max
        )
    else:
        records = []
        differences = {arm: np.zeros((0, 128)) for arm in ARMS}
    with pytest.raises((TypeError, ValueError)):
        fit(differences, records, steps=1)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "shot",
        "event_id",
        "family",
        "family_range",
        "disruptive",
        "event_family",
        "shot_outcome",
    ],
)
def test_invalid_or_inconsistent_event_metadata_are_rejected(change):
    records = _records()
    if change == "missing":
        del records[0]["shot"]
    elif change == "shot":
        records[0]["shot"] = True
    elif change == "event_id":
        records[0]["event_id"] = 1.5
    elif change == "family":
        records[0]["family"] = False
    elif change == "family_range":
        records[0]["family"] = 6
    elif change == "disruptive":
        records[0]["disruptive"] = 1
    elif change == "event_family":
        records[0]["family"] = 5
    else:
        records[0]["disruptive"] = True
    with pytest.raises((TypeError, ValueError)):
        fit(_differences(len(records)), records, steps=1)
