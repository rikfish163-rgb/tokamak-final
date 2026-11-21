"""CPU-only contract tests for persistent-update guard selection."""

import copy
import math

import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.lifecycle_stream_20261025 import learning as legacy
from experiments.risk_aligned_online.paired_gap_guard_20261031 import learning, rules
from experiments.risk_aligned_online.simple23 import Config, Simple23


LEGACY_FIELDS = (
    "accepted", "target_label_policy", "guard_before", "guard_after", "steps",
    "candidate_delta", "retained_delta", "guard_is_independent", "history_ids",
)


@pytest.fixture(autouse=True)
def math_attention_path():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture
def example():
    # Cached internal tokens and diagnostic deltas, without reading shot data.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31031)
        cfg = Config(dim=8, heads=2, layers=2, window=128, patch=8,
                     horizon=25, fast_site="tokens", input_view="source")
        model = ContextAdapter(Simple23(cfg, torch.zeros(23), torch.ones(23))).double().eval()
        with torch.no_grad():
            model.slow.weight.normal_(0, 0.03)
            model.memory.output.weight.normal_(0, 0.04)
            h = torch.randn(1, 17, 8, dtype=torch.float64)
            valid = torch.ones(1, 17, dtype=torch.bool)
            valid[0, 3] = False
            delta = torch.randn(1, 92, dtype=torch.float64)
            # For opposite target/history labels, the balanced objective moves
            # a negative logit upward and increases the guarded negative loss.
            model.base.risk_head.bias.add_(-1.25 - model.risk(h, valid, delta).item())

        def row(shot, label, flag):
            return dict(h=h.clone(), valid=valid.clone(), delta=delta.clone(),
                        y=torch.tensor([label], dtype=torch.float64),
                        shot=shot, flag=flag)

        return model, row


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def assert_examples_equal(actual, expected):
    for actual_group, expected_group in zip(actual, expected, strict=True):
        assert len(actual_group) == len(expected_group)
        for row, original in zip(actual_group, expected_group, strict=True):
            assert row.keys() == original.keys()
            for key, value in row.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                else:
                    assert value == original[key]


def assert_finite_logs(result):
    for losses in (result["guard_before"], result["guard_after"],
                   result["paired_guard"]["before_off"],
                   result["paired_guard"]["after_off"]):
        assert losses and all(math.isfinite(value) and value >= 0 for value in losses.values())
    assert all(math.isfinite(value) for entry in result["steps"] for value in entry.values())
    for key in ("candidate_delta", "retained_delta"):
        assert all(math.isfinite(value) and value >= 0 for value in result[key].values())


@pytest.mark.parametrize("target_label,expected_acceptance", [(0, True), (1, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_exactly_matches_legacy_for_real_accepted_and_rejected_candidates(
    example, target_label, expected_acceptance, dtype
):
    model, row = example
    model = model.to(dtype=dtype)

    def typed_row(*args):
        item = row(*args)
        return {key: value.to(dtype=dtype) if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value for key, value in item.items()}

    current = [typed_row(100, target_label, bool(target_label))]
    replay = [typed_row(10, 0, False)]
    source = [typed_row(20, 0, False)]
    original_examples = copy.deepcopy((current, replay, source))
    original_base = copy.deepcopy(model.base.state_dict())
    initial = model.adapted_state()
    reference = copy.deepcopy(model)

    expected = legacy.update(reference, current, replay, source, fast=True, steps=2,
                             guarded=True, policy="disruptive_positive_only")
    result = learning.update_with_guards(model, current, replay, source, rule="original")

    assert expected["accepted"] is expected_acceptance
    assert {key: result[key] for key in LEGACY_FIELDS} == expected
    assert set(result) == set(LEGACY_FIELDS) | {"paired_guard"}
    assert_state_equal(model.adapted_state(), reference.adapted_state())
    assert_state_equal(model.base.state_dict(), original_base)
    assert_examples_equal((current, replay, source), original_examples)
    assert result["candidate_delta"]["slow"] > 0
    assert result["candidate_delta"]["fast_output"] > 0
    assert result["paired_guard"]["eligible_count_before"] == 0
    assert result["paired_guard"]["eligible_count_after"] == int(expected_acceptance)
    assert result["paired_guard"]["checks"]["original_pass"] is expected_acceptance
    assert result["paired_guard"]["checks"] == rules.evaluate_guards(
        result["guard_before"], result["paired_guard"]["before_off"],
        result["guard_after"], result["paired_guard"]["after_off"])
    assert_finite_logs(result)
    if not expected_acceptance:
        assert_state_equal(model.adapted_state(), initial)
        assert result["guard_before"] != result["guard_after"]


@pytest.mark.parametrize("policy", ["all_points", "disruptive_positive_only"])
def test_original_matches_legacy_with_mixed_history_and_policy(example, policy):
    model, row = example

    def mixed(shot):
        item = row(shot, 1, True)
        item["h"] = torch.cat((item["h"], item["h"] + 0.25))
        item["valid"] = item["valid"].repeat(2, 1)
        item["delta"] = torch.cat((item["delta"], -item["delta"]))
        item["y"] = torch.tensor([0.0, 1.0], dtype=torch.float64)
        return item

    current = [mixed(100)]
    replay = [row(10, 0, False), mixed(11)]
    source = [mixed(20)]
    original_examples = copy.deepcopy((current, replay, source))
    reference = copy.deepcopy(model)
    with torch.no_grad():
        before_source = float(legacy.population_loss(model, source, True, "all_points"))
        before_disruptive = float(legacy.population_loss(model, [replay[1]], True, policy))
    expected = legacy.update(reference, current, replay, source, fast=True, steps=3,
                             guarded=True, policy=policy)
    result = learning.update_with_guards(model, current, replay, source, rule="original",
                                         eligible_so_far=4, policy=policy, steps=3)

    assert {key: result[key] for key in LEGACY_FIELDS} == expected
    assert_state_equal(model.adapted_state(), reference.adapted_state())
    assert_examples_equal((current, replay, source), original_examples)
    assert result["guard_before"]["source_event"] == before_source
    assert result["guard_before"]["historical_D"] == before_disruptive
    assert set(result["guard_before"]) == {"historical_S", "historical_D", "source_event"}
    assert result["paired_guard"]["eligible_count_after"] == 4 + int(expected["accepted"])
    assert_finite_logs(result)


def test_guards_use_only_existing_history_and_log_candidate_before_rollback(example, monkeypatch):
    model, row = example
    current = [row(100, 1, True)]
    replay = [row(10, 0, False), row(11, 1, True)]
    source = [row(20, 0, False), row(21, 1, True)]
    initial = model.adapted_state()
    original_examples = copy.deepcopy((current, replay, source))
    reference = copy.deepcopy(model)
    candidate = legacy.update(reference, current, replay, source, fast=True, steps=2,
                              guarded=False, policy="disruptive_positive_only")
    candidate_on = legacy.guard_losses(reference, replay, source, True, "disruptive_positive_only")
    candidate_off = legacy.guard_losses(reference, replay, source, False, "disruptive_positive_only")
    real_guard, real_evaluate = legacy.guard_losses, rules.evaluate_guards
    observed = []

    def record_guard(net, history, fixed_source, fast, policy="all_points"):
        assert history is replay and fixed_source is source
        assert all(item is not current[0] for item in (*history, *fixed_source))
        value = real_guard(net, history, fixed_source, fast, policy)
        observed.append((fast, policy, value))
        return value

    def reject(*args, **kwargs):
        checks = real_evaluate(*args, **kwargs)
        checks.update(original_pass=False, dual_absolute_pass=False, paired_pass=False)
        return checks

    monkeypatch.setattr(legacy, "guard_losses", record_guard)
    monkeypatch.setattr(rules, "evaluate_guards", reject)
    result = learning.update_with_guards(model, current, replay, source, rule="original")

    assert [(fast, policy) for fast, policy, _ in observed] == [
        (True, "disruptive_positive_only"), (False, "disruptive_positive_only"),
        (True, "disruptive_positive_only"), (False, "disruptive_positive_only"),
    ]
    assert result["guard_before"] == observed[0][2]
    assert result["paired_guard"]["before_off"] == observed[1][2]
    assert result["guard_after"] == observed[2][2] == candidate_on
    assert result["paired_guard"]["after_off"] == observed[3][2] == candidate_off
    assert set(result["guard_before"]) == {"historical_S", "historical_D", "source_event"}
    assert result["history_ids"] == [10, 11]
    assert result["candidate_delta"] == candidate["candidate_delta"]
    assert result["steps"] == candidate["steps"]
    assert result["retained_delta"] == {key: 0.0 for key in candidate["candidate_delta"]}
    assert not result["accepted"]
    assert_state_equal(model.adapted_state(), initial)
    assert_examples_equal((current, replay, source), original_examples)
    assert_finite_logs(result)


@pytest.mark.parametrize(
    "rule,original,dual,paired,accepted",
    [
        ("original", True, False, False, True),
        ("dual", True, False, False, False),
        ("dual", True, True, False, True),
        ("paired", True, True, False, False),
        ("paired", True, True, True, True),
        ("half", False, False, False, False),
    ],
)
def test_each_rule_selects_its_own_check(example, monkeypatch, rule, original, dual, paired, accepted):
    model, row = example
    initial = model.adapted_state()
    real_evaluate = rules.evaluate_guards

    def checks(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        result.update(original_pass=original, dual_absolute_pass=dual, paired_pass=paired)
        return result

    monkeypatch.setattr(rules, "evaluate_guards", checks)
    result = learning.update_with_guards(model, [row(100, 0, False)],
                                         [row(10, 0, False)], [row(20, 0, False)], rule=rule)
    assert result["accepted"] is accepted
    assert result["paired_guard"]["rule"] == rule
    assert result["paired_guard"]["eligible_count_before"] == 0
    assert result["paired_guard"]["eligible_count_after"] == int(original)
    if accepted:
        assert any(not torch.equal(model.adapted_state()[key], initial[key]) for key in initial)
        assert result["retained_delta"] == result["candidate_delta"]
    else:
        assert_state_equal(model.adapted_state(), initial)
        assert all(value == 0.0 for value in result["retained_delta"].values())
    assert_finite_logs(result)


def test_half_counts_only_original_eligible_proposals_in_its_own_history(example, monkeypatch):
    model, row = example
    real_evaluate = rules.evaluate_guards
    eligibility = iter([True, False, True, True, False, True])

    def checks(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        result["original_pass"] = next(eligibility)
        return result

    monkeypatch.setattr(rules, "evaluate_guards", checks)
    count, accepted, counts = 0, [], []
    for _ in range(6):
        initial = model.adapted_state()
        result = learning.update_with_guards(model, [row(100, 0, False)],
                                             [row(10, 0, False)], [row(20, 0, False)],
                                             rule="half", eligible_so_far=count)
        assert result["paired_guard"]["eligible_count_before"] == count
        count = result["paired_guard"]["eligible_count_after"]
        counts.append(count)
        accepted.append(result["accepted"])
        if not result["accepted"]:
            assert_state_equal(model.adapted_state(), initial)
    assert counts == [1, 1, 2, 3, 3, 4]
    assert accepted == [True, False, False, True, False, False]


def test_empty_replay_and_explicit_policy_and_steps(example):
    model, row = example
    result = learning.update_with_guards(model, [row(100, 0, False)], [],
                                         [row(20, 0, False)], rule="paired",
                                         eligible_so_far=7, policy="all_points", steps=1)
    assert result["history_ids"] == []
    assert result["target_label_policy"] == "all_points"
    assert len(result["steps"]) == 1
    assert set(result["guard_before"]) == {"source_event"}
    assert set(result["paired_guard"]["before_off"]) == {"source_event"}
    assert result["paired_guard"]["eligible_count_before"] == 7
    assert_finite_logs(result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rule": "other"}, {"rule": None}, {"rule": []},
        {"rule": "original", "eligible_so_far": -1},
        {"rule": "original", "eligible_so_far": 1.5},
        {"rule": "original", "eligible_so_far": True},
        {"rule": "original", "eligible_so_far": "1"},
        {"rule": "original", "steps": 0}, {"rule": "original", "steps": -1},
        {"rule": "original", "steps": 1.5}, {"rule": "original", "steps": True},
    ],
)
def test_invalid_controls_fail_before_mutation_or_guard_work(example, monkeypatch, kwargs):
    model, row = example
    initial = model.adapted_state()
    base = copy.deepcopy(model.base.state_dict())

    def unexpected(*args, **kw):
        pytest.fail("invalid controls must fail before guard or optimizer work")

    monkeypatch.setattr(legacy, "guard_losses", unexpected)
    monkeypatch.setattr(legacy, "update", unexpected)
    with pytest.raises((ValueError, TypeError)):
        learning.update_with_guards(model, [row(100, 0, False)],
                                    [row(10, 0, False)], [row(20, 0, False)], **kwargs)
    assert_state_equal(model.adapted_state(), initial)
    assert_state_equal(model.base.state_dict(), base)


@pytest.mark.parametrize("failure_site", ["candidate", "after_guard", "checks"])
def test_failures_restore_both_owned_matrices_and_reraise(example, monkeypatch, failure_site):
    model, row = example
    current, replay, source = [row(100, 0, False)], [row(10, 0, False)], [row(20, 0, False)]
    initial = model.adapted_state()
    original_base = copy.deepcopy(model.base.state_dict())
    original_examples = copy.deepcopy((current, replay, source))
    failure = RuntimeError("injected failure")
    real_guard = legacy.guard_losses
    calls = 0

    def fail_candidate(net, *args, **kwargs):
        with torch.no_grad():
            net.slow.weight.add_(1)
            net.memory.output.weight.sub_(1)
        raise failure

    def fail_after_guard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            assert any(not torch.equal(model.adapted_state()[key], initial[key]) for key in initial)
            raise failure
        return real_guard(*args, **kwargs)

    def fail_checks(*args, **kwargs):
        assert any(not torch.equal(model.adapted_state()[key], initial[key]) for key in initial)
        raise failure

    if failure_site == "candidate":
        monkeypatch.setattr(legacy, "update", fail_candidate)
    elif failure_site == "after_guard":
        monkeypatch.setattr(legacy, "guard_losses", fail_after_guard)
    else:
        monkeypatch.setattr(rules, "evaluate_guards", fail_checks)
    with pytest.raises(RuntimeError) as exc:
        learning.update_with_guards(model, current, replay, source, rule="paired")
    assert exc.value is failure
    assert_state_equal(model.adapted_state(), initial)
    assert_state_equal(model.base.state_dict(), original_base)
    assert_examples_equal((current, replay, source), original_examples)


def test_nonfinite_candidate_guard_is_rejected_and_restored(example, monkeypatch):
    model, row = example
    initial = model.adapted_state()
    real_guard = legacy.guard_losses
    calls = 0

    def invalid_after_off(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_guard(*args, **kwargs)
        if calls == 4:
            assert any(not torch.equal(model.adapted_state()[key], initial[key]) for key in initial)
            result["source_event"] = float("nan")
        return result

    monkeypatch.setattr(legacy, "guard_losses", invalid_after_off)
    with pytest.raises(ValueError, match="finite"):
        learning.update_with_guards(model, [row(100, 0, False)],
                                    [row(10, 0, False)], [row(20, 0, False)], rule="original")
    assert calls == 4
    assert_state_equal(model.adapted_state(), initial)
