from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "unified_online_value.py"
    spec = importlib.util.spec_from_file_location("unified_online_value", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_causal_update_value_is_bounded_missing_safe_and_monotonic() -> None:
    module = load_module()
    value = module.CausalUpdateValue()

    low = value.score(
        reconstruction_surprise=0.10,
        teacher_disagreement=0.10,
        physics_mismatch=0.10,
        private_innovation=0.10,
        availability_fraction=1.0,
        redundancy=0.0,
    )
    high = value.score(
        reconstruction_surprise=0.90,
        teacher_disagreement=0.90,
        physics_mismatch=0.90,
        private_innovation=0.90,
        availability_fraction=1.0,
        redundancy=0.0,
    )
    missing = value.score(
        reconstruction_surprise=1.0,
        teacher_disagreement=1.0,
        physics_mismatch=1.0,
        private_innovation=1.0,
        availability_fraction=0.0,
        redundancy=0.0,
    )

    assert 0.0 <= low < high <= 1.0
    assert missing == 0.0


def test_redundancy_and_unreliability_reduce_update_value() -> None:
    module = load_module()
    value = module.CausalUpdateValue()
    common = dict(
        reconstruction_surprise=0.8,
        teacher_disagreement=0.7,
        physics_mismatch=0.6,
        private_innovation=0.9,
    )

    novel = value.score(**common, availability_fraction=1.0, redundancy=0.0)
    redundant = value.score(**common, availability_fraction=1.0, redundancy=1.0)
    unreliable = value.score(**common, availability_fraction=0.25, redundancy=0.0)

    assert novel > redundant
    assert novel > unreliable


def test_learning_rate_scale_has_dead_zone_and_saturates() -> None:
    module = load_module()
    value = module.CausalUpdateValue(minimum_value=0.20)

    assert value.learning_rate_scale(0.10) == 0.0
    assert 0.0 < value.learning_rate_scale(0.60) < 1.0
    assert value.learning_rate_scale(1.0) == 1.0


def test_temporary_optimizer_scale_changes_step_size_and_restores_lr() -> None:
    module = load_module()

    def move(scale: float) -> float:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.2)
        loss = parameter.square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with module.temporary_optimizer_scale(optimizer, scale):
            optimizer.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
        return float(parameter.detach())

    unchanged = move(0.0)
    half = move(0.5)
    full = move(1.0)
    assert unchanged == pytest.approx(1.0)
    assert full < half < unchanged


def _alarm_trace(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(200, dtype=np.float64) / 1000.0
    margin = np.full(200, -0.4, dtype=np.float64)
    support = np.zeros(200, dtype=np.float64)
    if kind == "early_alarm":
        margin[80:100] = -0.1
        support[80:100] = 1.0
    elif kind == "missed":
        margin[150:180] = -0.3
        support[150:180] = 0.8
    else:
        raise ValueError(kind)
    return margin, support, times


def test_alarm_update_requires_seal_and_selected_query() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController()
    margin, support, times = _alarm_trace("early_alarm")

    with pytest.raises(RuntimeError, match="sealed"):
        controller.update_after_reveal(
            10, selected=True, is_disrupt=False, down_time_s=None,
            update_value=1.0,
        )
    controller.begin_shot(10)
    controller.predict_and_seal(10, margin, support, times)
    with pytest.raises(PermissionError, match="selected"):
        controller.update_after_reveal(
            10, selected=False, is_disrupt=False, down_time_s=None,
            update_value=1.0,
        )


def test_incremental_dual_alarm_matches_offline_causal_trace() -> None:
    module = load_module()
    from dual_threshold_alarm import dual_threshold_trace

    rng = np.random.default_rng(17)
    margin = rng.normal(-0.2, 0.3, 250)
    support = rng.uniform(0.0, 1.0, 250)
    state = module.IncrementalDualThresholdAlarm()
    online = [state.observe(margin[index], support[index], index / 1000.0)
              for index in range(len(margin))]
    offline = dual_threshold_trace(margin, support)

    np.testing.assert_array_equal(
        np.asarray([row["high"] for row in online]), offline["high_mask"]
    )
    np.testing.assert_array_equal(
        np.asarray([row["early"] for row in online]), offline["early_mask"]
    )
    np.testing.assert_allclose(
        np.asarray([row["early_score"] for row in online]),
        offline["early_score"], rtol=0, atol=1e-12,
    )


def test_projected_alarm_controller_can_seal_endpoint_by_endpoint() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController()
    margin, support, times = _alarm_trace("early_alarm")
    controller.begin_shot(10)
    first = None
    for value, physics, time_s in zip(margin, support, times):
        event = controller.observe_endpoint(10, value, physics, time_s)
        if event["new_alarm"] and first is None:
            first = event["index"]
    sealed = controller.seal_observed_shot(10)

    assert sealed["alarmed"] is True
    assert sealed["alarm_index"] == first


def test_alarm_decision_filter_can_only_delay_or_veto_before_final_seal() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController()
    margin, support, times = _alarm_trace("early_alarm")
    controller.begin_shot(10)
    for value, physics, time_s in zip(margin, support, times):
        controller.observe_endpoint(10, value, physics, time_s)
    sealed = controller.seal_observed_shot(10, decision_filter={
        "alarmed": False, "alarm_index": None, "route": None,
        "early_score": None, "filter_reason": "no_independent_branch_support",
    })

    assert sealed["alarmed"] is False
    assert sealed["unfiltered_alarm"]["alarmed"] is True
    assert sealed["filter_reason"] == "no_independent_branch_support"

    controller.discard_without_reveal(10)
    controller.begin_shot(20)
    for value, physics, time_s in zip(margin, support, times):
        controller.observe_endpoint(20, value, physics, time_s)
    with pytest.raises(ValueError, match="cannot advance"):
        controller.seal_observed_shot(20, decision_filter={
            "alarmed": True, "alarm_index": 0, "route": "early",
            "early_score": 8.0,
        })


def test_safe_false_alarm_tightens_future_alarm_without_rewriting_seal() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController(learning_rate=0.05)
    margin, support, times = _alarm_trace("early_alarm")
    before = controller.parameters_dict()
    controller.begin_shot(10)
    sealed = controller.predict_and_seal(10, margin, support, times)
    receipt = controller.update_after_reveal(
        10, selected=True, is_disrupt=False, down_time_s=None,
        update_value=1.0,
    )
    after = controller.parameters_dict()

    assert sealed["alarmed"] is True
    assert receipt["accepted"] is True
    assert after["early_threshold"] >= before["early_threshold"]
    assert after["physics_weight"] <= before["physics_weight"]
    assert receipt["sealed_alarm"] == sealed


def test_missed_disruption_relaxes_future_alarm_and_projects_bounds() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController(learning_rate=0.20)
    margin, support, times = _alarm_trace("missed")
    before = controller.parameters_dict()
    controller.begin_shot(10)
    controller.predict_and_seal(10, margin, support, times)
    receipt = controller.update_after_reveal(
        10, selected=True, is_disrupt=True, down_time_s=0.200,
        update_value=1.0,
    )
    after = controller.parameters_dict()

    assert receipt["discrepancy_group"] in {"FN", "TARDY"}
    assert after["early_threshold"] <= before["early_threshold"]
    assert after["physics_weight"] >= before["physics_weight"]
    bounds = module.ProjectedAlarmController.PARAMETER_BOUNDS
    for name, (lower, upper) in bounds.items():
        assert lower <= after[name] <= upper


def test_alarm_update_only_affects_later_shot() -> None:
    module = load_module()
    controller = module.ProjectedAlarmController(learning_rate=0.05)
    margin, support, times = _alarm_trace("early_alarm")
    controller.begin_shot(10)
    sealed = controller.predict_and_seal(10, margin, support, times)
    controller.update_after_reveal(
        10, selected=True, is_disrupt=False, down_time_s=None,
        update_value=1.0,
    )
    assert controller.sealed_receipt(10) == sealed
    controller.begin_shot(20)
    later = controller.predict_and_seal(20, margin, support, times)
    assert later["parameter_version"] > sealed["parameter_version"]


def test_query_value_weights_update_only_after_selected_reveal_and_stay_bounded() -> None:
    module = load_module()
    model = module.OnlineQueryValueModel(learning_rate=0.10)
    features = {
        "uncertainty": 0.9,
        "physics_contradiction": 0.8,
        "self_supervised_surprise": 0.7,
        "teacher_disagreement": 0.6,
        "private_innovation": 0.5,
        "expected_gradient": 0.4,
    }
    before = model.weights_dict()
    score = model.score_and_seal(10, features)
    assert np.isfinite(score)
    with pytest.raises(PermissionError):
        model.update_after_reveal(10, selected=False, utility_target=1.0)
    assert model.weights_dict() == before

    model = module.OnlineQueryValueModel(learning_rate=0.10)
    model.score_and_seal(10, features)
    receipt = model.update_after_reveal(10, selected=True, utility_target=1.0)
    assert receipt["updated"] is True
    bounds = model.weight_bounds()
    for name, value in model.weights_dict().items():
        assert bounds[name][0] <= value <= bounds[name][1]


def test_adaptive_query_selector_is_immediate_exact_budget_and_uses_past_cutoff() -> None:
    module = load_module()
    selector = module.AdaptiveBlockBudgetSelector(
        block_size=20, quota=4, history_size=80, quantile=0.80,
        exploration_offset=3, exploration_stride=7,
    )
    first = [selector.decide(score=0.10, stream_index=index) for index in range(20)]
    second = [
        selector.decide(score=0.20 if index == 0 else 0.05, stream_index=20 + index)
        for index in range(20)
    ]

    assert sum(row["selected"] for row in first) == 4
    assert sum(row["selected"] for row in second) == 4
    assert first[3]["selected"] is True
    assert first[3]["route"] == "exploration"
    assert second[0]["selected"] is True
    assert second[0]["route"] == "exploitation"
    assert second[0]["cutoff"] == pytest.approx(0.10)
    assert selector.total_selected == 8


def test_adaptive_query_selector_future_scores_cannot_change_prefix_decisions() -> None:
    module = load_module()

    def prefix(suffix: list[float]) -> list[bool]:
        selector = module.AdaptiveBlockBudgetSelector(
            block_size=20, quota=4, history_size=40, quantile=0.80,
            exploration_offset=3, exploration_stride=7,
        )
        scores = [0.10 + 0.01 * index for index in range(10)] + suffix
        return [
            selector.decide(score=score, stream_index=index)["selected"]
            for index, score in enumerate(scores)
        ][:10]

    assert prefix([0.0] * 10) == prefix([1.0] * 10)


def test_operational_query_value_adapts_inside_registered_robust_region() -> None:
    module = load_module()
    model = module.OperationalQueryValueModel(
        uncertainty_weight=0.65, learning_rate=0.20,
    )
    features = {
        "uncertainty": 0.90,
        "physics_contradiction": 0.10,
        "self_supervised_surprise": 1.0,
        "teacher_disagreement": 0.0,
        "private_innovation": 1.0,
        "expected_gradient": 0.5,
    }
    before = model.weights_dict()["uncertainty"]
    score = model.score_and_seal(10, features)
    assert score == pytest.approx(0.65 * 0.90 + 0.35 * 0.10)
    receipt = model.update_after_reveal(
        10, selected=True, discrepancy_group="FN",
    )

    after = model.weights_dict()["uncertainty"]
    assert receipt["utility_target"] == 1.0
    assert after > before
    assert 0.55 <= after <= 0.75
