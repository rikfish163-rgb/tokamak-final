from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_value_module():
    path = HERE / "online_episode_value_v3.py"
    spec = importlib.util.spec_from_file_location("online_episode_value_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode_features(value: float, *, width: int = 20) -> np.ndarray:
    features = np.zeros(width, dtype=np.float64)
    features[0] = value
    features[1] = value
    features[2] = value
    features[3] = 1.0 - value
    features[4:] = 0.25 * value
    return features


def test_explicit_query_cost_monotonically_reduces_exploitation() -> None:
    module = load_value_module()

    def selected_count(cost: float) -> int:
        selector = module.StratifiedRollingTokenBudgetSelector(
            query_rate=1.0, capacity=20.0, block_size=20, quota=20,
            history_size=20, quantile=0.70, initial_cutoff=0.10,
            exploration_per_block=0, query_cost=cost,
        )
        return sum(
            selector.decide(score=0.60, stream_index=index)["selected"]
            for index in range(20)
        )

    assert selected_count(0.0) == 20
    assert selected_count(0.50) == 20
    assert selected_count(0.61) == 0


def test_query_cost_is_reported_as_net_value() -> None:
    module = load_value_module()
    selector = module.StratifiedRollingTokenBudgetSelector(
        query_rate=1.0, capacity=2.0, block_size=2, quota=2,
        history_size=2, quantile=0.70, initial_cutoff=0.10,
        exploration_per_block=0, query_cost=0.40,
    )
    receipt = selector.decide(score=0.75, stream_index=0)
    assert receipt["selected"]
    assert receipt["query_cost"] == pytest.approx(0.40)
    assert receipt["net_value"] == pytest.approx(0.35)

    with pytest.raises(ValueError, match="query cost"):
        module.StratifiedRollingTokenBudgetSelector(query_cost=1.01)


def _teach_fp_head(module):
    head = module.CausalEpisodeFPHead(
        feature_dim=20,
        minimum_history=4,
        minimum_per_class=2,
        fp_probability_threshold=0.55,
        max_extra_points=3,
    )
    history = [
        (1, _episode_features(0.95), False, "FP"),
        (2, _episode_features(0.90), False, "FP"),
        (3, _episode_features(0.05), True, "TP"),
        (4, _episode_features(0.10), True, "TP"),
    ]
    for shot, features, disruptive, group in history:
        head.predict_and_seal(
            shot,
            candidate_mask=np.array([True]),
            feature_matrix=features[None],
            route=np.array(["early"], dtype=object),
            escape_mask=np.array([False]),
        )
        head.update_after_reveal(
            shot,
            selected=True,
            is_disrupt=disruptive,
            discrepancy_group=group,
        )
    return head


def test_episode_head_delays_fp_like_transient_but_never_creates_alarm() -> None:
    module = load_value_module()
    head = _teach_fp_head(module)

    transient = head.predict_and_seal(
        10,
        candidate_mask=np.array([True, False, False, False]),
        feature_matrix=np.stack([_episode_features(0.95)] * 4),
        route=np.array(["early"] * 4, dtype=object),
        escape_mask=np.zeros(4, dtype=bool),
    )
    assert transient["base_alarm_index"] == 0
    assert transient["alarmed"] is False
    assert transient["required_points"] > 1
    head.discard_without_reveal(10)

    no_candidate = head.predict_and_seal(
        11,
        candidate_mask=np.zeros(4, dtype=bool),
        feature_matrix=np.stack([_episode_features(0.05)] * 4),
        route=np.array([None] * 4, dtype=object),
        escape_mask=np.zeros(4, dtype=bool),
    )
    assert no_candidate["alarmed"] is False
    assert no_candidate["base_alarm_index"] is None


def test_learned_fp_episode_stays_blocked_until_independent_escape() -> None:
    module = load_value_module()
    head = _teach_fp_head(module)
    fp_like = np.stack([_episode_features(0.95)] * 8)

    blocked = head.predict_and_seal(
        12,
        candidate_mask=np.ones(8, dtype=bool),
        feature_matrix=fp_like,
        route=np.array(["early"] * 8, dtype=object),
        escape_mask=np.zeros(8, dtype=bool),
    )
    assert blocked["alarmed"] is False
    assert blocked["filter_reason"] == "learned_fp_episode_blocked"
    head.discard_without_reveal(12)

    escaped = head.predict_and_seal(
        13,
        candidate_mask=np.ones(8, dtype=bool),
        feature_matrix=fp_like,
        route=np.array(["early"] * 8, dtype=object),
        escape_mask=np.array([False, False, False, False, True, False, False, False]),
    )
    assert escaped["alarmed"] is True
    assert escaped["alarm_index"] == 4
    assert escaped["filter_reason"] == "independent_evidence_escape"


def test_disruption_like_episode_passes_after_fp_head_is_fitted() -> None:
    module = load_value_module()
    head = _teach_fp_head(module)
    disruptive = np.stack([_episode_features(0.05)] * 5)

    decision = head.predict_and_seal(
        14,
        candidate_mask=np.ones(5, dtype=bool),
        feature_matrix=disruptive,
        route=np.array(["early"] * 5, dtype=object),
        escape_mask=np.zeros(5, dtype=bool),
    )
    assert decision["alarmed"] is True
    assert decision["alarm_index"] == 0
    assert decision["filter_reason"] == "episode_model_cleared"


def test_current_reveal_cannot_rewrite_a_sealed_episode_decision() -> None:
    module = load_value_module()
    head = _teach_fp_head(module)
    decision = head.predict_and_seal(
        20,
        candidate_mask=np.array([True, True, False]),
        feature_matrix=np.stack([_episode_features(0.95)] * 3),
        route=np.array(["early", "early", None], dtype=object),
        escape_mask=np.zeros(3, dtype=bool),
    )
    sealed = dict(decision)
    head.update_after_reveal(
        20, selected=True, is_disrupt=False, discrepancy_group="FP",
    )
    assert decision == sealed


def test_episode_head_prefix_decision_is_invariant_to_unseen_future() -> None:
    module = load_value_module()
    source = _teach_fp_head(module)
    short = module.CausalEpisodeFPHead.from_state_dict(source.state_dict())
    long = module.CausalEpisodeFPHead.from_state_dict(source.state_dict())
    prefix_mask = np.array([True, True, True, True])
    prefix_features = np.stack([_episode_features(0.95)] * len(prefix_mask))
    prefix_routes = np.array(["early"] * len(prefix_mask), dtype=object)
    short_decision = short.predict_and_seal(
        30,
        candidate_mask=prefix_mask,
        feature_matrix=prefix_features,
        route=prefix_routes,
        escape_mask=np.zeros(len(prefix_mask), dtype=bool),
    )
    long_decision = long.predict_and_seal(
        30,
        candidate_mask=np.r_[prefix_mask, [False, True, True]],
        feature_matrix=np.vstack([
            prefix_features, np.stack([_episode_features(0.95)] * 3),
        ]),
        route=np.array(["early"] * 4 + [None, "high", "high"], dtype=object),
        escape_mask=np.zeros(7, dtype=bool),
    )
    assert short_decision["alarm_index"] == long_decision["alarm_index"]
    assert short_decision["required_points"] == long_decision["required_points"]


def test_delta_sequence_representation_is_strictly_causal() -> None:
    module = load_value_module()
    rng = np.random.default_rng(7)
    source = rng.normal(size=(48, 20))
    changed = source.copy()
    changed[24:] = rng.normal(size=changed[24:].shape) * 100.0

    original = module.causal_delta_sequence_matrix(source)
    counterfactual = module.causal_delta_sequence_matrix(changed)

    assert original.shape == (48, 60)
    np.testing.assert_allclose(original[:24], counterfactual[:24])
    np.testing.assert_allclose(original[:, :20], source)


def _teach_protected_sequence_head(module):
    head = module.CausalProtectedSequenceFPHead(
        base_feature_dim=20,
        minimum_history=4,
        minimum_per_class=2,
        l2=0.3,
        safety_margin=0.06,
        negative_quantile=0.9,
        safe_quantile=0.25,
        minimum_confidence=0.75,
    )
    history = [
        (1, 0.95, False, "FP"),
        (2, 0.90, False, "FP"),
        (3, 0.05, True, "TP"),
        (4, 0.10, True, "TP"),
    ]
    for shot, value, disruptive, group in history:
        head.predict_and_seal(
            shot,
            candidate_mask=np.array([True]),
            feature_matrix=_episode_features(value)[None],
            route=np.array(["early"], dtype=object),
            escape_mask=np.array([False]),
        )
        head.update_after_reveal(
            shot, selected=True, is_disrupt=disruptive,
            discrepancy_group=group,
        )
    return head


def test_protected_sequence_head_uses_past_class_boundary_not_fixed_probability() -> None:
    module = load_value_module()
    head = _teach_protected_sequence_head(module)
    matrix = np.stack([_episode_features(0.95)] * 8)
    blocked = head.predict_and_seal(
        10,
        candidate_mask=np.ones(8, dtype=bool),
        feature_matrix=matrix,
        route=np.array(["early"] * 8, dtype=object),
        escape_mask=np.zeros(8, dtype=bool),
    )

    assert blocked["alarmed"] is False
    assert blocked["filter_reason"] == "protected_sequence_episode_blocked"
    assert blocked["protection_threshold"] < 1.0
    assert blocked["safe_support_count"] == 2
    assert blocked["disruptive_support_count"] == 2


def test_protected_sequence_head_is_transparent_without_both_classes() -> None:
    module = load_value_module()
    head = module.CausalProtectedSequenceFPHead(
        base_feature_dim=20, minimum_history=4, minimum_per_class=2,
    )
    decision = head.predict_and_seal(
        1,
        candidate_mask=np.array([True, True]),
        feature_matrix=np.stack([_episode_features(0.95)] * 2),
        route=np.array(["early", "early"], dtype=object),
        escape_mask=np.zeros(2, dtype=bool),
    )

    assert decision["alarmed"] is True
    assert decision["alarm_index"] == 0
    assert decision["filter_reason"] == "episode_model_not_ready"
    assert decision["protection_threshold"] > 1.0


def test_protected_sequence_head_checkpoint_round_trip_is_exact() -> None:
    module = load_value_module()
    head = _teach_protected_sequence_head(module)
    restored = module.CausalProtectedSequenceFPHead.from_state_dict(
        head.state_dict()
    )
    matrix = np.stack([_episode_features(0.95)] * 8)
    kwargs = {
        "candidate_mask": np.ones(8, dtype=bool),
        "feature_matrix": matrix,
        "route": np.array(["early"] * 8, dtype=object),
        "escape_mask": np.zeros(8, dtype=bool),
    }
    assert head.predict_and_seal(20, **kwargs) == restored.predict_and_seal(
        20, **kwargs,
    )


def _query_features(**overrides: float) -> dict[str, float]:
    values = {
        "uncertainty": 0.7,
        "physics_contradiction": 0.7,
        "self_supervised_surprise": 0.5,
        "teacher_disagreement": 0.4,
        "private_innovation": 0.5,
        "expected_gradient": 0.6,
        "decision_changeability": 0.7,
        "fp_probability": 0.5,
        "fp_confidence": 0.0,
        "physics_reliability": 0.7,
        "stability_risk": 0.2,
    }
    values.update(overrides)
    return values


def test_decision_value_uses_changeability_novelty_and_realized_utility() -> None:
    module = load_value_module()
    value = module.CausalDecisionQueryValueModel(feature_dim=20)
    phenotype = _episode_features(0.9)
    high, high_receipt = value.score_and_seal(
        1,
        _query_features(decision_changeability=0.95, expected_gradient=0.9),
        phenotype=phenotype,
    )
    value.update_after_reveal(
        1, selected=True, discrepancy_group="FP", realized_utility=0.9,
    )
    low, low_receipt = value.score_and_seal(
        2,
        _query_features(decision_changeability=0.05, expected_gradient=0.1),
        phenotype=phenotype,
    )

    assert high > low
    assert set(high_receipt["experts"]) == {
        "error_correction", "information_gain", "physics_conflict",
        "gradient_actionability", "coverage_exploration",
    }
    assert 0.0 <= low_receipt["novelty"] <= 1.0
    assert 0.0 <= low_receipt["coverage_debt"] <= 1.0


def test_decision_value_rejects_unselected_label_supervision() -> None:
    module = load_value_module()
    value = module.CausalDecisionQueryValueModel(feature_dim=20)
    value.score_and_seal(1, _query_features(), phenotype=_episode_features(0.5))
    with pytest.raises(PermissionError, match="selected"):
        value.update_after_reveal(
            1, selected=False, discrepancy_group="FP", realized_utility=1.0,
        )


def test_decision_value_checkpoint_round_trip_is_exact() -> None:
    module = load_value_module()
    value = module.CausalDecisionQueryValueModel(feature_dim=20)
    phenotype = _episode_features(0.9)
    value.score_and_seal(1, _query_features(), phenotype=phenotype)
    value.update_after_reveal(
        1, selected=True, discrepancy_group="FP", realized_utility=0.9,
    )
    restored = module.CausalDecisionQueryValueModel.from_state_dict(
        value.state_dict()
    )

    original_score, original_receipt = value.score_and_seal(
        2, _query_features(), phenotype=phenotype,
    )
    restored_score, restored_receipt = restored.score_and_seal(
        2, _query_features(), phenotype=phenotype,
    )
    assert original_score == restored_score
    assert original_receipt == restored_receipt


def test_stratified_token_selector_reserves_real_exploration() -> None:
    module = load_value_module()
    selector = module.StratifiedRollingTokenBudgetSelector(
        query_rate=0.30,
        capacity=4.0,
        block_size=10,
        quota=3,
        history_size=20,
        quantile=0.8,
        initial_cutoff=0.95,
        exploration_per_block=1,
        exploration_offset=3,
        exploration_stride=3,
    )
    decisions = [
        selector.decide(score=0.05, stream_index=index)
        for index in range(20)
    ]
    exploration = [row for row in decisions if row["route"] == "exploration"]

    assert len(exploration) == 2
    assert selector.total_selected == 2
    assert all(row["route"] != "forced_fill" for row in decisions)
