from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "run_unified_online_active.py"
    spec = importlib.util.spec_from_file_location("run_unified_online_active", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def locked_v8_args(**overrides):
    values = {
        "locked_best_v8": True,
        "seed": 131,
        "max_shots": 360,
        "private_support_epochs": 8,
        "adaptation_events": 64,
        "intra_steps": 1,
        "inter_steps": 2,
        "intra_lr": 1e-4,
        "private_intra_lr": 1e-4,
        "intra_physics_weight": 0.10,
        "inter_lr": 5e-4,
        "private_inter_lr": 2e-4,
        "alarm_lr": 0.02,
        "query_block_size": 20,
        "query_quota": 4,
        "query_history_size": 80,
        "query_quantile": 0.80,
        "query_uncertainty_weight": 0.65,
        "query_value_lr": 0.05,
        "novel_early_min_physics_support": 0.0,
        "novel_early_min_trend": 0.0,
        "causal_error_memory": True,
        "error_memory_radius": 0.35,
        "error_memory_min_support": 2,
        "error_memory_weight": 0.0,
        "query_initial_cutoff": 0.35,
        "query_token_capacity": 12.0,
        "strict_window_active": True,
        "window_candidate_stride": 8,
        "window_update_rate": 0.25,
        "window_token_capacity": 4.0,
        "window_history_size": 64,
        "window_value_quantile": 0.80,
        "window_initial_cutoff": 0.40,
        "window_minimum_history": 8,
        "window_minimum_gap": 8,
        "typed_endpoint_memory": True,
        "endpoint_memory_radius": 0.35,
        "endpoint_memory_min_shots": 2,
        "endpoint_memory_positive": 0.35,
        "endpoint_memory_negative": 0.25,
        "endpoint_memory_physics_gate": 0.50,
        "endpoint_memory_protection_ratio": 1.0,
        "endpoint_memory_stride": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def episode_v3_args(**overrides):
    values = {
        "episode_value_v3": True,
        "development_search": True,
        "locked_best_v8": False,
        "frozen_online_config": None,
        "query_token_capacity": 12.0,
        "episode_fp_threshold": 0.65,
        "episode_minimum_confidence": 0.75,
        "episode_max_extra_points": 8,
        "episode_min_history": 12,
        "episode_min_per_class": 3,
        "episode_high_escape_margin": 0.75,
        "episode_physics_escape": 0.85,
        "replay_max_distance": 0.35,
        "episode_maximum_points": 64,
        "episode_shoulder_margin": 0.50,
        "query_exploration_per_block": 1,
        "episode_query_mode": "active",
        "episode_gradient_mode": "episode",
        "episode_protected_sequence": False,
        "episode_sequence_l2": 0.30,
        "episode_protection_margin": 0.06,
        "episode_negative_quantile": 0.90,
        "episode_safe_quantile": 0.25,
        "episode_isolate_learning_feedback": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_episode_v3_requires_explicit_development_mode_and_valid_geometry() -> None:
    module = load_module()
    module.validate_episode_v3_arguments(episode_v3_args())

    with pytest.raises(ValueError, match="development-search"):
        module.validate_episode_v3_arguments(
            episode_v3_args(development_search=False)
        )
    with pytest.raises(ValueError, match="confidence"):
        module.validate_episode_v3_arguments(
            episode_v3_args(episode_minimum_confidence=1.1)
        )
    with pytest.raises(ValueError, match="replay"):
        module.validate_episode_v3_arguments(
            episode_v3_args(replay_max_distance=0.0)
        )
    with pytest.raises(ValueError, match="query mode"):
        module.validate_episode_v3_arguments(
            episode_v3_args(episode_query_mode="replace_baseline")
        )
    with pytest.raises(ValueError, match="gradient mode"):
        module.validate_episode_v3_arguments(
            episode_v3_args(episode_gradient_mode="unknown")
        )


def test_protected_sequence_requires_registered_causal_calibration() -> None:
    module = load_module()
    module.validate_episode_v3_arguments(
        episode_v3_args(
            episode_protected_sequence=True,
            episode_high_escape_margin=99.0,
        )
    )

    with pytest.raises(ValueError, match="negative quantile"):
        module.validate_episode_v3_arguments(
            episode_v3_args(
                episode_protected_sequence=True,
                episode_negative_quantile=0.49,
            )
        )
    with pytest.raises(ValueError, match="safe quantile"):
        module.validate_episode_v3_arguments(
            episode_v3_args(
                episode_protected_sequence=True,
                episode_safe_quantile=0.75,
            )
        )
    with pytest.raises(ValueError, match="regularization"):
        module.validate_episode_v3_arguments(
            episode_v3_args(
                episode_protected_sequence=True,
                episode_sequence_l2=0.0,
            )
        )

    module.validate_episode_v3_arguments(
        episode_v3_args(
            episode_protected_sequence=True,
            episode_isolate_learning_feedback=True,
            episode_query_mode="shadow",
            episode_gradient_mode="legacy",
        )
    )
    with pytest.raises(ValueError, match="shadow query mode"):
        module.validate_episode_v3_arguments(
            episode_v3_args(
                episode_protected_sequence=True,
                episode_isolate_learning_feedback=True,
                episode_query_mode="active",
                episode_gradient_mode="legacy",
            )
        )
    with pytest.raises(ValueError, match="legacy gradient mode"):
        module.validate_episode_v3_arguments(
            episode_v3_args(
                episode_protected_sequence=True,
                episode_isolate_learning_feedback=True,
                episode_query_mode="shadow",
                episode_gradient_mode="episode",
            )
        )


def test_temporal_risk_forecast_requires_strict_prediction_before_update() -> None:
    module = load_module()
    args = SimpleNamespace(
        timescale_mode="intra_inter", query_cost=0.0,
        query_score_mode="current", episode_value_v3=False,
        fixed_query_manifest=None, query_token_capacity=0.0,
        order_manifest=None, diagnostic_reorder=False,
        development_search=True, strict_window_active=False,
        intra_temporal_risk_weight=1.0,
        intra_temporal_target_max_delta=0.5,
    )

    with pytest.raises(ValueError, match="prediction-before-update"):
        module.validate_timescale_ablation_arguments(args)

    args.strict_window_active = True
    module.validate_timescale_ablation_arguments(args)


def test_observed_physics_ttt_requires_strict_prediction_before_update() -> None:
    module = load_module()
    args = SimpleNamespace(
        timescale_mode="intra_inter", query_cost=0.0,
        query_score_mode="current", episode_value_v3=False,
        fixed_query_manifest=None, query_token_capacity=0.0,
        order_manifest=None, diagnostic_reorder=False,
        development_search=True, strict_window_active=False,
        intra_temporal_risk_weight=0.0,
        intra_temporal_target_max_delta=0.5,
        intra_observed_physics_weight=1.0,
    )

    with pytest.raises(ValueError, match="prediction-before-update"):
        module.validate_timescale_ablation_arguments(args)

    args.strict_window_active = True
    module.validate_timescale_ablation_arguments(args)


def test_representation_only_ttt_requires_strict_prediction_before_update() -> None:
    module = load_module()
    args = SimpleNamespace(
        timescale_mode="intra_inter", query_cost=0.0,
        query_score_mode="current", episode_value_v3=False,
        fixed_query_manifest=None, query_token_capacity=0.0,
        order_manifest=None, diagnostic_reorder=False,
        development_search=True, strict_window_active=False,
        intra_temporal_risk_weight=0.0,
        intra_temporal_target_max_delta=0.5,
        intra_observed_physics_weight=0.0,
        intra_parameter_scope="representation_only",
    )

    with pytest.raises(ValueError, match="prediction-before-update"):
        module.validate_timescale_ablation_arguments(args)

    args.strict_window_active = True
    module.validate_timescale_ablation_arguments(args)

    args.intra_parameter_scope = "risk_head_only"
    with pytest.raises(ValueError, match="parameter scope"):
        module.validate_timescale_ablation_arguments(args)


@pytest.mark.parametrize("name", [
    "intra_temporal_risk_weight", "intra_observed_physics_weight",
])
def test_intra_unlabeled_loss_weights_must_be_finite(name: str) -> None:
    module = load_module()
    args = SimpleNamespace(
        timescale_mode="intra_inter", query_cost=0.0,
        query_score_mode="current", episode_value_v3=False,
        fixed_query_manifest=None, query_token_capacity=0.0,
        order_manifest=None, diagnostic_reorder=False,
        development_search=True, strict_window_active=True,
        intra_temporal_risk_weight=0.0,
        intra_temporal_target_max_delta=0.5,
        intra_observed_physics_weight=0.0,
    )
    setattr(args, name, float("nan"))

    with pytest.raises(ValueError, match="finite"):
        module.validate_timescale_ablation_arguments(args)


def test_deployed_episode_group_is_separate_from_legacy_learning_group() -> None:
    module = load_module()
    times = np.asarray([1.0, 1.1, 1.2])
    deployed = {"alarmed": False, "alarm_index": None}
    learning = {"alarmed": True, "alarm_index": 1}

    groups = module.causal_discrepancy_groups(
        deployed_alarm=deployed, learning_alarm=learning, times_s=times,
        is_disrupt=False, down_time_s=None,
    )

    assert groups == {"deployed": "TN", "learning": "FP"}


def test_episode_deployment_overlay_cannot_create_or_advance_alarm() -> None:
    module = load_module()
    learning = {
        "alarmed": True, "alarm_index": 10, "route": "early",
        "early_score": 0.4, "parameter_version": 3,
    }
    veto = {
        "alarmed": False, "alarm_index": None, "route": None,
        "early_score": None, "filter_reason": "protected_fp_episode",
    }

    deployed = module.deploy_episode_decision(learning, veto)

    assert not deployed["alarmed"]
    assert deployed["alarm_index"] is None
    assert deployed["learning_alarm_reference"] == learning
    assert learning["alarmed"] and learning["alarm_index"] == 10

    with pytest.raises(ValueError, match="cannot advance"):
        module.deploy_episode_decision(
            learning,
            {
                "alarmed": True, "alarm_index": 9, "route": "early",
                "early_score": 0.5,
            },
        )
    with pytest.raises(ValueError, match="cannot create"):
        module.deploy_episode_decision(
            {**learning, "alarmed": False, "alarm_index": None},
            {
                "alarmed": True, "alarm_index": 10, "route": "early",
                "early_score": 0.5,
            },
        )


def test_realized_query_utility_values_fp_episode_learning_after_rollbacks() -> None:
    module = load_module()
    false_alarm = module.decision_query_realized_utility(
        discrepancy_group="FP", supervision_value=0.5,
        episode_updated=True, shared_accepted=False,
        private_accepted=False, novelty=0.2,
    )
    redundant_tn = module.decision_query_realized_utility(
        discrepancy_group="TN", supervision_value=0.0,
        episode_updated=False, shared_accepted=False,
        private_accepted=False, novelty=0.2,
    )

    assert false_alarm >= 0.70
    assert false_alarm > redundant_tn


def test_episode_supervision_strengthens_fp_without_changing_correct_groups() -> None:
    module = load_module()
    assert module.episode_supervision_scale(
        0.10, discrepancy_group="FP", episode_confidence=1.0,
    ) >= 0.45
    assert module.episode_supervision_scale(
        0.10, discrepancy_group="TN", episode_confidence=1.0,
    ) == pytest.approx(0.10)


def test_locked_best_v8_accepts_only_registered_configuration() -> None:
    module = load_module()
    module.validate_locked_best_v8_arguments(locked_v8_args())

    with pytest.raises(ValueError, match="locked best-v8"):
        module.validate_locked_best_v8_arguments(
            locked_v8_args(inter_lr=6e-4)
        )
    with pytest.raises(ValueError, match="locked best-v8"):
        module.validate_locked_best_v8_arguments(
            locked_v8_args(query_quota=6)
        )


def test_expected_query_count_requires_complete_exact_budget_blocks() -> None:
    module = load_module()
    assert module.expected_query_count(360, block_size=20, quota=2) == 36
    assert module.expected_query_count(360, block_size=20, quota=8) == 144
    with pytest.raises(ValueError, match="complete query blocks"):
        module.expected_query_count(359, block_size=20, quota=4)


def test_zero_inter_steps_becomes_a_no_gradient_finalize() -> None:
    module = load_module()
    assert module.inter_update_control(0, 0.8) == (1, 0.0)
    assert module.inter_update_control(2, 0.8) == (2, 0.8)
    with pytest.raises(ValueError, match="inter steps"):
        module.inter_update_control(-1, 0.8)


def test_error_phenotype_contains_risk_trend_without_labels() -> None:
    module = load_module()
    rising = module.error_memory_phenotype(
        risk=np.linspace(0.1, 0.9, 100), threshold=0.5,
        query_features={
            "uncertainty": 0.8, "physics_contradiction": 0.6,
            "self_supervised_surprise": 0.4,
            "teacher_disagreement": 0.2, "private_innovation": 0.7,
            "expected_gradient": 0.5,
        },
    )
    falling = module.error_memory_phenotype(
        risk=np.linspace(0.9, 0.1, 100), threshold=0.5,
        query_features={
            "uncertainty": 0.8, "physics_contradiction": 0.6,
            "self_supervised_surprise": 0.4,
            "teacher_disagreement": 0.2, "private_innovation": 0.7,
            "expected_gradient": 0.5,
        },
    )

    assert set(rising) == set(falling)
    assert rising["risk_late_minus_early"] > 0.5
    assert falling["risk_late_minus_early"] < 0.5
    assert all(0.0 <= value <= 1.0 for value in rising.values())


def test_frozen_online_contract_rejects_runtime_parameter_drift() -> None:
    module = load_module()
    args = locked_v8_args(
        locked_best_v8=False,
        novel_early_min_physics_support=0.80,
    )
    contract = {
        "allowed_seeds": [131, 133, 137],
        "parameters": {
            name: getattr(args, name) for name in module.FROZEN_PARAMETER_NAMES
        },
    }
    module.validate_frozen_online_arguments(args, contract)
    contract["parameters"]["query_quota"] = 6
    with pytest.raises(ValueError, match="frozen online configuration"):
        module.validate_frozen_online_arguments(args, contract)


def test_frozen_v2_contract_locks_window_query_and_endpoint_memory() -> None:
    module = load_module()
    args = locked_v8_args(
        locked_best_v8=False,
        query_quota=6,
        query_history_size=160,
        query_quantile=0.70,
        query_uncertainty_weight=0.55,
    )
    contract = {
        "allowed_seeds": [131, 133, 137],
        "locked_parameter_names": list(module.FROZEN_V2_PARAMETER_NAMES),
        "parameters": {
            name: getattr(args, name)
            for name in module.FROZEN_V2_PARAMETER_NAMES
        },
    }
    module.validate_frozen_online_arguments(args, contract)

    contract["parameters"]["endpoint_memory_radius"] = 0.45
    with pytest.raises(ValueError, match="endpoint_memory_radius"):
        module.validate_frozen_online_arguments(args, contract)

    contract["parameters"]["endpoint_memory_radius"] = 0.35
    contract["parameters"]["strict_window_active"] = False
    with pytest.raises(ValueError, match="strict_window_active"):
        module.validate_frozen_online_arguments(args, contract)


def test_static_phase_trace_must_match_requested_shot(tmp_path: Path) -> None:
    module = load_module()
    trace = tmp_path / "100.npz"
    module.seal_static_phase_trace(
        trace, shot_id=100, static_logit=np.array([-1.0, 0.5], np.float32)
    )
    loaded = module.load_static_phase_trace(trace, expected_shot_id=100)
    np.testing.assert_array_equal(loaded, np.array([-1.0, 0.5], np.float32))

    with pytest.raises(RuntimeError, match="shot identity"):
        module.load_static_phase_trace(trace, expected_shot_id=101)


def test_causal_raw8_zero_sentinel_does_not_use_future_observability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = load_module()
    path = tmp_path / "1.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for name in module.private23.DROPPED_RAW8:
            values = np.ones(20, dtype=np.float32)
            if name in ("MNM", "Mir_VV", "Mir_avg_amp"):
                values[:10] = 0.0
            data.create_dataset(name, data=values)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DATA", tmp_path / "data")
    expected = tmp_path / "data/EAST/unified_hdf5/0/1.hdf5"
    expected.parent.mkdir(parents=True)
    path.replace(expected)

    _values, available = module.causal_raw8_signal_only(1)
    for name in ("MNM", "Mir_VV", "Mir_avg_amp"):
        index = module.private23.DROPPED_RAW8.index(name)
        assert not available[:10, index].any()
        assert available[10:, index].all()


def test_private_window_features_do_not_change_when_only_future_changes() -> None:
    module = load_module()
    rng = np.random.default_rng(8)
    values = rng.normal(size=(300, 8)).astype(np.float32)
    available = np.ones_like(values, dtype=bool)
    base = np.zeros(173, dtype=np.float32)
    original = module.make_private_batch(values, available, np.array([20]), base)
    modified = values.copy()
    modified[200:] += 1000.0
    changed = module.make_private_batch(modified, available, np.array([20]), base)

    np.testing.assert_array_equal(original["raw"].numpy(), changed["raw"].numpy())


def test_endpoint_memory_key_does_not_read_future_risk_or_raw8() -> None:
    module = load_module()
    rng = np.random.default_rng(29)
    frozen = rng.normal(size=module.dha.FEATURE_DIM).astype(np.float32)
    risk_logit = rng.normal(size=100).astype(np.float32)
    raw = rng.normal(size=(240, 8)).astype(np.float32)
    available = np.ones_like(raw, dtype=bool)

    original = module.endpoint_memory_key(
        frozen_state=frozen, risk_logit=risk_logit, risk_index=40,
        physics_support=0.7, physics_known=True,
        raw=raw, raw_available=available, raw_endpoint=160,
    )
    changed_risk = risk_logit.copy()
    changed_risk[41:] += 1e6
    changed_raw = raw.copy()
    changed_raw[161:] -= 1e6
    changed = module.endpoint_memory_key(
        frozen_state=frozen, risk_logit=changed_risk, risk_index=40,
        physics_support=0.7, physics_known=True,
        raw=changed_raw, raw_available=available, raw_endpoint=160,
    )

    np.testing.assert_array_equal(original, changed)
    assert np.isfinite(original).all()


def test_supervision_value_prioritizes_miss_then_tardy_then_false_alarm() -> None:
    module = load_module()
    risk = np.full(10, 0.5)
    target = np.zeros(10)
    false_alarm = module.supervision_value(
        risk, target, group="FP", lead_ms=None,
    )
    tardy = module.supervision_value(
        risk, target, group="TARDY", lead_ms=0.0,
    )
    missed = module.supervision_value(
        risk, target, group="FN", lead_ms=None,
    )
    assert missed > tardy > false_alarm


def test_operational_temporal_target_marks_the_timely_alarm_band() -> None:
    module = load_module()
    shot = {
        "start": 0.0,
        "rate": 1000.0,
        "is_disrupt": True,
        "down": 1.0,
    }
    binary, mechanism, operational, timely = module.operational_temporal_targets(
        shot, risk_length=1000,
    )
    times = (
        shot["start"]
        + np.arange(module.alarm.W - 1, module.alarm.W - 1 + 1000) / shot["rate"]
    )
    ttd_ms = (shot["down"] - times) * 1000.0
    expected_timely = (ttd_ms >= 30.0) & (ttd_ms <= 125.0)
    expected_late = (ttd_ms >= 0.0) & (ttd_ms < 30.0)

    np.testing.assert_array_equal(timely, expected_timely)
    assert np.all(operational[expected_timely] == 1.0)
    assert np.all(operational[expected_late] == 1.0)
    assert np.all(operational[~binary] == 0.0)
    assert np.any(mechanism[expected_timely] < operational[expected_timely])


def test_supervised_selection_uses_closest_timely_alarm_evidence() -> None:
    module = load_module()
    risk = np.array([0.01, 0.20, 0.10, 0.90], dtype=np.float64)
    labels = np.array([True, True, True, False])
    timely = np.array([False, True, True, False])
    selected = module.select_supervised_indices(
        risk=risk,
        labels=labels,
        timely_labels=timely,
        margin=np.array([0.9, -0.4, -0.1, -1.0]),
        alarm_boundary_margin=np.zeros(4),
        physics_support=np.zeros(4),
        alarm_parameters={
            "high_threshold": 0.0,
            "physics_weight": 8.0,
            "trend_weight": 12.0,
            "early_threshold": 7.9,
        },
        group="TARDY",
        count=1,
    )

    np.testing.assert_array_equal(selected, np.array([2]))


def test_fp_episode_selection_covers_contiguous_alarm_shoulders() -> None:
    module = load_module()
    evidence = np.r_[
        np.full(20, -2.0),
        np.linspace(-0.45, 0.8, 20),
        np.linspace(0.8, -0.45, 20),
        np.full(20, -2.0),
    ]
    selected = module.select_fp_episode_indices(
        evidence=evidence,
        alarm_index=28,
        maximum_points=64,
        shoulder_margin=0.50,
    )

    assert len(selected) > 16
    assert 28 in selected
    assert np.all(np.diff(selected) == 1)
    assert evidence[selected[0]] >= -0.50
    assert evidence[selected[-1]] >= -0.50


def test_episode_feature_row_cannot_read_future_signal_or_risk() -> None:
    module = load_module()
    length = 40
    rng = np.random.default_rng(991)
    online = rng.normal(size=length)
    static = rng.normal(size=length)
    shared = rng.normal(size=length)
    memory = rng.normal(scale=0.05, size=length)
    physics = np.clip(rng.random(length), 0.0, 1.0)
    raw = rng.normal(size=(length + module.alarm.W - 1, 8))
    available = np.ones_like(raw, dtype=bool)
    parameters = {
        "high_threshold": 0.0,
        "physics_weight": 8.0,
        "trend_weight": 12.0,
        "early_threshold": 7.9,
    }
    trace = module.dual_alarm.dual_threshold_trace(
        online, physics, **parameters,
    )
    original = module.causal_episode_feature_matrix(
        online_logit=online,
        static_logit=static,
        shared_logit=shared,
        endpoint_memory_residual=memory,
        physics_support=physics,
        alarm_trace=trace,
        threshold_logit=0.0,
        alarm_parameters=parameters,
        raw=raw,
        raw_available=available,
        model_branch_mask=np.ones(length, dtype=bool),
        alarm_parameter_branch_mask=np.ones(length, dtype=bool),
    )
    changed_online = online.copy()
    changed_online[6:] += 1000.0
    changed_raw = raw.copy()
    changed_raw[module.alarm.W - 1 + 6:] += 1000.0
    changed_trace = module.dual_alarm.dual_threshold_trace(
        changed_online, physics, **parameters,
    )
    changed = module.causal_episode_feature_matrix(
        online_logit=changed_online,
        static_logit=static,
        shared_logit=shared,
        endpoint_memory_residual=memory,
        physics_support=physics,
        alarm_trace=changed_trace,
        threshold_logit=0.0,
        alarm_parameters=parameters,
        raw=changed_raw,
        raw_available=available,
        model_branch_mask=np.ones(length, dtype=bool),
        alarm_parameter_branch_mask=np.ones(length, dtype=bool),
    )

    np.testing.assert_allclose(original[5], changed[5], atol=0.0, rtol=0.0)
    assert original.shape == (length, 20)


def test_sealed_alarm_boundary_matches_either_registered_route() -> None:
    module = load_module()
    combined_logit = np.array([-2.0, -1.0, 0.5], dtype=np.float64)
    support = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    parameters = {
        "high_threshold": 0.0,
        "physics_weight": 8.0,
        "trend_weight": 0.0,
        "early_threshold": 7.9,
    }
    boundary = module.sealed_alarm_boundary_logit(
        combined_logit,
        physics_support=support,
        threshold_logit=-1.0,
        alarm_parameters=parameters,
    )

    # With no physics the immutable high route is limiting (-1 + 0).
    assert boundary[0] == pytest.approx(-1.0)
    # Full physics makes the early route slightly easier: -1 + (7.9 - 8).
    assert boundary[1] == pytest.approx(-1.1)


def test_supported_fusion_rejects_pure_model_alarm_parameter_synergy() -> None:
    module = load_module()
    physics = np.zeros(20, dtype=np.float64)
    current = {
        "high_threshold": -0.10,
        "physics_weight": 8.0,
        "trend_weight": 12.0,
        "early_threshold": 7.9,
    }
    # Adapted model (-0.05) is below the immutable high boundary (0.0), and
    # static model (-0.15) is below the adapted boundary (-0.10).  Only their
    # unvalidated sum of relaxations would alarm.
    rejected = module.supported_fusion_alarm(
        online_logit=np.full(20, -0.05),
        static_logit=np.full(20, -0.15),
        physics_support=physics,
        threshold_logit=0.0,
        alarm_parameters=current,
    )
    supported = module.supported_fusion_alarm(
        online_logit=np.full(20, 0.05),
        static_logit=np.full(20, -0.15),
        physics_support=physics,
        threshold_logit=0.0,
        alarm_parameters=current,
    )

    assert rejected["alarmed"] is False
    assert rejected["unfiltered_alarmed"] is True
    assert supported["alarmed"] is True
    assert supported["model_branch_support"] is True


def test_novel_online_early_alarm_requires_registered_physics_support() -> None:
    module = load_module()
    parameters = dict(module.INITIAL_ALARM_PARAMETERS)
    online = np.linspace(-1.0, -0.05, 200)
    static = np.full(200, -3.0)

    rejected = module.supported_fusion_alarm(
        online_logit=online, static_logit=static,
        physics_support=np.full(200, 0.75), threshold_logit=0.0,
        alarm_parameters=parameters, novel_early_min_physics_support=0.80,
    )
    admitted = module.supported_fusion_alarm(
        online_logit=online, static_logit=static,
        physics_support=np.full(200, 1.00), threshold_logit=0.0,
        alarm_parameters=parameters, novel_early_min_physics_support=0.80,
    )

    assert rejected["alarmed"] is False
    assert rejected["filter_reason"] == "novel_early_physics_guard"
    assert admitted["alarmed"] is True
    assert admitted["route"] == "early"


def test_novel_online_early_alarm_accepts_physics_or_strong_causal_trend() -> None:
    module = load_module()
    parameters = dict(module.INITIAL_ALARM_PARAMETERS)
    weak_trend = np.linspace(-1.0, -0.05, 200)
    strong_trend = np.linspace(-3.0, -0.05, 200)
    static = np.full(200, -4.0)
    support = np.full(200, 0.70)

    rejected = module.supported_fusion_alarm(
        online_logit=weak_trend, static_logit=static,
        physics_support=support, threshold_logit=0.0,
        alarm_parameters=parameters, novel_early_min_physics_support=0.75,
        novel_early_min_trend=0.40,
    )
    admitted_by_trend = module.supported_fusion_alarm(
        online_logit=strong_trend, static_logit=static,
        physics_support=support, threshold_logit=0.0,
        alarm_parameters=parameters, novel_early_min_physics_support=0.75,
        novel_early_min_trend=0.40,
    )
    admitted_by_physics = module.supported_fusion_alarm(
        online_logit=weak_trend, static_logit=static,
        physics_support=np.full(200, 0.80), threshold_logit=0.0,
        alarm_parameters=parameters, novel_early_min_physics_support=0.75,
        novel_early_min_trend=0.40,
    )

    assert rejected["alarmed"] is False
    assert rejected["filter_reason"] == "novel_early_support_or_trend_guard"
    assert admitted_by_trend["alarmed"] is True
    assert admitted_by_physics["alarmed"] is True


def test_query_features_are_label_free_and_bounded() -> None:
    module = load_module()
    features, components = module.query_features(
        risk=np.array([0.1, 0.4, 0.7]),
        physics_support=np.array([0.0, 0.8, 0.2]),
        physics_known=np.array([True, True, True]), threshold=0.5,
        update_values=[0.2, 0.8], reconstruction=[0.4, 0.9],
        teacher=[0.1, 0.2], private_innovation=[0.3, 0.7],
    )
    assert set(features) == {
        "uncertainty", "physics_contradiction", "self_supervised_surprise",
        "teacher_disagreement", "private_innovation", "expected_gradient",
    }
    assert all(0.0 <= value <= 1.0 for value in features.values())
    assert not {"label", "is_disrupt", "down_time_s"}.intersection(components)


def test_alarm_metrics_counts_very_early_alarm_as_timely() -> None:
    module = load_module()
    rows = [{
        "shot": 10, "alarmed": True, "alarm_time_s": 1.0,
    }, {
        "shot": 20, "alarmed": False, "alarm_time_s": None,
    }]
    outcomes = {
        10: {"is_disrupt": True, "down_time_s": 5.0},
        20: {"is_disrupt": False, "down_time_s": None},
    }
    result = module.alarm_metrics(rows, outcomes)
    assert result["tpr_at_30ms"] == 1.0
    assert result["over_3000ms_count"] == 1


def test_static_combined_logits_handles_more_than_one_inference_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    monkeypatch.setattr(module.alarm, "DEV", "cpu")
    shared = module.dynamic.DynamicHazardAdapter(
        feature_dim=2, physics_dim=2, hidden=4,
        max_logit_residual=1.0, min_physics_families=1,
    )
    private = module.private_continual.EastPrivateDynamicsResidual(
        channel_count=8, hidden=4,
    )
    count = 600
    result = module.static_combined_logits(
        shared, private,
        frozen=np.zeros((count, 2), np.float32),
        base_logit=np.zeros(count, np.float32),
        physics=np.zeros((count, 2), np.float32),
        physics_available=np.ones((count, 2), bool),
        raw=np.zeros((count + 140, 8), np.float32),
        raw_available=np.ones((count + 140, 8), bool),
        minimum_positive=0.1, risk_offset=8,
    )
    assert result.shape == (count,)
    assert torch.isfinite(torch.from_numpy(result)).all()


def test_sealed_parent_contract_rejects_seed_source_or_fidelity_drift() -> None:
    module = load_module()
    payload = {
        "schema_version": "equal-label-budget-frontload-v2",
        "seed": 131,
        "arm": "support_only",
        "source_checkpoint_sha256": "source-sha",
        "thresholds": {"frozen_parent_threshold": 0.42},
        "encoder": {"weight": torch.zeros(1)},
        "head": {"weight": torch.zeros(1)},
        "shared": {"weight": torch.zeros(1)},
        "private": {"weight": torch.zeros(1)},
    }
    fidelity = {
        "passed": True,
        "expected": {
            "encoder_head": "base-sha",
            "shared_adapter": "shared-sha",
            "private_adapter": "private-sha",
            "threshold": 0.42,
        },
        "actual": {
            "encoder_head": "base-sha",
            "shared_adapter": "shared-sha",
            "private_adapter": "private-sha",
            "threshold": 0.42,
        },
    }

    contract = module.validate_sealed_parent_contract(
        payload, fidelity, seed=131, source_checkpoint_sha256="source-sha",
    )

    assert contract["threshold"] == pytest.approx(0.42)
    assert contract["expected_base_sha256"] == "base-sha"
    with pytest.raises(RuntimeError, match="seed"):
        module.validate_sealed_parent_contract(
            payload, fidelity, seed=133, source_checkpoint_sha256="source-sha",
        )
    with pytest.raises(RuntimeError, match="source checkpoint"):
        module.validate_sealed_parent_contract(
            payload, fidelity, seed=131, source_checkpoint_sha256="other",
        )
    with pytest.raises(RuntimeError, match="fidelity"):
        module.validate_sealed_parent_contract(
            payload, {**fidelity, "passed": False}, seed=131,
            source_checkpoint_sha256="source-sha",
        )
