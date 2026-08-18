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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_locked_best_v8_accepts_only_registered_configuration() -> None:
    module = load_module()
    module.validate_locked_best_v8_arguments(locked_v8_args())

    with pytest.raises(ValueError, match="locked best-v8"):
        module.validate_locked_best_v8_arguments(
            locked_v8_args(inter_lr=6e-4)
        )


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
