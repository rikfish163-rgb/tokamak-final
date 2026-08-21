import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments/transfer_methods_2026/dual_threshold_alarm.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dual_threshold_alarm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_threshold_route_is_never_removed():
    module = load_module()
    margin = np.array([-1.0, -0.2, 0.1, -0.5])
    support = np.zeros(4)
    result = module.dual_threshold_alarm(
        margin, support, early_threshold=100.0
    )
    assert result["alarmed"] is True
    assert result["alarm_index"] == 2
    assert result["route"] == "high"


def test_early_route_can_confirm_a_weak_rising_risk():
    module = load_module()
    margin = np.r_[np.full(100, -2.0), np.linspace(-2.0, -0.4, 20)]
    support = np.r_[np.zeros(100), np.ones(20)]
    result = module.dual_threshold_alarm(
        margin,
        support,
        physics_weight=8.0,
        trend_weight=12.0,
        early_threshold=7.9,
    )
    assert result["alarmed"] is True
    assert result["route"] == "early"
    assert result["alarm_index"] >= 100


def test_decisions_are_prefix_causal():
    module = load_module()
    margin = np.r_[np.linspace(-2.0, -0.5, 120), np.full(20, 10.0)]
    support = np.r_[np.ones(120), np.zeros(20)]
    prefix = module.dual_threshold_trace(margin[:120], support[:120])
    full = module.dual_threshold_trace(margin, support)
    np.testing.assert_array_equal(prefix["alarm_mask"], full["alarm_mask"][:120])
    np.testing.assert_allclose(prefix["early_score"], full["early_score"][:120])


def test_inputs_must_align_and_be_finite_safe():
    module = load_module()
    with np.testing.assert_raises(ValueError):
        module.dual_threshold_alarm(np.zeros(3), np.zeros(2))
    result = module.dual_threshold_alarm(
        np.array([np.nan, -1.0]), np.array([1.0, np.nan])
    )
    assert result["alarmed"] is False


def test_persistent_precursor_requires_k_of_window_and_is_prefix_causal():
    module = load_module()
    risk = np.array([0.20, 0.45, 0.10, 0.50, 0.55, 0.10, 0.90])
    support = np.array([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.00])
    kwargs = {
        "decision_threshold": 0.95,
        "physics_weight": 8.0,
        "trend_weight": 0.0,
        "score_threshold": -10.0,
        "minimum_physics_support": 0.20,
        "minimum_risk_probability": 0.40,
        "required_points": 3,
        "window": 4,
    }

    prefix = module.persistent_precursor_trace(risk[:6], support[:6], **kwargs)
    full = module.persistent_precursor_trace(risk, support, **kwargs)

    assert prefix["alarm_index"] == 4
    assert prefix["route"] == "persistent_precursor"
    np.testing.assert_array_equal(
        prefix["alarm_mask"], full["alarm_mask"][:6]
    )
    np.testing.assert_array_equal(
        prefix["rolling_count"], full["rolling_count"][:6]
    )


def test_precursor_overlay_can_only_add_or_advance_the_deployment_alarm():
    module = load_module()
    base = {
        "alarmed": True, "alarm_index": 8, "route": "early",
        "early_score": 8.1, "filter_reason": "protected_sequence_cleared",
    }
    precursor = {
        "alarmed": True, "alarm_index": 5,
        "route": "persistent_precursor", "early_score": 4.7,
    }

    result = module.apply_precursor_overlay(base, precursor)

    assert result["alarmed"] is True
    assert result["alarm_index"] == 5
    assert result["route"] == "persistent_precursor"
    assert result["precursor_overlay_applied"] is True
    assert result["base_alarm_reference"] == base

    unchanged = module.apply_precursor_overlay(base, {
        "alarmed": True, "alarm_index": 10,
        "route": "persistent_precursor", "early_score": 5.0,
    })
    assert unchanged["alarm_index"] == 8
    assert unchanged["precursor_overlay_applied"] is False
