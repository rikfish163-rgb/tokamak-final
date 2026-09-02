from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_trace_features_are_signal_only_and_finite() -> None:
    policy = importlib.import_module("jtext_query_policy")
    rng = np.random.default_rng(131)
    values = rng.normal(size=(40, 17)).astype(np.float32)
    available = np.ones_like(values, dtype=bool)
    risk = np.linspace(0.1, 0.9, 40)
    logits = np.log(risk / (1.0 - risk))
    features = policy.trace_features(
        risk, logits, values, available, threshold=0.5,
    )
    assert features.ndim == 1
    assert len(features) == 80
    assert np.isfinite(features).all()


def test_validation_error_prior_is_deterministic_and_serializable() -> None:
    policy = importlib.import_module("jtext_query_policy")
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 5))
    labels = np.array([0] * 15 + [1] * 5)
    first = policy.ValidationErrorPrior.fit(features, labels, l2=1.0)
    second = policy.ValidationErrorPrior.fit(features, labels, l2=1.0)
    np.testing.assert_array_equal(first.score(features), second.score(features))
    assert first.sha256() == second.sha256()
    assert first.state_dict()["schema_version"] == policy.SCHEMA_VERSION


def test_validation_error_prior_requires_both_classes() -> None:
    policy = importlib.import_module("jtext_query_policy")
    with pytest.raises(ValueError, match="both outcome classes"):
        policy.ValidationErrorPrior.fit(np.zeros((3, 2)), [0, 0, 0])


def test_stage_balanced_inter_arm_samples_revealed_positive_and_hard_negative() -> None:
    runner = importlib.import_module("v3c_compat.run_jtext_online_active")
    logits = np.asarray([-4.0, -2.0, 0.2, 1.0, 3.0, 4.0])
    labels = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    indices = runner._advanced_inter_indices(logits, labels, count=4)

    # The post-reveal batch must contain an actual positive stage and the
    # highest-risk negative, rather than silently selecting only the earliest
    # (usually all-negative) endpoints.
    assert any(labels[index] > 0.5 for index in indices)
    assert 2 in indices
    assert len(indices) == 4


def test_stage_balanced_inter_arm_is_inter_only() -> None:
    runner = importlib.import_module("v3c_compat.run_jtext_online_active")

    assert "inter_active_stage_balanced" in runner.ARMS
    policy = runner._arm_policy("inter_active_stage_balanced")
    assert policy["intra"] is False
    assert policy["inter"] is True
    assert policy["active"] is True
    assert "inter_active_stage_balanced" in runner.STAGE_BALANCED_INTER_ARMS
