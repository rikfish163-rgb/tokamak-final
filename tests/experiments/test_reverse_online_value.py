from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments/transfer_methods_2026/reverse_online_value.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reverse_online_value", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_robust_feature_transform_does_not_use_shot_id():
    module = load_module()
    rows = [
        {"shot": 10, "fp": 0.0, "fn": 1.0},
        {"shot": 20, "fp": 1.0, "fn": 0.0},
        {"shot": 30, "fp": 2.0, "fn": 0.5},
    ]
    transformed, contract = module.robust_transform(rows, ("fp", "fn"))

    assert transformed.shape == (3, 2)
    assert set(contract) == {"fp", "fn"}
    assert "shot" not in contract


def test_value_policy_uses_exact_causal_block_budget():
    module = load_module()
    scores = np.asarray([0.0] * 16 + [1.0, 0.0, 0.0, 0.0] + [0.0] * 20)
    selected, forced = module.causal_block_select(
        scores, cutoff=0.9, block_size=20, quota=4
    )

    assert selected[:20].sum() == 4
    assert selected[20:].sum() == 4
    assert selected[16]
    assert forced.sum() >= 7


def test_fit_value_policy_prefers_error_informative_feature_and_returns_ranges():
    module = load_module()
    rows = []
    for index in range(120):
        error = index % 10 in (0, 1)
        rows.append({
            "stream_index": index,
            "shot": 1000 + index,
            "is_error": error,
            "is_fp": index % 10 == 0,
            "is_fn": index % 10 == 1,
            "informative": 1.0 if error else 0.0,
            "noise": float((index * 17) % 23) / 23.0,
        })
    result = module.fit_value_policy(
        rows,
        feature_names=("informative", "noise"),
        candidate_weights=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        cutoff_quantiles=(0.7, 0.8),
        fold_size=40,
    )

    assert result["reference_weights"]["informative"] > result["reference_weights"]["noise"]
    assert result["query_count"] == 24
    assert set(result["weight_ranges"]) == {"informative", "noise"}
    assert result["worst_fold_error_enrichment"] >= 1.0


def test_reverse_config_contains_no_per_shot_decisions():
    module = load_module()
    config = module.build_frozen_config(
        fit={
            "reference_weights": {"fp": 1.0},
            "weight_ranges": {"fp": [0.5, 1.0]},
            "cutoff": 0.8,
            "cutoff_quantile": 0.8,
        },
        feature_contract={"fp": {"center": 0.0, "scale": 1.0}},
        physics_ranges={"activity_threshold": [0.35, 0.65]},
    )

    encoded = str(config)
    assert "selected_shots" not in encoded
    assert "shot_id" not in encoded
    assert config["claim_boundary"] == "RETROSPECTIVE_DESIGNED_PREQUENTIAL_360"


def test_score_value_row_applies_frozen_transform_and_weights():
    module = load_module()
    config = {
        "feature_names": ["fp", "fn"],
        "reference_weights": {"fp": 0.25, "fn": 0.75},
        "feature_contract": {
            "fp": {"center": 0.0, "scale": 2.0},
            "fn": {"center": 1.0, "scale": 1.0},
        },
    }
    score, transformed = module.score_value_row({"fp": 2.0, "fn": 3.0}, config)

    assert transformed == {"fp": 1.0, "fn": 2.0}
    assert score == 1.75


def test_runtime_value_features_use_only_signal_side_information():
    module = load_module()
    features = module.runtime_value_features(
        query_components={
            "fp_contradiction": 0.2, "fn_contradiction": 0.3,
            "uncertainty": 0.4,
        },
        auxiliary_row={
            "aux_novelty": 1.2, "static_error": 0.5, "online_error": 0.25,
        },
        risk=np.asarray([0.1, 0.9, 0.1, 0.2, 0.1]),
    )

    assert set(features) == set(module.FEATURES)
    assert features["teacher_disagreement"] == 0.5
    assert features["risk_spikiness"] > 0.0
