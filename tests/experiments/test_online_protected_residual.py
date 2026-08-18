from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "online_protected_residual", HERE / "online_protected_residual.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_adapter():
    import dynamic_hazard_adapter as dynamic

    torch.manual_seed(7)
    adapter = dynamic.DynamicHazardAdapter(
        feature_dim=4, physics_dim=2, hidden=4, max_logit_residual=1.0
    )
    with torch.no_grad():
        adapter.residual_head[-1].weight.fill_(0.05)
    return adapter


def batch(labels=(0.0, 1.0)):
    count = len(labels)
    return {
        "frozen": torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 10,
        "physics": torch.ones(count, 2),
        "available": torch.ones(count, 2, dtype=torch.bool),
        "base_logit": torch.tensor([-1.0, 1.0])[:count],
        "label": torch.tensor(labels),
    }


def test_predict_must_be_sealed_before_outcome_and_current_prediction_is_immutable():
    module = load_module()
    learner = module.ProtectedResidualUpdater(
        make_adapter(), threshold_logit=0.0, guard_batch=batch(), lr=0.1
    )
    sealed = learner.predict_and_seal(101, batch())
    sealed_copy = sealed["adjusted_logit"].clone()

    with pytest.raises(RuntimeError, match="pending"):
        learner.predict_and_seal(102, batch())
    with pytest.raises(PermissionError, match="selected"):
        learner.reveal_and_update(
            101, selected=False, current_batch=batch(), replay_batch=batch(), query_score=1.0
        )

    learner.discard_without_reveal(101)
    assert torch.equal(sealed["adjusted_logit"], sealed_copy)


def test_only_residual_head_can_change_after_selected_reveal():
    module = load_module()
    adapter = make_adapter()
    immutable_before = module.module_subset_sha256(adapter, exclude_prefix="residual_head")
    residual_before = module.module_subset_sha256(adapter, include_prefix="residual_head")
    learner = module.ProtectedResidualUpdater(
        adapter, threshold_logit=0.0, guard_batch=batch(), lr=0.05,
        max_guard_harms=2,
    )
    learner.predict_and_seal(101, batch())
    receipt = learner.reveal_and_update(
        101, selected=True, current_batch=batch((1.0, 1.0)),
        replay_batch=batch(), query_score=0.8,
    )

    assert receipt["accepted"] is True
    assert module.module_subset_sha256(adapter, exclude_prefix="residual_head") == immutable_before
    assert module.module_subset_sha256(adapter, include_prefix="residual_head") != residual_before


def test_guard_harm_rolls_back_parameters_and_optimizer_state():
    module = load_module()
    adapter = make_adapter()
    learner = module.ProtectedResidualUpdater(
        adapter, threshold_logit=0.0, guard_batch=batch(), lr=5.0,
        max_guard_harms=0,
    )
    before = module.module_subset_sha256(adapter, include_prefix="residual_head")
    learner.predict_and_seal(101, batch())
    receipt = learner.reveal_and_update(
        101, selected=True, current_batch=batch((1.0, 0.0)),
        replay_batch=batch((1.0, 0.0)), query_score=1.0, steps=4,
    )

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "guard_harm"
    assert module.module_subset_sha256(adapter, include_prefix="residual_head") == before


def test_query_score_controls_only_a_bounded_update_weight():
    module = load_module()
    assert module.bounded_update_weight(-100.0) == pytest.approx(0.5)
    assert module.bounded_update_weight(0.5) == pytest.approx(1.0)
    assert module.bounded_update_weight(100.0) == pytest.approx(1.5)


def test_unselected_shot_and_intra_auxiliary_updates_leave_predictor_bitwise_unchanged():
    module = load_module()
    import online_auxiliary as auxiliary

    adapter = make_adapter()
    before = module.module_subset_sha256(adapter)
    sidecar = auxiliary.OnlineAuxiliaryLearner(
        feature_dim=4, physics_dim=2, hidden=4, lr=1e-2
    )
    sidecar.predict_and_seal(0, torch.zeros(4), torch.zeros(2), torch.ones(2, dtype=torch.bool))
    sidecar.observe_and_update(1, torch.ones(4), torch.ones(2), torch.ones(2, dtype=torch.bool))

    learner = module.ProtectedResidualUpdater(
        adapter, threshold_logit=0.0, guard_batch=batch(), lr=0.1
    )
    learner.predict_and_seal(101, batch())
    learner.discard_without_reveal(101)
    assert module.module_subset_sha256(adapter) == before


def test_shot_ids_must_be_strictly_increasing():
    module = load_module()
    learner = module.ProtectedResidualUpdater(
        make_adapter(), threshold_logit=0.0, guard_batch=batch(), lr=0.1
    )
    learner.predict_and_seal(101, batch())
    learner.discard_without_reveal(101)
    with pytest.raises(RuntimeError, match="increasing"):
        learner.predict_and_seal(100, batch())


def test_support_positive_gate_is_part_of_the_online_predictor():
    module = load_module()
    adapter = make_adapter()
    learner = module.ProtectedResidualUpdater(
        adapter, threshold_logit=0.0, guard_batch=batch(), lr=0.1,
        minimum_positive_residual=2.0,
    )
    sealed = learner.predict_and_seal(101, batch())
    assert torch.equal(sealed["adjusted_logit"], batch()["base_logit"])


def test_a1_query_contract_uses_predecision_seals_and_selected_query_receipts():
    sys.path.insert(0, str(HERE))
    import run_online_protected_residual as runner

    seals = [
        {"shot": 10, "query_score": 0.1, "queried": False},
        {"shot": 20, "query_score": 0.9, "queried": False},
    ]
    queries = [{"shot": 20, "query_score": 0.9, "revealed_is_disrupt": True}]
    merged = runner.merge_a1_query_contract(seals, queries, expected_queries=1)
    assert merged[10]["queried"] is False
    assert merged[20]["queried"] is True
    assert "revealed_is_disrupt" not in merged[20]


def test_revealed_safe_windows_cover_time_instead_of_one_peak_cluster():
    import run_online_protected_residual as runner

    logits = torch.zeros(160).numpy()
    logits[70:90] = 10.0
    chosen = runner.select_revealed_windows(logits, torch.zeros(160, dtype=torch.bool).numpy())
    occupied_bins = set((chosen // 10).tolist())
    assert len(chosen) == 16
    assert len(occupied_bins) == 16
