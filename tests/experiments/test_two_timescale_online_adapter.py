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
    path = HERE / "two_timescale_online_adapter.py"
    spec = importlib.util.spec_from_file_location("two_timescale_online_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_batch(n: int = 8):
    return {
        "frozen": torch.randn(n, 12),
        "physics": torch.randn(n, 4),
        "available": torch.ones(n, 4, dtype=torch.bool),
        "base_logit": torch.randn(n),
        "next_physics": torch.randn(n, 4),
        "next_available": torch.ones(n, 4, dtype=torch.bool),
        "label": torch.tensor(([0.0, 1.0] * (n // 2))[:n]),
    }


def make_controller(module):
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(7)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    with torch.no_grad():
        adapter.residual_head[-1].weight.fill_(0.1)
    return module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2, inter_lr=1e-2,
        max_guard_harms=16,
    )


def test_intra_update_changes_fast_path_only_and_future_prediction():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch()
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    prediction_before = controller.predict(batch).clone()
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=batch, steps=2
    )
    prediction_after = controller.predict(batch)

    assert receipt["updated"] is True
    assert controller.persistent_hash() == persistent_before
    assert not torch.equal(prediction_before, prediction_after)


def test_representation_only_ttt_freezes_auxiliary_and_risk_heads():
    """SSL error must adapt the representation instead of being absorbed by heads."""
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(701)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    with torch.no_grad():
        adapter.residual_head[-1].weight.fill_(0.1)
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, max_guard_harms=16,
        intra_parameter_scope="representation_only",
        intra_physics_weight=0.0, risk_anchor_weight=0.0,
    )
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    assert controller.fast is not None
    trainable = {
        name for name, value in controller.fast.named_parameters()
        if value.requires_grad
    }
    assert trainable
    assert all(
        name.startswith("projector.2") or name.startswith("projector.3")
        for name in trainable
    )
    head_before = {
        name: value.detach().clone()
        for name, value in controller.fast.state_dict().items()
        if name.startswith("next_head") or name.startswith("residual_head")
    }
    representation_before = {
        name: value.detach().clone()
        for name, value in controller.fast.state_dict().items()
        if name.startswith("projector.2") or name.startswith("projector.3")
    }
    probe = {name: value for name, value in make_batch(8).items() if name != "label"}
    prediction_before = controller.predict(probe)

    source = {
        name: value for name, value in make_batch(8).items()
        if name != "label"
    }
    source["sealed_pre_adapt_logit"] = controller.predict(source).clone()
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108,
        batch=source,
        update_scale=1.0, steps=3,
    )

    assert controller.fast is not None
    assert controller.persistent_hash() == persistent_before
    assert all(
        torch.equal(value, controller.fast.state_dict()[name])
        for name, value in head_before.items()
    )
    assert any(
        not torch.equal(value, controller.fast.state_dict()[name])
        for name, value in representation_before.items()
    )
    assert not torch.equal(prediction_before, controller.predict(probe))
    assert receipt["risk_anchor_source"] == "sealed_pre_adapt_logit"


def test_representation_only_ttt_requires_sealed_pre_adapt_output():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    controller = module.TwoTimescaleAdapterController(
        DynamicHazardAdapter(feature_dim=12, physics_dim=4, hidden=8),
        guard_batch=make_batch(16), intra_parameter_scope="representation_only",
    )
    controller.begin_shot(10)
    source = {name: value for name, value in make_batch(8).items() if name != "label"}
    with pytest.raises(KeyError, match="sealed_pre_adapt_logit"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=source, steps=1,
        )


def test_representation_only_ttt_rejects_outcome_fields():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    controller = module.TwoTimescaleAdapterController(
        DynamicHazardAdapter(feature_dim=12, physics_dim=4, hidden=8),
        guard_batch=make_batch(16), intra_parameter_scope="representation_only",
    )
    controller.begin_shot(10)
    with pytest.raises(ValueError, match="outcome field"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=make_batch(8), steps=1,
        )


def test_temporal_risk_forecast_uses_later_observed_teacher_to_move_fast_risk():
    """A sealed later window must provide a direct, non-label risk gradient."""
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(71)
    source = make_batch(8)
    observed = make_batch(8)
    source["base_logit"].fill_(-2.0)
    observed["base_logit"].fill_(2.0)
    source["physics"].zero_()
    source["available"].fill_(True)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, risk_anchor_weight=0.0,
        intra_physics_weight=0.0, max_guard_harms=16,
        temporal_risk_weight=1.0, temporal_target_max_delta=0.5,
    )
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    before = controller.predict(source)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=source,
        observed_batch=observed, steps=5,
    )
    after = controller.predict(source)

    assert receipt["temporal_risk_loss"] > 0
    assert receipt["temporal_target_delta"] == pytest.approx(0.5)
    assert torch.mean(after) > torch.mean(before) + 1e-4
    assert controller.persistent_hash() == persistent_before


def test_temporal_risk_forecast_fails_closed_without_observed_window():
    module = load_module()
    controller = make_controller(module)
    controller.temporal_risk_weight = 1.0
    controller.begin_shot(10)

    with pytest.raises(ValueError, match="observed_batch"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=make_batch(), steps=1,
        )


def test_observed_physics_ttt_uses_sealed_current_window_and_changes_fast_only():
    """Current real physics may train the fast path only after its prediction."""
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(73)
    source = make_batch(8)
    observed = make_batch(8)
    source["base_logit"].fill_(-2.0)
    source["physics"].zero_()
    observed["base_logit"].fill_(-2.0)
    observed["physics"].fill_(0.9)
    source["available"].fill_(True)
    observed["available"].fill_(True)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, risk_anchor_weight=0.0,
        intra_physics_weight=0.0, observed_physics_weight=1.0,
        max_guard_harms=16, threshold_logit=0.0,
    )
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    before = controller.predict(observed)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=source,
        observed_batch=observed, steps=5,
    )
    after = controller.predict(observed)

    assert receipt["observed_physics_loss"] > 0
    assert torch.mean(after) > torch.mean(before) + 1e-4
    assert controller.persistent_hash() == persistent_before


def test_observed_physics_ttt_fails_closed_without_current_window():
    module = load_module()
    controller = make_controller(module)
    controller.observed_physics_weight = 1.0
    controller.begin_shot(10)

    with pytest.raises(ValueError, match="observed_batch"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=make_batch(), steps=1,
        )


def test_observed_physics_ttt_is_invariant_to_forbidden_outcome_value():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter
    import copy

    torch.manual_seed(77)
    parent = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    source = make_batch(8)
    observed_zero = make_batch(8)
    observed_zero["base_logit"].fill_(-2.0)
    observed_zero["physics"].fill_(0.9)
    observed_zero["available"].fill_(True)
    observed_zero["label"].zero_()
    observed_one = {name: value.clone() for name, value in observed_zero.items()}
    observed_one["label"].fill_(1.0)

    def updated_hash(observed):
        torch.manual_seed(79)
        controller = module.TwoTimescaleAdapterController(
            copy.deepcopy(parent), guard_batch=make_batch(16),
            intra_lr=1e-2, inter_lr=1e-2, max_guard_harms=16,
            intra_physics_weight=0.0, observed_physics_weight=1.0,
            threshold_logit=0.0,
        )
        controller.begin_shot(10)
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108, batch=source,
            observed_batch=observed, steps=2,
        )
        return controller.fast_hash()

    assert updated_hash(observed_zero) == updated_hash(observed_one)


def test_zero_scale_temporal_update_has_complete_receipt_and_no_gradient():
    module = load_module()
    controller = make_controller(module)
    controller.temporal_risk_weight = 1.0
    controller.begin_shot(10)
    before = controller.fast_hash()
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=make_batch(),
        observed_batch=make_batch(), steps=1, update_scale=0.0,
    )

    assert receipt["temporal_risk_loss"] == 0.0
    assert receipt["temporal_target_delta"] == 0.0
    assert controller.fast_hash() == before


def test_temporal_risk_forecast_cannot_read_observed_outcome_label():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(79)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    source = make_batch(8)
    observed_zero = make_batch(8)
    observed_zero["label"].zero_()
    observed_one = {name: value.clone() for name, value in observed_zero.items()}
    observed_one["label"].fill_(1.0)

    def updated_hash(observed):
        controller = module.TwoTimescaleAdapterController(
            __import__("copy").deepcopy(adapter), guard_batch=make_batch(16),
            intra_lr=1e-2, inter_lr=1e-2, max_guard_harms=16,
            temporal_risk_weight=1.0,
        )
        controller.begin_shot(10)
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108, batch=source,
            observed_batch=observed, steps=2,
        )
        return controller.fast_hash()

    torch.manual_seed(83)
    first = updated_hash(observed_zero)
    torch.manual_seed(83)
    second = updated_hash(observed_one)
    assert first == second


def test_temporal_risk_velocity_gain_extrapolates_only_the_sealed_delta():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(89)
    source = make_batch(4)
    observed = {name: value.clone() for name, value in source.items()}
    source["base_logit"].zero_()
    observed["base_logit"].fill_(0.1)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, max_guard_harms=16,
        temporal_risk_weight=1.0, temporal_target_max_delta=0.5,
        temporal_target_gain=4.0,
    )
    controller.begin_shot(10)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=source,
        observed_batch=observed, steps=1,
    )

    assert receipt["temporal_target_delta"] == pytest.approx(0.4, abs=1e-5)


def test_zero_value_intra_update_is_a_true_noop():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch()
    controller.begin_shot(10)
    before = controller.fast_hash()
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=batch, steps=2,
        update_scale=0.0,
    )

    assert receipt["updated"] is False
    assert receipt["update_scale"] == 0.0
    assert controller.fast_hash() == before


def test_intra_update_rejects_noncausal_target_endpoint():
    module = load_module()
    controller = make_controller(module)
    controller.begin_shot(10)
    with pytest.raises(ValueError, match="strictly later"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=100, batch=make_batch(), steps=1
        )


def test_next_shot_resets_fast_state_to_persistent_state():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch()
    controller.begin_shot(10)
    controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=batch, steps=2
    )
    controller.seal_shot(10)
    controller.discard_without_reveal(10)
    controller.begin_shot(20)

    assert controller.fast_hash() == controller.persistent_hash()


def test_inter_update_requires_seal_and_changes_persistent_for_next_shot():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch(16)
    before = controller.persistent_hash()
    with pytest.raises(RuntimeError, match="sealed shot"):
        controller.reveal_and_inter_update(
            10, selected=True, current_batch=batch, replay_batch=batch, steps=2
        )
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=batch, replay_batch=batch, steps=2
    )

    assert receipt["accepted"] is True
    assert controller.persistent_hash() != before
    controller.begin_shot(20)
    assert controller.fast_hash() == controller.persistent_hash()


def test_inter_update_remains_trainable_at_zero_increment_with_deployment_gate():
    """The hard signed deployment gate must not kill the supervised gradient."""
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(17)
    current = make_batch(16)
    current["label"].zero_()
    current["base_logit"].fill_(1.0)
    current["physics"].zero_()
    current["available"].fill_(True)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=current, intra_lr=1e-2, inter_lr=1e-2,
        max_guard_harms=16, physics_increment_gate=True,
        threshold_logit=0.0, negative_max_physics_support=0.10,
        positive_min_physics_support=0.65,
    )
    before = controller.persistent_hash()
    controller.begin_shot(10)
    prediction_before = controller.predict_persistent(current)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=current, replay_batch=current,
        steps=2, update_scale=1.0,
    )

    assert receipt["accepted"] is True
    assert receipt["updated"] is True
    assert controller.persistent_hash() != before
    controller.begin_shot(20)
    prediction_after = controller.predict_persistent(current)
    assert torch.mean(prediction_after) < torch.mean(prediction_before) - 1e-5


def test_zero_value_inter_update_finalizes_without_changing_persistent():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch(16)
    before = controller.persistent_hash()
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=batch, replay_batch=batch, steps=2,
        update_scale=0.0,
    )

    assert receipt["accepted"] is True
    assert receipt["updated"] is False
    assert receipt["update_scale"] == 0.0
    assert controller.persistent_hash() == before


def test_unselected_shot_cannot_run_supervised_inter_update():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch()
    controller.begin_shot(10)
    controller.seal_shot(10)
    with pytest.raises(PermissionError):
        controller.reveal_and_inter_update(
            10, selected=False, current_batch=batch, replay_batch=batch, steps=1
        )


def test_revealed_correct_shot_can_finalize_as_protection_without_update():
    module = load_module()
    controller = make_controller(module)
    before = controller.persistent_hash()
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.finalize_revealed_protection(10, group="TP")

    assert receipt["event"] == "revealed_protection_only"
    assert receipt["group"] == "TP"
    assert receipt["updated"] is False
    assert controller.persistent_hash() == before
    controller.begin_shot(20)


def test_physics_gate_blocks_only_unsupported_positive_fast_increment():
    module = load_module()
    fast = torch.tensor([0.7, 0.7, 0.3, 0.7])
    persistent = torch.tensor([0.5, 0.5, 0.5, 0.5])
    result = module.protect_positive_fast_increment(
        fast, persistent,
        physics_support=torch.tensor([0.1, 0.3, 0.0, 0.3]),
        physics_known=torch.tensor([True, True, False, False]),
        minimum_support=0.2,
    )

    torch.testing.assert_close(result, torch.tensor([0.5, 0.7, 0.3, 0.5]))


def test_causal_replay_buffer_samples_only_previous_revealed_shots():
    module = load_module()
    buffer = module.CausalReplayBuffer(capacity_shots=2)
    first = make_batch(8)
    second = make_batch(8)
    assert buffer.sample(count=4, seed=1) is None
    buffer.add(10, first)
    sampled = buffer.sample(count=4, seed=1)
    assert sampled is not None
    assert buffer.shots == [10]
    buffer.add(20, second)
    buffer.add(30, first)
    assert buffer.shots == [20, 30]
    with pytest.raises(RuntimeError, match="increasing"):
        buffer.add(25, second)


def test_causal_replay_can_sample_one_prior_opposite_outcome_bag():
    module = load_module()
    buffer = module.CausalReplayBuffer(capacity_shots=4)
    safe = make_batch(16)
    safe["label"].zero_()
    disruptive = make_batch(16)
    buffer.add(10, safe)
    buffer.add(20, disruptive)

    opposite_to_disruptive = buffer.sample_opposite_bag(
        current_is_disruptive=True, count=8, seed=3
    )
    opposite_to_safe = buffer.sample_opposite_bag(
        current_is_disruptive=False, count=8, seed=3
    )

    assert opposite_to_disruptive is not None
    assert opposite_to_safe is not None
    assert torch.count_nonzero(opposite_to_disruptive["label"]) == 0
    assert torch.count_nonzero(opposite_to_safe["label"]) > 0


def test_causal_replay_retrieves_nearest_prior_same_error_group():
    module = load_module()
    buffer = module.CausalReplayBuffer(capacity_shots=4)
    near = make_batch(16)
    far = make_batch(16)
    buffer.add(10, far, discrepancy_group="FP", phenotype=[0.0, 0.0])
    buffer.add(20, near, discrepancy_group="FP", phenotype=[0.8, 0.9])
    buffer.add(30, make_batch(16), discrepancy_group="FN", phenotype=[0.9, 0.9])

    sampled = buffer.sample_similar_error_bag(
        discrepancy_group="FP", phenotype=[0.75, 0.85], count=8, seed=7,
    )

    assert sampled is not None
    assert sampled["source_shot"] == 20
    assert sampled["distance"] < 0.1
    assert sampled["batch"]["label"].shape[0] == 8


def test_similar_error_replay_rejects_nearest_bag_outside_radius():
    module = load_module()
    buffer = module.CausalReplayBuffer(capacity_shots=4)
    batch = make_batch(8)
    batch["label"].zero_()
    buffer.add(10, batch, discrepancy_group="FP", phenotype=[0.0, 0.0])

    rejected = buffer.sample_similar_error_bag(
        discrepancy_group="FP", phenotype=[1.0, 1.0], count=2, seed=9,
        max_distance=0.35,
    )
    accepted = buffer.sample_similar_error_bag(
        discrepancy_group="FP", phenotype=[0.1, 0.1], count=2, seed=9,
        max_distance=0.35,
    )

    assert rejected is None
    assert accepted is not None
    assert accepted["distance"] <= 0.35


def test_replay_radius_uses_dimension_normalized_distance():
    module = load_module()
    buffer = module.CausalReplayBuffer(capacity_shots=4)
    batch = make_batch(8)
    buffer.add(
        10, batch, discrepancy_group="FP", phenotype=[0.0] * 11,
    )

    accepted = buffer.sample_similar_error_bag(
        discrepancy_group="FP", phenotype=[0.2] * 11,
        count=2, seed=9, max_distance=0.35,
    )

    assert accepted is not None
    assert accepted["distance"] == pytest.approx(0.2)


def test_shot_mil_loss_matches_frozen_alarm_boundary():
    module = load_module()
    threshold = -1.0

    safe_bad = module.shot_level_mil_hazard_loss(
        torch.tensor([-2.0, 0.5, -1.5]), torch.zeros(3),
        threshold_logit=threshold, top_k=2,
    )
    safe_good = module.shot_level_mil_hazard_loss(
        torch.tensor([-2.0, -1.8, -1.5]), torch.zeros(3),
        threshold_logit=threshold, top_k=2,
    )
    disruptive_bad = module.shot_level_mil_hazard_loss(
        torch.tensor([-2.0, -1.8, -1.5]), torch.tensor([0.0, 1.0, 0.0]),
        threshold_logit=threshold, top_k=2,
    )
    disruptive_good = module.shot_level_mil_hazard_loss(
        torch.tensor([-2.0, 0.5, -1.5]), torch.tensor([0.0, 1.0, 0.0]),
        threshold_logit=threshold, top_k=2,
    )

    assert safe_good < safe_bad
    assert disruptive_good < disruptive_bad


def test_shot_mil_does_not_accept_illegal_peak_as_disruption_evidence():
    module = load_module()
    labels = torch.tensor([0.0, 1.0, 0.0])
    legal_peak = module.shot_level_mil_hazard_loss(
        torch.tensor([-2.0, 0.5, -2.0]), labels,
        threshold_logit=-1.0, top_k=2,
    )
    illegal_peak = module.shot_level_mil_hazard_loss(
        torch.tensor([2.0, -2.0, -2.0]), labels,
        threshold_logit=-1.0, top_k=2,
    )

    assert legal_peak < illegal_peak


def test_operational_alarm_boundary_loss_matches_any_point_alarm_semantics():
    module = load_module()
    boundary = torch.zeros(3)
    positive = torch.tensor([True, True, True])
    missed = module.operational_alarm_boundary_loss(
        torch.tensor([-2.0, -1.0, -0.5]), boundary, positive,
    )
    rescued = module.operational_alarm_boundary_loss(
        torch.tensor([-2.0, 0.5, -0.5]), boundary, positive,
    )
    false_alarm = module.operational_alarm_boundary_loss(
        torch.tensor([-2.0, 0.5, -0.5]), boundary, torch.zeros(3, dtype=torch.bool),
    )
    safe = module.operational_alarm_boundary_loss(
        torch.tensor([-2.0, -1.0, -0.5]), boundary, torch.zeros(3, dtype=torch.bool),
    )

    assert rescued < missed
    assert safe < false_alarm


def test_inter_receipt_declares_shot_level_mil_objective():
    module = load_module()
    controller = make_controller(module)
    current = make_batch(16)
    replay = make_batch(16)
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=current, replay_batch=replay, steps=1
    )

    assert receipt["objective"] == "shot_level_mil_hazard"
    assert receipt["current_positive_windows"] == 8


def test_inter_update_records_causal_opposite_outcome_replay():
    module = load_module()
    controller = make_controller(module)
    current = make_batch(16)
    current["label"].zero_()
    opposite = make_batch(8)
    replay = make_batch(8)
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=current, replay_batch=replay,
        opposite_batch=opposite, steps=1,
    )

    assert receipt["opposite_outcome_replay"] is True


def test_online_increment_is_signed_relative_to_gated_parent_anchor():
    module = load_module()
    controller = make_controller(module)
    controller.minimum_positive_residual = 0.95
    batch = make_batch(8)
    controller.begin_shot(10)

    initial = controller.predict_persistent(batch)
    with torch.no_grad():
        _adjusted, parent_delta, _gate, _latent = controller.persistent(
            batch["frozen"], batch["physics"], batch["available"], batch["base_logit"]
        )
        gated_parent = torch.where(
            (parent_delta > 0) & (parent_delta < 0.95),
            torch.zeros_like(parent_delta), parent_delta,
        )
    torch.testing.assert_close(
        initial, batch["base_logit"] + gated_parent, atol=1e-6, rtol=0
    )

    with torch.no_grad():
        controller.persistent.residual_head[-1].bias.add_(0.10)
    raised = controller.predict_persistent(batch)
    assert torch.all(raised > initial)

    with torch.no_grad():
        controller.persistent.residual_head[-1].bias.sub_(0.20)
    lowered = controller.predict_persistent(batch)
    assert torch.all(lowered < initial)


def test_replay_distillation_anchors_predictions_without_class_prior():
    module = load_module()
    anchor = torch.tensor([-2.0, -0.5, 1.0])
    same = module.replay_distillation_loss(anchor.clone(), anchor)
    shifted = module.replay_distillation_loss(anchor + 0.5, anchor)

    assert same.item() == pytest.approx(0.0)
    assert shifted > same


def test_physics_constraint_is_fail_closed_when_families_are_unknown():
    module = load_module()
    batch = make_batch(4)
    batch["available"].zero_()
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0], requires_grad=True)
    loss = module.physics_consistency_loss(
        logits, batch, threshold_logit=-1.0,
        activity_threshold=0.5, minimum_families=2,
    )

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_physics_constraint_respects_legal_positive_and_low_support_negative():
    module = load_module()
    batch = make_batch(2)
    batch["physics"] = torch.tensor([[0.9, 0.8, 0.7, 0.9], [0.0, 0.1, 0.0, 0.1]])
    batch["available"].fill_(True)
    batch["label"] = torch.tensor([1.0, 0.0])

    aligned = module.physics_consistency_loss(
        torch.tensor([1.0, -2.0]), batch, threshold_logit=-1.0,
        activity_threshold=0.5, minimum_families=2,
    )
    contradicted = module.physics_consistency_loss(
        torch.tensor([-2.0, 1.0]), batch, threshold_logit=-1.0,
        activity_threshold=0.5, minimum_families=2,
    )

    assert aligned < contradicted


def test_signed_increment_gate_localizes_fp_and_fn_corrections():
    module = load_module()
    increment = torch.tensor([-0.4, -0.4, 0.4, 0.4, 0.4])
    base_logit = torch.tensor([1.0, 1.0, -2.0, -2.0, -2.0])
    support = torch.tensor([0.05, 0.8, 0.8, 0.1, 0.8])
    known = torch.tensor([True, True, True, True, False])
    gated = module.gate_signed_online_increment(
        increment, base_logit=base_logit, threshold_logit=-1.0,
        physics_support=support, physics_known=known,
        negative_max_support=0.10, positive_min_support=0.65,
    )

    torch.testing.assert_close(gated, torch.tensor([-0.4, 0.0, 0.4, 0.0, 0.0]))


def test_unlabeled_physics_pseudo_loss_prefers_contradiction_correction():
    module = load_module()
    batch = make_batch(2)
    batch["base_logit"] = torch.tensor([1.0, -2.0])
    batch["physics"] = torch.tensor([[0.0, 0.1, 0.0, 0.1], [0.9, 0.8, 0.7, 0.9]])
    batch["available"].fill_(True)
    aligned = module.unlabeled_physics_pseudo_loss(
        torch.tensor([-2.0, 1.0]), batch, threshold_logit=-1.0,
        activity_threshold=0.5, minimum_families=2,
        negative_max_support=0.10, positive_min_support=0.65,
    )
    contradicted = module.unlabeled_physics_pseudo_loss(
        torch.tensor([1.0, -2.0]), batch, threshold_logit=-1.0,
        activity_threshold=0.5, minimum_families=2,
        negative_max_support=0.10, positive_min_support=0.65,
    )

    assert aligned < contradicted


def test_intra_physics_update_changes_fast_only_before_current_prediction():
    module = load_module()
    controller = make_controller(module)
    batch = make_batch(4)
    batch["base_logit"].fill_(1.0)
    batch["physics"].fill_(0.0)
    batch["available"].fill_(True)
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    prediction_before = controller.predict(batch)
    receipt = controller.intra_physics_update(batch=batch, steps=2)
    prediction_after = controller.predict(batch)

    assert receipt["event"] == "intra_physics_pseudo_update"
    assert controller.persistent_hash() == persistent_before
    assert not torch.equal(prediction_before, prediction_after)


def test_causal_intra_update_uses_physics_pseudo_gradient_when_registered():
    module = load_module()
    from dynamic_hazard_adapter import DynamicHazardAdapter

    torch.manual_seed(23)
    batch = make_batch(8)
    batch["base_logit"].fill_(1.0)
    batch["physics"].zero_()
    batch["available"].fill_(True)
    adapter = DynamicHazardAdapter(
        feature_dim=12, physics_dim=4, hidden=8,
        max_logit_residual=1.0, min_physics_families=2,
    )
    controller = module.TwoTimescaleAdapterController(
        adapter, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, max_guard_harms=16,
        intra_physics_weight=1.0, physics_increment_gate=True,
        threshold_logit=0.0, negative_max_physics_support=0.10,
        positive_min_physics_support=0.65,
    )
    controller.begin_shot(10)
    before = controller.predict(batch)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=batch, steps=2,
    )
    after = controller.predict(batch)

    assert receipt["pseudo_loss"] > 0
    assert torch.mean(after) < torch.mean(before) - 1e-5
