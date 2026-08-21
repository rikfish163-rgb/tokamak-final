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
    path = HERE / "east_private_continual.py"
    spec = importlib.util.spec_from_file_location("east_private_continual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_batch(n: int = 8, *, channels: int = 8):
    return {
        "raw": torch.randn(n, 128, channels),
        "available": torch.ones(n, 128, channels, dtype=torch.bool),
        "base_logit": torch.randn(n),
        "next_features": torch.randn(n, channels * 6),
        "target": torch.tensor(([0.0, 1.0] * (n // 2))[:n]),
        "sealed_logit": torch.randn(n),
        "label": torch.tensor(([0.0, 1.0] * (n // 2))[:n]),
    }


def make_controller(module):
    torch.manual_seed(11)
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.05)
    return module.PrivateTwoTimescaleController(
        model, guard_batch=make_batch(16), intra_lr=1e-2, inter_lr=1e-2,
        threshold_logit=0.0, max_guard_harms=16,
    )


def test_private_dynamics_outputs_bounded_risk_and_next_features() -> None:
    module = load_module()
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    batch = make_batch(4)
    adjusted, residual, latent, features = model(
        batch["raw"], batch["available"], batch["base_logit"]
    )
    predicted_next = model.predict_next(latent)

    assert adjusted.shape == residual.shape == (4,)
    assert latent.shape == (4, 12)
    assert features.shape == predicted_next.shape == (4, 48)
    assert torch.all(residual.abs() <= model.maximum_logit_residual + 1e-6)


def test_private_dynamics_is_missing_safe() -> None:
    module = load_module()
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    batch = make_batch(4)
    batch["available"].zero_()
    adjusted, residual, latent, features = model(
        batch["raw"], batch["available"], batch["base_logit"]
    )
    assert torch.isfinite(adjusted).all()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(latent).all()
    assert torch.count_nonzero(features) == 0


def test_private_intra_update_changes_fast_only() -> None:
    module = load_module()
    controller = make_controller(module)
    batch = make_batch()
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    before = controller.predict(batch)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=batch,
        update_scale=1.0, steps=2,
    )
    after = controller.predict(batch)

    assert receipt["updated"] is True
    assert controller.persistent_hash() == persistent_before
    assert not torch.equal(before, after)


def test_private_representation_only_ttt_freezes_auxiliary_and_risk_heads() -> None:
    module = load_module()
    torch.manual_seed(703)
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.05)
    controller = module.PrivateTwoTimescaleController(
        model, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, teacher_weight=0.0, max_guard_harms=16,
        intra_parameter_scope="representation_only",
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
        name.startswith("trunk.2") or name.startswith("trunk.3")
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
        if name.startswith("trunk.2") or name.startswith("trunk.3")
    }
    probe = {
        name: value for name, value in make_batch(8).items()
        if name not in {"label", "target", "sealed_logit"}
    }
    prediction_before = controller.predict(probe)

    source = {
        name: value for name, value in make_batch(8).items()
        if name not in {"label", "target", "sealed_logit"}
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


def test_private_representation_only_ttt_requires_sealed_pre_adapt_output() -> None:
    module = load_module()
    controller = module.PrivateTwoTimescaleController(
        module.EastPrivateDynamicsResidual(channel_count=8, hidden=12),
        guard_batch=make_batch(16), intra_parameter_scope="representation_only",
    )
    controller.begin_shot(10)
    source = {
        name: value for name, value in make_batch(8).items()
        if name not in {"label", "target", "sealed_logit"}
    }
    with pytest.raises(KeyError, match="sealed_pre_adapt_logit"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=source, steps=1,
        )


def test_private_representation_only_ttt_rejects_outcome_fields() -> None:
    module = load_module()
    controller = module.PrivateTwoTimescaleController(
        module.EastPrivateDynamicsResidual(channel_count=8, hidden=12),
        guard_batch=make_batch(16), intra_parameter_scope="representation_only",
    )
    controller.begin_shot(10)
    with pytest.raises(ValueError, match="outcome field"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=make_batch(8), steps=1,
        )


def test_private_temporal_risk_forecast_directly_updates_fast_residual() -> None:
    module = load_module()
    torch.manual_seed(73)
    source = make_batch(8)
    observed = make_batch(8)
    source["base_logit"].fill_(-2.0)
    observed["base_logit"].fill_(2.0)
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    controller = module.PrivateTwoTimescaleController(
        model, guard_batch=make_batch(16), intra_lr=1e-2,
        inter_lr=1e-2, teacher_weight=0.0, next_weight=0.0,
        max_guard_harms=16, temporal_risk_weight=1.0,
        temporal_target_max_delta=0.5,
    )
    persistent_before = controller.persistent_hash()
    controller.begin_shot(10)
    before = controller.predict(source)
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=source,
        observed_batch=observed, update_scale=1.0, steps=5,
    )
    after = controller.predict(source)

    assert receipt["temporal_risk_loss"] > 0
    assert receipt["temporal_target_delta"] == pytest.approx(0.5)
    assert torch.mean(after) > torch.mean(before) + 1e-4
    assert controller.persistent_hash() == persistent_before


def test_private_temporal_risk_forecast_requires_observed_window() -> None:
    module = load_module()
    controller = make_controller(module)
    controller.temporal_risk_weight = 1.0
    controller.begin_shot(10)

    with pytest.raises(ValueError, match="observed_batch"):
        controller.intra_update(
            source_endpoint=100, observed_endpoint=108,
            batch=make_batch(), steps=1,
        )


def test_private_temporal_weight_must_be_finite() -> None:
    module = load_module()
    model = module.EastPrivateDynamicsResidual(channel_count=8, hidden=12)
    with pytest.raises(ValueError, match="invalid private"):
        module.PrivateTwoTimescaleController(
            model, guard_batch=make_batch(16),
            temporal_risk_weight=float("nan"),
        )


def test_private_zero_scale_temporal_update_has_complete_receipt() -> None:
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


def test_private_zero_value_intra_update_is_noop() -> None:
    module = load_module()
    controller = make_controller(module)
    controller.begin_shot(10)
    before = controller.fast_hash()
    receipt = controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=make_batch(),
        update_scale=0.0, steps=2,
    )
    assert receipt["updated"] is False
    assert controller.fast_hash() == before


def test_private_fast_state_resets_at_next_shot() -> None:
    module = load_module()
    controller = make_controller(module)
    controller.begin_shot(10)
    controller.intra_update(
        source_endpoint=100, observed_endpoint=108, batch=make_batch(),
        update_scale=1.0, steps=2,
    )
    controller.seal_shot(10)
    controller.discard_without_reveal(10)
    controller.begin_shot(20)
    assert controller.fast_hash() == controller.persistent_hash()


def test_private_inter_update_requires_selected_sealed_shot_and_changes_future() -> None:
    module = load_module()
    controller = make_controller(module)
    batch = make_batch(16)
    with pytest.raises(RuntimeError, match="sealed"):
        controller.reveal_and_inter_update(
            10, selected=True, current_batch=batch, replay_batch=batch,
        )
    controller.begin_shot(10)
    controller.seal_shot(10)
    before = controller.persistent_hash()
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=batch, replay_batch=batch,
        update_scale=1.0, steps=2,
    )

    assert receipt["accepted"] is True
    assert receipt["updated"] is True
    assert controller.persistent_hash() != before
    controller.begin_shot(20)
    assert controller.fast_hash() == controller.persistent_hash()


def test_private_fixed_compute_update_emits_exact_shape_signature() -> None:
    module = load_module()
    controller = make_controller(module)
    current = make_batch(16)
    current["alarm_boundary"] = torch.zeros(16)
    replay = make_batch(16)
    controller.begin_shot(10)
    controller.seal_shot(10)
    receipt = controller.reveal_and_inter_update(
        10, selected=True, current_batch=current, replay_batch=replay,
        steps=2, update_scale=1.0, fixed_compute_objective=True,
    )

    assert receipt["objective"] == "fixed_shape_pointwise_alarm_bce"
    signature = receipt["fixed_compute_signature"]
    assert {
        name: signature[name] for name in (
            "current_windows", "replay_windows", "guard_windows", "steps",
            "objective",
        )
    } == {
        "current_windows": 16, "replay_windows": 16,
        "guard_windows": 16, "steps": 2,
        "objective": "fixed_shape_pointwise_alarm_bce",
    }
    assert signature["optimizer_steps_attempted"] == 2
    assert signature["trainable_scope"]["parameter_count"] > 0


def test_private_unqueried_shot_cannot_use_supervision() -> None:
    module = load_module()
    controller = make_controller(module)
    controller.begin_shot(10)
    controller.seal_shot(10)
    with pytest.raises(PermissionError):
        controller.reveal_and_inter_update(
            10, selected=False, current_batch=make_batch(),
            replay_batch=make_batch(),
        )
