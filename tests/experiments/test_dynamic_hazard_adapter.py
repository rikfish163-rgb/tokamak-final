from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
MODULE = HERE / "dynamic_hazard_adapter.py"
sys.path.insert(0, str(HERE))


def load_module():
    spec = importlib.util.spec_from_file_location("dynamic_hazard_adapter", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_initialized_residual_exactly_preserves_base_logit() -> None:
    module = load_module()
    adapter = module.DynamicHazardAdapter(feature_dim=4, physics_dim=2, hidden=8)
    base = torch.tensor([-2.0, 0.0, 2.0])
    adjusted, delta, gate, _ = adapter(
        torch.zeros(3, 4), torch.zeros(3, 2),
        torch.ones(3, 2, dtype=torch.bool), base,
    )

    assert torch.equal(adjusted, base)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(gate, torch.ones_like(gate))


def test_missing_physics_families_force_exact_base_fallback() -> None:
    module = load_module()
    adapter = module.DynamicHazardAdapter(feature_dim=4, physics_dim=3, hidden=8)
    with torch.no_grad():
        adapter.residual_head[-1].bias.fill_(5.0)
    base = torch.tensor([0.3, -0.7])
    available = torch.tensor([[True, False, False], [False, False, False]])
    adjusted, _, gate, _ = adapter(
        torch.zeros(2, 4), torch.zeros(2, 3), available, base,
    )

    assert torch.equal(adjusted, base)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_residual_is_bounded_in_logit_space() -> None:
    module = load_module()
    adapter = module.DynamicHazardAdapter(
        feature_dim=4, physics_dim=2, hidden=8, max_logit_residual=0.75
    )
    with torch.no_grad():
        adapter.residual_head[-1].bias.fill_(100.0)
    base = torch.zeros(5)
    adjusted, delta, _, _ = adapter(
        torch.zeros(5, 4), torch.zeros(5, 2),
        torch.ones(5, 2, dtype=torch.bool), base,
    )

    assert torch.all(delta <= 0.75)
    assert torch.all(delta >= -0.75)
    torch.testing.assert_close(adjusted, delta)


def test_horizon_labels_use_hdf5_downtime_and_125ms_boundary() -> None:
    module = load_module()
    times = np.asarray([0.800, 0.874, 0.875, 0.900, 1.001])
    labels = module.alarm_window_labels(
        times, is_disrupt=True, down_time_s=1.0, horizon_ms=125.0
    )
    safe = module.alarm_window_labels(
        times, is_disrupt=False, down_time_s=None, horizon_ms=125.0
    )

    np.testing.assert_array_equal(labels, np.asarray([False, False, True, True, False]))
    np.testing.assert_array_equal(safe, np.zeros(len(times), dtype=bool))


def test_adapter_training_step_does_not_change_frozen_base_modules() -> None:
    module = load_module()
    import grid_search as grid
    import online_auxiliary as auxiliary

    torch.manual_seed(11)
    enc = grid.Enc(3).cpu().eval()
    head = grid.Head().cpu().eval()
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    base_hash = auxiliary.module_sha256(enc, head)
    adapter = module.DynamicHazardAdapter(feature_dim=4, physics_dim=2, hidden=8)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    base_logit = torch.tensor([0.2, -0.4])
    adjusted, delta, _, latent = adapter(
        torch.randn(2, 4), torch.randn(2, 2),
        torch.ones(2, 2, dtype=torch.bool), base_logit,
    )
    loss = F.binary_cross_entropy_with_logits(adjusted, torch.tensor([1.0, 0.0]))
    loss = loss + 0.1 * delta.square().mean() + 0.1 * latent.square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    assert auxiliary.module_sha256(enc, head) == base_hash
    assert all(parameter.grad is None for parameter in enc.parameters())
    assert all(parameter.grad is None for parameter in head.parameters())


def test_positive_residual_gate_removes_only_weak_positive_evidence() -> None:
    module = load_module()
    delta = torch.tensor([-0.3, 0.2, 0.6])
    gated = module.apply_positive_residual_gate(delta, minimum_positive=0.5)

    torch.testing.assert_close(gated, torch.tensor([-0.3, 0.0, 0.6]))


def test_median_normalized_margin_uses_each_models_own_threshold() -> None:
    module = load_module()
    scores = np.asarray([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    thresholds = np.asarray([0.5, 0.25, 0.75])
    alarm = module.median_margin_alarm(scores, thresholds)

    np.testing.assert_array_equal(alarm, np.asarray([True, False]))


def test_support_weak_margin_contract_preserves_registered_disruption_fraction() -> None:
    module = load_module()
    contract = module.support_weak_margin_contract(
        [-0.2, 0.0, 0.1, 0.2, 0.3], preserve_fraction=0.8
    )
    assert contract["preserve_fraction"] == 0.8
    assert contract["weak_margin_max"] == pytest.approx(0.1)
    assert contract["protected_support_disruption_count"] == 4
    assert contract["excluded_missed_disruption_count"] == 1
