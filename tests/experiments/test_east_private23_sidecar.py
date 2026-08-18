from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "east_private23_sidecar.py"
    spec = importlib.util.spec_from_file_location("east_private23_sidecar", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    path = HERE / "run_east_private23_online.py"
    spec = importlib.util.spec_from_file_location("run_east_private23_online", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_zero_initialized_private_branch_exactly_preserves_shared_risk():
    module = load_module()
    model = module.EastPrivateResidual(channel_count=8, hidden=16)
    raw = torch.randn(4, 128, 8)
    available = torch.ones_like(raw, dtype=torch.bool)
    base = torch.randn(4)
    adjusted, residual, _features = model(raw, available, base)

    torch.testing.assert_close(adjusted, base, atol=0, rtol=0)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0, rtol=0)
    assert sum(parameter.numel() for parameter in model.parameters()) < 5000


def test_private_features_fail_closed_for_unavailable_channel():
    module = load_module()
    raw = torch.randn(2, 128, 8)
    available = torch.ones_like(raw, dtype=torch.bool)
    available[:, :, 3] = False
    features = module.causal_private_features(raw, available)

    unavailable = features.reshape(2, 8, -1)[:, 3]
    torch.testing.assert_close(unavailable[:, :-1], torch.zeros_like(unavailable[:, :-1]))
    torch.testing.assert_close(unavailable[:, -1], torch.zeros(2))


def test_sealed_temporal_error_sets_opposite_fp_fn_directions():
    module = load_module()
    sealed = torch.tensor([2.0, -2.0])
    current = sealed.clone().requires_grad_(True)
    target = torch.tensor([0.0, 1.0])
    loss = module.sealed_temporal_residual_loss(
        current, sealed, target, correction_gain=1.0, maximum_target_shift=0.5,
    )
    loss.backward()

    assert current.grad[0] > 0  # optimizer lowers the FP logit
    assert current.grad[1] < 0  # optimizer raises the FN logit


def test_same_error_replay_never_substitutes_opposite_error_type():
    module = load_module()
    replay = module.SameErrorReplayBuffer(capacity_shots=4)
    batch = {"sealed_logit": torch.zeros(3), "target": torch.zeros(3)}
    replay.add(10, error_group="FP", batch=batch)

    assert replay.sample(error_group="FN", count=2, seed=1) is None
    sampled = replay.sample(error_group="FP", count=2, seed=1)
    assert sampled is not None
    assert sampled["sealed_logit"].shape == (2,)


def test_error_coreset_contains_only_sealed_temporal_mistakes():
    runner = load_runner()
    probability = np.asarray([0.1, 0.8, 0.2, 0.9], np.float32)
    labels = np.asarray([False, False, True, True])
    chosen = runner.select_error_coreset(
        probability, labels, threshold=0.5, per_error_type=16
    )

    assert chosen.tolist() == [1, 2]
