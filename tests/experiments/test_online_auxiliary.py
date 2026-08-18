from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
MODULE = HERE / "online_auxiliary.py"
sys.path.insert(0, str(HERE))


def load_module():
    spec = importlib.util.spec_from_file_location("online_auxiliary", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_token_path_reproduces_base_cls_bitwise() -> None:
    module = load_module()
    import grid_search as grid

    torch.manual_seed(7)
    enc = grid.Enc(3).cpu().eval()
    x = torch.randn(2, 128, 3)
    before = module.module_sha256(enc)
    with torch.no_grad():
        expected = enc(x)
        cls, last = module.frozen_token_features(enc, x)

    assert torch.equal(cls, expected)
    assert last.shape == expected.shape
    assert module.module_sha256(enc) == before


def test_online_learner_requires_predict_before_observe_and_strict_next_step() -> None:
    module = load_module()
    learner = module.OnlineAuxiliaryLearner(feature_dim=4, physics_dim=2, hidden=8)
    frozen = torch.zeros(4)
    physics = torch.zeros(2)
    available = torch.ones(2, dtype=torch.bool)

    with pytest.raises(RuntimeError):
        learner.observe_and_update(1, frozen, physics, available)
    learner.predict_and_seal(0, frozen, physics, available)
    with pytest.raises(RuntimeError):
        learner.observe_and_update(2, frozen, physics, available)


def test_sidecar_update_changes_only_sidecar_and_emits_ordered_events() -> None:
    module = load_module()
    import grid_search as grid

    torch.manual_seed(9)
    enc = grid.Enc(3).cpu().eval()
    head = grid.Head().cpu().eval()
    learner = module.OnlineAuxiliaryLearner(feature_dim=4, physics_dim=2, hidden=8)
    base_before = module.module_sha256(enc, head)
    sidecar_before = module.module_sha256(learner.sidecar)

    learner.predict_and_seal(
        0, torch.tensor([0.0, 0.1, 0.2, 0.3]),
        torch.tensor([0.1, 0.2]), torch.tensor([True, True]),
    )
    receipt = learner.observe_and_update(
        1, torch.tensor([0.2, 0.1, 0.0, -0.1]),
        torch.tensor([0.7, 0.8]), torch.tensor([True, True]),
    )

    assert [event["event"] for event in learner.events] == ["predict_seal", "observe_update"]
    assert receipt["updated"] is True
    assert np.isfinite(receipt["next_loss"])
    assert module.module_sha256(enc, head) == base_before
    assert module.module_sha256(learner.sidecar) != sidecar_before
    assert all(parameter.grad is None for parameter in enc.parameters())
    assert all(parameter.grad is None for parameter in head.parameters())


def test_missing_next_physics_skips_next_loss_but_direction_stays_finite() -> None:
    module = load_module()
    learner = module.OnlineAuxiliaryLearner(feature_dim=4, physics_dim=2, hidden=8)
    learner.predict_and_seal(
        0, torch.zeros(4), torch.zeros(2), torch.zeros(2, dtype=torch.bool)
    )
    receipt = learner.observe_and_update(
        1, torch.ones(4), torch.zeros(2), torch.zeros(2, dtype=torch.bool)
    )

    assert receipt["updated"] is True
    assert receipt["next_valid_count"] == 0
    assert receipt["next_loss"] == 0.0
    assert np.isfinite(receipt["direction_loss"])


def test_reset_shot_clears_pending_transition_without_resetting_weights() -> None:
    module = load_module()
    learner = module.OnlineAuxiliaryLearner(feature_dim=4, physics_dim=2, hidden=8)
    learner.predict_and_seal(0, torch.zeros(4), torch.zeros(2), torch.ones(2, dtype=torch.bool))
    before = module.module_sha256(learner.sidecar)
    learner.reset_shot()

    assert module.module_sha256(learner.sidecar) == before
    with pytest.raises(RuntimeError):
        learner.observe_and_update(1, torch.zeros(4), torch.zeros(2), torch.ones(2, dtype=torch.bool))


def test_observe_without_update_is_a_weight_immutable_static_control() -> None:
    module = load_module()
    learner = module.OnlineAuxiliaryLearner(feature_dim=4, physics_dim=2, hidden=8)
    learner.predict_and_seal(0, torch.zeros(4), torch.zeros(2), torch.ones(2, dtype=torch.bool))
    before = module.module_sha256(learner.sidecar, learner.teacher)
    receipt = learner.observe_and_update(
        1, torch.ones(4), torch.ones(2), torch.ones(2, dtype=torch.bool), update=False
    )

    assert receipt["updated"] is False
    assert module.module_sha256(learner.sidecar, learner.teacher) == before


def test_support_novelty_contract_uses_only_registered_quantiles() -> None:
    module = load_module()
    contract = module.support_novelty_contract([1.0, 2.0, 3.0, 4.0, 5.0])
    assert contract["quiet_quantile"] == 0.20
    assert contract["reference_quantile"] == 0.80
    assert contract["quiet_normalized"] == pytest.approx(
        contract["quiet_raw"] / contract["reference_raw"]
    )


def test_causal_candidate_requires_risk_then_observed_physics_confirmation() -> None:
    module = load_module()
    result = module.causal_candidate_confirmation(
        [0.1, 0.8, 0.2, 0.1], threshold=0.5,
        novelty_events=[(0, 0.2), (2, 0.9), (3, 0.9)],
        quiet_raw=0.5, minimum_events=2,
    )
    assert result["candidate_index"] == 1
    assert result["confirmed"] is True
    assert result["confirmation_index"] == 2


def test_causal_candidate_never_uses_future_event_to_move_confirmation_earlier() -> None:
    module = load_module()
    base = module.causal_candidate_confirmation(
        [0.1, 0.8, 0.2], threshold=0.5,
        novelty_events=[(0, 0.1), (2, 0.9)], quiet_raw=0.4, minimum_events=2,
    )
    changed = module.causal_candidate_confirmation(
        [0.1, 0.8, 0.2], threshold=0.5,
        novelty_events=[(0, 0.1), (2, 0.9), (99, 1000.0)],
        quiet_raw=0.4, minimum_events=2,
    )
    assert changed["confirmation_index"] == base["confirmation_index"] == 2
