from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jtext_episodic_adapter as episodic  # noqa: E402


def _prototype(dimension: int = 8) -> np.ndarray:
    value = np.linspace(0.2, 1.0, dimension, dtype=np.float32)
    return value / np.linalg.norm(value)


def _descriptor(width: int = 11) -> np.ndarray:
    value = np.linspace(0.1, 1.1, width, dtype=np.float32)
    return value / np.linalg.norm(value)


def test_prototype_adapter_is_identity_before_fit_and_below_parameter_budget() -> None:
    adapter = episodic.PrototypeConditionedRiskAdapter(
        prototype=_prototype(), rank=2, max_logit_delta=2.0,
    )
    hidden = torch.randn(5, 8)
    base = torch.randn(5)
    context = torch.randn(5, 3)

    adjusted, transformed, residual = adapter(hidden, base, context)

    torch.testing.assert_close(adjusted, base, rtol=0.0, atol=0.0)
    torch.testing.assert_close(transformed, hidden, rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual, torch.zeros_like(base), rtol=0.0, atol=0.0)
    assert episodic.adapter_parameter_count(adapter) < 5_000


def test_prototype_adapter_fit_executes_neural_update_with_bounded_receipt() -> None:
    rng = np.random.default_rng(131)
    hidden = rng.normal(size=(12, 8)).astype(np.float32)
    prototype = episodic.hidden_prototype(hidden)
    descriptors = rng.normal(size=(12, 11)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    adapter = episodic.PrototypeConditionedRiskAdapter(
        prototype=prototype, descriptor_prototype=descriptors,
        rank=2, max_logit_delta=2.0,
    )
    before = episodic.module_sha256(adapter)
    receipt = episodic.fit_episode_adapter(
        adapter,
        embeddings=hidden,
        base_logits=np.full(12, 1.5, dtype=np.float32),
        contexts=np.zeros((12, 3), dtype=np.float32),
        labels=np.zeros(12, dtype=np.float32),
        device="cpu", learning_rate=0.05, steps=8,
        max_relative_change=1.0,
    )

    assert receipt["accepted"] is True
    assert receipt["mean_abs_logit_residual"] > 0.0
    assert receipt["mean_representation_delta"] > 0.0
    assert receipt["relative_parameter_change"] <= 1.0 + 1e-6
    assert episodic.module_sha256(adapter) != before


def test_ordered_bank_routes_only_strict_past_and_correct_blocker_can_veto() -> None:
    descriptor = _descriptor()
    adapter = episodic.PrototypeConditionedRiskAdapter(
        prototype=_prototype(), rank=2,
    )
    bank = episodic.OrderedEpisodicBank(
        threshold=0.5, minimum_similarity=0.8,
        minimum_margin=0.05, risk_margin=1.0,
    )
    bank.add_error_episode(
        shot_id=101, stream_index=3, group="FP",
        descriptors=descriptor[None, :], adapter=adapter,
        source_risk_min=0.6, source_risk_max=0.8,
    )

    episode, receipt = bank.route(
        stream_index=3, descriptor=descriptor, source_probability=0.7,
    )
    assert episode is None
    assert receipt.accepted is False
    assert receipt.outcome_free is True

    episode, receipt = bank.route(
        stream_index=4, descriptor=descriptor, source_probability=0.7,
    )
    assert episode is not None
    assert receipt.accepted is True
    assert set(receipt.as_dict()).isdisjoint({"outcome", "is_disrupt", "label"})

    bank.add_blocker(
        shot_id=102, stream_index=4, descriptors=descriptor[None, :],
    )
    episode, receipt = bank.route(
        stream_index=5, descriptor=descriptor, source_probability=0.7,
    )
    assert episode is None
    assert receipt.accepted is False


def test_ordered_bank_requires_repeated_past_support_before_promotion() -> None:
    descriptor = _descriptor()
    adapter = episodic.PrototypeConditionedRiskAdapter(
        prototype=_prototype(), rank=2,
    )
    bank = episodic.OrderedEpisodicBank(
        threshold=0.5, minimum_similarity=0.8,
        minimum_margin=0.0, risk_margin=1.0, minimum_episode_support=2,
    )
    bank.add_error_episode(
        shot_id=201, stream_index=1, group="FP",
        descriptors=descriptor[None, :], adapter=adapter,
        source_risk_min=0.6, source_risk_max=0.8,
    )
    episode, receipt = bank.route(
        stream_index=2, descriptor=descriptor, source_probability=0.7,
    )
    assert episode is None
    assert receipt.reason == "insufficient_past_episode_support"

    bank.add_error_episode(
        shot_id=202, stream_index=2, group="FP",
        descriptors=descriptor[None, :], adapter=adapter,
        source_risk_min=0.6, source_risk_max=0.8,
    )
    episode, receipt = bank.route(
        stream_index=3, descriptor=descriptor, source_probability=0.7,
    )
    assert episode is not None
    assert receipt.support_count == 2


def test_hidden_prototype_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="prototype"):
        episodic.hidden_prototype(np.zeros((3, 8), dtype=np.float32))
