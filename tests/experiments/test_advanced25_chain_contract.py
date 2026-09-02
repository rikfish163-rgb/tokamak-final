from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"


def _runner():
    return importlib.import_module("run_jtext_rebuilt17_online")


def test_outcome_oracle_requires_explicit_query_after_seal(monkeypatch) -> None:
    runner = _runner()
    oracle = runner.OutcomeOracle([10])
    oracle.seal(10)
    with pytest.raises(PermissionError, match="sealed and queried"):
        oracle.reveal(10)
    oracle.query(10)
    monkeypatch.setattr(
        runner,
        "read_outcome",
        lambda shot: {"is_disrupt": False, "down_time_s": None},
    )
    assert oracle.reveal(10)["is_disrupt"] is False
    oracle.open_final_evaluation()
    assert oracle.evaluation_open is True


def test_causal_embedding_prefix_is_invariant_to_future_signal() -> None:
    runner = _runner()

    class Encoder(nn.Module):
        def forward(self, x, valid_lengths=None):
            # A deterministic 64-dimensional representation of the last row.
            assert valid_lengths is not None
            assert valid_lengths.shape[0] == x.shape[0]
            return x[:, -1, :1].repeat(1, 64)

    class Head(nn.Module):
        def forward(self, z):
            return z[:, 0]

    rng = np.random.default_rng(131)
    values = rng.normal(size=(runner.WINDOW + 12, 17)).astype(np.float32)
    changed = values.copy()
    changed[runner.WINDOW + 3 :] += 1000.0
    endpoint, embedding, logits = runner.causal_embeddings(
        Encoder(), Head(), values, device="cpu", batch_size=16
    )
    endpoint_changed, embedding_changed, logits_changed = runner.causal_embeddings(
        Encoder(), Head(), changed, device="cpu", batch_size=16
    )
    # The online runner must use the same all-endpoint policy as the source
    # evaluator, including left-padded prefixes and short-shot endpoints.
    assert len(endpoint) == len(values)
    assert int(endpoint[0]) == 0
    assert int(endpoint[-1]) == len(values) - 1
    np.testing.assert_array_equal(endpoint, endpoint_changed)
    first_prefix = np.flatnonzero(endpoint < runner.WINDOW + 3)
    np.testing.assert_array_equal(embedding[first_prefix], embedding_changed[first_prefix])
    np.testing.assert_array_equal(logits[first_prefix], logits_changed[first_prefix])


def test_advanced25_command_plan_uses_static_source_for_both_east_stages() -> None:
    chain = importlib.import_module("reproduce_advanced25_chain")
    output = Path("/tmp/adv25-contract-test")
    plan = chain.command_plan(
        seed=131,
        output_root=output,
        retrain_source=True,
    )
    assert "--jtext-label-policy advanced_floor25" in " ".join(plan["source"])
    assert "run_jtext_rebuilt17_online.py" in " ".join(plan["jtext_online"])
    source = output / "seed131" / "source" / "source_checkpoint.pt"
    for name in ("jtext_online", "a1", "adapter", "east_online"):
        command = " ".join(plan[name])
        assert str(source) in command
    assert str(output / "seed131" / "jtext_online") not in " ".join(plan["a1"])
    assert str(output / "seed131" / "jtext_online") not in " ".join(plan["east_online"])
    east_command = " ".join(plan["east_online"])
    assert "--locked-best-v8" in east_command
    assert "--query-quota 6" not in east_command


def test_development_v3c_lineage_is_explicit_and_not_the_default() -> None:
    chain = importlib.import_module("reproduce_advanced25_chain")
    output = Path("/tmp/adv25-development-lineage-contract-test")
    default_plan = chain.command_plan(
        seed=131,
        output_root=output / "default",
        retrain_source=True,
    )
    diagnostic_plan = chain.command_plan(
        seed=131,
        output_root=output / "diagnostic",
        retrain_source=True,
        online_lineage="development_v3c",
    )
    default_command = " ".join(default_plan["east_online"])
    diagnostic_command = " ".join(diagnostic_plan["east_online"])
    assert "--locked-best-v8" in default_command
    assert "--locked-best-v8" not in diagnostic_command
    assert "--query-quota 6" in diagnostic_command
    assert "--development-search" in diagnostic_command


def test_dynamic_checkpoint_contract_rejects_fixed_policy(tmp_path: Path) -> None:
    chain = importlib.import_module("reproduce_advanced25_chain")
    path = tmp_path / "source_checkpoint.pt"
    torch.save(
        {
            "seed": 131,
            "training_label": {"policy": "fixed125"},
            "test_shards_read": 0,
        },
        path,
    )
    with pytest.raises(ValueError, match="Adv25 labels"):
        chain._require_dynamic_checkpoint(path, seed=131)


def test_dynamic_checkpoint_contract_rejects_incomplete_adv25_metadata(tmp_path: Path) -> None:
    chain = importlib.import_module("reproduce_advanced25_chain")
    path = tmp_path / "source_checkpoint.pt"
    torch.save(
        {
            "seed": 131,
            "training_label": {"policy": "advanced_floor25"},
            "test_shards_read": 0,
        },
        path,
    )
    with pytest.raises(ValueError, match="AdvancedTime contract metadata"):
        chain._require_dynamic_checkpoint(path, seed=131)


def test_chain_main_rejects_local_artifact_root_before_any_stage_write(
    monkeypatch,
) -> None:
    """The orchestrator must enforce the mounted-disk rule, not only runners."""
    chain = importlib.import_module("reproduce_advanced25_chain")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reproduce_advanced25_chain.py",
            "--seed",
            "131",
            "--stage",
            "source",
            "--output-root",
            "/home/hetaisheng/fuison_ext4_runs",
            "--source-checkpoint",
            "/tmp/not-created.pt",
        ],
    )
    with pytest.raises(RuntimeError, match="legacy local artifact root"):
        chain.main()


def test_jtext_intra_update_has_causal_risk_gradient_and_bounded_projection() -> None:
    """A legal one-step observation must not silently become a no-op.

    The adapter is identity-initialized.  The regression checks the failure
    that previously made every intra-shot step roll back: a finite gradient
    is applied to the next-endpoint risk path, then projected into the
    registered radius rather than rejected solely because the raw Adam step
    is larger than that radius.
    """
    runner = _runner()
    torch.manual_seed(131)
    adapter = runner.LowRankRiskAdapter().to("cpu")
    before = runner.module_sha256(adapter)
    previous_z = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    current_z = np.linspace(1.0, -1.0, 64, dtype=np.float32)
    receipt = runner.intra_update(
        adapter,
        previous_z,
        current_z,
        [previous_z],
        [np.asarray([[-1.5]], dtype=np.float32).reshape(-1)],
        device="cpu",
        learning_rate=1e-3,
        steps=1,
        max_relative_change=1e-3,
        current_base_logit=np.asarray([1.5], dtype=np.float32),
        risk_forecast_weight=0.5,
    )
    assert receipt["applied"] is True
    assert receipt["projected_to_radius"] is True
    assert receipt["raw_relative_parameter_change"] > 1e-3
    assert receipt["relative_parameter_change"] <= 1e-3 + 1e-7
    assert receipt["parameter_sha256"] != before


def test_causal_block_query_budget_does_not_push_all_queries_to_the_suffix() -> None:
    runner = _runner()
    selector = runner.CausalBlockBudgetSelector(
        total_shots=220, fraction=0.20, cutoff=0.50, block_size=20,
    )
    selected = []
    for index in range(220):
        value, _reason = selector.decide(
            score=1.0 if index % 20 == 0 else 0.0,
            stream_index=index,
        )
        selected.append(value)
    assert selector.selected == 44
    assert [sum(selected[start:start + 20]) for start in range(0, 220, 20)] == [4] * 11
    assert max(index for index, value in enumerate(selected) if value) < 220


def test_causal_block_query_budget_rejects_incomplete_roster() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="complete roster"):
        runner.CausalBlockBudgetSelector(
            total_shots=221, fraction=0.20, cutoff=0.50, block_size=20,
        )
