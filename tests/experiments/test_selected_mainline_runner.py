from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
CONFIG = (
    HERE
    / "paper_grade_package/configs/primary_chain_dynamic_source_v1.json"
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


selected = importlib.import_module("selected_mainline_contract")
runner = importlib.import_module("run_selected_full_mainline")
a1 = importlib.import_module("run_online_active_a1")
dha = importlib.import_module("run_dynamic_hazard_adapter")
unified = importlib.import_module("run_unified_online_active")


def _literal_threshold_receipt(
    *, scope: str, roster: list[int], threshold: float, target_tpr: float,
) -> dict:
    body = {
        "schema_version": "selected-threshold-receipt-v1",
        "scope": scope,
        "roster": list(roster),
        "roster_sha256": selected.sha256_json(list(roster)),
        "data_manifest_sha256": "6" * 64,
        "outcome_manifest_sha256": "7" * 64,
        "selection_rule": (
            "meet target TPR, then minimize FPR, then maximize finite threshold"
        ),
        "target_tpr": float(target_tpr),
        "predictions": [
            {
                "shot": int(shot),
                "score": float(threshold if index % 2 == 0 else 0.1),
                "outcome": bool(index % 2 == 0),
            }
            for index, shot in enumerate(roster)
        ],
        "candidates": [
            {
                "threshold": float(threshold), "tpr": 1.0, "fpr": 0.0,
            },
            {"threshold": 0.1, "tpr": 1.0, "fpr": 1.0},
        ],
        "chosen_threshold": float(threshold),
    }
    return {**body, "receipt_sha256": selected.sha256_json(body)}


def _preprocessing_objects() -> dict[str, dict]:
    return {
        "target_scaler": {"center": [0.0, 0.0], "scale": [1.0, 1.0]},
        "raw8_scaler": {"center": [0.0], "scale": [1.0]},
        "physics_scales": {"ip": 1.0},
        "reconstruction_constants": {
            "minimum_positive_residual": 0.1,
            "positive_support_windows": 1,
            "shared_reconstruction_q80": 0.2,
            "private_reconstruction_q80": 0.3,
        },
    }


def _contract() -> selected.SelectedMainlineContract:
    source_receipt = _literal_threshold_receipt(
        scope="J-TEXT VAL160", roster=list(range(160)),
        threshold=0.5, target_tpr=0.90,
    )
    east_receipt = _literal_threshold_receipt(
        scope="EAST support110", roster=[101, 102],
        threshold=0.42, target_tpr=0.80,
    )
    preprocessing = _preprocessing_objects()
    return selected.SelectedMainlineContract(
        seed=131,
        selection_role="post_adjudication_user_selected_development_candidate",
        historical_admission="FAIL",
        confirmation_eligible=False,
        corrected_ablation_report_sha256=(
            "896584467254d2ae1df894f6f4a0e4ed3a024ec556dcc1ffeb7e083f04db3d04"
        ),
        selection_metric="mean balanced360 operational_TPR - FPR",
        historical_failure_reason="seed133 TPR@30 < 0.80",
        source_checkpoint_sha256="a" * 64,
        source_label_policy="advanced_floor",
        source_label_map_sha256="b" * 64,
        source_split_sha256="c" * 64,
        source_scaler_sha256="d" * 64,
        source_input_contract_sha256="1" * 64,
        source_preprocessing_sha256="2" * 64,
        source_val_threshold_receipt_sha256=source_receipt["receipt_sha256"],
        east_target_label_policy="fixed125",
        east_target_horizon_ms=125.0,
        east_support_roster_sha256=east_receipt["roster_sha256"],
        east_support_data_manifest_sha256=east_receipt["data_manifest_sha256"],
        east_support_outcome_manifest_sha256=(
            east_receipt["outcome_manifest_sha256"]
        ),
        east_threshold_receipt_sha256=east_receipt["receipt_sha256"],
        target_scaler_sha256=selected.sha256_json(
            preprocessing["target_scaler"]
        ),
        raw8_scaler_sha256=selected.sha256_json(preprocessing["raw8_scaler"]),
        physics_scales_sha256=selected.sha256_json(
            preprocessing["physics_scales"]
        ),
        reconstruction_constants_sha256=selected.sha256_json(
            preprocessing["reconstruction_constants"]
        ),
        jtext_test_roster_sha256=selected.sha256_json(list(range(220))),
        east_test_roster_sha256=selected.sha256_json(list(range(360))),
        claim_status="TEST_DRIVEN_EXPLORATION",
    )


def _parent_payload(contract: selected.SelectedMainlineContract) -> dict:
    encoder = {"weight": torch.tensor([1.0])}
    head = {"weight": torch.tensor([2.0])}
    shared = {"weight": torch.tensor([3.0])}
    private = {"weight": torch.tensor([4.0])}
    source_receipt = _literal_threshold_receipt(
        scope="J-TEXT VAL160", roster=list(range(160)),
        threshold=0.5, target_tpr=0.90,
    )
    east_receipt = _literal_threshold_receipt(
        scope="EAST support110", roster=[101, 102],
        threshold=0.42, target_tpr=0.80,
    )
    preprocessing = _preprocessing_objects()
    payload = {
        "schema_version": selected.SELECTED_PARENT_SCHEMA,
        "selected_label_arm": selected.SELECTED_LABEL_ARM,
        "lineage": {"source": "source_dynamic", "target": "east_fixed"},
        **contract.to_dict(),
        "east_threshold": 0.42,
        "source_val_threshold_receipt": source_receipt,
        "east_threshold_receipt": east_receipt,
        **preprocessing,
        "encoder": encoder,
        "head": head,
        "shared": shared,
        "private": private,
        "target_encoder_sha256": selected.state_dict_sha256(encoder),
        "target_head_sha256": selected.state_dict_sha256(head),
        "target_base_sha256": selected.state_bundle_sha256(encoder, head),
        "shared_parent_sha256": selected.state_dict_sha256(shared),
        "private_parent_sha256": selected.state_dict_sha256(private),
    }
    payload["canonical_parent_fingerprint_sha256"] = (
        selected.canonical_parent_fingerprint(payload)
    )
    return payload


def test_selected_config_freezes_source_dynamic_east_fixed_values() -> None:
    config = runner.load_selected_config(CONFIG)
    frozen_v3c = json.loads((
        HERE / "paper_grade_package/configs/primary_chain_v3c.json"
    ).read_text(encoding="utf-8"))

    assert config["schema_version"] == "primary-chain-dynamic-source-v1"
    assert config["selected_label_arm"] == "source_dynamic__east_fixed"
    assert config["source"]["label_policy"] == "advanced_floor"
    assert config["source"]["positive_horizon"] == (
        "max(25 ms, AdvancedTime_ms); missing -> 25 ms"
    )
    assert config["east"]["target_label_policy"] == "fixed125"
    assert config["east"]["target_horizon_ms"] == 125.0
    assert config["east"]["threshold_source"] == "EAST support110"
    assert config["online"] == frozen_v3c["online"]
    assert config["jtext_online"] == {
        "arm": "full_active",
        "budget_percent": 20,
        "state_transfer_to_east_allowed": False,
    }
    assert config["claim_status"] == "TEST_DRIVEN_EXPLORATION"
    assert config["selection_role"] == (
        "post_adjudication_user_selected_development_candidate"
    )
    assert config["historical_admission"] == "FAIL"
    assert config["confirmation_eligible"] is False
    assert config["stage_sequence"] == [
        "source-train", "jtext-static", "jtext-online", "east-parent",
        "east-static", "east-online",
    ]


def test_selected_east_support_training_uses_historical_shot_first_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source")
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    support = [
        {"shot": 1, "is_disrupt": True, "v": np.zeros((128, 2), np.float32)},
        {"shot": 2, "is_disrupt": False, "v": np.zeros((128, 2), np.float32)},
    ]
    calls: list[str] = []

    monkeypatch.setattr(
        a1.selected_contract, "load_selected_source_checkpoint",
        lambda *_args, **_kwargs: (encoder, head, {
            "input_contract": {"schema": "jtext-rebuilt17-prefix-causal-v2"},
        }),
    )
    monkeypatch.setattr(
        a1.alarm, "_source_contract",
        lambda *_args, **_kwargs: (
            source_path.resolve(), np.zeros(2), np.ones(2),
            {"input_contract": {"schema": "jtext-rebuilt17-prefix-causal-v2"}},
        ),
    )
    monkeypatch.setattr(
        a1.alarm.GS, "load_east_selected_support", lambda: (support, []),
    )
    monkeypatch.setattr(a1.alarm.GS, "transform", lambda rows, _mode: rows)
    monkeypatch.setattr(a1.alarm.GS, "apply_scale", lambda *_args: None)
    monkeypatch.setattr(
        a1.alarm, "prepare_east_training_labels",
        lambda rows, **_kwargs: (rows, {"policy": "fixed125"}),
    )
    monkeypatch.setattr(
        a1.alarm.GS, "build_sampler",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("class-first sampler must not be used")
        ),
    )
    monkeypatch.setattr(
        a1.alarm, "historical_shot_first_sampler",
        lambda *_args, **_kwargs: calls.append("shot-first") or object(),
        raising=False,
    )
    monkeypatch.setattr(
        a1.alarm.GS, "train_adapt", lambda *_args, **_kwargs: (encoder, head),
    )

    a1.alarm.train_support_only(131, source_path)

    assert calls == ["shot-first"]


def test_jtext_online_stage_command_uses_static_source_only(tmp_path: Path) -> None:
    config = runner.load_selected_config(CONFIG)
    command = runner.jtext_online_stage_command(
        seed=131,
        output=tmp_path / "jtext-online",
        source_checkpoint=tmp_path / "source.pt",
        data_root=tmp_path / "data",
        config=config,
        python=Path(sys.executable),
    )

    assert command[command.index("--source-checkpoint") + 1] == str(
        tmp_path / "source.pt"
    )
    assert command[command.index("--arm") + 1] == "full_active"
    assert command[command.index("--budget") + 1] == "20"
    assert "east-parent" not in " ".join(command)


def test_selected_config_is_portable_and_contains_no_absolute_media_path() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def strings(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)
        elif isinstance(value, str):
            yield value

    assert not any(value.startswith("/media/") for value in strings(payload))


def test_selected_contract_is_in_mainline_code_manifest_dependency_closure() -> None:
    manifest = json.loads((ROOT / "config/mainline_code_manifest.json").read_text())
    assert (
        "experiments/transfer_methods_2026/selected_mainline_contract.py"
        in manifest["tracked_files"]
    )
    assert manifest["entrypoints"]["selected_dynamic_mainline"] == (
        "experiments/transfer_methods_2026/run_selected_full_mainline.py"
    )


def test_source_stage_command_trains_dynamic_labels_without_test_read(
    tmp_path: Path,
) -> None:
    config = runner.load_selected_config(CONFIG)
    command = runner.source_stage_command(
        seed=131,
        output=tmp_path / "source",
        config=config,
        python=Path(sys.executable),
    )

    assert command[0] == sys.executable
    assert command[command.index("--label-policy") + 1] == "advanced_floor"
    assert "--skip-test" in command
    assert "--evaluate-test" not in command
    assert "fixed125" not in command


def test_selected_support_parent_requires_explicit_dynamic_source_and_fixed125(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "dynamic.pt"
    source_path.write_bytes(b"source")
    calls: list[dict] = []
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    support = [
        {"shot": 1, "is_disrupt": True, "v": np.zeros((128, 2), np.float32)},
        {"shot": 2, "is_disrupt": False, "v": np.zeros((128, 2), np.float32)},
    ]

    monkeypatch.setattr(
        a1.selected_contract,
        "load_selected_source_checkpoint",
        lambda path, seed: (encoder, head, {
            "training_label": {"policy": "advanced_floor"},
        }),
    )

    def support_only(seed, source_checkpoint, **kwargs):
        calls.append({
            "seed": seed,
            "source_checkpoint": source_checkpoint,
            **kwargs,
        })
        return encoder, head, support, np.zeros(2), np.ones(2), {
            "policy": kwargs["target_label_policy"],
        }

    monkeypatch.setattr(a1.alarm, "train_support_only", support_only)
    monkeypatch.setattr(
        a1.alarm, "full_risk",
        lambda _encoder, _head, shot: np.asarray(
            [0.9 if shot["is_disrupt"] else 0.1], dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        a1, "fit_support_physics_scales", lambda _support: {"ip": 1.0},
    )

    with pytest.raises(ValueError, match="explicit source checkpoint"):
        a1.train_selected_support_parent(seed=131, source_checkpoint=None)
    with pytest.raises(ValueError, match="fixed125"):
        a1.train_selected_support_parent(
            seed=131, source_checkpoint=source_path,
            target_label_policy="advanced_floor",
        )

    result = a1.train_selected_support_parent(
        seed=131, source_checkpoint=source_path,
        target_label_policy="fixed125", target_tpr=0.80,
    )

    assert calls == [{
        "seed": 131,
        "source_checkpoint": source_path.resolve(),
        "target_label_policy": "fixed125",
        "return_label_audit": True,
    }]
    assert result["threshold"] == pytest.approx(0.9)
    assert result["threshold_source"] == "EAST support110"
    assert result["target_label_policy"] == "fixed125"
    assert result["source_checkpoint"] == source_path.resolve()
    receipt = selected.validate_threshold_receipt(
        result["threshold_receipt"],
        expected_scope="EAST support110",
        expected_roster=[1, 2],
        expected_threshold=0.9,
    )
    assert result["support_roster_sha256"] == receipt["roster_sha256"]
    assert result["support_data_manifest_sha256"] == receipt[
        "data_manifest_sha256"
    ]
    assert result["support_outcome_manifest_sha256"] == receipt[
        "outcome_manifest_sha256"
    ]


def test_selected_shared_adapter_never_calls_legacy_full_test_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy shot_alarm.train must not be called")

    monkeypatch.setattr(dha.alarm, "train", forbidden)
    monkeypatch.setattr(
        dha, "build_support_cache", lambda *_args, **_kwargs: [{"label": np.ones(1)}],
    )
    monkeypatch.setattr(
        dha, "train_adapter", lambda adapter, *_args, **_kwargs: [{"loss": 0.0}],
    )
    monkeypatch.setattr(dha.alarm, "DEV", "cpu")
    base = {
        "encoder": torch.nn.Linear(2, 2),
        "head": torch.nn.Linear(2, 1),
        "support": [{"shot": 1}],
        "physics_scales": {"ip": 1.0},
        "threshold": 0.5,
    }

    result = dha.fit_selected_shared_parent(base, seed=131)

    assert result["support_cache"] == [{"label": np.ones(1)}]
    assert result["training_history"] == [{"loss": 0.0}]
    assert result["shared"].training is False


def test_parent_payload_is_new_schema_not_relabelled_legacy() -> None:
    contract = _contract()
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    shared = torch.nn.Linear(2, 2)
    private = torch.nn.Linear(2, 2)

    payload = runner.make_selected_parent_payload(
        contract=contract,
        threshold=0.42,
        encoder=encoder,
        head=head,
        shared=shared,
        private=private,
        target_label_audit={"policy": "fixed125"},
    )

    assert payload["schema_version"] == "dynamic-source-east-fixed-parent-v1"
    assert payload["schema_version"] != "equal-label-budget-frontload-v2"
    assert payload["lineage"] == {
        "source": "source_dynamic", "target": "east_fixed",
    }
    assert payload["east_target_label_policy"] == "fixed125"
    selected.validate_selected_parent(payload, contract)


def test_support_shared_and_private_parent_fit_never_open_balanced360_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source")
    outcome_reads: list[int] = []
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    shared = torch.nn.Linear(2, 2)
    private = torch.nn.Linear(2, 2)
    support = [{"shot": 101}, {"shot": 102}]
    source_payload = {
        "training_label": {"policy": "advanced_floor"},
        "source_checkpoint_sha256": "a" * 64,
        "source_label_map_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "source_scaler_sha256": "d" * 64,
        "source_input_contract_sha256": "1" * 64,
        "source_preprocessing_sha256": "2" * 64,
    }

    monkeypatch.setattr(
        runner.selected,
        "load_selected_source_checkpoint",
        lambda *_args, **_kwargs: (encoder, head, source_payload),
    )
    monkeypatch.setattr(
        runner.a1,
        "train_selected_support_parent",
        lambda **_kwargs: {
            "encoder": encoder, "head": head, "support": support,
            "center": np.zeros(2), "scale": np.ones(2),
            "physics_scales": {"ip": 1.0}, "threshold": 0.42,
            "threshold_source": "EAST support110",
            "target_label_policy": "fixed125",
            "target_label_audit": {"policy": "fixed125"},
            "source_checkpoint": source_path,
        },
    )
    monkeypatch.setattr(
        runner.dha,
        "fit_selected_shared_parent",
        lambda *_args, **_kwargs: {"shared": shared, "support_cache": []},
    )
    monkeypatch.setattr(
        runner.unified,
        "fit_causal_raw8_scaler",
        lambda _ids: (np.zeros(2), np.ones(2)),
    )
    monkeypatch.setattr(
        runner.unified,
        "fit_private_model",
        lambda *_args, **_kwargs: (private, [], {"status": "fit"}),
    )
    monkeypatch.setattr(
        runner.positive_gate,
        "support_positive_threshold", lambda *_args: (0.0, 1),
    )
    monkeypatch.setattr(
        runner.two_runner,
        "read_hdf5_outcome",
        lambda shot: outcome_reads.append(int(shot)),
    )
    monkeypatch.setattr(
        runner.common,
        "DataSpec",
        lambda: (_ for _ in ()).throw(
            AssertionError("balanced360 label manifest must not be parsed")
        ),
    )
    monkeypatch.setattr(
        runner.dha.alarm,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy alarm.train must not run")
        ),
    )
    monkeypatch.setattr(
        runner.a1.alarm.ATL,
        "load_advanced_time_ms",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("EAST AdvancedTime must not be read")
        ),
    )
    contract = replace(
        _contract(),
        east_support_roster_sha256=selected.sha256_json([101, 102]),
    )

    result = runner.build_selected_east_parent(
        seed=131,
        source_checkpoint=source_path,
        contract=contract,
        output=tmp_path / "east-parent",
        private_support_epochs=0,
        balanced360_roster=list(range(360)),
    )

    assert outcome_reads == []
    assert result["payload"]["schema_version"] == selected.SELECTED_PARENT_SCHEMA
    assert result["payload"]["east_target_label_policy"] == "fixed125"
    assert result["parent_path"].is_file()


def test_first_balanced360_outcome_read_is_after_closed_hashed_static_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "stream_index": index,
            "shot": 60_000 + index,
            "alarmed": False,
            "risk_trace": np.asarray([0.1, 0.2], dtype=np.float32),
        }
        for index in range(360)
    ]
    path = tmp_path / "static_predictions.jsonl"
    events: list[tuple[str, int | str]] = []
    original_hash = runner.selected.sha256_file

    def tracked_hash(target: Path) -> str:
        digest = original_hash(target)
        events.append(("hash", digest))
        return digest

    def outcome_reader(shot: int) -> dict:
        assert path.is_file()
        assert len(path.read_text().splitlines()) == 360
        with path.open("a", encoding="utf-8") as handle:
            assert handle.closed is False
        events.append(("read", int(shot)))
        return {"is_disrupt": False, "down_time_s": None}

    monkeypatch.setattr(runner.selected, "sha256_file", tracked_hash)

    result = runner.seal_static_predictions_then_read_outcomes(
        rows=rows,
        seal_path=path,
        stage="east-static",
        expected_count=360,
        outcome_oracle_factory=lambda: outcome_reader,
        threshold_receipt_sha256="a" * 64,
        parent_fingerprint_sha256="b" * 64,
    )

    assert events[0][0] == "hash"
    assert [kind for kind, _value in events].count("read") == 360
    assert result["prediction_seal_sha256"] in [
        value for kind, value in events if kind == "hash"
    ]
    assert result["seal_manifest_sha256"] in [
        value for kind, value in events if kind == "hash"
    ]
    assert len(result["outcomes"]) == 360


def test_static_seal_rejects_partial_or_duplicate_roster(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="360"):
        runner.seal_static_predictions_then_read_outcomes(
            rows=[{"stream_index": 0, "shot": 1}],
            seal_path=tmp_path / "partial.jsonl",
            stage="east-static",
            expected_count=360,
            outcome_oracle_factory=lambda: (lambda _shot: {}),
            threshold_receipt_sha256="a" * 64,
            parent_fingerprint_sha256="b" * 64,
        )
    duplicate = [
        {"stream_index": index, "shot": 1, "alarmed": False}
        for index in range(360)
    ]
    with pytest.raises(RuntimeError, match="unique"):
        runner.seal_static_predictions_then_read_outcomes(
            rows=duplicate,
            seal_path=tmp_path / "duplicate.jsonl",
            stage="east-static",
            expected_count=360,
            outcome_oracle_factory=lambda: (lambda _shot: {}),
            threshold_receipt_sha256="a" * 64,
            parent_fingerprint_sha256="b" * 64,
        )


def test_unified_validator_routes_selected_schema_through_full_contract() -> None:
    contract = _contract()
    payload = _parent_payload(contract)

    lineage = unified.validate_sealed_parent_contract(
        payload,
        None,
        seed=131,
        source_checkpoint_sha256=contract.source_checkpoint_sha256,
        selected_contract=contract,
    )

    assert lineage["schema_version"] == selected.SELECTED_PARENT_SCHEMA
    drifted = copy.deepcopy(payload)
    drifted["source_label_map_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source_label_map_sha256"):
        unified.validate_sealed_parent_contract(
            drifted,
            None,
            seed=131,
            source_checkpoint_sha256=contract.source_checkpoint_sha256,
            selected_contract=contract,
        )
    with pytest.raises(RuntimeError, match="selected mainline contract"):
        unified.validate_sealed_parent_contract(
            payload,
            None,
            seed=131,
            source_checkpoint_sha256=contract.source_checkpoint_sha256,
        )


def test_unified_stage_commands_bind_source_parent_contract_and_static_mode(
    tmp_path: Path,
) -> None:
    common = {
        "seed": 131,
        "source_checkpoint": tmp_path / "source.pt",
        "parent_state": tmp_path / "parent.pt",
        "run_contract": tmp_path / "run_contract.json",
        "roster_manifest": tmp_path / "balanced360.jsonl",
        "config": runner.load_selected_config(CONFIG),
        "python": Path(sys.executable),
    }
    static = runner.unified_stage_command(
        stage="east-static", output=tmp_path / "static", **common,
    )
    online = runner.unified_stage_command(
        stage="east-online", output=tmp_path / "online", **common,
    )

    for command in (static, online):
        assert command[command.index("--source-checkpoint") + 1] == str(
            common["source_checkpoint"]
        )
        assert command[command.index("--sealed-parent-state") + 1] == str(
            common["parent_state"]
        )
        assert command[command.index("--selected-mainline-contract") + 1] == str(
            common["run_contract"]
        )
        assert command[command.index("--roster-manifest") + 1] == str(
            common["roster_manifest"]
        )
    assert "--static-only" in static
    assert "--static-only" not in online


def test_unified_stage_command_forwards_registered_alarm_initialization(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(runner.load_selected_config(CONFIG))
    config["online"].update({
        "initial_high_threshold": -0.10,
        "initial_physics_weight": 6.0,
        "initial_trend_weight": 13.5,
        "initial_early_threshold": 7.5,
    })
    command = runner.unified_stage_command(
        stage="east-online", seed=131, output=tmp_path / "online",
        source_checkpoint=tmp_path / "source.pt",
        parent_state=tmp_path / "parent.pt",
        run_contract=tmp_path / "run_contract.json",
        roster_manifest=tmp_path / "balanced360.jsonl",
        config=config, python=Path(sys.executable),
    )

    assert command[command.index("--initial-high-threshold") + 1] == "-0.1"
    assert command[command.index("--initial-physics-weight") + 1] == "6.0"
    assert command[command.index("--initial-trend-weight") + 1] == "13.5"
    assert command[command.index("--initial-early-threshold") + 1] == "7.5"


def test_completed_stage_resume_requires_identical_bindings_and_outputs(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "stage_receipts.jsonl"
    artifact = tmp_path / "source.pt"
    artifact.write_bytes(b"source")
    binding = {
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "source_split_manifest_sha256": "c" * 64,
        "source_label_map_sha256": "d" * 64,
        "jtext_test_roster_sha256": "e" * 64,
        "east_support_roster_sha256": "f" * 64,
        "east_test_roster_sha256": "1" * 64,
        "hdf5_content_manifest_sha256": "2" * 64,
        "parent_sha256": None,
    }
    runner.append_stage_receipt(
        ledger,
        stage="source-train",
        binding=binding,
        outputs={"source_checkpoint": artifact},
        command=["python", "train.py"],
    )

    receipt = runner.require_resumable_stage(
        ledger, stage="source-train", binding=binding,
    )
    assert receipt["status"] == "completed"
    assert receipt["outputs"]["source_checkpoint"]["sha256"] == (
        selected.sha256_file(artifact)
    )

    with pytest.raises(RuntimeError, match="resume binding mismatch"):
        runner.require_resumable_stage(
            ledger,
            stage="source-train",
            binding={**binding, "code_sha256": "f" * 64},
        )
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        runner.require_resumable_stage(
            ledger, stage="source-train", binding=binding,
        )


def _write_minimal_outcome_hdf5(path: Path, *, disruptive: bool) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        signal = data.create_dataset("ip", data=np.asarray([1.0, 2.0], np.float32))
        signal.attrs["SampleRate"] = 1000.0
        signal.attrs["StartTime"] = 0.0
        meta = handle.create_group("meta")
        meta.create_dataset("IsDisrupt", data=np.asarray([disruptive], np.bool_))
        meta.create_dataset(
            "DownTime", data=np.asarray([1.5 if disruptive else 0.0], np.float64)
        )


@pytest.mark.parametrize(
    ("stage", "count", "allow_advanced_time"),
    [
        pytest.param("jtext-static", 220, True, id="J-TEXT220"),
        pytest.param("east-static", 360, False, id="EAST360"),
    ],
)
def test_c_mut_01_real_hdf5_meta_and_advanced_time_follow_complete_trace_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    count: int,
    allow_advanced_time: bool,
) -> None:
    disruptive_path = tmp_path / "disruptive.hdf5"
    safe_path = tmp_path / "safe.hdf5"
    _write_minimal_outcome_hdf5(disruptive_path, disruptive=True)
    _write_minimal_outcome_hdf5(safe_path, disruptive=False)
    advanced_path = tmp_path / "AdvancedTime.json"
    advanced_path.write_text(json.dumps({"1": 37.0}), encoding="utf-8")
    roster = list(range(1, count + 1))
    output = tmp_path / stage
    events: list[tuple[str, str | int]] = []
    seal_hashed = False

    original_getitem = h5py.Dataset.__getitem__
    original_close = runner.a1.JsonlSealWriter.close
    original_fsync = os.fsync
    original_hash = selected.sha256_file

    def guarded_getitem(dataset, key):
        if dataset.name.startswith("/meta/"):
            events.append(("meta", dataset.name))
            if not seal_hashed:
                raise AssertionError("pre-seal /meta access")
        return original_getitem(dataset, key)

    def tracked_close(writer):
        original_close(writer)
        events.append(("close", writer.path.name))

    def tracked_fsync(fd):
        events.append(("fsync", int(fd)))
        return original_fsync(fd)

    def tracked_hash(path):
        nonlocal seal_hashed
        digest = original_hash(path)
        events.append(("hash", Path(path).name))
        if Path(path).name == "static_seal_manifest.json":
            seal_hashed = True
            events.append(("seal-hash", digest))
        return digest

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(runner.a1.JsonlSealWriter, "close", tracked_close)
    monkeypatch.setattr(runner.os, "fsync", tracked_fsync)
    monkeypatch.setattr(selected, "sha256_file", tracked_hash)

    def predict(stream_index: int, shot: int) -> dict:
        return {
            "stream_index": stream_index,
            "shot": shot,
            "score": float(stream_index) / count,
            "risk_trace": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        }

    def oracle_factory():
        prediction_path = output / "static_predictions.jsonl"
        assert prediction_path.is_file()
        assert len(prediction_path.read_text().splitlines()) == count
        assert seal_hashed

        def read(shot: int) -> dict:
            path = disruptive_path if shot % 2 else safe_path
            with h5py.File(path, "r") as handle:
                disruptive = bool(
                    np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0]
                )
                down = float(
                    np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0]
                )
            if allow_advanced_time:
                events.append(("advanced-time", int(shot)))
                json.loads(advanced_path.read_text(encoding="utf-8"))
            return {"is_disrupt": disruptive, "down_time_s": down}

        return read

    result = runner.execute_static_stage_boundary(
        stage=stage,
        roster=roster,
        output=output,
        prediction_builder=predict,
        outcome_oracle_factory=oracle_factory,
        threshold_receipt_sha256="a" * 64,
        parent_fingerprint_sha256="b" * 64,
    )

    manifest = json.loads(result["seal_manifest_path"].read_text())
    assert manifest["row_count"] == count
    assert len(manifest["risk_traces"]) == count
    assert all(
        selected.sha256_file(output / row["path"]) == row["sha256"]
        for row in manifest["risk_traces"]
    )
    first_forbidden = min(
        index for index, event in enumerate(events)
        if event[0] in {"meta", "advanced-time"}
    )
    assert any(event[0] == "close" for event in events[:first_forbidden])
    assert any(event[0] == "fsync" for event in events[:first_forbidden])
    assert any(event[0] == "seal-hash" for event in events[:first_forbidden])
    if stage == "east-static":
        assert not any(event[0] == "advanced-time" for event in events)


def test_c_mut_01_direct_h5py_meta_read_mutant_fails_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "shot.hdf5"
    _write_minimal_outcome_hdf5(fixture, disruptive=True)
    output = tmp_path / "mutant"
    sealed = False
    original_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, key):
        if dataset.name.startswith("/meta/") and not sealed:
            raise AssertionError("pre-seal /meta access")
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)

    def mutant_predict(stream_index: int, shot: int) -> dict:
        with h5py.File(fixture, "r") as handle:
            handle["meta/IsDisrupt"][()]
        return {
            "stream_index": stream_index,
            "shot": shot,
            "risk_trace": np.asarray([0.1], dtype=np.float32),
        }

    with pytest.raises(AssertionError, match="pre-seal /meta access"):
        runner.execute_static_stage_boundary(
            stage="jtext-static",
            roster=list(range(1, 221)),
            output=output,
            prediction_builder=mutant_predict,
            outcome_oracle_factory=lambda: (lambda _shot: {}),
            threshold_receipt_sha256="a" * 64,
            parent_fingerprint_sha256="b" * 64,
        )

    assert not (output / "static_seal_manifest.json").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "stage_receipts.jsonl").exists()


def test_postseal_jtext_metrics_use_first_crossing_and_strict_ten_ms(
    tmp_path: Path,
) -> None:
    rows = []
    traces = {
        1: np.asarray([0.1, 0.8], dtype=np.float32),
        2: np.asarray([0.1, 0.8], dtype=np.float32),
        3: np.asarray([0.1, 0.8], dtype=np.float32),
        4: np.asarray([0.1, 0.2], dtype=np.float32),
    }
    for stream_index, shot in enumerate((1, 2, 3, 4)):
        relative = Path("risk_traces") / f"{stream_index:04d}_{shot}.npz"
        digest = selected.write_trace_file(
            tmp_path / relative, shot=shot, risk_trace=traces[shot],
        )
        rows.append({
            "stream_index": stream_index,
            "shot": shot,
            "score": float(traces[shot].max()),
            "risk_trace_path": str(relative),
            "risk_trace_sha256": digest,
            "start_time_s": 0.0,
            "sample_rate_hz": 1000.0,
        })
    outcomes = {
        1: {"is_disrupt": True, "down_time_s": 0.020},
        2: {"is_disrupt": True, "down_time_s": 0.010},
        3: {"is_disrupt": False, "down_time_s": None},
        4: {"is_disrupt": False, "down_time_s": None},
    }

    metrics = runner.compute_postseal_jtext_static_metrics(
        sealed_rows=rows, outcomes=outcomes, output=tmp_path, threshold=0.5,
    )

    assert metrics["TP"] == metrics["FN"] == 1
    assert metrics["FP"] == metrics["TN"] == 1
    assert metrics["raw_tpr"] == 1.0
    assert metrics["timely_tpr"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["median_lead_ms"] == pytest.approx(19.0)
    assert metrics["minimum_lead_ms"] == 10.0
    assert metrics["minimum_run"] == 1


@pytest.mark.parametrize(
    ("forbidden_key", "forbidden_value"),
    [
        ("is_disrupt", True),
        ("class", "disruptive"),
        ("DownTime", 1.0),
        ("AdvancedTime", 25.0),
        ("outcome", False),
    ],
)
def test_label_free_roster_rejects_every_outcome_bearing_field(
    tmp_path: Path, forbidden_key: str, forbidden_value,
) -> None:
    path = tmp_path / "roster.jsonl"
    path.write_text(
        json.dumps({
            "stream_index": 0, "shot": 1, forbidden_key: forbidden_value,
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label-free roster"):
        selected.load_label_free_roster(path, expected_count=1)


def test_label_free_roster_loader_never_constructs_label_bearing_dataspec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "roster.jsonl"
    path.write_text(
        "".join(
            json.dumps({"stream_index": index, "shot": shot}) + "\n"
            for index, shot in enumerate((11, 12))
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner.common,
        "DataSpec",
        lambda: (_ for _ in ()).throw(
            AssertionError("label-bearing DataSpec must not be constructed")
        ),
    )

    assert selected.load_label_free_roster(path, expected_count=2) == [11, 12]


def test_selected_timebase_metadata_missing_sample_rate_or_start_time_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-timebase.hdf5"
    with h5py.File(path, "w") as handle:
        signal = handle.create_dataset("data/ip", data=np.ones(2, np.float32))
        signal.attrs["SampleRate"] = 1000.0

    with h5py.File(path, "r") as handle:
        with pytest.raises(RuntimeError, match="StartTime"):
            selected.read_required_timebase(handle["data/ip"])


@pytest.mark.parametrize("phase", ["east-static", "east-online"])
@pytest.mark.parametrize(
    "mutation",
    ["tensor_bitflip", "scaler", "threshold", "roster"],
)
def test_c_mut_03_selected_startup_recomputes_parent_before_first_ledger_row(
    tmp_path: Path, phase: str, mutation: str,
) -> None:
    contract = _contract()
    parent = _parent_payload(contract)
    online_copy = copy.deepcopy(parent)
    roster = list(range(360))
    if mutation == "tensor_bitflip":
        online_copy["shared"]["weight"].view(-1)[0] += 1.0
    elif mutation == "scaler":
        online_copy["target_scaler"]["center"][0] += 1.0
    elif mutation == "threshold":
        online_copy["east_threshold"] = 0.41
    elif mutation == "roster":
        roster[-1] = 999
    ledger = tmp_path / f"{phase}.jsonl"

    with pytest.raises(RuntimeError, match="selected startup .* mismatch"):
        unified.validate_selected_startup_inputs(
            payload=online_copy,
            contract=contract,
            actual_roster=roster,
            seed=contract.seed,
            source_checkpoint_sha256=contract.source_checkpoint_sha256,
        )

    assert not ledger.exists()


def _snapshot_tree(path: Path) -> dict[str, bytes]:
    return {
        str(candidate.relative_to(path)): candidate.read_bytes()
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


def _two_shot_resume_fixture(tmp_path: Path) -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    code_files = []
    for name in ("source_trainer.py", "rebuilt_harness.py", "preprocess.py"):
        path = inputs / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        code_files.append(path)
    config = inputs / "config.json"
    config.write_text('{"mode":"cpu-smoke"}\n', encoding="utf-8")
    checkpoint = inputs / "checkpoint.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    parent = inputs / "parent.pt"
    torch.save({"weight": torch.tensor([2.0])}, parent)
    roster = inputs / "roster.jsonl"
    roster.write_text(
        '{"stream_index":0,"shot":1}\n'
        '{"stream_index":1,"shot":2}\n',
        encoding="utf-8",
    )
    threshold = inputs / "threshold.json"
    threshold.write_text('{"threshold":0.5}\n', encoding="utf-8")
    scaler = inputs / "scaler.json"
    scaler.write_text('{"center":[0.0],"scale":[1.0]}\n', encoding="utf-8")
    shot1 = inputs / "1.hdf5"
    shot2 = inputs / "2.hdf5"
    _write_minimal_outcome_hdf5(shot1, disruptive=True)
    _write_minimal_outcome_hdf5(shot2, disruptive=False)
    return {
        "output": tmp_path / "run",
        "config_path": config,
        "checkpoint_path": checkpoint,
        "parent_path": parent,
        "roster_path": roster,
        "threshold_receipt_path": threshold,
        "scaler_receipt_path": scaler,
        "hdf5_paths": [shot1, shot2],
        "code_files": code_files,
    }


def test_c_mut_04_real_two_shot_cpu_stage_resumes_without_rewriting_outputs(
    tmp_path: Path,
) -> None:
    fixture = _two_shot_resume_fixture(tmp_path)

    first = runner.run_two_shot_cpu_resume_stage(**fixture)
    before = _snapshot_tree(fixture["output"])
    second = runner.run_two_shot_cpu_resume_stage(**fixture)

    assert first["status"] == "completed"
    assert second["status"] == "resumed"
    assert _snapshot_tree(fixture["output"]) == before
    assert len(
        (fixture["output"] / "static_predictions.jsonl").read_text().splitlines()
    ) == 2


@pytest.mark.parametrize(
    ("component", "fixture_key", "list_index"),
    [
        ("code_closure_sha256", "code_files", 0),
        ("config_sha256", "config_path", None),
        ("checkpoint_sha256", "checkpoint_path", None),
        ("parent_sha256", "parent_path", None),
        ("roster_sha256", "roster_path", None),
        ("threshold_receipt_sha256", "threshold_receipt_path", None),
        ("scaler_receipt_sha256", "scaler_receipt_path", None),
        ("hdf5_contents_sha256", "hdf5_paths", 0),
    ],
)
def test_c_mut_04_one_byte_resume_drift_fails_before_any_existing_write(
    tmp_path: Path,
    component: str,
    fixture_key: str,
    list_index: int | None,
) -> None:
    fixture = _two_shot_resume_fixture(tmp_path)
    runner.run_two_shot_cpu_resume_stage(**fixture)
    before = _snapshot_tree(fixture["output"])
    value = fixture[fixture_key]
    target = value[list_index] if list_index is not None else value
    target.write_bytes(target.read_bytes() + b"x")

    with pytest.raises(runner.ResumeBindingMismatch, match=component):
        runner.run_two_shot_cpu_resume_stage(**fixture)

    assert _snapshot_tree(fixture["output"]) == before


def test_resume_code_closure_includes_source_and_preprocessing_dependencies() -> None:
    relative = {
        str(path.relative_to(ROOT)) for path in runner.selected_code_closure_paths()
    }
    assert {
        "experiments/transfer_methods_2026/train_jtext_source_transformer.py",
        "experiments/transfer_methods_2026/rebuilt_harness.py",
        "experiments/transfer_methods_2026/grid_search.py",
        "experiments/transfer_methods_2026/selected_mainline_contract.py",
        "experiments/transfer_methods_2026/run_two_timescale_online.py",
        "experiments/transfer_methods_2026/run_unified_online_active.py",
        "experiments/transfer_methods_2026/run_selected_full_mainline.py",
        "config/mainline_code_manifest.json",
    }.issubset(relative)


def _tiny_live_manifest(
    tmp_path: Path,
) -> tuple[Path, Path, dict, Path]:
    data_root = tmp_path / "data"
    relative = Path("J-TEXT/unified_hdf5/0/1.hdf5")
    target = data_root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"registered-hdf5-bytes")
    file_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    row = {
        "relative_path": relative.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": file_sha,
    }
    aggregate = hashlib.sha256(
        f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode()
    ).hexdigest()
    manifest = tmp_path / "hdf5_manifest.json"
    manifest.write_text(json.dumps({
        "content_manifest": {
            "algorithm": "sha256",
            "file_count": 1,
            "aggregate_sha256": aggregate,
            "files": [row],
        },
    }, sort_keys=True) + "\n", encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["data_contract"]["hdf5_content_manifest_sha256"] = (
        selected.sha256_file(manifest)
    )
    config["data_contract"]["hdf5_content_aggregate_sha256"] = aggregate
    config_path = tmp_path / "selected_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return data_root, manifest, config, config_path


def _track_runtime_root_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    for module, attribute in (
        (runner.common, "DATA"),
        (runner.rebuilt, "DATA"),
        (runner.source_trainer, "DATA_ROOT"),
        (runner.two_runner, "DATA"),
        (runner.two_runner.reverse_value, "DATA"),
        (runner.unified, "DATA"),
        (runner.unified.old_private, "DATA"),
    ):
        monkeypatch.setattr(module, attribute, getattr(module, attribute))
    monkeypatch.setenv(
        "FUSION_DATA_ROOT",
        "/media/hetaisheng/044A81D94A81C83E/Fuison/data",
    )


def test_formal_main_rejects_live_hdf5_byte_drift_before_any_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, manifest, config, config_path = _tiny_live_manifest(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "effective_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "prior-artifact.bin").write_bytes(b"immutable")
    before = _snapshot_tree(output)
    registered = data_root / "J-TEXT/unified_hdf5/0/1.hdf5"
    registered.write_bytes(registered.read_bytes() + b"x")
    _track_runtime_root_mutations(monkeypatch)
    monkeypatch.setattr(runner, "HDF5_CONTENT_MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", [
        str(runner.__file__),
        "--seed", "131",
        "--data-root", str(data_root),
        "--output", str(output),
        "--stage", "source-train",
        "--config", str(config_path),
    ])

    with pytest.raises(runner.ResumeBindingMismatch, match="HDF5 live content"):
        runner.main()

    assert _snapshot_tree(output) == before


def test_selected_runtime_rebinds_every_inprocess_reader_to_cli_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "explicit-cli-data"
    data_root.mkdir()
    _track_runtime_root_mutations(monkeypatch)

    receipt = runner.bind_runtime_data_root(data_root)

    assert set(receipt) == {
        "common.DATA",
        "rebuilt_harness.DATA",
        "train_jtext_source_transformer.DATA_ROOT",
        "run_two_timescale_online.DATA",
        "run_two_timescale_online.reverse_online_value.DATA",
        "run_unified_online_active.DATA",
        "run_unified_online_active.run_east_private23_online.DATA",
    }
    assert all(Path(value) == data_root.resolve() for value in receipt.values())
    assert runner.a1.rebuilt.DATA == data_root.resolve()
    assert runner.a1.alarm.GS.RH.DATA == data_root.resolve()


def test_formal_source_stage_refuses_unreceipted_preexisting_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    source_dir = output / "source-train"
    source_dir.mkdir(parents=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    (output / "effective_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save({"fabricated": torch.tensor([1.0])}, source_dir / "source_checkpoint.pt")
    before = _snapshot_tree(output)
    _track_runtime_root_mutations(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        str(runner.__file__),
        "--seed", "131",
        "--data-root", "/media/hetaisheng/044A81D94A81C83E/Fuison/data",
        "--output", str(output),
        "--stage", "source-train",
        "--config", str(CONFIG),
    ])

    with pytest.raises(
        runner.ResumeBindingMismatch,
        match="unreceipted pre-existing output",
    ):
        runner.main()

    assert _snapshot_tree(output) == before


def test_unified_production_static_gate_revalidates_before_outcome_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = [70_000 + index for index in range(360)]
    rows = [
        {
            "stream_index": index,
            "shot": shot,
            "alarmed": False,
            "risk_trace": np.asarray([0.1, 0.2], dtype=np.float32),
        }
        for index, shot in enumerate(roster)
    ]
    sealed = runner.seal_static_predictions_then_read_outcomes(
        rows=rows,
        seal_path=tmp_path / "static_predictions.jsonl",
        stage="east-static",
        expected_count=360,
        outcome_oracle_factory=lambda: (
            lambda _shot: {"is_disrupt": False, "down_time_s": None}
        ),
        threshold_receipt_sha256="a" * 64,
        parent_fingerprint_sha256="b" * 64,
    )
    events: list[str] = []
    original_validate = selected.validate_static_seal_manifest

    def tracked_validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        events.append("validated")
        return result

    def outcome_reader(_shot: int) -> dict:
        assert events and events[0] == "validated"
        events.append("outcome")
        return {"is_disrupt": False, "down_time_s": None}

    monkeypatch.setattr(
        unified.selected_mainline,
        "validate_static_seal_manifest",
        tracked_validate,
    )
    outcomes = unified.read_selected_static_outcomes_after_seal(
        static_seal_manifest=sealed["seal_manifest_path"],
        roster=roster,
        outcome_reader=outcome_reader,
    )

    assert len(outcomes) == 360
    assert events == ["validated"] + ["outcome"] * 360


def test_static_seal_does_not_monkeypatch_h5py_dataset_getitem(
    tmp_path: Path,
) -> None:
    original = h5py.Dataset.__getitem__
    result = runner.seal_static_predictions_then_read_outcomes(
        rows=[{
            "stream_index": 0,
            "shot": 1,
            "risk_trace": np.asarray([0.1], dtype=np.float32),
        }],
        seal_path=tmp_path / "static_predictions.jsonl",
        stage="jtext-static",
        expected_count=1,
        outcome_oracle_factory=lambda: (
            lambda _shot: {"is_disrupt": False, "down_time_s": None}
        ),
        threshold_receipt_sha256="a" * 64,
        parent_fingerprint_sha256="b" * 64,
    )

    assert result["outcomes"][1]["is_disrupt"] is False
    assert h5py.Dataset.__getitem__ is original
