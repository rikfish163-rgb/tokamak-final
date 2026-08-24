from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import hashlib
import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


selected = importlib.import_module("selected_mainline_contract")
grid = importlib.import_module("grid_search")
source_trainer = importlib.import_module("train_jtext_source_transformer")
jtext_io = importlib.import_module("jtext_online_io")
a1 = importlib.import_module("run_online_active_a1")
auxiliary = importlib.import_module("online_auxiliary")


def _literal_threshold_receipt(
    *, scope: str, roster: list[int], chosen_threshold: float,
    target_tpr: float,
) -> dict:
    predictions = [
        {
            "shot": int(shot),
            "score": float(chosen_threshold if index % 2 == 0 else 0.1),
            "outcome": bool(index % 2 == 0),
        }
        for index, shot in enumerate(roster)
    ]
    body = {
        "schema_version": "selected-threshold-receipt-v1",
        "scope": scope,
        "roster": [int(shot) for shot in roster],
        "roster_sha256": selected.sha256_json([int(shot) for shot in roster]),
        "data_manifest_sha256": "6" * 64,
        "outcome_manifest_sha256": "7" * 64,
        "selection_rule": (
            "meet target TPR, then minimize FPR, then maximize finite threshold"
        ),
        "target_tpr": float(target_tpr),
        "predictions": predictions,
        "candidates": [
            {
                "threshold": float(chosen_threshold),
                "tpr": 1.0,
                "fpr": 0.0,
            },
            {
                "threshold": 0.1,
                "tpr": 1.0,
                "fpr": 1.0,
            },
        ],
        "chosen_threshold": float(chosen_threshold),
    }
    return {**body, "receipt_sha256": selected.sha256_json(body)}


def _input_contract() -> dict:
    return jtext_io.build_source_input_contract(ROOT)


def _source_payload(*, seed: int = 131) -> dict:
    torch.manual_seed(seed)
    encoder = grid.Enc(len(grid.RH.REBUILD))
    head = grid.Head()
    roles = source_trainer.load_canonical_jtext_roles(ROOT)
    advanced_time_path = ROOT / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json"
    validation_receipt = _literal_threshold_receipt(
        scope="J-TEXT VAL160",
        roster=[int(shot) for shot in roles["val"]],
        chosen_threshold=0.5,
        target_tpr=0.90,
    )
    return {
        "schema_version": grid.SOURCE_CHECKPOINT_SCHEMA,
        "architecture_version": grid.SOURCE_ARCHITECTURE_VERSION,
        "feature_names": list(grid.RH.REBUILD),
        "model_config": grid.source_model_config(len(grid.RH.REBUILD)),
        "seed": seed,
        "epoch": 1,
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "scaler_center": torch.zeros(len(grid.RH.REBUILD)),
        "scaler_scale": torch.ones(len(grid.RH.REBUILD)),
        "split_counts": {"train": 1353, "val": 160, "test": 220},
        "split_sha256": source_trainer.split_sha256(roles),
        "input_contract": _input_contract(),
        "training_label": {
            "policy": "advanced_floor",
            "outcome_authority": "HDF5 meta/IsDisrupt",
            "down_time_authority": "HDF5 meta/DownTime",
            "advanced_time_role": "positive-window upper boundary only",
            "advanced_time_used": True,
            "advanced_time_sha256": hashlib.sha256(
                advanced_time_path.read_bytes()
            ).hexdigest(),
            "formula": "upper_ms=max(25, AdvancedTime_ms); missing->25",
            "audit": {
                "device": "J-TEXT",
                "policy": "advanced_floor",
                "unit": "ms",
                "floor_ms": 25.0,
            },
        },
        "validation_threshold": 0.5,
        "validation_threshold_receipt": validation_receipt,
        "test_shards_read": 0,
    }


def _write_source(path: Path, payload: dict | None = None) -> dict:
    value = _source_payload() if payload is None else payload
    torch.save(value, path)
    return value


def _contract(source_path: Path, payload: dict) -> selected.SelectedMainlineContract:
    target_scaler = {"center": [0.0, 0.0], "scale": [1.0, 1.0]}
    raw8_scaler = {"center": [0.0], "scale": [1.0]}
    physics_scales = {"ip": 1.0}
    reconstruction_constants = {
        "minimum_positive_residual": 0.1,
        "positive_support_windows": 1,
        "shared_reconstruction_q80": 0.2,
        "private_reconstruction_q80": 0.3,
    }
    east_receipt = _literal_threshold_receipt(
        scope="EAST support110", roster=[101, 102],
        chosen_threshold=0.42, target_tpr=0.80,
    )
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
        source_checkpoint_sha256=selected.sha256_file(source_path),
        source_label_policy="advanced_floor",
        source_label_map_sha256=payload["training_label"]["advanced_time_sha256"],
        source_split_sha256=payload["split_sha256"],
        source_scaler_sha256=selected.scaler_sha256(
            payload["scaler_center"], payload["scaler_scale"]
        ),
        source_input_contract_sha256=selected.sha256_json(
            payload["input_contract"]
        ),
        source_preprocessing_sha256=payload["input_contract"][
            "preprocessing_sha256"
        ],
        source_val_threshold_receipt_sha256=(
            payload["validation_threshold_receipt"]["receipt_sha256"]
        ),
        east_target_label_policy="fixed125",
        east_target_horizon_ms=125.0,
        east_support_roster_sha256=east_receipt["roster_sha256"],
        east_support_data_manifest_sha256=east_receipt["data_manifest_sha256"],
        east_support_outcome_manifest_sha256=(
            east_receipt["outcome_manifest_sha256"]
        ),
        east_threshold_receipt_sha256=east_receipt["receipt_sha256"],
        target_scaler_sha256=selected.sha256_json(target_scaler),
        raw8_scaler_sha256=selected.sha256_json(raw8_scaler),
        physics_scales_sha256=selected.sha256_json(physics_scales),
        reconstruction_constants_sha256=selected.sha256_json(
            reconstruction_constants
        ),
        jtext_test_roster_sha256=selected.sha256_json(list(range(220))),
        east_test_roster_sha256=selected.sha256_json(list(range(360))),
        claim_status="TEST_DRIVEN_EXPLORATION",
    )


def _parent_payload(
    contract: selected.SelectedMainlineContract,
) -> dict:
    encoder = {"weight": torch.tensor([1.0])}
    head = {"weight": torch.tensor([2.0])}
    shared = {"weight": torch.tensor([3.0])}
    private = {"weight": torch.tensor([4.0])}
    target_scaler = {"center": [0.0, 0.0], "scale": [1.0, 1.0]}
    raw8_scaler = {"center": [0.0], "scale": [1.0]}
    physics_scales = {"ip": 1.0}
    reconstruction_constants = {
        "minimum_positive_residual": 0.1,
        "positive_support_windows": 1,
        "shared_reconstruction_q80": 0.2,
        "private_reconstruction_q80": 0.3,
    }
    source_receipt = _literal_threshold_receipt(
        scope="J-TEXT VAL160",
        roster=[
            int(shot)
            for shot in source_trainer.load_canonical_jtext_roles(ROOT)["val"]
        ],
        chosen_threshold=0.5, target_tpr=0.90,
    )
    east_receipt = _literal_threshold_receipt(
        scope="EAST support110", roster=[101, 102],
        chosen_threshold=0.42, target_tpr=0.80,
    )
    payload = {
        "schema_version": "dynamic-source-east-fixed-parent-v1",
        "selected_label_arm": "source_dynamic__east_fixed",
        "lineage": {"source": "source_dynamic", "target": "east_fixed"},
        **contract.to_dict(),
        "seed": contract.seed,
        "source_checkpoint_sha256": contract.source_checkpoint_sha256,
        "source_label_policy": "advanced_floor",
        "source_label_map_sha256": contract.source_label_map_sha256,
        "source_split_sha256": contract.source_split_sha256,
        "source_scaler_sha256": contract.source_scaler_sha256,
        "source_input_contract_sha256": contract.source_input_contract_sha256,
        "source_preprocessing_sha256": contract.source_preprocessing_sha256,
        "east_target_label_policy": "fixed125",
        "east_target_horizon_ms": 125.0,
        "east_support_roster_sha256": contract.east_support_roster_sha256,
        "east_support_data_manifest_sha256": (
            contract.east_support_data_manifest_sha256
        ),
        "east_support_outcome_manifest_sha256": (
            contract.east_support_outcome_manifest_sha256
        ),
        "source_val_threshold_receipt_sha256": (
            contract.source_val_threshold_receipt_sha256
        ),
        "east_threshold_receipt_sha256": contract.east_threshold_receipt_sha256,
        "source_val_threshold_receipt": source_receipt,
        "east_threshold_receipt": east_receipt,
        "east_threshold": 0.42,
        "target_scaler": target_scaler,
        "target_scaler_sha256": selected.sha256_json(target_scaler),
        "raw8_scaler": raw8_scaler,
        "raw8_scaler_sha256": selected.sha256_json(raw8_scaler),
        "physics_scales": physics_scales,
        "physics_scales_sha256": selected.sha256_json(physics_scales),
        "reconstruction_constants": reconstruction_constants,
        "reconstruction_constants_sha256": selected.sha256_json(
            reconstruction_constants
        ),
        "encoder": encoder,
        "head": head,
        "shared": shared,
        "private": private,
        "target_encoder_sha256": selected.state_dict_sha256(encoder),
        "target_head_sha256": selected.state_dict_sha256(head),
        "target_base_sha256": selected.state_bundle_sha256(encoder, head),
        "shared_parent_sha256": selected.state_dict_sha256(shared),
        "private_parent_sha256": selected.state_dict_sha256(private),
        "claim_status": "TEST_DRIVEN_EXPLORATION",
    }
    payload["canonical_parent_fingerprint_sha256"] = (
        selected.canonical_parent_fingerprint(payload)
    )
    return payload


def test_selected_source_requires_an_explicit_existing_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit source checkpoint"):
        selected.load_selected_source_checkpoint(None, 131)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        selected.load_selected_source_checkpoint(tmp_path / "missing.pt", 131)
    with pytest.raises(ValueError, match="explicit source checkpoint"):
        grid.load_source_checkpoint(
            None, channels=len(grid.RH.REBUILD), expected_seed=131,
            device="cpu", allow_fallback=False,
        )


def test_selected_source_rejects_fixed_label_policy(tmp_path: Path) -> None:
    payload = _source_payload()
    payload["training_label"]["policy"] = "fixed125"
    path = tmp_path / "source.pt"
    _write_source(path, payload)

    with pytest.raises(ValueError, match="advanced_floor"):
        selected.load_selected_source_checkpoint(path, 131)


def test_selected_source_rejects_old_checkpoint_without_input_contract(
    tmp_path: Path,
) -> None:
    payload = _source_payload()
    del payload["input_contract"]
    path = tmp_path / "old-dynamic-source.pt"
    _write_source(path, payload)

    with pytest.raises(ValueError, match="input contract"):
        selected.load_selected_source_checkpoint(path, 131)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema", "jtext-rebuilt17-prefix-causal-v1"),
        ("raw_signal_names", ["Bt", "ip"]),
        ("feature_names", list(reversed(grid.RH.REBUILD))),
        ("window_points", 64),
        ("history_alignment", "left"),
        ("left_padding", "repeat_first"),
        ("valid_lengths_required", False),
        ("valid_length_range", [0, 128]),
        ("patch_mask_rule", "none"),
        ("availability", "finite-whole-shot-v1"),
        ("preprocessing_sha256", None),
    ],
)
def test_selected_source_rejects_input_contract_drift(
    tmp_path: Path, field: str, invalid,
) -> None:
    payload = _source_payload()
    payload["input_contract"][field] = invalid
    path = tmp_path / f"invalid-{field}.pt"
    _write_source(path, payload)

    with pytest.raises(ValueError, match="input contract"):
        selected.load_selected_source_checkpoint(path, 131)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(seed=133), "seed"),
        (lambda value: value.update(split_sha256="0" * 64), "split"),
        (
            lambda value: value["training_label"].update(
                advanced_time_sha256="0" * 64
            ),
            "label-map",
        ),
        (lambda value: value.update(test_shards_read=1), "TEST220"),
        (
            lambda value: value.update(
                scaler_scale=torch.zeros(len(grid.RH.REBUILD))
            ),
            "scaler",
        ),
    ],
)
def test_selected_source_rejects_lineage_drift(
    tmp_path: Path, mutation, message: str,
) -> None:
    payload = _source_payload()
    mutation(payload)
    path = tmp_path / "source.pt"
    _write_source(path, payload)

    with pytest.raises(ValueError, match=message):
        selected.load_selected_source_checkpoint(path, 131)


def test_selected_source_accepts_complete_dynamic_lineage(tmp_path: Path) -> None:
    path = tmp_path / "source.pt"
    payload = _write_source(path)

    encoder, head, loaded = selected.load_selected_source_checkpoint(path, 131)

    assert loaded["training_label"]["policy"] == "advanced_floor"
    assert loaded["source_checkpoint_sha256"] == selected.sha256_file(path)
    assert loaded["source_label_map_sha256"] == (
        payload["training_label"]["advanced_time_sha256"]
    )
    assert loaded["source_scaler_sha256"] == selected.scaler_sha256(
        payload["scaler_center"], payload["scaler_scale"]
    )
    assert loaded["source_input_contract_sha256"] == selected.sha256_json(
        payload["input_contract"]
    )
    assert loaded["source_preprocessing_sha256"] == (
        payload["input_contract"]["preprocessing_sha256"]
    )
    assert encoder.state_dict().keys() == payload["encoder"].keys()
    assert head.state_dict().keys() == payload["head"].keys()


def test_selected_contract_is_immutable_and_rejects_missing_source_sha() -> None:
    values = {
        "seed": 131,
        "selection_role": "post_adjudication_user_selected_development_candidate",
        "historical_admission": "FAIL",
        "confirmation_eligible": False,
        "corrected_ablation_report_sha256": (
            "896584467254d2ae1df894f6f4a0e4ed3a024ec556dcc1ffeb7e083f04db3d04"
        ),
        "selection_metric": "mean balanced360 operational_TPR - FPR",
        "historical_failure_reason": "seed133 TPR@30 < 0.80",
        "source_checkpoint_sha256": "a" * 64,
        "source_label_policy": "advanced_floor",
        "source_label_map_sha256": "b" * 64,
        "source_split_sha256": "c" * 64,
        "source_scaler_sha256": "d" * 64,
        "source_input_contract_sha256": "1" * 64,
        "source_preprocessing_sha256": "2" * 64,
        "source_val_threshold_receipt_sha256": "3" * 64,
        "east_target_label_policy": "fixed125",
        "east_target_horizon_ms": 125.0,
        "east_support_roster_sha256": "e" * 64,
        "east_support_data_manifest_sha256": "4" * 64,
        "east_support_outcome_manifest_sha256": "5" * 64,
        "east_threshold_receipt_sha256": "6" * 64,
        "target_scaler_sha256": "7" * 64,
        "raw8_scaler_sha256": "8" * 64,
        "physics_scales_sha256": "9" * 64,
        "reconstruction_constants_sha256": "0" * 64,
        "claim_status": "TEST_DRIVEN_EXPLORATION",
        "jtext_test_roster_sha256": "a" * 64,
        "east_test_roster_sha256": "b" * 64,
    }
    contract = selected.SelectedMainlineContract(**values)
    with pytest.raises(FrozenInstanceError):
        contract.seed = 133  # type: ignore[misc]
    with pytest.raises(ValueError, match="source_checkpoint_sha256"):
        selected.SelectedMainlineContract(
            **{**values, "source_checkpoint_sha256": None}
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("selection_role", "historical_primary", "selection role"),
        ("historical_admission", "PASS", "historical admission"),
        ("confirmation_eligible", True, "confirmation"),
        ("corrected_ablation_report_sha256", "f" * 64, "ablation"),
        ("selection_metric", "AUC", "selection metric"),
        ("historical_failure_reason", "none", "failure reason"),
    ],
)
def test_selected_contract_rejects_post_adjudication_role_drift(
    tmp_path: Path, field: str, invalid, message: str,
) -> None:
    source_path = tmp_path / "source.pt"
    payload = _write_source(source_path)
    values = _contract(source_path, payload).to_dict()

    with pytest.raises(ValueError, match=message):
        selected.SelectedMainlineContract(**{**values, field: invalid})


def test_threshold_receipt_binds_roster_predictions_rule_and_manifests() -> None:
    predictions = [
        {"shot": 101, "score": 0.9, "outcome": True},
        {"shot": 102, "score": 0.1, "outcome": False},
    ]
    receipt = selected.build_threshold_receipt(
        scope="EAST support110",
        predictions=predictions,
        data_manifest_sha256="6" * 64,
        outcome_manifest_sha256="7" * 64,
        target_tpr=0.80,
    )

    validated = selected.validate_threshold_receipt(
        receipt,
        expected_scope="EAST support110",
        expected_roster=[101, 102],
        expected_threshold=0.9,
    )
    assert validated["chosen_threshold"] == pytest.approx(0.9)
    assert validated["receipt_sha256"] == receipt["receipt_sha256"]
    assert validated["selection_rule"] == (
        "meet target TPR, then minimize FPR, then maximize finite threshold"
    )

    for field, value in (
        ("chosen_threshold", 0.8),
        ("data_manifest_sha256", "0" * 64),
        ("receipt_sha256", "1" * 64),
    ):
        drifted = copy.deepcopy(receipt)
        drifted[field] = value
        with pytest.raises((ValueError, RuntimeError), match="threshold receipt"):
            selected.validate_threshold_receipt(
                drifted,
                expected_scope="EAST support110",
                expected_roster=[101, 102],
                expected_threshold=0.9,
            )


def test_threshold_receipt_rejects_an_incomplete_recomputed_candidate_set() -> None:
    receipt = selected.build_threshold_receipt(
        scope="EAST support110",
        predictions=[
            {"shot": 101, "score": 0.9, "outcome": True},
            {"shot": 102, "score": 0.1, "outcome": False},
        ],
        data_manifest_sha256="6" * 64,
        outcome_manifest_sha256="7" * 64,
        target_tpr=0.80,
    )
    incomplete = copy.deepcopy(receipt)
    incomplete["candidates"] = [incomplete["candidates"][0]]
    body = {key: value for key, value in incomplete.items() if key != "receipt_sha256"}
    incomplete["receipt_sha256"] = selected.sha256_json(body)

    with pytest.raises(RuntimeError, match="candidate recomputation mismatch"):
        selected.validate_threshold_receipt(
            incomplete,
            expected_scope="EAST support110",
            expected_roster=[101, 102],
            expected_threshold=0.9,
        )


@pytest.mark.parametrize(
    "missing",
    ["jtext_test_roster_sha256", "east_test_roster_sha256"],
)
def test_selected_contract_requires_full_chain_test_roster_hashes(
    tmp_path: Path, missing: str,
) -> None:
    source_path = tmp_path / "source.pt"
    source_payload = _write_source(source_path)
    values = _contract(source_path, source_payload).to_dict()
    values.pop(missing)

    with pytest.raises(ValueError, match=missing):
        selected.SelectedMainlineContract.from_dict(values)


def test_canonical_state_hash_matches_parent_writer_and_unified_helpers() -> None:
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(2, 1)

    assert selected.state_dict_sha256(first.state_dict()) == a1.model_sha256(first)
    assert selected.state_dict_sha256(first.state_dict()) == auxiliary.module_sha256(
        first
    )
    assert selected.state_bundle_sha256(
        first.state_dict(), second.state_dict()
    ) == a1.model_sha256(first, second)


def test_selected_parent_rejects_legacy_schema_and_source_sha_drift(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pt"
    source_payload = _write_source(source_path)
    contract = _contract(source_path, source_payload)
    parent = _parent_payload(contract)

    legacy = {**parent, "schema_version": "equal-label-budget-frontload-v2"}
    with pytest.raises(RuntimeError, match="schema"):
        selected.validate_selected_parent(legacy, contract)

    mismatched = {**parent, "source_checkpoint_sha256": "f" * 64}
    with pytest.raises(RuntimeError, match="source checkpoint"):
        selected.validate_selected_parent(mismatched, contract)


def test_selected_parent_rejects_policy_or_state_hash_drift(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pt"
    source_payload = _write_source(source_path)
    contract = _contract(source_path, source_payload)
    parent = _parent_payload(contract)

    with pytest.raises(RuntimeError, match="target label policy"):
        selected.validate_selected_parent(
            {**parent, "east_target_label_policy": "advanced_floor"}, contract,
        )
    with pytest.raises(RuntimeError, match="target encoder hash"):
        selected.validate_selected_parent(
            {**parent, "target_encoder_sha256": "0" * 64}, contract,
        )
    drifted_preprocessing = copy.deepcopy(parent)
    drifted_preprocessing["physics_scales"]["ip"] = 2.0
    with pytest.raises(RuntimeError, match="physics scales"):
        selected.validate_selected_parent(drifted_preprocessing, contract)
    drifted_receipt = copy.deepcopy(parent)
    drifted_receipt["east_threshold_receipt"]["chosen_threshold"] = 0.41
    with pytest.raises(RuntimeError, match="threshold receipt"):
        selected.validate_selected_parent(drifted_receipt, contract)


def test_valid_source_dynamic_east_fixed_parent_is_accepted(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pt"
    source_payload = _write_source(source_path)
    contract = _contract(source_path, source_payload)
    parent = _parent_payload(contract)

    lineage = selected.validate_selected_parent(parent, contract)

    assert lineage == {
        "schema_version": "dynamic-source-east-fixed-parent-v1",
        "threshold": 0.42,
        "expected_base_sha256": parent["target_base_sha256"],
        "expected_encoder_sha256": parent["target_encoder_sha256"],
        "expected_head_sha256": parent["target_head_sha256"],
        "expected_shared_sha256": parent["shared_parent_sha256"],
        "expected_private_sha256": parent["private_parent_sha256"],
        "east_threshold_receipt_sha256": contract.east_threshold_receipt_sha256,
        "claim_status": "TEST_DRIVEN_EXPLORATION",
    }
