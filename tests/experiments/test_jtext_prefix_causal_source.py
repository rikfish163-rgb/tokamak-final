from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

jio = importlib.import_module("jtext_online_io")
grid = importlib.import_module("grid_search")
selected = importlib.import_module("selected_mainline_contract")


def _raw(points: int = 20) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for index, name in enumerate(selected.SOURCE_RAW_SIGNAL_NAMES):
        row = np.linspace(1.0, 2.0, points, dtype=np.float32) + index
        if name == "Mir_avg_fre":
            row[:5] = 0.0
        if name == "mode_number_n":
            row[:9] = 0.0
        values[name] = row
    return values


def test_prefix_availability_waits_for_first_nonzero_evidence() -> None:
    values, availability = jio.prefix_causal_rebuild_features(_raw())
    mir = jio.REBUILD_FEATURE_NAMES.index("Mir_avg_fre")
    mode = jio.REBUILD_FEATURE_NAMES.index("mode_number_n")

    assert not availability[:5, mir].any()
    assert availability[5:, mir].all()
    assert not availability[:9, mode].any()
    assert availability[9:, mode].all()
    assert np.all(values[:5, mir] == 0.0)
    assert np.all(values[:9, mode] == 0.0)


def test_suffix_mutation_is_bitwise_invariant_through_embedding_and_risk() -> None:
    torch.manual_seed(131)
    endpoint = 10
    raw_a = _raw()
    raw_b = copy.deepcopy(raw_a)
    for name in ("Mir_avg_fre", "mode_number_n", "n=1 amplitude"):
        raw_b[name][endpoint + 1 :] = -777.0
    values_a, available_a = jio.prefix_causal_rebuild_features(raw_a)
    values_b, available_b = jio.prefix_causal_rebuild_features(raw_b)
    values_a = np.where(available_a, values_a, 0.0).astype(np.float32)
    values_b = np.where(available_b, values_b, 0.0).astype(np.float32)
    window_a, valid_a = jio.causal_prefix_window(values_a, endpoint, 128)
    window_b, valid_b = jio.causal_prefix_window(values_b, endpoint, 128)

    np.testing.assert_array_equal(values_a[: endpoint + 1], values_b[: endpoint + 1])
    np.testing.assert_array_equal(
        available_a[: endpoint + 1], available_b[: endpoint + 1],
    )
    np.testing.assert_array_equal(window_a, window_b)
    assert valid_a == valid_b == endpoint + 1
    encoder = grid.Enc(len(jio.REBUILD_FEATURE_NAMES)).eval()
    head = grid.Head().eval()
    with torch.no_grad():
        tensor_a = torch.from_numpy(window_a[None])
        tensor_b = torch.from_numpy(window_b[None])
        embedding_a = encoder(tensor_a, valid_lengths=torch.tensor([valid_a]))
        embedding_b = encoder(tensor_b, valid_lengths=torch.tensor([valid_b]))
        risk_a = torch.sigmoid(head(embedding_a))
        risk_b = torch.sigmoid(head(embedding_b))
    torch.testing.assert_close(embedding_a, embedding_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(risk_a, risk_b, rtol=0.0, atol=0.0)


def test_causal_window_uses_zero_left_padding_and_valid_length_mask() -> None:
    values = np.ones((4, len(jio.REBUILD_FEATURE_NAMES)), dtype=np.float32)
    window, valid = jio.causal_prefix_window(values, endpoint=3, window_points=128)

    assert valid == 4
    assert np.count_nonzero(window[:-4]) == 0
    np.testing.assert_array_equal(window[-4:], values)
    torch.manual_seed(133)
    encoder = grid.Enc(len(jio.REBUILD_FEATURE_NAMES)).eval()
    x = torch.from_numpy(window[None])
    with torch.no_grad():
        masked = encoder(x, valid_lengths=torch.tensor([valid]))
        unmasked = encoder(x)
    assert not torch.equal(masked, unmasked)


def test_input_contract_sha_is_recomputed_from_real_preprocessing_closure() -> None:
    contract = jio.build_source_input_contract(ROOT)
    current = jio.build_preprocessing_manifest(ROOT)

    assert contract["schema"] == "jtext-rebuilt17-prefix-causal-v2"
    assert contract["preprocessing_manifest"] == current
    assert contract["preprocessing_sha256"] == selected.sha256_json(current)
    payload = {"input_contract": contract}
    assert selected.validate_source_input_contract(payload) == contract

    arbitrary = copy.deepcopy(contract)
    arbitrary["preprocessing_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        selected.validate_source_input_contract({"input_contract": arbitrary})


def test_input_contract_rejects_manifest_file_hash_drift() -> None:
    contract = jio.build_source_input_contract(ROOT)
    drifted = copy.deepcopy(contract)
    drifted["preprocessing_manifest"]["files"][0]["sha256"] = "0" * 64
    drifted["preprocessing_sha256"] = selected.sha256_json(
        drifted["preprocessing_manifest"]
    )

    with pytest.raises(ValueError, match="preprocessing manifest"):
        selected.validate_source_input_contract({"input_contract": drifted})
