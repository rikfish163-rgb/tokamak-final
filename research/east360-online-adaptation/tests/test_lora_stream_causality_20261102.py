"""CPU-only, synthetic end-to-end checks of the LoRA stream's causal contract."""

import hashlib
import re

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import source_basis
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.lora_stream_20261102.run import predict
from experiments.risk_aligned_online.simple23 import Config, Simple23


ARMS = ("cold", "real", "reverse", "hold")
FEEDBACK_ARMS = ARMS[1:]
RECORD_FIELDS = {
    "origin",
    "target_first",
    "target_last",
    "risk_sealed_through",
    "first_affected",
    "forecast_sha256",
    "mask_sha256",
    "target_sha256",
    "state256_before",
    "state256_after",
    "gradient256",
    "gradient_norm",
    "observed",
    "accepted",
}


@pytest.fixture(scope="module")
def case():
    old_threads = torch.get_num_threads()
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    old_rng = torch.get_rng_state()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(37)
    model = ContextAdapter(
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            np.zeros(23),
            np.ones(23),
        )
    ).cpu().eval().requires_grad_(False)
    rng = np.random.default_rng(38)
    signal = {
        "values": rng.normal(size=(384, 23)).astype(np.float32),
        "available": rng.random((384, 23)) > 0.05,
    }
    try:
        yield model, signal, source_basis(model)
    finally:
        torch.set_rng_state(old_rng)
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
        torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def full(case):
    model, signal, basis = case
    return predict(model, signal, basis)


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def expected_feedback(signal, origin, end):
    """Independently reconstruct the established common-support controls."""
    values = signal["values"]
    available = signal["available"]
    future = values[origin + 1 : end + 1]
    mask = available[origin + 1 : end + 1].copy()
    mask &= available[origin]
    mask[:, 10] = False  # The discrete mode channel is not a forecast target.
    mask &= mask[::-1].copy()
    targets = {
        "real": future,
        "reverse": future[::-1],
        "hold": np.repeat(values[origin : origin + 1], 25, axis=0),
    }
    return mask, targets


def assert_curves_equal(left, right, stop):
    assert set(left) == set(right) == set(ARMS)
    for arm in ARMS:
        np.testing.assert_array_equal(left[arm][:stop], right[arm][:stop], err_msg=arm)


def test_receipt_clock_and_arm_feedback_support(case, full):
    _, signal, _ = case
    curves, receipt = full
    assert set(curves) == set(ARMS)
    assert receipt["reset_each_shot"] is True
    assert receipt["temporary_parameters"] == 256
    assert set(receipt["updates"]) == set(FEEDBACK_ARMS)
    for arm in ARMS:
        assert np.asarray(curves[arm]).shape == (384,)
        np.testing.assert_array_equal(curves[arm][:160], curves["cold"][:160])

    for arm in FEEDBACK_ARMS:
        records = receipt["updates"][arm]
        assert isinstance(records, list)
        assert [r["target_last"] for r in records] == [159, 287]
        # Make sure the synthetic causality/reset checks exercise an actual
        # first update in each arm rather than passing on an always-cold path.
        assert records[0]["accepted"]
        assert np.any(curves[arm][160:288] != curves["cold"][160:288])
        np.testing.assert_array_equal(records[0]["state256_before"], np.zeros(256))
        for index, (record, end) in enumerate(zip(records, (159, 287))):
            assert RECORD_FIELDS <= set(record)
            origin = end - 25
            assert record["origin"] == origin
            assert record["target_first"] == origin + 1
            assert record["target_last"] == record["risk_sealed_through"] == end
            assert record["first_affected"] == end + 1
            mask, targets = expected_feedback(signal, origin, end)
            assert mask.any()
            assert record["observed"] == int(mask.sum())
            assert record["mask_sha256"] == digest(mask)
            assert record["target_sha256"] == digest(targets[arm])
            assert re.fullmatch(r"[0-9a-f]{64}", record["forecast_sha256"])
            if index:
                np.testing.assert_array_equal(
                    record["state256_before"], records[index - 1]["state256_after"]
                )


def test_same_input_restarts_temporary_state_exactly(case, full):
    model, signal, basis = case
    repeated, repeated_receipt = predict(model, signal, basis)
    curves, receipt = full
    assert_curves_equal(curves, repeated, 384)
    assert repeated_receipt == receipt


@pytest.mark.parametrize("extra_key", ["metadata", "sample_time", "unexpected"])
def test_signal_rejects_any_extra_metadata(case, extra_key):
    model, signal, basis = case
    with pytest.raises(ValueError):
        predict(model, {**signal, extra_key: 1}, basis)


def test_same_length_future_change_cannot_affect_prior_batch_positions(case, full):
    model, signal, basis = case
    changed = {key: value.copy() for key, value in signal.items()}
    changed["values"][263:] += 1000.0
    changed["available"][263:] = ~changed["available"][263:]
    altered, altered_receipt = predict(model, changed, basis)
    curves, receipt = full
    assert_curves_equal(curves, altered, 263)
    for arm in FEEDBACK_ARMS:
        before = receipt["updates"][arm]
        after = altered_receipt["updates"][arm]
        assert len(before) == len(after) == 2
        assert before[0] == after[0]
        # The second forecast was issued at 262, before any changed sample.
        assert before[1]["origin"] == after[1]["origin"] == 262
        assert before[1]["forecast_sha256"] == after[1]["forecast_sha256"]
        np.testing.assert_array_equal(before[1]["state256_before"], after[1]["state256_before"])


def test_truncation_to_256_preserves_every_prefix_prediction(case, full):
    model, signal, basis = case
    truncated = {key: value[:256].copy() for key, value in signal.items()}
    prefix, prefix_receipt = predict(model, truncated, basis)
    curves, receipt = full
    assert_curves_equal(curves, prefix, 256)
    for arm in FEEDBACK_ARMS:
        assert prefix_receipt["updates"][arm] == receipt["updates"][arm][:1]


@pytest.mark.parametrize("length", [1, 127, 159])
def test_shorter_than_160_has_no_updates_and_all_arms_are_cold(case, length):
    model, signal, basis = case
    short = {key: value[:length].copy() for key, value in signal.items()}
    curves, receipt = predict(model, short, basis)
    assert set(curves) == set(ARMS)
    assert set(receipt["updates"]) == set(FEEDBACK_ARMS)
    for arm in ARMS:
        assert np.asarray(curves[arm]).shape == (length,)
        np.testing.assert_array_equal(curves[arm], curves["cold"])
    assert all(receipt["updates"][arm] == [] for arm in FEEDBACK_ARMS)


def test_exact_160_boundary_seals_update_without_altering_prefix(case, full):
    model, signal, basis = case
    signal160 = {key: value[:160].copy() for key, value in signal.items()}
    curves, receipt = predict(model, signal160, basis)
    reference, reference_receipt = full
    assert_curves_equal(curves, reference, 160)
    for arm in FEEDBACK_ARMS:
        np.testing.assert_array_equal(curves[arm], curves["cold"])
        assert receipt["updates"][arm] == reference_receipt["updates"][arm][:1]


def test_no_observed_feedback_has_zero_gradient_and_never_updates(case):
    model, signal, basis = case
    unobserved = {key: value.copy() for key, value in signal.items()}
    unobserved["available"][135:160] = False
    unobserved["available"][263:288] = False
    curves, receipt = predict(model, unobserved, basis)
    assert set(curves) == set(ARMS)
    assert set(receipt["updates"]) == set(FEEDBACK_ARMS)
    for arm in FEEDBACK_ARMS:
        np.testing.assert_array_equal(curves[arm], curves["cold"])
        records = receipt["updates"][arm]
        assert [record["target_last"] for record in records] == [159, 287]
        for record in records:
            assert record["observed"] == 0
            assert record["accepted"] is False
            assert record["gradient_norm"] == 0
            np.testing.assert_array_equal(record["gradient256"], np.zeros(256))
            np.testing.assert_array_equal(record["state256_before"], record["state256_after"])
