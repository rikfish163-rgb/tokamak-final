import numpy as np
import pytest

from src.atlas23.contract import (
    CIII_MISSING_ATOL as VERIFIED_CIII_MISSING_ATOL,
    CIII_MISSING_VALUE as VERIFIED_CIII_MISSING_VALUE,
)
from experiments.risk_aligned_online.causal_input import (
    PrefixAlignment,
    CIII_MISSING_ATOL,
    CIII_MISSING_VALUE,
    apply_input_contract,
)


def _source_stats():
    return np.arange(23, dtype=np.float64), np.full(23, 2.0, dtype=np.float64)


def test_candidate_defaults_match_verified_local_contract():
    assert CIII_MISSING_VALUE == VERIFIED_CIII_MISSING_VALUE == -10.0
    assert CIII_MISSING_ATOL == VERIFIED_CIII_MISSING_ATOL == 1e-2


def test_apply_input_contract_is_pointwise_and_does_not_mutate_input():
    values = np.zeros((4, 23), dtype=np.float32)
    values[1, 1] = -10.0
    values[3, 1] = -9.9997
    values[2, 4] = -10.0
    available = np.ones_like(values, dtype=bool)
    shot = {"shot": 7, "device": "EAST", "values": values, "available": available}
    original_values = values.copy()
    original_available = available.copy()

    changed = apply_input_contract(shot)

    assert changed is not shot
    assert changed["values"] is not shot["values"]
    assert changed["available"] is not shot["available"]
    assert not changed["available"][1, 1]
    assert not changed["available"][3, 1]
    assert changed["available"][0, 1]
    assert changed["available"][2, 4]
    np.testing.assert_array_equal(changed["values"], original_values)
    np.testing.assert_array_equal(changed["available"][:, 0], original_available[:, 0])
    np.testing.assert_array_equal(shot["values"], original_values)
    np.testing.assert_array_equal(shot["available"], original_available)


def test_real_zero_and_unclassified_channels_are_preserved():
    values = np.zeros((5, 23), dtype=np.float32)
    values[:, 5] = -10.0
    values[1, 1] = -10.011
    values[2, 1] = -9.989
    available = np.ones_like(values, dtype=bool)
    shot = {"values": values, "available": available}

    changed = apply_input_contract(shot)

    assert changed["available"][0, 1]
    assert changed["available"][1, 1]
    assert changed["available"][2, 1]
    assert changed["available"][3:, 1].all()
    assert changed["available"][:, 5].all()
    assert changed["values"][0, 1] == 0.0
    np.testing.assert_array_equal(changed["values"], values)


def test_prefix_alignment_uses_only_first_prefix_observations_and_is_frozen():
    center, scale = _source_stats()
    prefix = np.zeros((6, 23), dtype=np.float32)
    prefix[:, 0] = [10.0, 11.0, 12.0, 13.0, 10_000.0, -10_000.0]
    prefix_available = np.ones_like(prefix, dtype=bool)
    current = np.array([[10.0] + [0.0] * 22, [13.0] + [0.0] * 22], dtype=np.float32)
    current_available = np.ones_like(current, dtype=bool)
    original_current = current.copy()
    original_current_available = current_available.copy()

    adapter = PrefixAlignment(center, scale, channels=[0], min_points=4)
    adapter.fit_prefix(prefix, prefix_available)
    first, first_audit = adapter.transform(current, current_available)

    changed_prefix = prefix.copy()
    changed_prefix[:4, 0] = [500.0, 501.0, 502.0, 503.0]
    adapter.fit_prefix(changed_prefix, prefix_available)
    second, second_audit = adapter.transform(current, current_available)

    np.testing.assert_allclose(first, second)
    assert first_audit["channels"][0]["status"] == "aligned"
    assert second_audit["channels"][0]["frozen"] is True
    np.testing.assert_array_equal(prefix[:, 0], [10.0, 11.0, 12.0, 13.0, 10_000.0, -10_000.0])
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(current_available, original_current_available)


def test_prefix_alignment_waits_until_min_points_then_starts():
    center, scale = _source_stats()
    prefix = np.zeros((3, 23), dtype=np.float32)
    prefix[:, 0] = [2.0, 4.0, 6.0]
    available = np.ones_like(prefix, dtype=bool)
    current = prefix.copy()

    adapter = PrefixAlignment(center, scale, channels=[0], min_points=3)
    adapter.fit_prefix(prefix[:2], available[:2])
    waiting, waiting_audit = adapter.transform(current, available)
    np.testing.assert_array_equal(waiting, current)
    assert waiting_audit["channels"][0]["status"] == "insufficient_prefix"
    assert waiting_audit["ready"] is False

    adapter.fit_prefix(prefix, available)
    aligned, aligned_audit = adapter.transform(current, available)
    assert aligned_audit["channels"][0]["status"] == "aligned"
    assert aligned[0, 0] != current[0, 0]


def test_constant_channel_is_not_calibrated_and_is_recorded():
    center, scale = _source_stats()
    prefix = np.zeros((4, 23), dtype=np.float32)
    prefix[:, 0] = 7.0
    available = np.ones_like(prefix, dtype=bool)
    current = prefix.copy()
    current[1, 0] = 9.0

    adapter = PrefixAlignment(center, scale, channels=[0], min_points=4)
    adapter.fit_prefix(prefix, available)
    mapped, audit = adapter.transform(current, available)

    np.testing.assert_array_equal(mapped[:, 0], current[:, 0])
    channel_audit = audit["channels"][0]
    assert channel_audit["status"] == "constant_unalignable"
    assert channel_audit["scale_aligned"] is False
    assert channel_audit["iqr"] == 0.0


def test_new_prefix_alignment_object_starts_unfitted():
    center, scale = _source_stats()
    prefix = np.arange(4 * 23, dtype=np.float32).reshape(4, 23)
    available = np.ones_like(prefix, dtype=bool)
    current = prefix[:2].copy()

    fitted = PrefixAlignment(center, scale, channels=[0], min_points=4)
    fitted.fit_prefix(prefix, available)
    mapped, fitted_audit = fitted.transform(current, available[:2])
    fresh, fresh_audit = PrefixAlignment(center, scale, channels=[0], min_points=4).transform(
        current, available[:2]
    )

    assert fitted_audit["ready"] is True
    assert fresh_audit["channels"][0]["status"] == "insufficient_prefix"
    np.testing.assert_array_equal(fresh, current)
    assert not np.array_equal(mapped[:, 0], fresh[:, 0])


def test_transform_accepts_only_a_single_n_by_23_window():
    center, scale = _source_stats()
    adapter = PrefixAlignment(center, scale, channels=[0])
    with pytest.raises(ValueError, match="N x 23"):
        adapter.transform(np.zeros((1, 8, 23)), np.zeros((1, 8, 23), dtype=bool))
