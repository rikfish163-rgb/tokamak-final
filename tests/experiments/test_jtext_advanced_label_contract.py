from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jtext_label_contract as labels  # noqa: E402
import train_jtext_source_transformer as source  # noqa: E402


def test_effective_horizon_uses_floor_and_explicit_missing_fallback() -> None:
    values = {101: 12.0, 102: 40.0}

    assert labels.effective_horizon_ms(101, values) == (25.0, 12.0, False)
    assert labels.effective_horizon_ms(102, values) == (40.0, 40.0, False)
    assert labels.effective_horizon_ms(103, values) == (25.0, None, True)


def test_advanced_time_loader_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "advanced.json"
    path.write_text(json.dumps({"101": "not-a-number"}), encoding="utf-8")

    try:
        labels.load_advanced_time_ms(path)
    except ValueError as exc:
        assert "invalid J-TEXT AdvancedTime" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid AdvancedTime was accepted")


def _shot(*, horizon: float | None) -> dict:
    row = {
        "shot": 101,
        "v": np.zeros((240, 17), dtype=np.float32),
        "a": np.ones((240, 17), dtype=bool),
        "is_disrupt": True,
        "down": 0.199,
        "start": 0.0,
        "rate": 1000.0,
    }
    if horizon is not None:
        row["positive_horizon_ms"] = horizon
    return row


def test_sampler_can_switch_to_per_shot_horizon_without_changing_default() -> None:
    dynamic = source.BalancedSourceSampler(
        [_shot(horizon=100.0)], seed=131, per_shot_horizon=True
    )
    fixed = source.BalancedSourceSampler(
        [_shot(horizon=100.0)], seed=131, per_shot_horizon=False
    )

    assert int(dynamic.positive[0].min()) == 99
    assert int(fixed.positive[0].min()) == 74
    assert int(dynamic.positive[0].max()) == int(fixed.positive[0].max())


def test_advanced_floor25_ignores_transition_context_until_125_ms() -> None:
    """Changing the positive horizon must not create contradictory negatives.

    For a disruptive shot with H=25 ms, the 25--125 ms context is unknown for
    the new source contract.  It must be excluded from both classes; only
    ttd>125 ms remains a negative.  This is deliberately tested at exact
    boundary points as well as in the interior.
    """
    shot = _shot(horizon=25.0)
    sampler = source.BalancedSourceSampler(
        [shot], seed=131, per_shot_horizon=True, ambiguous_until_ms=125.0
    )

    # start=0, rate=1000 Hz, down=199 ms: ttd_ms = 199 - endpoint.
    assert 199 in sampler.positive[0]       # ttd=0 ms
    assert 174 in sampler.positive[0]       # ttd=25 ms
    assert 173 not in sampler.positive[0]   # ttd=26 ms
    assert 173 in sampler.ambiguous[0]      # ttd=26 ms, transition context
    assert 74 in sampler.ambiguous[0]       # ttd=125 ms, inclusive boundary
    assert 73 in sampler.negative[0]        # ttd=126 ms, true negative
    assert 89 not in sampler.negative[0]    # ttd=110 ms must not be negative


def test_advanced_floor25_main_policy_uses_horizon_as_negative_boundary() -> None:
    """The requested Adv25 experiment changes only the positive horizon.

    The interval after H is a negative under the literal contract.  The
    optional 125-ms ambiguity guard is a separate diagnostic policy and must
    not silently enter the main source checkpoint.
    """
    shot = _shot(horizon=25.0)
    sampler = source.BalancedSourceSampler(
        [shot], seed=131, per_shot_horizon=True, ambiguous_until_ms=None
    )

    assert 174 in sampler.positive[0]       # ttd=25 ms
    assert 173 in sampler.negative[0]       # ttd=26 ms
    assert 89 in sampler.negative[0]        # ttd=110 ms
    assert not sampler.ambiguous


def test_fixed_horizon_sampler_retains_historical_negative_boundary() -> None:
    """The old sampler remains unchanged when no ambiguity guard is set."""
    sampler = source.BalancedSourceSampler(
        [_shot(horizon=None)], seed=131, horizon_ms=25.0,
        per_shot_horizon=False,
    )

    # With H=25 ms and no guard, ttd=110 ms is historically a negative.
    assert 89 in sampler.negative[0]
    assert 174 in sampler.positive[0]


def test_contract_metadata_records_only_requested_roles(tmp_path: Path) -> None:
    path = tmp_path / "advanced.json"
    path.write_text(json.dumps({"101": 40.0}), encoding="utf-8")

    metadata = labels.contract_metadata(
        path=path,
        used_shot_ids=[101, 102],
        advanced_times_ms={101: 40.0},
    )

    assert metadata["used_shot_count"] == 2
    assert metadata["raw_advanced_time_present_count"] == 1
    assert metadata["fallback_shot_ids"] == [102]


def test_contract_metadata_does_not_count_safe_shots_as_horizon_fallbacks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "advanced.json"
    path.write_text(json.dumps({"101": 40.0}), encoding="utf-8")

    metadata = labels.contract_metadata(
        path=path,
        used_shot_ids=[101, 102, 103],
        disruptive_shot_ids=[101, 102],
        advanced_times_ms={101: 40.0},
    )

    assert metadata["used_shot_count"] == 3
    assert metadata["used_disruptive_shot_count"] == 2
    assert metadata["fallback_shot_count"] == 1
    assert metadata["fallback_shot_ids"] == [102]
