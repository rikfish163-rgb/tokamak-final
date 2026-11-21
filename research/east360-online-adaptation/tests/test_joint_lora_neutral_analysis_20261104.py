import numpy as np
import pytest

from joint_lora_neutral_analysis_20261104 import (
    ARMS,
    paired_transitions,
    scan_curves,
    trace_accounting,
)


def test_score_distinguishes_ultra_early_from_timely_rescue():
    times = np.arange(3201) / 1000
    down = 3.2
    flat = np.full(len(times), 0.3, np.float32)
    curves = {arm: flat.copy() for arm in ARMS}
    curves["intra"][135] = 0.8
    curves["hold"][3130] = 0.8
    result, _ = scan_curves(times, curves, True, down, 0.7, frozen=False)
    assert result["intra"]["lead_ms"] > 3000
    assert result["hold"]["lead_ms"] == 70
    per = {a: [dict(shot=7, disruptive=True, **r)] for a, r in result.items()}
    assert paired_transitions(per["intra"], per["static"])["early_added"] == [7]
    assert paired_transitions(per["intra"], per["static"])["capped_gained"] == []
    assert paired_transitions(per["hold"], per["static"])["capped_gained"] == [7]


def test_early_change_and_static_drift_are_rejected():
    times = np.arange(140) / 1000
    curves = {arm: np.full(140, 0.3, np.float32) for arm in ARMS}
    curves["reverse"][134] = 0.4
    with pytest.raises(ValueError, match="before"):
        scan_curves(times, curves, False, None, 0.7, frozen=False)
    curves["reverse"][134] = 0.3
    curves["static"][135] = 0.4
    with pytest.raises(ValueError, match="native"):
        scan_curves(times, curves, False, None, 0.7, frozen=False)


def test_forecast_clock_preserves_pending_and_rejects_unsupported_update():
    origins = np.array([127, 159], dtype=np.int64)
    releases = np.array([135, -1], dtype=np.int64)
    target = np.full((2, 168), np.nan, np.float32)
    mask = np.zeros((2, 168), dtype=bool)
    target[0, 0] = 1.0
    mask[0, 0] = True
    data, history = {}, {}
    for arm in ARMS[1:]:
        for name, value in dict(
            origin=origins,
            released_at=releases,
            target=target,
            mask=mask,
            prediction=np.zeros_like(target),
        ).items():
            data[arm + "__" + name] = value.copy()
        history[arm] = [
            dict(
                origin=127,
                arrival=135,
                supported=1,
                updated=arm != "static",
                gradient_norm=None if arm == "static" else 0.1,
                parameter_displacement=0.0 if arm == "static" else 0.001,
            )
        ]
    stats = trace_accounting(data, history, 167)
    assert stats["intra"]["pending"] == 1
    assert stats["intra"]["mse"] == 1
    assert stats["intra"]["updates"] == 1
    history["reverse"][0]["supported"] = 0
    with pytest.raises(ValueError, match="support"):
        trace_accounting(data, history, 167)
