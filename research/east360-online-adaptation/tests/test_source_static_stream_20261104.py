import numpy as np
import pytest

from experiments.risk_aligned_online.source_anticipation_20261104 import static_stream


def test_map_is_not_applied_until_preparation_shots_have_finished():
    raw, mapped = np.array([0.1, 0.2]), np.array([0.7, 0.8])
    assert static_stream.operational_curve(1, raw, mapped) is raw
    assert static_stream.operational_curve(20, raw, mapped) is raw
    assert static_stream.operational_curve(21, raw, mapped) is mapped
    with pytest.raises(ValueError):
        static_stream.operational_curve(0, raw, mapped)
    with pytest.raises(ValueError):
        static_stream.operational_curve(21, raw, mapped[:1])


def test_late_and_missed_alarms_remain_distinct_and_strict30():
    rows = [
        dict(shot=1, disruptive=True, alarm=True, timely=True, lead_ms=31.0),
        dict(shot=2, disruptive=True, alarm=True, timely=False, lead_ms=30.0),
        dict(shot=3, disruptive=True, alarm=False, timely=False, lead_ms=None),
        dict(shot=4, disruptive=False, alarm=True, timely=False, lead_ms=None),
        dict(shot=5, disruptive=False, alarm=False, timely=False, lead_ms=None),
    ]
    got = static_stream.timing_decomposition(rows)
    assert got["tp"] == got["tardy"] == got["miss"] == got["fp"] == 1
    assert got["tpr"] == pytest.approx(1 / 3) and got["fpr"] == 0.5
    assert got["diagnostic_oracle_advance_ms"]["25"] == 2
    assert got["diagnostic_oracle_advance_ms"]["0"] == 1
    assert got["oracle_is_deployable"] is False


def test_bad_truth_or_duplicate_shots_cannot_silently_change_the_denominator():
    row = dict(shot=1, disruptive=True, alarm=True, timely=False, lead_ms=31.0)
    with pytest.raises(ValueError, match="timely"):
        static_stream.timing_decomposition([row])
    row["timely"] = True
    with pytest.raises(ValueError, match="duplicate"):
        static_stream.timing_decomposition([row, row])


def test_source_diagnostic_keeps_its_own_strict10ms_deadline():
    rows = [
        dict(shot=1, disruptive=True, alarm=True, timely=True, lead_ms=10.1),
        dict(shot=2, disruptive=True, alarm=True, timely=False, lead_ms=10.0),
        dict(shot=3, disruptive=False, alarm=False, timely=False, lead_ms=None),
    ]
    got = static_stream.timing_decomposition(rows, deadline_ms=10)
    assert got["tp"] == got["tardy"] == 1 and got["tpr"] == 0.5
