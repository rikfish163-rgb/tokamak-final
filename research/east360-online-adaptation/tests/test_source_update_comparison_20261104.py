from copy import deepcopy

import pytest

from experiments.risk_aligned_online.source_anticipation_20261104 import comparison


def rows():
    return [
        dict(
            view="native",
            head=head,
            mode=mode,
            shot=1,
            query=100,
            before_logit=0.0,
            after_logit=value,
            stratum="safe",
            state_sha256=str(value),
        )
        for head in ("native_reference", "anticipation125")
        for mode, value in (("real", 0.1), ("hold", 0.08), ("reverse", 0.09))
    ]


def test_exact_baseline_and_frozen_reference_are_required():
    data = rows()
    got = comparison.paired_check(data, data)
    assert got == dict(records=6, static_exact=6, native_reference_exact=3)
    changed = deepcopy(data)
    changed[0]["before_logit"] = 0.001
    with pytest.raises(ValueError, match="static"):
        comparison.paired_check(data, changed)
    changed = deepcopy(data)
    changed[0]["after_logit"] = 0.11
    with pytest.raises(ValueError, match="reference"):
        comparison.paired_check(data, changed)


def test_duplicate_or_missing_records_fail_instead_of_being_overwritten():
    data = rows()
    with pytest.raises(ValueError, match="duplicate"):
        comparison.paired_check(data, data + [data[0]])
    with pytest.raises(ValueError, match="grid"):
        comparison.paired_check(data, data[:-1])


def test_feedback_dependence_and_crossings_are_point_diagnostics_only():
    got = comparison.feedback_dependence(rows(), cutoff=0.05)
    item = got["native"]["safe"]
    assert item["records"] == 1 and item["risk_up"] == 1
    assert item["cross_up"] == 1 and item["cross_down"] == 0
    assert item["mean_absolute_real_hold_logit_difference"] == pytest.approx(0.02)
    assert item["mean_absolute_logit_move"] == pytest.approx(0.1)
    assert item["real_hold_identical_states"] == 0
