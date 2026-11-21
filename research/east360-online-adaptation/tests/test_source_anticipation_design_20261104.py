from copy import deepcopy

import numpy as np
import pytest

from experiments.risk_aligned_online.source_anticipation_20261104 import design


def records():
    rows = [
        dict(
            shot=1,
            query=q,
            role="train",
            hdf5_disruptive=False,
            original_source_point_label=0,
            future_disruption_125ms=0,
            hdf5_source_point_valid=True,
            anticipation_stratum="safe",
        )
        for q in (153, 178)
    ]
    for q in range(8):
        stratum = (
            "disruptive_earlier_than_125ms",
            "disruptive_30_to_125ms",
            "disruptive_0_to_30ms",
        )[q % 3]
        rows.append(
            dict(
                shot=2,
                query=200 + q * 25,
                role="train",
                hdf5_disruptive=True,
                original_source_point_label=int(stratum == "disruptive_0_to_30ms"),
                future_disruption_125ms=int(stratum != "disruptive_earlier_than_125ms"),
                hdf5_source_point_valid=True,
                anticipation_stratum=stratum,
            )
        )
    return rows


def test_matched_targets_keep_truth_fields_and_per_shot_positive_counts():
    rows = records()
    original = deepcopy(rows)
    labels, receipt = design.make_targets(rows)
    assert rows == original and set(labels) == set(design.ARMS)
    for shot in (1, 2):
        used = np.array([r["shot"] == shot for r in rows])
        assert (
            labels["anticipation125"][used].sum()
            == labels["time_shuffled125"][used].sum()
        )
    assert not labels["time_shuffled125"][:2].any()
    assert receipt["changed_records"] > 0
    again, _ = design.make_targets(rows)
    for name in labels:
        np.testing.assert_array_equal(labels[name], again[name])


def test_one_schedule_is_independent_of_which_teacher_is_trained():
    rows = records()
    schedule = design.make_schedule(rows, steps=6)
    assert len(schedule) == 6
    assert [s["task"] for s in schedule] == ["native", "no_MNM", "no_MNM_CIII"] * 2
    for step in schedule:
        assert len(step["indices"]) == 4
        assert [rows[i]["anticipation_stratum"] for i in step["indices"]] == list(
            design.STRATA
        )
    assert schedule == design.make_schedule(rows, steps=6)


def test_time_control_preserves_actual_repeated_training_exposure_not_only_corpus():
    rows = records()
    schedule = design.make_schedule(rows, steps=128)
    labels, receipt = design.make_targets(rows, schedule=schedule)
    used = np.array([i for step in schedule for i in step["indices"]])
    assert receipt["exposure_matched"] is True
    assert receipt["training_draws"] == 512
    for sid in {r["shot"] for r in rows}:
        chosen = used[np.array([rows[i]["shot"] == sid for i in used])]
        assert (
            labels["anticipation125"][chosen].sum()
            == labels["time_shuffled125"][chosen].sum()
        )
    assert receipt["future_positive_draws"] == receipt["shuffled_positive_draws"]


@pytest.mark.parametrize(
    "bad", ["val", "bad_label", "missing_group", "nonbinary_flag", "unsafe_label"]
)
def test_design_cannot_fit_holdout_or_inconsistent_authoritative_targets(bad):
    rows = records()
    if bad == "val":
        rows[0]["role"] = "val"
    elif bad == "bad_label":
        rows[0]["future_disruption_125ms"] = 2
    elif bad == "missing_group":
        rows = [r for r in rows if r["anticipation_stratum"] != "safe"]
    elif bad == "nonbinary_flag":
        rows[0]["hdf5_disruptive"] = 0
    else:
        rows[0]["future_disruption_125ms"] = 1
    with pytest.raises(ValueError):
        design.make_schedule(rows, steps=3)
