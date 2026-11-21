import numpy as np
import pytest
import torch

from joint_lora_hindsight_capacity_20261104 import (
    choose_cases,
    collateral,
    fixed_support,
    held_splice,
    hindsight_labels,
    project_joint,
    timeline,
)


def test_labels_use_only_hdf_outcome_and_fixed_125ms_window():
    times = np.array([0.874, 0.875, 0.970, 1.0, 1.001])
    labels, valid = hindsight_labels(times, True, 1.0)
    assert labels.tolist() == [0, 1, 1, 1, 0]
    assert valid.tolist() == [True, True, True, True, False]
    safe, safe_valid = hindsight_labels(times, False, None)
    assert not safe.any() and safe_valid.all()


def test_selects_first_cases_per_stratum_and_excludes_startup():
    cases = [
        dict(index=2, shot=1, kind="close", margin=0.01),
        dict(index=21, shot=2, kind="deep", margin=0.41),
        dict(index=22, shot=3, kind="close", margin=0.03),
        dict(index=23, shot=4, kind="safe", margin=None),
        dict(index=24, shot=5, kind="deep", margin=0.70),
        dict(index=25, shot=6, kind="close", margin=0.04),
        dict(index=26, shot=7, kind="close", margin=0.05),
        dict(index=27, shot=8, kind="safe", margin=None),
    ]
    result = choose_cases(cases)
    assert [r["shot"] for r in result] == [2, 3, 4, 5, 6, 8]
    with pytest.raises(ValueError, match="stratum"):
        choose_cases([cases[1], cases[2]])


def test_disruption_support_is_balanced_distinct_and_timely():
    times = np.arange(1200) / 1000
    indices = fixed_support(times, True, 1.199, np.zeros(1200), 0.7)
    assert len(set(indices)) == 8
    assert min(indices) >= 135
    lead = np.round((1.199 - times[indices]) * 1000, 9)
    assert np.all(lead[:4] > 125)
    assert np.all((lead[4:] > 30) & (lead[4:] <= 125))


def test_safe_support_is_before_and_around_first_false_alarm():
    times = np.arange(500) / 1000
    probs = np.zeros(500)
    probs[250] = 0.7
    indices = fixed_support(times, False, None, probs, 0.7)
    assert len(set(indices)) == 8
    assert all(135 <= i < 250 for i in indices[:4])
    assert indices[4:] == [250, 266, 282, 298]


def test_support_fail_closed_when_positive_window_absent():
    with pytest.raises(ValueError):
        fixed_support(np.arange(140) / 1000, True, 0.139, np.zeros(140), 0.7)


def test_prefix_and_support_are_spliced_without_changing_held_points():
    baseline = np.zeros(150)
    adapted = np.ones(150)
    mixed = held_splice(baseline, adapted, [140, 145])
    assert np.all(mixed[:135] == baseline[:135])
    assert mixed[140] == mixed[145] == 0
    assert mixed[135] == mixed[149] == 1
    with pytest.raises(ValueError):
        held_splice(baseline, adapted, [134])


def test_first_alarm_and_strict_30ms_boundary():
    times = np.array([0.000, 0.020, 0.030, 0.040])
    values = np.array([0.1, 0.7, 0.9, 0.9])
    assert timeline(times, values, True, 0.05, 0.7)["status"] == "tardy"
    assert timeline(times, values, True, 0.051, 0.7)["status"] == "tp"
    assert timeline(times, values, False, None, 0.7)["status"] == "fp"


def test_earlier_negative_collateral_excludes_positive_and_support():
    times = np.arange(300) / 1000
    baseline = np.zeros(300)
    adapted = baseline.copy()
    adapted[140], adapted[150], adapted[200] = 0.8, 0.9, 0.9
    row = collateral(times, baseline, adapted, True, 0.300, 0.7, [150])
    assert row["new_crossings"] == 1
    assert row["first_new_crossing_index"] == 140


def test_projection_is_joint_and_preserves_initial_center():
    a = torch.nn.Parameter(torch.tensor([4.0, 0.0]))
    b = torch.nn.Parameter(torch.tensor([0.0, 3.0]))
    center = [torch.tensor([1.0, 0.0]), torch.zeros(2)]
    project_joint([a, b], center, 0.5)
    displacement = torch.cat([a.detach() - center[0], b.detach() - center[1]])
    assert 0.4999 <= torch.linalg.vector_norm(displacement).item() <= 0.5
    assert a[0].item() > 1 and b[1].item() > 0
