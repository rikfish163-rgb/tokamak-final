from copy import deepcopy

import pytest

from experiments.risk_aligned_online.source_anticipation_20261104.design import (
    make_schedule,
    make_targets,
)
from experiments.risk_aligned_online.source_anticipation_20261104.train import (
    validate_design_content,
)
from experiments.risk_aligned_online.source_anticipation_20261104 import train


def fixture():
    names = (
        "safe",
        "disruptive_30_to_125ms",
        "disruptive_earlier_than_125ms",
        "disruptive_0_to_30ms",
    )
    rows = [
        dict(
            shot=1 if i == 0 else 2,
            query=200 + i * 25,
            role="train",
            hdf5_disruptive=i != 0,
            hdf5_source_point_valid=True,
            anticipation_stratum=name,
            original_source_point_label=int(i == 3),
            future_disruption_125ms=int(i in (1, 3)),
        )
        for i, name in enumerate(names)
    ]
    schedule = make_schedule(rows, steps=3)
    labels, _ = make_targets(rows, schedule=schedule)
    return rows, schedule, {k: v.tolist() for k, v in labels.items()}


def test_fit_accepts_only_a_matched_authoritative_fixed_design():
    rows, schedule, labels = fixture()
    got = validate_design_content(rows, schedule, labels, {"exposure_matched": True})
    assert set(got) == set(labels)


@pytest.mark.parametrize(
    "bad", ["old_corpus_control", "target_relabel", "exposure", "validation_role"]
)
def test_fit_rejects_old_or_confounded_packets(bad):
    rows, schedule, labels = fixture()
    protocol = {"exposure_matched": True}
    if bad == "old_corpus_control":
        protocol = {}
    elif bad == "target_relabel":
        labels["point_contract"][1] = 1
    elif bad == "exposure":
        labels["time_shuffled125"] = deepcopy(labels["anticipation125"])
        labels["time_shuffled125"][2] = 1
    else:
        rows[0]["role"] = "val"
    with pytest.raises(ValueError):
        validate_design_content(rows, schedule, labels, protocol)


def test_validation_window_loader_checks_val_membership_before_raw_io(monkeypatch):
    calls = []

    def roles(role):
        calls.append(role)
        return [], [11], {}

    monkeypatch.setattr(train, "load_records", roles)
    rows = [dict(shot=99, role="val")]
    with pytest.raises(ValueError, match="signal identities"):
        train.load_windows(rows, [{"indices": [0]}], "cpu", role="val")
    assert calls == ["val"]


@pytest.mark.parametrize("role", ["test", "EAST", None])
def test_source_window_loader_never_authorizes_other_roles(monkeypatch, role):
    monkeypatch.setattr(
        train, "load_records", lambda _: pytest.fail("forbidden role lookup")
    )
    with pytest.raises(ValueError):
        train.load_windows([], [], "cpu", role=role)


@pytest.mark.parametrize("name", ["cosine", "risk_utility"])
def test_teaching_objective_selector_preserves_the_two_distinct_losses(name):
    from experiments.risk_aligned_online.source_anticipation_20261104 import meta, utility

    function, plan = train.teaching_objective(name)
    expected = meta.objective if name == "cosine" else utility.objective
    assert function is expected
    assert plan.name == (
        "SOURCE_ANTICIPATION_TRAIN_PLAN_20261104.md"
        if name == "cosine"
        else "SOURCE_UPDATE_UTILITY_PLAN_20261104.md"
    )
    assert plan.is_file()


@pytest.mark.parametrize("name", [None, "", "utility", "validation_selected"])
def test_unknown_teaching_objectives_fail_before_loading_data(name):
    with pytest.raises(ValueError, match="teaching objective"):
        train.teaching_objective(name)
