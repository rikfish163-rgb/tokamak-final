import csv

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_prefix_probe import Event
from experiments.risk_aligned_online.mask_repair_20261104 import training


def test_short_and_event_enriched_endpoints_never_invent_future_samples():
    times = np.arange(90) / 1000
    events = [Event(1, 1, 0, 0.020, 0.040), Event(1, 2, 1, 0.200, 0.210)]
    points, omitted = training.endpoints(times, events)
    assert points[0]["query"] == 0 and points[-1]["query"] == 89
    assert all(0 <= r["query"] < 90 for r in points)
    assert {7, 31, 63, 20, 39} <= {r["query"] for r in points}
    assert omitted == [2]
    assert len({r["query"] for r in points}) == len(points)


def test_event_sampler_ignores_csv_hard_outcome_and_other_roles(tmp_path):
    path = tmp_path / "events.csv"
    fields = [
        "shot_no",
        "annotation_status",
        "event_index",
        "event_class",
        "event_start_s",
        "event_end_s",
        "disruption_label",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(
                shot_no=1,
                annotation_status="event",
                event_index=1,
                event_class="A",
                event_start_s=0.02,
                event_end_s=0.04,
                disruption_label="NOT_A_HARD_LABEL",
            )
        )
        writer.writerow(
            dict(
                shot_no=2,
                annotation_status="event",
                event_index="invalid",
                event_class="unknown",
                event_start_s="invalid",
                event_end_s="invalid",
                disruption_label="invalid",
            )
        )
    events = training.read_training_events(path, {1})
    assert set(events) == {1} and events[1][0].family == 0


def test_schedule_uses_only_source_train_ids_and_never_labels():
    records = [dict(shot=sid, query=q, role="train") for sid in (1, 2) for q in (0, 40)]
    one = training.make_schedule(records, steps=4, batch_size=5)
    assert one == training.make_schedule(records, steps=4, batch_size=5)
    assert [r["view"] for r in one] == ["no_MNM", "no_MNM_CIII"] * 2
    assert all(
        len(r["indices"]) == 5 and all(0 <= i < 4 for i in r["indices"]) for r in one
    )
    records[0]["role"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        training.make_schedule(records)


def test_embedding_loss_detaches_teacher_and_excludes_padding():
    before = torch.zeros(1, 2, 64)
    after = torch.ones(1, 2, 64, requires_grad=True)
    teacher = torch.full((1, 2, 64), 2.0, requires_grad=True)
    support = torch.tensor([[True, False]])
    loss, info = training.embedding_loss(before, after, teacher, support)
    assert loss.item() == pytest.approx(1.01)
    loss.backward()
    assert teacher.grad is None and after.grad[0, 1].count_nonzero() == 0
    assert info["tokens"] == 1 and info["before_mse"].item() == 4
    with pytest.raises(ValueError, match="support"):
        training.embedding_loss(before, after, teacher, torch.zeros_like(support))


def test_cached_teacher_embeddings_match_the_real_source_path():
    from experiments.risk_aligned_online.mask_repair_20261104 import core
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
        load_parent,
        DEFAULT_PARENT,
    )

    torch.set_num_threads(1)
    base, _ = load_parent(DEFAULT_PARENT)
    generator = torch.Generator().manual_seed(131)
    x = base.center + base.scale * torch.randn(2, 128, 23, generator=generator)
    a = torch.rand(2, 128, 23, generator=generator) > 0.1
    a[0, :121] = False
    repair = core.TokenRepair()
    with torch.no_grad():
        expected = core.forward(base, repair, x, a, enabled=False)["embedding_before"]
    actual = training.native_embedding(base, x, a)
    assert torch.equal(expected, actual) and not actual.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
