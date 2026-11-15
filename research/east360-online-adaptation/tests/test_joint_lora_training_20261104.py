from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cold20_compare_20261104.data import make_schedule
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.training import (
    ended_records,
    train_schedule,
    eos_schedule,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base, p = load_parent(DEFAULT_PARENT)
    rng = np.random.default_rng(4)
    shots, truth = {}, {1: (True, 0.3), 2: (False, None)}
    for sid in truth:
        values = base.center.numpy() + base.scale.numpy() * rng.normal(
            0, 0.1, (310, 23)
        ).astype(np.float32)
        shots[sid] = dict(
            shot=sid,
            device="EAST",
            times=np.arange(310) / 1000,
            values=values,
            available=np.ones((310, 23), bool),
        )
    return base, p, shots, truth


def test_records_use_only_complete_targets_and_strict_timely_positive(fixture):
    _, _, shots, truth = fixture
    rows, pools = ended_records(shots, truth, [1, 2])
    assert set(pools) == {"positive", "safe", "earlier"}
    for row in rows:
        q = row["query"]
        assert (
            q >= 135
            and (q - 135) % 8 == 0
            and q + 8 < len(shots[row["shot"]]["values"])
        )
        if row["group"] == "positive":
            lead = round((0.3 - shots[1]["times"][q]) * 1000, 9)
            assert 30 < lead <= 125
    with pytest.raises(ValueError):
        ended_records(shots, truth, [1])


def test_training_changes_only_two_explicit_scopes_and_preserves_source(fixture):
    base, parent, shots, truth = fixture
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model = JointLoRA(base, torch.zeros(168, 70))
    frozen = {k: v.clone() for k, v in model.named_parameters() if not v.requires_grad}
    w0 = model.W0.clone()
    rows, pools = ended_records(shots, truth, [1, 2])
    schedule = make_schedule(pools, steps=2, batch_size=8)
    history = train_schedule(
        model,
        shots,
        rows,
        schedule,
        threshold=parent["threshold"],
        device="cpu",
        lr=0.001,
    )
    assert len(history) == 2 and history[-1]["gradient_norm"] > 0
    assert any(p.abs().max() > 0 for p in model.decoder_parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())
    assert torch.equal(w0, model.W0)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_eos_draws_are_deterministic_current_plus_previous_only(fixture):
    _, _, shots, truth = fixture
    rows, _ = ended_records(shots, truth, [1, 2])
    current = [r for r in rows if r["shot"] == 1]
    previous = [r for r in rows if r["shot"] == 2]
    merged, schedule = eos_schedule(current, previous, seed=131, steps=3, batch_size=8)
    merged2, schedule2 = eos_schedule(
        current, previous, seed=131, steps=3, batch_size=8
    )
    assert merged == merged2 and schedule == schedule2
    for draw in schedule:
        selected = [merged[i] for i in draw["indices"]]
        assert sum(r["shot"] == 1 for r in selected) == 4
        assert sum(r["shot"] == 2 for r in selected) == 4


def test_nonfinite_or_unsupported_truth_does_not_become_safe(fixture):
    _, _, shots, truth = fixture
    invalid = deepcopy(truth)
    invalid[1] = (True, None)
    with pytest.raises(ValueError):
        ended_records(shots, invalid, [1, 2])
