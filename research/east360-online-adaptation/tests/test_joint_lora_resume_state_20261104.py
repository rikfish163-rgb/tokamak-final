from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.training import (
    ended_records,
    eos_schedule,
    train_schedule,
)
from experiments.risk_aligned_online.joint_lora_resume_20261104 import state as resume
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import state_digest
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(scope="module")
def experiment():
    old_threads = torch.get_num_threads()
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(3401)
    base = Simple23(
        Config(fast_site="tokens", fast_rule="normalized", input_view="source"),
        torch.zeros(23),
        torch.ones(23),
    ).eval().requires_grad_(False)
    w0 = torch.randn(168, 70) * 0.01
    fixed = JointLoRA(base, w0)
    with torch.no_grad():
        fixed.residual_decoder.weight.normal_(0, 0.01)
        fixed.residual_decoder.bias.normal_(0, 0.01)
    fixed.eval().requires_grad_(False)

    order = list(range(1, 23))
    truth = {sid: (False, None) for sid in order}
    rng = np.random.default_rng(874)
    shots = {
        sid: dict(
            shot=sid,
            device="EAST",
            times=np.arange(168, dtype=np.float64) / 1000,
            values=(rng.normal(size=(168, 23)) * 0.1).astype(np.float32),
            available=np.ones((168, 23), dtype=bool),
        )
        for sid in order
    }
    prefix = order[:20]
    replay, _ = ended_records(
        {sid: shots[sid] for sid in prefix},
        {sid: truth[sid] for sid in prefix},
        prefix,
    )
    continuous = deepcopy(fixed)
    logs = []
    for index in range(21, 23):
        sid = order[index - 1]
        current, _ = ended_records({sid: shots[sid]}, {sid: truth[sid]}, [sid])
        records, schedule = eos_schedule(
            current, replay, seed=131 + index, steps=2, batch_size=8
        )
        before = state_digest(continuous)
        history = train_schedule(
            continuous,
            {s: shots[s] for s in order[:index]},
            records,
            schedule,
            threshold=0.5,
            device="cpu",
            lr=1e-4,
        )
        used = sorted({i for draw in schedule for i in draw["indices"]})
        remap = {original: i for i, original in enumerate(used)}
        compact = [records[i] for i in used]
        logs.append(
            dict(
                shot=sid,
                available_ended_shots=order[:index],
                training_shots=sorted({r["shot"] for r in compact}),
                shared_slow_for=["inter", "both"],
                parent_before=before,
                parent_after=state_digest(continuous),
                current_records=len(current),
                records=compact,
                replay_buffer_indices=used,
                schedule=[
                    dict(step=draw["step"], indices=[remap[i] for i in draw["indices"]])
                    for draw in schedule
                ],
                history=history,
            )
        )
        replay.extend(current)
    warm = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in fixed.state_dict().items()}, warm)
    value = dict(
        base=base,
        fixed=fixed,
        continuous=continuous,
        warm_bytes=warm.getvalue(),
        receipt=dict(label_shots=order[:20], fitted_state_sha256=state_digest(fixed)),
        order=order,
        shots=shots,
        truths=truth,
        logs=logs,
    )
    yield value
    torch.backends.mha.set_fastpath_enabled(old_fastpath)
    torch.set_num_threads(old_threads)


@pytest.fixture
def virtual_files(monkeypatch, experiment):
    files = {"/virtual/warm.pt": experiment["warm_bytes"]}
    files.update(
        {f"/virtual/{i}.json": json.dumps(row).encode() for i, row in enumerate(experiment["logs"])}
    )
    original_open = Path.open

    def virtual_open(path, mode="r", *args, **kwargs):
        key = str(path)
        if key not in files:
            return original_open(path, mode, *args, **kwargs)
        if mode == "rb":
            return io.BytesIO(files[key])
        if mode == "r":
            return io.StringIO(files[key].decode("utf-8"))
        raise AssertionError("warm and EOS inputs must never be opened for writing")

    monkeypatch.setattr(Path, "open", virtual_open)
    return files


def replay(experiment, *, paths=None, **overrides):
    kwargs = dict(
        fixed=experiment["fixed"],
        shots=experiment["shots"],
        truths=experiment["truths"],
        order=experiment["order"],
        log_paths=["/virtual/0.json", "/virtual/1.json"] if paths is None else paths,
        threshold=0.5,
        device="cpu",
        expected_steps=2,
        expected_batch=8,
    )
    kwargs.update(overrides)
    return resume.replay_updates(**kwargs)


def edit_log(files, index, edit):
    path = f"/virtual/{index}.json"
    log = json.loads(files[path])
    edit(log)
    files[path] = json.dumps(log).encode()


def test_restore_warm_strict_exact_and_source_immutable(experiment, virtual_files):
    source_before = state_digest(experiment["base"])
    restored = resume.restore_warm(
        experiment["base"], "/virtual/warm.pt", experiment["receipt"], device="cpu"
    )
    assert state_digest(restored) == experiment["receipt"]["fitted_state_sha256"]
    assert state_digest(experiment["base"]) == source_before
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    assert restored.W0.data_ptr() != experiment["fixed"].W0.data_ptr()
    for name, value in experiment["fixed"].state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_restore_rejects_bad_digest_missing_state_or_wrong_source(experiment, virtual_files):
    receipt = dict(experiment["receipt"], fitted_state_sha256="0" * 64)
    with pytest.raises((ValueError, RuntimeError)):
        resume.restore_warm(experiment["base"], "/virtual/warm.pt", receipt, device="cpu")
    wrong_base = deepcopy(experiment["base"])
    with torch.no_grad():
        wrong_base.embedding.weight[0, 0] += 1
    with pytest.raises((ValueError, RuntimeError)):
        resume.restore_warm(wrong_base, "/virtual/warm.pt", experiment["receipt"], device="cpu")
    saved = torch.load(io.BytesIO(virtual_files["/virtual/warm.pt"]), weights_only=True)
    del saved["W0"]
    payload = io.BytesIO()
    torch.save(saved, payload)
    virtual_files["/virtual/warm.pt"] = payload.getvalue()
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        resume.restore_warm(experiment["base"], "/virtual/warm.pt", experiment["receipt"], device="cpu")


def test_restore_rejects_strict_state_key_mismatch(experiment, virtual_files):
    saved = torch.load(io.BytesIO(virtual_files["/virtual/warm.pt"]), weights_only=True)
    saved["unexpected"] = torch.zeros(1)
    payload = io.BytesIO()
    torch.save(saved, payload)
    virtual_files["/virtual/warm.pt"] = payload.getvalue()
    with pytest.raises((ValueError, RuntimeError)):
        resume.restore_warm(experiment["base"], "/virtual/warm.pt", experiment["receipt"], device="cpu")


def test_replay_matches_uninterrupted_training_and_never_changes_inputs(experiment, virtual_files, monkeypatch):
    before_files = dict(virtual_files)
    fixed_before = state_digest(experiment["fixed"])
    source_before = state_digest(experiment["base"])
    original = resume.train_schedule
    calls = []

    def checked(*args, **kwargs):
        calls.append((len(args[3]), kwargs["lr"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(resume, "train_schedule", checked)
    slow, receipts = replay(experiment)
    assert calls == [(2, 1e-4), (2, 1e-4)]
    assert state_digest(slow) == state_digest(experiment["continuous"])
    assert [r["shot"] for r in receipts] == [21, 22]
    assert [r["parent_after"] for r in receipts] == [x["parent_after"] for x in experiment["logs"]]
    assert all(torch.equal(value, slow.state_dict()[name]) for name, value in experiment["continuous"].state_dict().items())
    assert slow is not experiment["fixed"]
    assert not slow.training and all(not p.requires_grad for p in slow.parameters())
    assert state_digest(experiment["fixed"]) == fixed_before
    assert state_digest(experiment["base"]) == source_before
    assert virtual_files == before_files


def test_zero_eos_logs_returns_independent_warm_copy(experiment, virtual_files):
    ids = experiment["order"][:20]
    slow, receipts = replay(
        experiment,
        paths=[],
        order=ids,
        shots={s: experiment["shots"][s] for s in ids},
        truths={s: experiment["truths"][s] for s in ids},
    )
    assert receipts == [] and slow is not experiment["fixed"]
    assert state_digest(slow) == state_digest(experiment["fixed"])


def test_one_revealed_tail_may_be_pending_without_accessing_its_data(experiment, virtual_files):
    class NoPendingValueAccess(dict):
        def __getitem__(self, key):
            if key == 22:
                raise AssertionError("the pending shot must not enter EOS replay")
            return super().__getitem__(key)

    slow, receipts = replay(
        experiment,
        paths=["/virtual/0.json"],
        shots=NoPendingValueAccess(experiment["shots"]),
        truths=NoPendingValueAccess(experiment["truths"]),
    )
    assert [row["shot"] for row in receipts] == [21]
    assert state_digest(slow) == experiment["logs"][0]["parent_after"]


def test_single_step_protocol_budget_is_supported(experiment, virtual_files):
    log = deepcopy(experiment["logs"][0])
    draw = log["schedule"][0]
    used = sorted(set(draw["indices"]))
    remap = {old: new for new, old in enumerate(used)}
    records = [log["records"][index] for index in used]
    schedule = [dict(step=1, indices=[remap[index] for index in draw["indices"]])]
    reference = deepcopy(experiment["fixed"])
    train_schedule(
        reference, {sid: experiment["shots"][sid] for sid in experiment["order"][:21]},
        records, schedule, threshold=0.5, device="cpu", lr=1e-4,
    )
    log.update(
        records=records,
        schedule=schedule,
        training_shots=sorted({row["shot"] for row in records}),
        parent_after=state_digest(reference),
    )
    virtual_files["/virtual/0.json"] = json.dumps(log).encode()
    slow, receipts = replay(experiment, paths=["/virtual/0.json"], expected_steps=1)
    assert state_digest(slow) == state_digest(reference)
    assert receipts[0]["steps"] == 1


@pytest.mark.parametrize("field", ["parent_before", "parent_after"])
def test_tampered_parent_digest_rejected_and_fixed_unchanged(experiment, virtual_files, field):
    edit_log(virtual_files, 0, lambda log: log.update({field: "0" * 64}))
    digest = state_digest(experiment["fixed"])
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment)
    assert state_digest(experiment["fixed"]) == digest


@pytest.mark.parametrize("field,value", [("query", 136), ("y", 1.0), ("group", "positive")])
def test_tampered_record_fields_are_rejected(experiment, virtual_files, field, value):
    edit_log(virtual_files, 0, lambda log: log["records"][0].update({field: value}))
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment)


def test_future_shot_record_rejected_even_if_in_later_revealed_prefix(experiment, virtual_files):
    future, _ = ended_records(
        {22: experiment["shots"][22]}, {22: experiment["truths"][22]}, [22]
    )
    edit_log(virtual_files, 0, lambda log: log["records"].__setitem__(0, future[0]))
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment)


def test_incorrect_revealed_truth_invalidates_saved_record(experiment, virtual_files):
    truth = dict(experiment["truths"], **{})
    truth[21] = (True, 0.150)
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment, truths=truth)


@pytest.mark.parametrize("mutation", ["wrong_shot", "wrong_prefix", "training_shots", "step", "batch", "balance", "negative", "unused"])
def test_inconsistent_log_rejected(experiment, virtual_files, mutation):
    def corrupt(log):
        if mutation == "wrong_shot":
            log["shot"] = 22
        elif mutation == "wrong_prefix":
            log["available_ended_shots"] = log["available_ended_shots"][:-1]
        elif mutation == "training_shots":
            log["training_shots"] = []
        elif mutation == "step":
            log["schedule"][0]["step"] = 2
        elif mutation == "batch":
            log["schedule"][0]["indices"].pop()
        elif mutation == "balance":
            indices = log["schedule"][0]["indices"]
            current = next(i for i, row in enumerate(log["records"]) if row["shot"] == 21)
            old_position = next(j for j, i in enumerate(indices) if log["records"][i]["shot"] != 21)
            indices[old_position] = current
        elif mutation == "negative":
            log["schedule"][0]["indices"][0] = -1
        else:
            log["records"].append(dict(log["records"][0]))

    edit_log(virtual_files, 0, corrupt)
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment)


def test_log_gap_tail_omission_and_reordering_are_rejected(experiment, virtual_files):
    for paths in ([], ["/virtual/1.json", "/virtual/0.json"], ["/virtual/1.json"]):
        with pytest.raises((ValueError, RuntimeError)):
            replay(experiment, paths=paths)


@pytest.mark.parametrize("problem", ["duplicate", "missing_shot", "extra_shot", "missing_truth"])
def test_prefix_must_match_shots_and_truth_exactly(experiment, virtual_files, problem):
    order = list(experiment["order"])
    shots, truths = dict(experiment["shots"]), dict(experiment["truths"])
    if problem == "duplicate":
        order[-1] = order[-2]
    elif problem == "missing_shot":
        shots.pop(22)
    elif problem == "extra_shot":
        shots[23] = shots[22]
    else:
        truths.pop(22)
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment, order=order, shots=shots, truths=truths)


@pytest.mark.parametrize("kwargs", [dict(expected_steps=1), dict(expected_batch=64), dict(expected_steps=0), dict(expected_batch=3), dict(threshold=float("nan"))])
def test_wrong_budget_or_threshold_rejected(experiment, virtual_files, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        replay(experiment, **kwargs)
