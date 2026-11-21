"""Restore a warm parent and replay only verified, completed EOS updates.

Signal and outcome arrays must already have been revealed and supplied by the
caller. This module reads only the supplied warm checkpoint and EOS JSON logs;
it neither opens HDF files nor performs any prediction or additional training.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from numbers import Integral, Real
from pathlib import Path

import torch

from ..joint_lora_20261104.model import JointLoRA
from ..joint_lora_20261104.run import frozen_digest
from ..joint_lora_20261104.training import ended_records, train_schedule
from ..mask_separated_intra_20261103.run import state_digest


def restore_warm(base, warm_path, warm_receipt, *, device):
    """Load a strictly matching warm state without changing the source model.

    ``warm_receipt`` is the already-parsed warm.json mapping. The caller must
    independently bind its label_shots and the input files to the sealed run.
    """
    if not isinstance(warm_receipt, Mapping):
        raise ValueError("a parsed warm receipt is required")
    expected = warm_receipt.get("fitted_state_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("a fitted warm state digest is required")
    source_before = state_digest(base)
    with Path(warm_path).open("rb") as stream:
        state = torch.load(stream, map_location=device, weights_only=True)
    if not isinstance(state, Mapping) or "W0" not in state:
        raise ValueError("warm checkpoint must be a state dictionary with W0")
    model = JointLoRA(base, state["W0"]).to(device)
    original_frozen = frozen_digest(model)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if frozen_digest(model) != original_frozen:
        raise ValueError("warm checkpoint differs from the supplied frozen source")
    if state_digest(model) != expected:
        raise ValueError("warm checkpoint does not match its fitted state digest")
    if state_digest(base) != source_before:
        raise RuntimeError("source changed during warm restoration")
    return model


def replay_updates(
    fixed, shots, truths, order, log_paths, *, threshold, device,
    expected_steps=8, expected_batch=64,
):
    """Recreate a slow model by applying the exact logged EOS batches.

    The prefix contains the original 20 warm shots and consecutive EOS logs,
    optionally followed by one sealed/revealed shot with no completed log.
    That pending shot is not learned here; the caller must handle it exactly
    once before advancing the stream. Returned receipts contain shot,
    parent_before, parent_after, steps, and batch_size. No original model,
    array, receipt, or file is mutated.
    """
    if not isinstance(fixed, JointLoRA):
        raise ValueError("a frozen JointLoRA warm parent is required")
    if any(p.requires_grad for p in fixed.parameters()) or any(
        module.training for module in fixed.modules()
    ):
        raise ValueError("the warm parent must be frozen and in eval mode")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
        or type(expected_steps) is not int
        or expected_steps < 1
        or type(expected_batch) is not int
        or expected_batch < 2
        or expected_batch % 2
    ):
        raise ValueError("a finite threshold and positive, even replay budget are required")
    try:
        ids = list(order)
        paths = list(log_paths)
    except TypeError as error:
        raise ValueError("ordered prefix and log paths are required") from error
    if (
        len(ids) < 20
        or any(isinstance(s, bool) or not isinstance(s, Integral) or s <= 0 for s in ids)
        or len(set(ids)) != len(ids)
        or not 0 <= len(ids) - 20 - len(paths) <= 1
        or not isinstance(shots, Mapping)
        or not isinstance(truths, Mapping)
        or set(shots) != set(ids)
        or set(truths) != set(ids)
    ):
        raise ValueError("EOS logs must be contiguous with at most one pending tail shot")
    ids = [int(sid) for sid in ids]

    fixed_before = state_digest(fixed)
    immutable = frozen_digest(fixed)
    slow = deepcopy(fixed).to(device)
    if state_digest(slow) != fixed_before:
        raise RuntimeError("copying the warm parent changed its state")
    cache = {}
    receipts = []

    def records_for(sid):
        if sid not in cache:
            rows, _ = ended_records({sid: shots[sid]}, {sid: truths[sid]}, [sid])
            cache[sid] = {row["query"]: row for row in rows}
        return cache[sid]

    for offset, path in enumerate(paths):
        sid = ids[20 + offset]
        ended = ids[:21 + offset]
        ended_set = set(ended)
        with Path(path).open("r", encoding="utf-8") as stream:
            log = json.load(stream)
        if not isinstance(log, dict):
            raise ValueError(f"shot {sid}: EOS log must be an object")
        if (
            type(log.get("shot")) is not int
            or log["shot"] != sid
            or not isinstance(log.get("available_ended_shots"), list)
            or any(type(value) is not int for value in log["available_ended_shots"])
            or log["available_ended_shots"] != ended
        ):
            raise ValueError(f"shot {sid}: noncontiguous EOS log or revealed prefix")
        before = state_digest(slow)
        if log.get("parent_before") != before:
            raise ValueError(f"shot {sid}: parent_before digest mismatch")

        records = log.get("records")
        schedule = log.get("schedule")
        if not isinstance(records, list) or not records or not isinstance(schedule, list):
            raise ValueError(f"shot {sid}: records and schedule are required")
        if len(schedule) != expected_steps:
            raise ValueError(f"shot {sid}: unexpected number of EOS steps")
        seen = set()
        for row in records:
            if not isinstance(row, dict) or set(row) != {"shot", "query", "y", "group"}:
                raise ValueError(f"shot {sid}: invalid training record")
            source, query, label = row["shot"], row["query"], row["y"]
            if (
                type(source) is not int
                or source not in ended_set
                or type(query) is not int
                or type(label) not in (int, float)
                or not math.isfinite(label)
                or (source, query) in seen
            ):
                raise ValueError(f"shot {sid}: invalid, future, or duplicate record")
            seen.add((source, query))
            actual = records_for(source).get(query)
            if actual is None or row != actual:
                raise ValueError(f"shot {sid}: record disagrees with ended truth")

        used = set()
        for number, draw in enumerate(schedule, 1):
            if (
                not isinstance(draw, dict)
                or type(draw.get("step")) is not int
                or draw["step"] != number
                or not isinstance(draw.get("indices"), list)
                or len(draw["indices"]) != expected_batch
                or any(type(index) is not int or not 0 <= index < len(records)
                       for index in draw["indices"])
            ):
                raise ValueError(f"shot {sid}: invalid EOS step or batch")
            used.update(draw["indices"])
            current = sum(records[index]["shot"] == sid for index in draw["indices"])
            if current != expected_batch // 2:
                raise ValueError(f"shot {sid}: EOS batch must be half current and half past")
        if used != set(range(len(records))):
            raise ValueError(f"shot {sid}: compact log contains unused records")
        used_shots = {records[index]["shot"] for index in used}
        training_shots = log.get("training_shots")
        if (
            not isinstance(training_shots, list)
            or any(type(value) is not int for value in training_shots)
            or len(training_shots) != len(set(training_shots))
            or set(training_shots) != used_shots
        ):
            raise ValueError(f"shot {sid}: training_shots disagrees with used records")
        if "current_records" in log and (
            type(log["current_records"]) is not int
            or log["current_records"] != len(records_for(sid))
        ):
            raise ValueError(f"shot {sid}: current record count mismatch")

        train_schedule(
            slow, {s: shots[s] for s in ended}, records, schedule,
            threshold=threshold, device=device, lr=1e-4,
        )
        after = state_digest(slow)
        if after != log.get("parent_after"):
            raise ValueError(f"shot {sid}: parent_after digest mismatch")
        if frozen_digest(slow) != immutable or state_digest(fixed) != fixed_before:
            raise RuntimeError(f"shot {sid}: a frozen parent or source changed")
        receipts.append(dict(
            shot=sid, parent_before=before, parent_after=after,
            steps=expected_steps, batch_size=expected_batch,
        ))
    return slow, receipts
