"""Offline TRAIN-only embedding repair; no hard outcomes or source updates."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..east_context_operator_20261031.run import configure
from ..event_prefix_probe import DEFAULT_LABELS, Event, assign_events
from ..history_gradient_metric_20261102.run import sha, write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .core import TokenRepair, forward

PLAN = io23.ROOT / "SOURCE_MASK_REPAIR_TRAIN_PLAN_20261104.md"
VIEWS = {"no_MNM": (2,), "no_MNM_CIII": (1, 2)}


def read_training_events(path, allowed):
    """Use only selected TRAIN interval fields; CSV outcome is not consulted."""
    events = defaultdict(list)
    required = {
        "shot_no",
        "annotation_status",
        "event_index",
        "event_class",
        "event_start_s",
        "event_end_s",
    }
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not required <= set(rows.fieldnames or []):
            raise ValueError("event interval columns are required")
        for row in rows:
            sid = int(row["shot_no"])
            if sid not in allowed:
                continue
            status = row["annotation_status"]
            if status == "event":
                name = row["event_class"]
                if name not in tuple("ABCDEF"):
                    raise ValueError("nominal event family must be A-F")
                events[sid].append(
                    Event(
                        sid,
                        int(row["event_index"]),
                        ord(name) - ord("A"),
                        float(row["event_start_s"]),
                        float(row["event_end_s"]),
                    )
                )
            elif status not in ("no_event_detected", "excluded_short_record"):
                raise ValueError("unknown annotation status")
            elif any(
                row[key]
                for key in (
                    "event_index",
                    "event_class",
                    "event_start_s",
                    "event_end_s",
                )
            ):
                raise ValueError("non-event row contains an event target")
    for rows in events.values():
        rows.sort(key=lambda e: e.index)
        if rows[0].index != 1:
            raise ValueError("event numbering must start at one")
        assign_events(np.array([0.0]), rows)
    return dict(events)


def endpoints(times, events):
    times = np.asarray(times)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("nonempty finite increasing source clock required")
    chosen, omitted = defaultdict(set), []
    for q in np.linspace(0, len(times) - 1, 8).astype(int):
        chosen[int(q)].add("coverage")
    for q in (7, 31, 63, 127):
        if q < len(times):
            chosen[q].add("prefix")
    compared = np.round(times, 9)
    for e in events:
        idx = np.flatnonzero((compared >= e.start) & (compared < e.end))
        if not len(idx):
            omitted.append(e.index)
            continue
        for name, q in (
            ("first", idx[0]),
            ("middle", idx[len(idx) // 2]),
            ("last", idx[-1]),
        ):
            chosen[int(q)].add(f"event:{e.index}:{name}")
    return [
        dict(query=q, reasons=sorted(tags)) for q, tags in sorted(chosen.items())
    ], omitted


def make_schedule(records, *, steps=512, batch_size=32, seed=131):
    if (
        not records
        or type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 1
        or any(r.get("role") != "train" for r in records)
    ):
        raise ValueError(
            "nonempty source TRAIN records and positive fixed budget required"
        )
    by_shot = defaultdict(list)
    for i, r in enumerate(records):
        by_shot[r["shot"]].append(i)
    rng = np.random.default_rng(seed)
    ids = sorted(by_shot)
    result = []
    for step in range(steps):
        # Deliberate shot-uniform sampling with replacement, not window-count
        # weighting or outcome-based balancing. Labels are not read.
        shots = rng.choice(ids, size=batch_size, replace=True)
        result.append(
            dict(
                step=step + 1,
                view=tuple(VIEWS)[step % 2],
                indices=[int(rng.choice(by_shot[int(s)])) for s in shots],
            )
        )
    return result


def embedding_loss(before, after, teacher, support):
    if (
        before.shape != after.shape
        or teacher.shape != after.shape
        or after.ndim != 3
        or after.shape[-1] != 64
        or support.shape != after.shape[:2]
        or support.dtype != torch.bool
        or not support.any()
    ):
        raise ValueError("matching patch tensors and nonempty boolean support required")
    if not all(torch.isfinite(t[support]).all() for t in (before, after, teacher)):
        raise ValueError("finite supported embeddings required")
    target = teacher.detach()
    mse = (after[support] - target[support]).square().mean()
    regularizer = (after[support] - before[support]).square().mean()
    return mse + 0.01 * regularizer, dict(
        mse=mse,
        regularizer=regularizer,
        before_mse=(before.detach()[support] - target[support]).square().mean(),
        tokens=int(support.sum()),
    )


def native_embedding(base, x, a):
    """Exact native source-view patch construction; no Transformer or labels."""
    if base.cfg.input_view != "source" or base.cfg.window != 128 or base.cfg.patch != 8:
        raise ValueError("qualified source-view W128/P8 required")
    with torch.no_grad():
        clean = torch.where(a, x, base.center)
        level = torch.asinh((clean - base.center) / base.scale)
        joined = torch.cat((level, a.to(level.dtype)), -1)
        return base.embedding(joined.reshape(len(x), 16, 368)).detach()


def main(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    ids = io23.source_roles()["train"]
    events = read_training_events(DEFAULT_LABELS, set(ids))
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    original = state_digest(base)
    records, xs, masks, raw_hashes, omissions = [], [], [], {}, []
    reader = io23.read_signals

    def checked(device, sid, *args, **kwargs):
        if device != "J-TEXT" or sid not in ids:
            raise ValueError("only locked J-TEXT TRAIN signals allowed")
        return reader(device, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=checked),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("hard outcomes not used for embedding teaching"),
        ),
    ):
        for index, sid in enumerate(ids, 1):
            path = io23.path_for("J-TEXT", sid)
            digest = sha(path)
            shot = simple_data.read_shot("J-TEXT", sid)
            points, omitted = endpoints(shot["times"], events.get(sid, []))
            omissions.extend(
                dict(shot=sid, event_index=i, reason="interval_outside_record")
                for i in omitted
            )
            for point in points:
                x, a = simple_data.window(shot, point["query"], 128)
                if not a.any():
                    omissions.append(
                        dict(shot=sid, query=point["query"], reason="no_observed_token")
                    )
                    continue
                xs.append(x)
                masks.append(a)
                records.append(
                    dict(
                        shot=sid,
                        role="train",
                        query_time=float(shot["times"][point["query"]]),
                        **point,
                    )
                )
            if sha(path) != digest:
                raise RuntimeError("source file changed during window preparation")
            raw_hashes[str(sid)] = digest
            if index % 128 == 0 or index == len(ids):
                print(
                    dict(stage="TRAIN_windows", shots=index, windows=len(records)),
                    flush=True,
                )
    x = torch.as_tensor(np.stack(xs), device="cuda")
    a = torch.as_tensor(np.stack(masks), device="cuda")
    del xs, masks
    schedule = make_schedule(records)
    teacher = torch.cat(
        [
            native_embedding(base, x[i : i + 256], a[i : i + 256])
            for i in range(0, len(x), 256)
        ]
    )
    repair = TokenRepair().to(base.center.device)
    with torch.no_grad():
        first = a[:4].clone()
        first[:, :, 2] = False
        initial = forward(base, repair, x[:4], first)
        native = base(x[:4], first)
        torch.testing.assert_close(initial["logit"], native["logit"], atol=0, rtol=0)
        torch.testing.assert_close(
            initial["embedding_before"],
            native_embedding(base, x[:4], first),
            atol=0,
            rtol=0,
        )
    output.mkdir(parents=True)
    code = imported_code_hashes()
    bound = {
        str(p): sha(p) for p in (PLAN, DEFAULT_PARENT, io23.JTEXT_INDEX, DEFAULT_LABELS)
    }
    protocol = dict(
        scope="OFFLINE_SOURCE_TRAIN_PRIVILEGED_EMBEDDING_REPAIR_NOT_JTEXT_ONLINE_OR_EAST_GAIN",
        source_sha256=PARENT_SHA256,
        source_state_sha256=original,
        source_parameter_updates=0,
        source_signal_shots=len(raw_hashes),
        train_windows=len(records),
        val_signal_reads=0,
        test_signal_reads=0,
        east_signal_reads=0,
        hard_outcome_reads=0,
        steps=512,
        batch_size=32,
        lr=0.001,
        optimizer="Adam",
        weight_decay=0,
        clip_grad_norm=1.0,
        seed=131,
        objective="patch embedding MSE to native same-prefix teacher + .01 correction MSE; no risk loss",
        event_use="TRAIN interval endpoint enrichment only; no event category or CSV outcome in model/loss",
        partial_prefixes_included=True,
        fixed_last_iterate=True,
        threshold=float(parent["threshold"]),
        raw_sha256=raw_hashes,
        input_sha256=bound,
        code_sha256=code,
    )
    write(output / "protocol.json", protocol)
    write(output / "records.json", records)
    write(output / "schedule.json", schedule)
    write(output / "omissions.json", omissions)
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    optimizer = torch.optim.Adam(repair.parameters(), lr=0.001)
    history = []
    for step in schedule:
        idx = step["indices"]
        mask = a[idx].clone()
        mask[:, :, VIEWS[step["view"]]] = False
        optimizer.zero_grad(set_to_none=True)
        result = forward(base, repair, x[idx], mask)
        loss, info = embedding_loss(
            result["embedding_before"],
            result["embedding_after"],
            teacher[idx],
            result["repair_tokens"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite supported patch teaching loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            repair.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            dict(
                step=step["step"],
                view=step["view"],
                loss=float(loss.detach()),
                mse=float(info["mse"].detach()),
                before_mse=float(info["before_mse"].detach()),
                gradient_norm=float(norm),
                tokens=info["tokens"],
            )
        )
        if step["step"] % 128 == 0:
            print(dict(stage="repair_fit", **history[-1]), flush=True)
    optimizer.zero_grad(set_to_none=True)
    repair.requires_grad_(False).eval()
    constants = {}
    with torch.no_grad():
        for view, columns in VIEWS.items():
            total = base.center.new_zeros(64, dtype=torch.float64)
            count = 0
            for lo in range(0, len(x), 256):
                xx, aa = x[lo : lo + 256], a[lo : lo + 256].clone()
                aa[:, :, columns] = False
                result = forward(base, repair, xx, aa)
                support = result["repair_tokens"]
                total += (
                    (teacher[lo : lo + 256] - result["embedding_before"])[support]
                    .double()
                    .sum(0)
                )
                count += int(support.sum())
            if not count:
                raise ValueError("constant baseline has no TRAIN token support")
            constants[view] = (total / count).to(base.center.dtype).cpu()
    torch.save(
        dict(
            repair={k: v.cpu() for k, v in repair.state_dict().items()},
            constant_delta=constants,
            source_sha256=PARENT_SHA256,
            trained_not_validated=True,
        ),
        output / "repair.pt",
    )
    write(output / "training.json", history)
    if state_digest(base) != original or any(
        p.grad is not None for p in base.parameters()
    ):
        raise AssertionError("qualified source changed during auxiliary fit")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound producer/input changed during fitting")
    result = dict(
        status="TRAINED_EMBEDDING_REPAIR_NOT_VALIDATED",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        source_state_sha256=original,
        source_signal_shots=len(raw_hashes),
        windows=len(records),
        steps=512,
        auxiliary_parameters=sum(p.numel() for p in repair.parameters()),
        val_signal_reads=0,
        test_signal_reads=0,
        east_signal_reads=0,
        hard_outcome_reads=0,
        classification_gain_measured=False,
        online_neural_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", result)
    print(
        {k: result[k] for k in ("status", "seconds", "windows", "source_signal_shots")},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
