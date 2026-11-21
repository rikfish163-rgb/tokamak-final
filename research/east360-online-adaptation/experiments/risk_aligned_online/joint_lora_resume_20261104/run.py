"""Continue a verified timed-out flow without rewriting sealed predictions."""

import argparse
import json
from pathlib import Path
import os
import shutil
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..history_gradient_metric_20261102.run import read, sha, write
from ..hdf5_score_recheck_20261103 import score_curve
from ..joint_lora_20261104.run import ARMS, native_curve, paired, frozen_digest
from ..joint_lora_20261104.timeline import run_shot
from ..joint_lora_20261104.training import ended_records, eos_schedule, train_schedule
from ..mask_repair_20261104.east_evaluation import (
    configure,
    load_inputs,
    prepare_actual,
    check_reference,
)
from ..mask_repair_20261104.evaluation import _write as durable_write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..source_anticipation_20261104.static_stream import timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes

PLAN = io23.ROOT / "JOINT_LORA_RESUME_PLAN_20261104.md"


def process_is_live(pid, start_ticks):
    path = Path(f"/proc/{pid}/stat")
    try:
        text = path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[19]) == start_ticks and fields[0] not in ("Z", "X")


def _checkpoint(output, index, sid, slow, fixed_state):
    temp = output / "checkpoint.next.pt"
    with temp.open("xb") as stream:
        torch.save(
            dict(
                completed_index=index,
                shot=sid,
                fixed_state_sha256=fixed_state,
                slow={k: v.detach().cpu() for k, v in slow.state_dict().items()},
            ),
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, output / "checkpoint.pt")


def _ended_update(
    output, prefix, index, sid, slow, past, truths, replay, *, threshold, device, steps
):
    current, _ = ended_records({sid: past[sid]}, {sid: truths[sid]}, [sid])
    records, schedule = eos_schedule(
        current, replay, seed=131 + index, steps=steps, batch_size=64
    )
    before = state_digest(slow)
    history = train_schedule(
        slow, past, records, schedule, threshold=threshold, device=device, lr=0.0001
    )
    used = sorted({i for draw in schedule for i in draw["indices"]})
    local = {old: new for new, old in enumerate(used)}
    selected = [records[i] for i in used]
    write(
        output / "inter" / (prefix + ".json"),
        dict(
            shot=sid,
            available_ended_shots=list(past),
            training_shots=sorted({r["shot"] for r in selected}),
            shared_slow_for=["inter", "both"],
            parent_before=before,
            parent_after=state_digest(slow),
            current_records=len(current),
            records=selected,
            replay_buffer_indices=used,
            schedule=[
                dict(step=d["step"], indices=[local[i] for i in d["indices"]])
                for d in schedule
            ],
            history=history,
        ),
    )
    return current


def run(
    source, output, *, source_pid, source_start_ticks, source_exit_code, device="cuda"
):
    if process_is_live(source_pid, source_start_ticks):
        raise RuntimeError(
            "source producer is still active; never run concurrent recovery"
        )
    if source_exit_code != 124:
        raise ValueError("only a tool-confirmed hard timeout124 may use this protocol")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or (source / "completed.json").exists():
        raise FileExistsError(
            "new output and an actually incomplete source are required"
        )
    from .state import restore_warm, replay_updates

    started = time.monotonic()
    p = read(source / "protocol.json")
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device).eval().requires_grad_(False)
    original = state_digest(base)
    threshold = float(parent["threshold"])
    if (
        p["order"] != inputs["order"]
        or p["threshold"] != threshold
        or p["source_state_sha256"] != original
        or p["arms"] != list(ARMS)
    ):
        raise ValueError("the exact source, order, threshold and arms are required")
    for path, digest in {**p["code_sha256"], **p["input_sha256"]}.items():
        if sha(Path(path)) != digest:
            raise ValueError("original producer/input changed")
    names = {f.name for f in (source / "shots").glob("*_seal.json")}
    count = 0
    for index, sid in enumerate(ids, 1):
        if f"{index:04d}_{sid}_seal.json" not in names:
            break
        count = index
    if len(names) != count or count < 20:
        raise ValueError("a contiguous prefix with complete warm20 is required")
    eos = []
    partial_eos = None
    for index, sid in enumerate(ids[20:count], 21):
        path = source / "inter" / f"{index:04d}_{sid}.json"
        if not path.is_file():
            break
        try:
            read(path)
        except json.JSONDecodeError:
            # A verified timeout may interrupt the final JSON write after its
            # prediction was sealed. Preserve this partial file as evidence,
            # never consume it as a completed update. Earlier corruption or
            # a syntactically valid but invalid log is NOT recoverable here.
            if index != count:
                raise
            partial_eos = path
            break
        eos.append(path)
    if len(list((source / "inter").glob("*.json"))) != len(eos) + int(
        partial_eos is not None
    ) or count - 20 - len(eos) not in (0, 1):
        raise ValueError("only the final sealed EOS may be unfinished, no gaps")
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    (output / "inter").mkdir()
    copied = {}

    def copy(src, dst):
        before = sha(src)
        shutil.copy2(src, dst)
        if sha(src) != before or sha(dst) != before:
            raise ValueError("prefix changed during copy")
        copied[str(src.relative_to(source))] = before

    for filename in ("warm.json", "warm.pt", "code.zip"):
        copy(source / filename, output / filename)
    for index, sid in enumerate(ids[:count], 1):
        prefix = f"{index:04d}_{sid}"
        seal = read(source / "shots" / (prefix + "_seal.json"))
        curve = source / "shots" / (prefix + ".npz")
        if (
            seal["shot"] != sid
            or seal["current_outcome_used_in_prediction"] is not False
            or sha(curve) != seal["curve_sha256"]
        ):
            raise ValueError("original prefix prediction seal mismatch")
        for suffix in (".npz", "_seal.json", "_updates.json"):
            copy(
                source / "shots" / (prefix + suffix),
                output / "shots" / (prefix + suffix),
            )
        if seal["forecast_sha256"] is not None:
            path = source / "shots" / (prefix + "_forecasts.npz")
            if sha(path) != seal["forecast_sha256"]:
                raise ValueError("original forecast seal mismatch")
            copy(path, output / "shots" / path.name)
    for path in eos:
        copy(path, output / "inter" / path.name)
    if partial_eos is not None:
        copy(partial_eos, output / "interrupted_last_eos.json.partial")
    code = imported_code_hashes()
    code.update({str(f): sha(f) for f in Path(__file__).parent.glob("*.py")})
    bounds = {
        **inputs["bound"],
        str(PLAN): sha(PLAN),
        str(source / "protocol.json"): sha(source / "protocol.json"),
    }
    protocol = {
        **p,
        "resume": dict(
            source=str(source),
            source_pid=source_pid,
            source_start_ticks=source_start_ticks,
            tool_reported_exit_code=source_exit_code,
            sealed_prefix=count,
            completed_eos=len(eos),
            prefix_predictions_recomputed=False,
            learning_budget_changed=False,
        ),
        "code_sha256": {**p["code_sha256"], **code},
        "input_sha256": {**p["input_sha256"], **bounds},
    }
    write(output / "protocol.json", protocol)
    with zipfile.ZipFile(
        output / "resume_code.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for filename in code:
            archive.write(filename, str(Path(filename).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    write(
        output / "resume_provenance.json",
        dict(
            source=str(source),
            source_exit_code=source_exit_code,
            copied_sha256=copied,
            source_process_confirmed_terminal=True,
            source_not_modified=True,
        ),
    )
    past, truths, per = {}, {}, {arm: [] for arm in ARMS}
    for index, sid in enumerate(ids[:count], 1):
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError("ended HDF changed")
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        past[sid] = dict(
            prepare_actual(raw, 21, inputs["mapping"]),
            device="EAST",
            times=raw["times"],
        )
        truths[sid] = io23.read_outcome("EAST", sid)
        prefix = f"{index:04d}_{sid}"
        with np.load(
            output / "shots" / (prefix + ".npz"), allow_pickle=False
        ) as values:
            assert np.array_equal(values["times"], raw["times"])
            flag, down = truths[sid]
            for arm in ARMS:
                per[arm].append(
                    dict(
                        shot=sid,
                        disruptive=flag,
                        down_s=down,
                        **score_curve(
                            values["times"], values[arm], flag, down, threshold
                        ),
                    )
                )
    warm = read(output / "warm.json")
    if warm["label_shots"] != ids[:20]:
        raise ValueError("warm shot membership changed")
    records, _ = ended_records(
        {s: past[s] for s in ids[:20]}, {s: truths[s] for s in ids[:20]}, ids[:20]
    )
    if records != warm["records"]:
        raise ValueError("warm records differ from fresh ended HDF truth")
    fixed = restore_warm(base, output / "warm.pt", warm, device=device)
    fixed_state = state_digest(fixed)
    immutable = frozen_digest(fixed)
    with (
        patch.object(
            io23,
            "read_signals",
            side_effect=RuntimeError("no new signals in historical state replay"),
        ),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("only revealed outcomes in replay"),
        ),
    ):
        completed_prefix = ids[: 20 + len(eos)]
        slow, receipts = replay_updates(
            fixed,
            {s: past[s] for s in completed_prefix},
            {s: truths[s] for s in completed_prefix},
            completed_prefix,
            [output / "inter" / f.name for f in eos],
            threshold=threshold,
            device=device,
            expected_steps=p["eos_steps"],
            expected_batch=p["eos_batch"],
        )
        replay = list(records)
        for index, sid in enumerate(ids[20:count], 21):
            seal = read(output / "shots" / f"{index:04d}_{sid}_seal.json")
            if seal["shot_start_parents"]["static"] != fixed_state:
                raise ValueError("fixed parent chain mismatch")
            if index - 21 < len(eos):
                log = read(output / "inter" / eos[index - 21].name)
                if seal["shot_start_parents"]["slow"] != log["parent_before"]:
                    raise ValueError("prefix slow seal differs from EOS parent")
                current, _ = ended_records({sid: past[sid]}, {sid: truths[sid]}, [sid])
            else:
                if seal["shot_start_parents"]["slow"] != state_digest(slow):
                    raise ValueError("unfinished final EOS parent mismatch")
                allowed = {s: past[s] for s in ids[:index]}
                current = _ended_update(
                    output,
                    f"{index:04d}_{sid}",
                    index,
                    sid,
                    slow,
                    allowed,
                    truths,
                    replay,
                    threshold=threshold,
                    device=device,
                    steps=p["eos_steps"],
                )
            replay.extend(current)
    if frozen_digest(slow) != immutable:
        raise ValueError("frozen source weights changed in reconstruction")
    write(
        output / "reconstruction.json",
        dict(
            replayed_eos=receipts,
            last_sealed_eos_completed_after_interrupt=count - 20 > len(eos),
            restored_slow_sha256=state_digest(slow),
            past_predictions_recomputed=0,
        ),
    )
    _checkpoint(output, count, ids[count - 1], slow, fixed_state)
    print(
        dict(
            stage="joint_lora_exact_resume",
            sealed_prefix=count,
            verified_eos=len(receipts),
            seconds=time.monotonic() - started,
        ),
        flush=True,
    )
    reader = io23.read_signals
    calls = []
    for index, sid in enumerate(ids[count:], count + 1):
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError("current HDF changed")

        def checked(domain, requested, *args, **kwargs):
            if domain != "EAST" or requested != sid:
                raise ValueError("only current signals allowed")
            calls.append(sid)
            return reader(domain, requested, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23, "read_outcome", side_effect=RuntimeError("no outcome before seal")
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = dict(
                prepare_actual(raw, index, inputs["mapping"]),
                device="EAST",
                times=raw["times"],
            )
            parents = dict(static=state_digest(fixed), slow=state_digest(slow))
            curves = {"native": native_curve(base, signal, device=device)}
            updates = {}
            forecasts = {}
            for arm, model, mode in (
                ("static", fixed, "none"),
                ("intra", fixed, "real"),
                ("inter", slow, "none"),
                ("both", slow, "real"),
                ("hold", fixed, "hold"),
                ("reverse", fixed, "reverse"),
            ):
                curves[arm], updates[arm], forecast = run_shot(
                    model, signal, mode, device=device
                )
                forecasts.update(
                    {arm + "__" + key: value for key, value in forecast.items()}
                )
            if (
                state_digest(fixed) != parents["static"]
                or state_digest(slow) != parents["slow"]
            ):
                raise ValueError("fast update escaped its local copy")
            prefix = f"{index:04d}_{sid}"
            reference = inputs["baseline_shots"] / prefix
            check_reference(
                reference.with_suffix(".npz"),
                reference.with_name(prefix + "_seal.json"),
                inputs["seals"][index - 1],
                raw["times"],
                curves["native"],
            )
            npz = output / "shots" / (prefix + ".npz")
            with npz.open("xb") as stream:
                np.savez_compressed(stream, times=raw["times"], **curves)
                stream.flush()
                os.fsync(stream.fileno())
            forecast_path = output / "shots" / (prefix + "_forecasts.npz")
            with forecast_path.open("xb") as stream:
                np.savez_compressed(stream, **forecasts)
                stream.flush()
                os.fsync(stream.fileno())
            update_path = output / "shots" / (prefix + "_updates.json")
            durable_write(update_path, updates)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                forecast_sha256=sha(forecast_path),
                updates_sha256=sha(update_path),
                current_outcome_used_in_prediction=False,
                shot_start_parents=parents,
                same_slow_for_inter_and_both=True,
            )
            durable_write(output / "shots" / (prefix + "_seal.json"), seal)
            fd = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        assert sha(npz) == seal["curve_sha256"]
        flag, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            per[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    down_s=down,
                    **score_curve(raw["times"], curves[arm], flag, down, threshold),
                )
            )
        past[sid] = signal
        truths[sid] = (flag, down)
        with (
            patch.object(
                io23,
                "read_signals",
                side_effect=RuntimeError("no new signals in ended training"),
            ),
            patch.object(
                io23,
                "read_outcome",
                side_effect=RuntimeError("only revealed outcomes in EOS"),
            ),
        ):
            current = _ended_update(
                output,
                prefix,
                index,
                sid,
                slow,
                past,
                truths,
                replay,
                threshold=threshold,
                device=device,
                steps=p["eos_steps"],
            )
            replay.extend(current)
        if frozen_digest(slow) != immutable:
            raise ValueError("frozen weights changed in continuation")
        _checkpoint(output, index, sid, slow, fixed_state)
        if index % 10 == 0 or index == len(ids):
            print(
                dict(
                    stage="joint_lora_continuation",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids[count:]
        or state_digest(base) != original
        or state_digest(fixed) != fixed_state
    ):
        raise ValueError("source, fixed or signal boundary changed")
    for path, digest in {**protocol["code_sha256"], **protocol["input_sha256"]}.items():
        if sha(Path(path)) != digest:
            raise ValueError("bound code/input changed")
    for name, digest in copied.items():
        if sha(source / name) != digest:
            raise ValueError("original prefix was modified")
    summary = {
        arm: {
            name: timing_decomposition(rows, deadline_ms=30)
            for name, rows in (
                ("full360", per[arm]),
                ("startup20", per[arm][:20]),
                ("suffix340", per[arm][20:]),
            )
        }
        for arm in ARMS
    }
    write(output / "per_shot.json", per)
    write(output / "summary.json", summary)
    write(output / "paired.json", paired(per))
    torch.save(
        {k: v.detach().cpu() for k, v in slow.state_dict().items()},
        output / "final_slow.pt",
    )
    done = dict(
        status="COMPLETED_JOINT_LORA_PAIRED_DEVELOPMENT",
        resumed=True,
        resumed_prefix_shots=count,
        recomputed_past_predictions=0,
        last_sealed_eos_completed_after_interrupt=count - 20 > len(eos),
        postseal_outcomes=len(ids),
        source_unchanged=True,
        static_parent_unchanged=True,
        independent_validation=False,
        original_process_exit_code=source_exit_code,
        continuation_seconds=time.monotonic() - started,
        artifact_sha256={
            str(f.relative_to(output)): sha(f) for f in output.rglob("*") if f.is_file()
        },
    )
    write(output / "completed.json", done)
    print(
        dict(
            status=done["status"], full360={a: s["full360"] for a, s in summary.items()}
        ),
        flush=True,
    )
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--source-start-ticks", type=int, required=True)
    parser.add_argument("--source-exit-code", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = parser.parse_args()
    if a.device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
    run(
        a.source,
        a.output,
        source_pid=a.source_pid,
        source_start_ticks=a.source_start_ticks,
        source_exit_code=a.source_exit_code,
        device=a.device,
    )
