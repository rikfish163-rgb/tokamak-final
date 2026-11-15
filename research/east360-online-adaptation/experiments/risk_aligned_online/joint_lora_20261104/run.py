"""Paired complete EAST flow: fixed alarm, internal fast and slow learning."""

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..cold20_compare_20261104.data import make_schedule
from ..direct_fast_memory_20261104.timeline import window_batch
from ..hdf5_score_recheck_20261103 import score_curve
from ..history_gradient_metric_20261102.run import sha, write
from ..mask_repair_20261104.east_evaluation import (
    configure,
    load_inputs,
    prepare_actual,
    check_reference,
)
from ..mask_repair_20261104.evaluation import _write as durable_write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_anticipation_20261104.static_stream import timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .model import JointLoRA
from .training import ended_records, train_schedule, eos_schedule

ARMS = ("native", "static", "intra", "inter", "both", "hold", "reverse")
WARM_STEPS, EOS_STEPS = 512, 8
PLAN = io23.ROOT / "JOINT_LORA_PLAN_20261104.md"
PRIOR = io23.ROOT / "probe_outputs/direct_fast_memory_ridge360_20261104_v1/fitted.pt"
PRIOR_SHA = "86ed0b2315ced2fb13ebe4e1b8fcd1bb202760903edc14b1054c371f4e365c47"


def load_initial(ids, device):
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError("pinned ended20 diagnostic forecaster changed")
    value = torch.load(PRIOR, map_location=device, weights_only=False)
    if value["label_shots"] != ids or value["source_sha256"] != PARENT_SHA256:
        raise ValueError("the exact same ended20 and source are required")
    return value["initial"].detach().clone()


def frozen_digest(model):
    allowed = {id(p) for p in (*model.neural_parameters(), *model.decoder_parameters())}
    digest = hashlib.sha256()
    for name, p in (*model.named_parameters(), *model.named_buffers()):
        if id(p) not in allowed:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def native_curve(base, shot, *, device):
    values = np.empty(len(shot["values"]), np.float32)
    for lo in range(0, len(values), 256):
        hi = min(lo + 256, len(values))
        x, a = window_batch(shot, range(lo, hi), device=device)
        values[lo:hi] = base(x, a)["logit"].sigmoid().cpu().numpy()
    return values


def paired(per):
    result = {}
    for candidate, reference in (
        ("intra", "static"),
        ("inter", "static"),
        ("both", "inter"),
        ("both", "intra"),
        ("intra", "hold"),
        ("intra", "reverse"),
    ):
        rows = list(zip(per[candidate], per[reference], strict=True))
        result[candidate + "_vs_" + reference] = dict(
            timely_gained=[
                a["shot"]
                for a, b in rows
                if a["disruptive"] and a["timely"] and not b["timely"]
            ],
            timely_lost=[
                a["shot"]
                for a, b in rows
                if a["disruptive"] and not a["timely"] and b["timely"]
            ],
            false_alarms_added=[
                a["shot"]
                for a, b in rows
                if not a["disruptive"] and a["alarm"] and not b["alarm"]
            ],
            false_alarms_removed=[
                a["shot"]
                for a, b in rows
                if not a["disruptive"] and not a["alarm"] and b["alarm"]
            ],
        )
    return result


def run(output, *, device="cuda"):
    from .timeline import run_shot

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device).eval().requires_grad_(False)
    source_state = state_digest(base)
    threshold = float(parent["threshold"])
    if (
        inputs["baseline_protocol"]["threshold"] != threshold
        or inputs["baseline_protocol"]["source_state_sha256"] != source_state
    ):
        raise ValueError("unchanged qualified source and fixed threshold required")
    code = imported_code_hashes()
    code.update({str(p): sha(p) for p in Path(__file__).parent.glob("*.py")})
    bound = {
        **inputs["bound"],
        str(PLAN): sha(PLAN),
        str(DEFAULT_PARENT): sha(DEFAULT_PARENT),
        str(PRIOR): sha(PRIOR),
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    (output / "inter").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="JOINT_LORA_FAST_SLOW_PAIRED_DEVELOPMENT",
            source_sha256=PARENT_SHA256,
            source_state_sha256=source_state,
            threshold=threshold,
            order=inputs["order"],
            arms=ARMS,
            warm_steps=WARM_STEPS,
            warm_batch=128,
            warm_lr=0.001,
            eos_steps=EOS_STEPS,
            eos_batch=64,
            eos_lr=0.0001,
            intra_lr=0.0001,
            intra_radius=0.5,
            intra_parameters=512,
            diagnostic_head_parameters=32424,
            original_risk_head_frozen=True,
            source_checkpoint_immutable=True,
            prediction_horizon_samples=8,
            update_interval_samples=32,
            inter_and_both_share_one_slow_model=True,
            fast_never_promoted=True,
            first20_original_predictions=True,
            ended_outcomes_automatically_revealed=True,
            expensive_query_policy_tested=False,
            alarm_parameters_updated=False,
            whole_shot_io_but_causal_windows=True,
            independent_future_validation=False,
            input_sha256=bound,
            code_sha256=code,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for filename in code:
            z.write(filename, str(Path(filename).relative_to(io23.ROOT)))
        z.write(PLAN, PLAN.name)
    per = {a: [] for a in ARMS}
    past, truths, replay, calls, seals, inter_ids = {}, {}, [], [], [], []
    fixed = slow = None
    fixed_state = immutable = None
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError("raw HDF changed")

        def checked(domain, requested, *args, **kwargs):
            if domain != "EAST" or requested != sid:
                raise ValueError("only current EAST signals may be opened")
            calls.append(sid)
            return reader(domain, requested, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23, "read_outcome", side_effect=RuntimeError("no outcome before seal")
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            prepared = prepare_actual(raw, index, inputs["mapping"])
            signal = dict(prepared, device="EAST", times=raw["times"])
            curves = {"native": native_curve(base, signal, device=device)}
            updates, forecasts = {}, {}
            parents = (
                None
                if fixed is None
                else {"static": state_digest(fixed), "slow": state_digest(slow)}
            )
            if fixed is None:
                curves.update({arm: curves["native"].copy() for arm in ARMS[1:]})
            else:
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
                    raise AssertionError(
                        "shot-local fast updates escaped into a parent"
                    )
            prefix = f"{index:04d}_{sid}"
            ref = inputs["baseline_shots"] / prefix
            check_reference(
                ref.with_suffix(".npz"),
                ref.with_name(prefix + "_seal.json"),
                inputs["seals"][index - 1],
                raw["times"],
                curves["native"],
            )
            npz = output / "shots" / (prefix + ".npz")
            with npz.open("xb") as stream:
                np.savez_compressed(stream, times=raw["times"], **curves)
                stream.flush()
                os.fsync(stream.fileno())
            forecast_hash = None
            if forecasts:
                forecast_path = output / "shots" / (prefix + "_forecasts.npz")
                with forecast_path.open("xb") as stream:
                    np.savez_compressed(stream, **forecasts)
                    stream.flush()
                    os.fsync(stream.fileno())
                forecast_hash = sha(forecast_path)
            durable_write(output / "shots" / (prefix + "_updates.json"), updates)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                forecast_sha256=forecast_hash,
                current_outcome_used_in_prediction=False,
                shot_start_parents=parents,
                same_slow_for_inter_and_both=True,
            )
            durable_write(output / "shots" / (prefix + "_seal.json"), seal)
            directory = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            seals.append(seal)
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
        truths[sid] = (flag, down)
        # Raw prefix is retained until its ended20 map is available. Later
        # replay stores only already-ended, mapped arrays in RAM, not on disk.
        past[sid] = raw if index <= 20 else signal
        with (
            patch.object(
                io23,
                "read_signals",
                side_effect=RuntimeError("no new signals in ended training"),
            ),
            patch.object(
                io23,
                "read_outcome",
                side_effect=RuntimeError("use only revealed in-memory outcomes"),
            ),
        ):
            if index == 20:
                past = {
                    s: dict(
                        prepare_actual(past[s], 21, inputs["mapping"]),
                        device="EAST",
                        times=past[s]["times"],
                    )
                    for s in ids[:20]
                }
                records, pools = ended_records(past, truths, ids[:20])
                schedule = make_schedule(
                    pools, steps=WARM_STEPS, batch_size=128, seed=131
                )
                fixed = JointLoRA(base, load_initial(ids[:20], device))
                immutable = frozen_digest(fixed)
                history = train_schedule(
                    fixed,
                    past,
                    records,
                    schedule,
                    threshold=threshold,
                    device=device,
                    lr=0.001,
                )
                fixed_state = state_digest(fixed)
                slow = deepcopy(fixed)
                replay = list(records)
                write(
                    output / "warm.json",
                    dict(
                        label_shots=ids[:20],
                        records=records,
                        schedule=schedule,
                        history=history,
                        fitted_state_sha256=fixed_state,
                    ),
                )
                torch.save(
                    {k: v.detach().cpu() for k, v in fixed.state_dict().items()},
                    output / "warm.pt",
                )
            elif index > 20:
                current, _ = ended_records({sid: signal}, {sid: truths[sid]}, [sid])
                records, schedule = eos_schedule(
                    current, replay, seed=131 + index, steps=EOS_STEPS, batch_size=64
                )
                before = state_digest(slow)
                history = train_schedule(
                    slow,
                    past,
                    records,
                    schedule,
                    threshold=threshold,
                    device=device,
                    lr=0.0001,
                )
                # Save the exact used windows and remap the schedule, not a
                # quadratic sequence of whole replay-buffer copies on disk.
                used = sorted({i for draw in schedule for i in draw["indices"]})
                local_index = {original: i for i, original in enumerate(used)}
                selected_records = [records[i] for i in used]
                saved_schedule = [
                    dict(
                        step=draw["step"],
                        indices=[local_index[i] for i in draw["indices"]],
                    )
                    for draw in schedule
                ]
                write(
                    output / "inter" / (prefix + ".json"),
                    dict(
                        shot=sid,
                        available_ended_shots=ids[:index],
                        training_shots=sorted({r["shot"] for r in selected_records}),
                        shared_slow_for=["inter", "both"],
                        parent_before=before,
                        parent_after=state_digest(slow),
                        current_records=len(current),
                        records=selected_records,
                        replay_buffer_indices=used,
                        schedule=saved_schedule,
                        history=history,
                    ),
                )
                replay.extend(current)
                inter_ids.append(sid)
                if frozen_digest(slow) != immutable:
                    raise AssertionError(
                        "an original encoder/head weight or W0 changed"
                    )
        if index % 10 == 0 or index == len(ids):
            print(
                dict(
                    stage="joint_lora_flow",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids
        or fixed is None
        or state_digest(base) != source_state
        or state_digest(fixed) != fixed_state
        or frozen_digest(fixed) != immutable
    ):
        raise AssertionError("source, fixed parent or causal signal access changed")
    if any(sha(Path(p)) != value for p, value in {**bound, **code}.items()):
        raise AssertionError("producer or bound inputs changed")
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
        seconds=time.monotonic() - started,
        source_unchanged=True,
        static_parent_unchanged=True,
        postseal_outcomes=len(seals),
        fit_label_shots=ids[:20],
        inter_shots=inter_ids,
        independent_validation=False,
        artifact_sha256={
            str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
        },
    )
    write(output / "completed.json", done)
    print(
        dict(
            status=done["status"],
            seconds=done["seconds"],
            full360={a: s["full360"] for a, s in summary.items()},
        ),
        flush=True,
    )
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
    run(args.output, device=args.device)
