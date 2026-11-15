"""Same360 replay: crossed connection training and delayed commit policies."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..cold20_compare_20261104.data import make_schedule
from ..direct_fast_memory_20261104.model import predict
from ..direct_fast_memory_20261104.timeline import all_keys, window_batch
from ..hdf5_score_recheck_20261103 import score_curve
from ..history_gradient_metric_20261102.run import sha, write, read
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
from .timeline import trace
from .training import records_and_pools, fit_connection

PLAN = io23.ROOT / "SHADOW_COMMIT_PLAN_20261104.md"
PREFIT = io23.ROOT / "probe_outputs/direct_fast_memory_ridge360_20261104_v1"
PREFIT_DONE_SHA = "faeeaa386e3d160b4da4a2582adda79d3a3b226029c0767fe187bcdd90e732f6"
STEPS = 512
TRACES = {
    "guard_real": ("real", "guard"),
    "delay_real": ("real", "delay"),
    "guard_hold": ("hold", "guard"),
    "guard_reverse": ("reverse", "guard"),
}
ARM_MAP = {
    "guard_guard": ("guard", "guard_real"),
    "guard_delay": ("guard", "delay_real"),
    "delay_guard": ("delay", "guard_real"),
    "delay_delay": ("delay", "delay_real"),
    "guard_hold": ("guard", "guard_hold"),
    "guard_reverse": ("guard", "guard_reverse"),
}
ARMS = ("native", *ARM_MAP)


def load_initial(ids, past, truth, device):
    """Only called after the same20 have ended; no future-fitted state allowed."""
    if set(past) != set(truth) or set(past) != set(ids) or len(ids) != 20:
        raise ValueError("exactly twenty ended identities required before W0 access")
    done_path = PREFIT / "completed.json"
    if sha(done_path) != PREFIT_DONE_SHA:
        raise ValueError("pinned RLS reference completion changed")
    done = read(done_path)
    bound = {str(done_path): PREFIT_DONE_SHA}
    for name in ("fitted.pt", "fit.json", "protocol.json"):
        want = done["artifact_sha256"][name]
        if sha(PREFIT / name) != want:
            raise ValueError("ended20 W0 reference changed")
        bound[str(PREFIT / name)] = want
    saved = torch.load(PREFIT / "fitted.pt", map_location="cpu", weights_only=True)
    fit = read(PREFIT / "fit.json")
    if (
        saved["label_shots"] != ids
        or saved["source_sha256"] != PARENT_SHA256
        or saved["update_rule"] != "ridge"
        or fit["suffix_training_shots"] != 0
        or fit["label_shots"] != ids
    ):
        raise ValueError("same source and ended20-only initial predictor required")
    origin = saved["initial"].to(device)
    if (
        origin.shape != (168, 70)
        or origin.dtype != torch.float32
        or not torch.isfinite(origin).all()
    ):
        raise ValueError("finite original direct predictor required")
    return origin, fit["training_records"], bound


@torch.no_grad()
def predict_curves(base, initial, connections, shot, *, device):
    n = len(shot["values"])
    curves = {arm: np.empty(n, np.float32) for arm in ARMS}
    histories, receipts, innovations = {}, {}, {}
    if initial is not None:
        if set(connections) != {"guard", "delay"}:
            raise ValueError("both crossed connections required")
        keys = all_keys(base, shot, device=device)
        for name, (feedback, policy) in TRACES.items():
            innovations[name], histories[name], receipts[name] = trace(
                base, shot, initial, feedback, policy=policy, device=device, keys=keys
            )
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        x, a = window_batch(shot, range(lo, hi), device=device)
        curves["native"][lo:hi] = base(x, a)["logit"].sigmoid().cpu().numpy()
        for arm, (connection, trajectory) in ARM_MAP.items():
            curves[arm][lo:hi] = (
                curves["native"][lo:hi]
                if initial is None
                else predict(
                    base, connections[connection], x, a, innovations[trajectory][lo:hi]
                )
                .sigmoid()
                .cpu()
                .numpy()
            )
    if any(not np.isfinite(s).all() for s in curves.values()):
        raise FloatingPointError("nonfinite shadow risk")
    return curves, histories, receipts


def fit_pair(base, past, truth, ids, *, threshold, device, output):
    initial, expected_records, bound = load_initial(ids, past, truth, device)
    records, pools = records_and_pools(past, truth, ids)
    if records != expected_records:
        raise ValueError("new ended20 records differ from the matched RLS reference")
    schedule = make_schedule(pools, steps=STEPS, batch_size=128, seed=131)
    write(output / "training_records.json", records)
    write(output / "training_schedule.json", schedule)
    write(output / "fit_input_hashes.json", bound)
    innovations = {p: {} for p in ("guard", "delay")}
    histories = {p: {} for p in ("guard", "delay")}
    for sid in ids:
        keys = all_keys(base, past[sid], device=device)
        for policy in innovations:
            delta, history, _ = trace(
                base,
                past[sid],
                initial,
                "real",
                policy=policy,
                device=device,
                keys=keys,
            )
            innovations[policy][sid] = delta
            histories[policy][str(sid)] = history
    connections = {}
    fit_history = {}
    for policy in innovations:
        connections[policy], fit_history[policy] = fit_connection(
            base,
            past,
            records,
            schedule,
            innovations[policy],
            threshold=threshold,
            device=device,
            name=policy,
        )
    torch.save(
        dict(
            initial=initial.cpu(),
            connections={
                p: {k: v.cpu() for k, v in m.state_dict().items()}
                for p, m in connections.items()
            },
            label_shots=ids,
            source_sha256=PARENT_SHA256,
        ),
        output / "fitted.pt",
    )
    write(
        output / "fit.json",
        dict(
            label_shots=ids,
            source_training_shots=0,
            suffix_training_shots=0,
            records=len(records),
            sample_counts={g: sum(map(len, v.values())) for g, v in pools.items()},
            steps=STEPS,
            batch_size=128,
            parameter_counts={
                p: sum(x.numel() for x in m.parameters())
                for p, m in connections.items()
            },
            forecast_histories=histories,
            connection_history=fit_history,
            previous_W0_sha256=bound,
        ),
    )
    return initial, connections, bound


def run(output, *, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device).eval().requires_grad_(False)
    original = state_digest(base)
    threshold = float(parent["threshold"])
    if (
        inputs["baseline_protocol"]["threshold"] != threshold
        or inputs["baseline_protocol"]["source_state_sha256"] != original
    ):
        raise ValueError("same qualified source and threshold required")
    code = imported_code_hashes()
    bound = {
        **inputs["bound"],
        str(PLAN): sha(PLAN),
        str(DEFAULT_PARENT): sha(DEFAULT_PARENT),
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="SHADOW_DELAYED_COMMIT_CROSSED_DEVELOPMENT",
            source_sha256=PARENT_SHA256,
            source_state_sha256=original,
            threshold=threshold,
            order=inputs["order"],
            arms=ARMS,
            arm_map=ARM_MAP,
            traces=TRACES,
            startup20_fixed=True,
            first_possible_state_change=167,
            horizon_samples=8,
            update_interval_samples=32,
            explicit_issue_before_target=True,
            validation_uses_actual_signal_only=True,
            training_after_ended20_only=True,
            fitting_steps=STEPS,
            independent_validation=False,
            prior_W0_reference=str(PREFIT),
            prior_completion_sha256=PREFIT_DONE_SHA,
            input_sha256=bound,
            code_sha256=code,
            whole_shot_io_causal_replay=True,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in code:
            z.write(path, str(Path(path).relative_to(io23.ROOT)))
        z.write(PLAN, PLAN.name)
    per = {arm: [] for arm in ARMS}
    seals = []
    calls = []
    past = {}
    truth = {}
    initial = connections = fit_states = None
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        rawpath = io23.path_for("EAST", sid)
        if sha(rawpath) != inputs["hashes"][str(sid)]:
            raise ValueError("raw HDF changed")

        def checked(domain, requested, *args, **kwargs):
            if domain != "EAST" or requested != sid:
                raise ValueError("only current signal allowed")
            calls.append(sid)
            return reader(domain, requested, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23, "read_outcome", side_effect=RuntimeError("no outcome before seal")
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = prepare_actual(raw, index, inputs["mapping"])
            curves, history, forecasts = predict_curves(
                base, initial, connections, signal, device=device
            )
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
            with npz.open("xb") as f:
                np.savez_compressed(f, times=raw["times"], **curves)
                f.flush()
                os.fsync(f.fileno())
            forecast_sha = None
            if forecasts:
                forecastpath = output / "shots" / (prefix + "_forecasts.npz")
                arrays = {
                    name + "__" + field: value
                    for name, receipt in forecasts.items()
                    for field, value in receipt.items()
                }
                with forecastpath.open("xb") as f:
                    np.savez_compressed(f, **arrays)
                    f.flush()
                    os.fsync(f.fileno())
                forecast_sha = sha(forecastpath)
            historypath = output / "shots" / (prefix + "_updates.json")
            durable_write(historypath, history)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                forecast_sha256=forecast_sha,
                history_sha256=sha(historypath),
                current_outcome_used_in_prediction=False,
                adaptation_enabled=index > 20,
                shot_state_reset=True,
            )
            durable_write(output / "shots" / (prefix + "_seal.json"), seal)
            directory = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            seals.append(seal)
        if sha(npz) != seal["curve_sha256"]:
            raise RuntimeError("sealed curve changed")
        with np.load(npz, allow_pickle=False) as a:
            times = a["times"].copy()
            scores = {arm: a[arm].copy() for arm in ARMS}
        flag, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            per[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    down_s=down,
                    **score_curve(times, scores[arm], flag, down, threshold),
                )
            )
        if index <= 20:
            past[sid] = raw
            truth[sid] = (flag, down)
        if index == 20:
            mapped = {
                s: dict(
                    prepare_actual(past[s], 21, inputs["mapping"]),
                    times=past[s]["times"],
                )
                for s in ids[:20]
            }
            with (
                patch.object(
                    io23,
                    "read_signals",
                    side_effect=RuntimeError("no new signals during fit"),
                ),
                patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("only ended truth during fit"),
                ),
            ):
                initial, connections, fit_bound = fit_pair(
                    base,
                    mapped,
                    truth,
                    ids[:20],
                    threshold=threshold,
                    device=device,
                    output=output,
                )
            bound.update(fit_bound)
            fit_states = {p: state_digest(m) for p, m in connections.items()}
            initial_state = initial.clone()
            past.clear()
            mapped.clear()
        if index % 20 == 0 or index == len(ids):
            print(
                dict(
                    stage="shadow_commit_replay",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids
        or initial is None
        or not torch.equal(initial, initial_state)
        or state_digest(base) != original
        or any(p.grad is not None for p in base.parameters())
        or any(
            state_digest(m) != fit_states[p]
            or any(x.requires_grad or x.grad is not None for x in m.parameters())
            for p, m in connections.items()
        )
    ):
        raise AssertionError(
            "source, fitted connection, initial state or I/O contract changed"
        )
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound inputs or producer changed")
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
    done = dict(
        status="COMPLETED_SHADOW_COMMIT_DIAGNOSTIC",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        native_exact_baseline=True,
        fit_label_shots=ids[:20],
        suffix_training_shots=0,
        postseal_outcomes=len(seals),
        seals=seals,
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
    run(args.output, device=args.device)
