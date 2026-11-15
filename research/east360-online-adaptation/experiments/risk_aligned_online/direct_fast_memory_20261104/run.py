"""Complete same360 replay with actual diagnostic-driven shot-local weights."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
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
from .fit import fit_prefix
from .model import predict
from .timeline import all_keys, trace, window_batch

ARMS = ("native", "real", "hold", "rotated", "constant")
PLAN = io23.ROOT / "DIRECT_FAST_MEMORY_PLAN_20261104.md"
RIDGE_PLAN = io23.ROOT / "DIRECT_FAST_MEMORY_RLS_PLAN_20261104.md"
STEPS = 512


@torch.no_grad()
def predict_curves(base, initial, injection, constant, shot, *, device, rule="lms"):
    n = len(shot["values"])
    curves = {m: np.empty(n, np.float32) for m in ARMS}
    history, innovations = {}, {}
    if initial is not None:
        keys = all_keys(base, shot, device=device)
        for arm in ("real", "hold", "rotated"):
            innovations[arm], history[arm] = trace(
                base, shot, initial, arm, device=device, keys=keys, rule=rule
            )
        innovations["constant"] = constant.expand(n, -1).clone()
        innovations["constant"][:135] = 0
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        x, a = window_batch(shot, range(lo, hi), device=device)
        curves["native"][lo:hi] = base(x, a)["logit"].sigmoid().cpu().numpy()
        for arm in ARMS[1:]:
            if initial is None:
                curves[arm][lo:hi] = curves["native"][lo:hi]
            else:
                curves[arm][lo:hi] = (
                    predict(base, injection, x, a, innovations[arm][lo:hi])
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
    if any(not np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError("nonfinite direct-memory risk")
    return curves, history


def run(output, *, device="cuda", rule="lms"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if rule not in ("lms", "ridge"):
        raise ValueError("LMS or recursive ridge must be explicit")
    plan = RIDGE_PLAN if rule == "ridge" else PLAN
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
        raise ValueError("unchanged qualified parent and threshold required")
    code = imported_code_hashes()
    bound = {
        **inputs["bound"],
        str(plan): sha(plan),
        str(DEFAULT_PARENT): sha(DEFAULT_PARENT),
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="DIRECT_DIAGNOSTIC_FAST_WEIGHTS_INTRA_ONLY_DEVELOPMENT",
            source_sha256=PARENT_SHA256,
            source_state_sha256=original,
            threshold=threshold,
            arms=ARMS,
            order=inputs["order"],
            first20_original_prediction=True,
            fit_after_shot=20,
            fitting_steps=STEPS,
            horizon_samples=8,
            update_interval_samples=32,
            fast_weight_shape=[168, 70],
            update_rule=rule,
            fast_lr=0.5 if rule == "lms" else None,
            covariance_prior_precision=1.0 if rule == "ridge" else None,
            fast_radius=2.0,
            base_weights_and_risk_head_frozen=True,
            internal_site="pre-Transformer token gain/bias",
            direct_target="21 continuous asinh coordinates, excluding MNM and discrete mode_number_n",
            query_labels_used_intra=False,
            inter_updates_after_prefix=0,
            full360_including_startup=True,
            whole_shot_io_causal_keys_and_arrived_targets=True,
            independent_validation=False,
            input_sha256=bound,
            code_sha256=code,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in code:
            z.write(path, str(Path(path).relative_to(io23.ROOT)))
        z.write(plan, plan.name)
    per = {m: [] for m in ARMS}
    past, truth, seals, calls = {}, {}, [], []
    initial = injection = constant = fit_state = None
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        raw_path = io23.path_for("EAST", sid)
        if sha(raw_path) != inputs["hashes"][str(sid)]:
            raise ValueError("raw HDF changed")

        def checked(domain, requested, *args, **kwargs):
            if domain != "EAST" or requested != sid:
                raise ValueError("only the current shot signal may be read")
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
            curves, history = predict_curves(
                base, initial, injection, constant, signal, device=device, rule=rule
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
            durable_write(output / "shots" / (prefix + "_updates.json"), history)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                current_outcome_used_in_prediction=False,
                weights_reset_to_W0=True,
                adaptation_enabled=index > 20,
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
        with np.load(npz, allow_pickle=False) as f:
            times = f["times"].copy()
            scores = {m: f[m].copy() for m in ARMS}
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
            mapped_past = {
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
                    side_effect=RuntimeError("only ended in-memory truth in fit"),
                ),
            ):
                initial, injection, constant, fit_info = fit_prefix(
                    base,
                    mapped_past,
                    truth,
                    ids[:20],
                    threshold=threshold,
                    device=device,
                    steps=STEPS,
                    rule=rule,
                )
            write(output / "fit.json", fit_info)
            torch.save(
                dict(
                    initial=initial.cpu(),
                    injection={k: v.cpu() for k, v in injection.state_dict().items()},
                    constant=constant.cpu(),
                    label_shots=ids[:20],
                    source_sha256=PARENT_SHA256,
                    update_rule=rule,
                ),
                output / "fitted.pt",
            )
            fit_state = state_digest(injection)
            initial_state, constant_state = initial.clone(), constant.clone()
            past.clear()
            mapped_past.clear()
        if index % 20 == 0 or index == len(ids):
            print(
                dict(
                    stage="direct_memory_replay",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids
        or initial is None
        or state_digest(base) != original
        or state_digest(injection) != fit_state
        or not torch.equal(initial, initial_state)
        or not torch.equal(constant, constant_state)
        or any(p.grad is not None for p in base.parameters())
    ):
        raise AssertionError(
            "frozen source, reset memory or causal I/O contract changed"
        )
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
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
    completed = dict(
        status="COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        native_exact_baseline=True,
        fit_label_shots=ids[:20],
        source_training_shots=0,
        suffix_training_shots=0,
        postseal_outcomes=len(seals),
        seals=seals,
        artifact_sha256={
            str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
        },
    )
    write(output / "completed.json", completed)
    print(
        dict(
            status=completed["status"],
            seconds=completed["seconds"],
            full360={m: s["full360"] for m, s in summary.items()},
        ),
        flush=True,
    )
    return completed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--rule", choices=("lms", "ridge"), default="lms")
    args = p.parse_args()
    if args.device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
    run(args.output, device=args.device, rule=args.rule)
