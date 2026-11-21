"""Same frozen source, full360, factorial density/amplitude input control."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
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
from .calibration import source_reference, fit_gain

ARMS = ("unmapped", "density", "n1", "both")
PLAN = io23.ROOT / "N1_ALIGNMENT_PLAN_20261104.md"


def make_views(raw, index, mapping, gain):
    if type(index) is not int or index < 1:
        raise ValueError("positive stream position required")
    unmapped = prepare_actual(raw, 1, mapping)
    if index <= 20:
        return {k: unmapped for k in ARMS}
    if gain is None or not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must have been calibrated from the ended prefix")
    density = prepare_actual(raw, index, mapping)

    def scale_n1(s):
        values = s["values"].copy()
        values[:, 11] *= np.float32(gain)
        if not np.isfinite(values[s["available"]]).all():
            raise FloatingPointError("n1 alignment overflow")
        return dict(s, values=values)

    return dict(
        unmapped=unmapped,
        density=density,
        n1=scale_n1(unmapped),
        both=scale_n1(density),
    )


@torch.no_grad()
def predict_curves(base, views, *, device):
    if set(views) != set(ARMS):
        raise ValueError("all four fixed input controls required")
    n = len(views["unmapped"]["values"])
    scores = {a: np.empty(n, np.float32) for a in ARMS}
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        for arm in ARMS:
            x, a = window_batch(views[arm], range(lo, hi), device=device)
            scores[arm][lo:hi] = base(x, a)["logit"].sigmoid().cpu().numpy()
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError("nonfinite static control scores")
    return scores


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
        inputs["baseline_protocol"]["source_state_sha256"] != original
        or inputs["baseline_protocol"]["threshold"] != threshold
    ):
        raise ValueError("fixed qualified source and threshold required")
    reference = source_reference(io23.source_roles()["train"])
    if not np.isfinite(reference["value"]) or reference["value"] <= 0:
        raise ValueError("specified source amplitude reference is not applicable")
    code = imported_code_hashes()
    bound = {
        **inputs["bound"],
        str(PLAN): sha(PLAN),
        str(DEFAULT_PARENT): sha(DEFAULT_PARENT),
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(output / "source_reference.json", reference)
    source_stats = {
        str(io23.path_for("J-TEXT", s)): (
            io23.path_for("J-TEXT", s).stat().st_size,
            io23.path_for("J-TEXT", s).stat().st_mtime_ns,
        )
        for s in reference["source_safe_ids"]
    }
    write(output / "source_file_states.json", source_stats)
    write(
        output / "protocol.json",
        dict(
            scope="NORMAL_REFERENCE_N1_INPUT_ALIGNMENT_NOT_NEURAL_GAIN",
            source_sha256=PARENT_SHA256,
            source_state_sha256=original,
            order=inputs["order"],
            threshold=threshold,
            arms=ARMS,
            first20_unmapped=True,
            calibration_after20=True,
            neural_updates=0,
            source_safe_reference=reference["value"],
            reference_method="ratio_of_equal_shot_safe_medians",
            physical_unit_conversion_claim=False,
            independent_validation=False,
            input_sha256=bound,
            code_sha256=code,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in code:
            z.write(p, str(Path(p).relative_to(io23.ROOT)))
        z.write(PLAN, PLAN.name)
    per = {a: [] for a in ARMS}
    seals = []
    past = {}
    truth = {}
    calls = []
    gain = None
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        path = io23.path_for("EAST", sid)
        if sha(path) != inputs["hashes"][str(sid)]:
            raise ValueError("raw file changed")

        def checked(domain, requested, *args, **kwargs):
            if domain != "EAST" or requested != sid:
                raise ValueError("only current EAST signal allowed")
            calls.append(sid)
            return reader(domain, requested, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23,
                "read_outcome",
                side_effect=RuntimeError("no current outcome before seal"),
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            views = make_views(raw, index, inputs["mapping"], gain)
            scores = predict_curves(base, views, device=device)
            prefix = f"{index:04d}_{sid}"
            ref = inputs["baseline_shots"] / prefix
            check_reference(
                ref.with_suffix(".npz"),
                ref.with_name(prefix + "_seal.json"),
                inputs["seals"][index - 1],
                raw["times"],
                scores["density"],
            )
            with np.load(ref.with_suffix(".npz"), allow_pickle=False) as baseline:
                if not np.array_equal(baseline["unmapped"], scores["unmapped"]):
                    raise AssertionError("unmapped original baseline changed")
            target = output / "shots" / (prefix + ".npz")
            with target.open("xb") as f:
                np.savez_compressed(f, times=raw["times"], **scores)
                f.flush()
                os.fsync(f.fileno())
            seal = dict(
                shot=sid,
                curve_sha256=sha(target),
                current_outcome_used_in_prediction=False,
                calibrated=index > 20,
                neural_updates=0,
            )
            durable_write(output / "shots" / (prefix + "_seal.json"), seal)
            directory = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            seals.append(seal)
        if sha(target) != seal["curve_sha256"]:
            raise RuntimeError("sealed score changed")
        with np.load(target, allow_pickle=False) as a:
            times = a["times"].copy()
            curves = {arm: a[arm].copy() for arm in ARMS}
        flag, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            per[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    down_s=down,
                    **score_curve(times, curves[arm], flag, down, threshold),
                )
            )
        if index <= 20:
            past[sid] = raw
            truth[sid] = (flag, down)
        if index == 20:
            with (
                patch.object(
                    io23,
                    "read_signals",
                    side_effect=RuntimeError(
                        "only in-memory ended prefix in calibration"
                    ),
                ),
                patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("only revealed truth in calibration"),
                ),
            ):
                calibration = fit_gain(reference, past, truth, ids[:20])
            gain = float(calibration["gain"])
            write(output / "calibration.json", calibration)
            past.clear()
            print(
                dict(
                    stage="n1_calibration",
                    gain=gain,
                    target_reference=calibration["target_reference"],
                ),
                flush=True,
            )
        if index % 40 == 0 or index == 20 or index == len(ids):
            print(
                dict(
                    stage="n1_static_control",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids
        or state_digest(base) != original
        or any(p.grad is not None for p in base.parameters())
    ):
        raise AssertionError("source or signal boundary changed")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound inputs changed")
    for path, stat in source_stats.items():
        now = Path(path).stat()
        if (now.st_size, now.st_mtime_ns) != tuple(stat):
            raise AssertionError("source reference raw file changed")
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
        status="COMPLETED_N1_ALIGNMENT_STATIC_CONTROL",
        seconds=time.monotonic() - started,
        gain=gain,
        source_unchanged=True,
        neural_updates=0,
        baseline_exact=True,
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
            gain=gain,
            full360={arm: s["full360"] for arm, s in summary.items()},
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
        torch.backends.mha.set_fastpath_enabled(False)
    run(args.output, device=args.device)
