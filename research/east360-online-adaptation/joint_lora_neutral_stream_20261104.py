"""Run the fixed-parent EAST360 diagnostic, changing only within-shot Q/V."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import (
    read,
    sha,
)
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.run import native_curve
from experiments.risk_aligned_online.joint_lora_20261104.timeline import run_shot
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    check_reference,
    configure,
    load_inputs,
    prepare_actual,
)
from experiments.risk_aligned_online.mask_repair_20261104.evaluation import (
    _write as durable_write,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
)
from experiments.risk_aligned_online.source_anticipation_20261104.static_stream import (
    timing_decomposition,
)
from experiments.risk_aligned_online.source_event_tasks_20261102.context_pair import (
    imported_code_hashes,
)

ARMS = ("native", "static", "intra", "hold", "reverse")
MODES = dict(static="none", intra="real", hold="hold", reverse="reverse")
ROOT = io23.ROOT
PLAN = ROOT / "JOINT_LORA_NEUTRAL_STREAM_PLAN_20261104.md"
WARM = ROOT / "probe_outputs/joint_lora_decoder_only_warm_20261104_v1"
PARITY_ATOL = 3e-6


def require_unmoved_risk(pristine, fitted):
    reference, actual = pristine.state_dict(), fitted.state_dict()
    if reference.keys() != actual.keys() or any(
        not torch.equal(value, actual[key])
        for key, value in reference.items()
        if not key.startswith("residual_decoder.")
    ):
        raise ValueError("non-decoder state must exactly equal the unadapted parent")


def load_risk_neutral(base, directory, expected_ids, revealed_ids, threshold):
    if len(expected_ids) != 20 or list(revealed_ids) != list(expected_ids):
        raise ValueError("the first 20 outcomes must be revealed in order")
    done = read(directory / "completed.json")
    hashes = done["artifact_sha256"]
    if (
        done["status"] != "COMPLETED_RISK_NEUTRAL_DECODER_ONLY_PREPARATION"
        or set(hashes) != {"warm.json", "warm.pt"}
        or any(sha(directory / name) != value for name, value in hashes.items())
    ):
        raise ValueError("verified decoder-only preparation required")
    receipt = read(directory / "warm.json")
    if (
        receipt["label_shots"] != list(expected_ids)
        or receipt["lora_updates"] != 0
        or receipt["threshold"] != threshold
        or receipt["source_state_sha256"] != state_digest(base)
        or any(sha(Path(p)) != value for p, value in receipt["input_sha256"].items())
    ):
        raise ValueError("decoder-only preparation input contract changed")
    state = torch.load(
        directory / "warm.pt", map_location=base.center.device, weights_only=True
    )
    pristine = JointLoRA(base, state["W0"])
    fitted = JointLoRA(base, state["W0"])
    fitted.load_state_dict(state, strict=True)
    require_unmoved_risk(pristine, fitted)
    if (
        state_digest(fitted) != receipt["fitted_state_sha256"]
        or state_digest(fitted.residual_decoder) != receipt["decoder_after"]
        or receipt["decoder_after"] == receipt["decoder_before"]
    ):
        raise ValueError("decoder-only fitted state digest changed")
    return fitted.eval().requires_grad_(False)


def compare_shot(base, fixed, signal, *, device):
    scores = {"native": native_curve(base, signal, device=device)}
    updates, forecasts = {}, {}
    if fixed is None:
        scores.update({arm: scores["native"].copy() for arm in MODES})
        return scores, updates, forecasts, 0.0
    parent_sha = state_digest(fixed)
    for arm, mode in MODES.items():
        scores[arm], updates[arm], trace = run_shot(fixed, signal, mode, device=device)
        if arm != "static":
            if not np.array_equal(scores[arm][:135], scores["static"][:135]):
                raise AssertionError("a fast arm changed risk before feedback maturity")
            for key in ("origin", "released_at", "target", "mask"):
                if not np.array_equal(
                    trace[key], forecasts["static__" + key], equal_nan=True
                ):
                    raise AssertionError(
                        "control clocks and observed targets must match"
                    )
        forecasts.update({arm + "__" + key: v for key, v in trace.items()})
    gap = float(np.max(np.abs(scores["native"] - scores["static"])))
    if gap > PARITY_ATOL or state_digest(fixed) != parent_sha:
        raise AssertionError(
            "decoder altered pre-adaptation risk or fast state escaped"
        )
    return scores, updates, forecasts, gap


def write_npz(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def run(output, *, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device).eval().requires_grad_(False)
    parent_state = state_digest(base)
    threshold = float(parent["threshold"])
    baseline = inputs["baseline_protocol"]
    if (baseline["source_state_sha256"], baseline["threshold"]) != (
        parent_state,
        threshold,
    ):
        raise ValueError("qualified source and native threshold required")
    code = imported_code_hashes()
    inputs_sha = {
        **inputs["bound"],
        str(PLAN): sha(PLAN),
        str(DEFAULT_PARENT): sha(DEFAULT_PARENT),
    }
    shots = output / "shots"
    shots.mkdir(parents=True)
    durable_write(
        output / "protocol.json",
        dict(
            scope="FIXED_RISK_NEUTRAL_DECODER_ONLY_INTRA_DEVELOPMENT",
            order=inputs["order"],
            source_checkpoint_sha256=PARENT_SHA256,
            source_state_sha256=parent_state,
            arms=ARMS,
            threshold=threshold,
            decoder_enabled_after=20,
            fixed_inter_updates=0,
            feedback_horizon=8,
            origin_first=127,
            origin_stride=32,
            intra_lr=1e-4,
            intra_radius=0.5,
            parity_atol=PARITY_ATOL,
            independent_validation=False,
            code_sha256=code,
            input_sha256=inputs_sha,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in (*code, str(PLAN)):
            archive.write(path, Path(path).relative_to(ROOT))
    fixed = None
    fixed_state = None
    revealed, observed_calls, output_hashes, gaps = [], [], {}, []
    per = {arm: [] for arm in ARMS}
    original_read = io23.read_signals
    for index, sid in enumerate(ids, 1):
        if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
            raise ValueError(f"raw input changed: {sid}")

        def current_only(domain, shot, *args, **kwargs):
            if (domain, shot) != ("EAST", sid):
                raise ValueError("only the current EAST signal is allowed")
            observed_calls.append(sid)
            return original_read(domain, shot, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=current_only),
            patch.object(
                io23, "read_outcome", side_effect=RuntimeError("outcome before seal")
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = prepare_actual(raw, index, inputs["mapping"])
            times = raw["times"]
            scores, updates, forecasts, gap = compare_shot(
                base, fixed, signal, device=device
            )
            prefix = f"{index:04d}_{sid}"
            ref = inputs["baseline_shots"] / prefix
            check_reference(
                ref.with_suffix(".npz"),
                ref.with_name(prefix + "_seal.json"),
                inputs["seals"][index - 1],
                times,
                scores["native"],
            )
            curve = shots / (prefix + ".npz")
            curve_sha = write_npz(curve, times=times, **scores)
            forecast_sha = None
            if forecasts:
                path = shots / (prefix + "_forecasts.npz")
                forecast_sha = write_npz(path, **forecasts)
                output_hashes[str(path.relative_to(output))] = forecast_sha
            update = shots / (prefix + "_updates.json")
            durable_write(update, updates)
            seal = shots / (prefix + "_seal.json")
            durable_write(
                seal,
                dict(
                    shot=sid,
                    curve_sha256=curve_sha,
                    forecast_sha256=forecast_sha,
                    updates_sha256=sha(update),
                    current_outcome_used_in_prediction=False,
                    starting_fixed_state_sha256=fixed_state,
                    max_native_static_gap=gap,
                ),
            )
            for artifact in (curve, update, seal):
                output_hashes[str(artifact.relative_to(output))] = sha(artifact)
        flag, down = io23.read_outcome("EAST", sid)
        revealed.append(sid)
        for arm in ARMS:
            per[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    down_s=down,
                    **score_curve(times, scores[arm], flag, down, threshold),
                )
            )
        if per["native"][-1] != per["static"][-1]:
            raise AssertionError(f"risk-neutral controls disagree in scoring: {sid}")
        gaps.append(gap)
        if index == 20:
            fixed = load_risk_neutral(base, WARM, ids[:20], revealed, threshold)
            fixed_state = state_digest(fixed)
            warm_hash = {
                name: sha(WARM / name)
                for name in ("completed.json", "warm.json", "warm.pt")
            }
            durable_write(
                output / "parent_activation.json",
                dict(
                    after_sealed_revealed_shots=list(revealed),
                    parent_sha256=fixed_state,
                    preparation=str(WARM),
                    artifact_sha256=warm_hash,
                ),
            )
        if index % 10 == 0:
            print(dict(shots=index, seconds=time.monotonic() - started), flush=True)
    if (
        observed_calls != ids
        or revealed != ids
        or state_digest(base) != parent_state
        or state_digest(fixed) != fixed_state
        or any(sha(Path(p)) != h for p, h in {**code, **inputs_sha}.items())
        or any(sha(WARM / name) != h for name, h in warm_hash.items())
        or any(sha(output / name) != h for name, h in output_hashes.items())
    ):
        raise ValueError("current shot access, parent, or bound inputs/outputs changed")
    summary = {
        arm: {
            block: timing_decomposition(rows[region], deadline_ms=30)
            for block, region in (
                ("full360", slice(None)),
                ("startup20", slice(20)),
                ("suffix340", slice(20, None)),
            )
        }
        for arm, rows in per.items()
    }
    durable_write(output / "per_shot.json", per)
    durable_write(output / "summary.json", summary)
    for name in (
        "protocol.json",
        "code.zip",
        "parent_activation.json",
        "per_shot.json",
        "summary.json",
    ):
        output_hashes[name] = sha(output / name)
    durable_write(
        output / "completed.json",
        dict(
            status="COMPLETED_RISK_NEUTRAL_LORA_INTRA_DIAGNOSTIC",
            scored_shots=len(revealed),
            max_native_static_gap=max(gaps),
            inter_updates=0,
            seconds=time.monotonic() - started,
            independent_validation=False,
            artifact_sha256=output_hashes,
        ),
    )
    print({arm: x["full360"] for arm, x in summary.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    run(args.output)
