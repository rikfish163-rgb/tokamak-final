"""Past20-only threshold controls on fixed completed models and sealed curves."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_repair_20261104.evaluation import _write as durable_write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_anticipation_20261104.static_stream import timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..static_threshold_diagnostic_20261104 import cutoff_from_safe
from ..hdf5_score_recheck_20261103 import score_curve
from .run import (
    MODES,
    STARTUP,
    configure,
    load_inputs,
    make_adapter,
    make_readout,
    predict_curves,
    prepare_actual,
)

REFERENCE = io23.ROOT / "probe_outputs/cold20_readout_temporal360_20261104_v1"
PLAN = io23.ROOT / "COLD20_CALIBRATION_PLAN_20261104.md"
CAPS = (0.10, 0.15, 0.20)


def threshold_at(index, source_threshold, learned_threshold):
    if type(index) is not int or index < 1:
        raise ValueError("one-based shot index required")
    return source_threshold if index <= STARTUP else learned_threshold


def choose_thresholds(maxima, safe_ids):
    if (
        not safe_ids
        or len(safe_ids) != len(set(safe_ids))
        or not maxima
        or any(set(v) != set(safe_ids) for v in maxima.values())
    ):
        raise ValueError(
            "only the same explicit ended safe shots may choose thresholds"
        )
    return {
        str(cap): {
            mode: cutoff_from_safe([values[s] for s in safe_ids], cap)
            for mode, values in maxima.items()
        }
        for cap in CAPS
    }


def main(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    done, protocol = (
        read(REFERENCE / "completed.json"),
        read(REFERENCE / "protocol.json"),
    )
    if (
        done["status"] != "COMPLETED_ENDED20_READOUT_TEMPORAL_CONTROL_NOT_ONLINE_GAIN"
        or not done["source_unchanged"]
        or protocol["source_sha256"] != PARENT_SHA256
    ):
        raise ValueError("completed fixed parent/end20 experiment required")
    inputs = load_inputs()
    ids = inputs["ids"]
    if done["fit_label_shots"] != ids[:STARTUP] or protocol["order"] != inputs["order"]:
        raise ValueError("same ended20 and subsequent stream required")
    files = ["protocol.json", "fit/scope.json", "fit/fitted.json"] + [
        f"fit/{m}.pt" for m in MODES[1:]
    ]
    for name in files:
        if sha(REFERENCE / name) != done["artifact_sha256"][name]:
            raise ValueError("fit artifact changed")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    original = state_digest(base)
    source_threshold = float(parent["threshold"])
    fitted = read(REFERENCE / "fit/fitted.json")
    models = {}
    for mode in MODES[1:]:
        model = (
            make_readout(base, mode)
            if mode in ("bias", "head")
            else make_adapter(base, mode)
        )
        state = torch.load(
            REFERENCE / f"fit/{mode}.pt", map_location="cuda", weights_only=True
        )
        if state["source_sha256"] != PARENT_SHA256 or state["mode"] != mode:
            raise ValueError("target checkpoint binding changed")
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        if state_digest(model) != fitted[mode]["fitted_state_sha256"]:
            raise ValueError("wrong fitted model state")
        models[mode] = model
    code = imported_code_hashes()
    bound = {
        **inputs["bound"],
        **{
            str(p): sha(p)
            for p in (
                PLAN,
                DEFAULT_PARENT,
                REFERENCE / "completed.json",
                *[REFERENCE / name for name in files],
            )
        },
    }
    output.mkdir(parents=True)
    (output / "past_calibration_scores").mkdir()
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    write(
        output / "protocol.json",
        dict(
            scope="PAST20_IN_SAMPLE_THRESHOLD_CONTROL_NOT_NEURAL_OR_INTRA_GAIN",
            source_sha256=PARENT_SHA256,
            neural_updates=0,
            label_shots=ids[:STARTUP],
            calibration_caps=list(CAPS),
            prefix_actual_threshold=source_threshold,
            training_models_and_map_fixed=True,
            same_data_fit_and_calibration=True,
            calibration_does_not_guarantee_suffix_fpr=True,
            uses_suffix_for_selection=False,
            suffix_raw_signal_arrays_read=0,
            independent_future_validation=False,
            input_sha256=bound,
            code_sha256=code,
        ),
    )
    maxima = {mode: {} for mode in MODES}
    reader = io23.read_signals
    safe_ids, calls = [], []
    for index, sid in enumerate(ids[:STARTUP], 1):
        path = io23.path_for("EAST", sid)
        if sha(path) != inputs["hashes"][str(sid)]:
            raise ValueError("past calibration raw input changed")

        def checked(domain, shot, *args, **kwargs):
            if domain != "EAST" or shot not in ids[:STARTUP]:
                raise ValueError("no suffix signal allowed for calibration")
            calls.append(shot)
            return reader(domain, shot, *args, **kwargs)

        with patch.object(io23, "read_signals", side_effect=checked):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        signal = prepare_actual(raw, STARTUP + 1, inputs["mapping"])
        curves = predict_curves(base, models, signal, device="cuda")
        np.savez_compressed(
            output / "past_calibration_scores" / f"{index:04d}_{sid}.npz",
            times=raw["times"],
            **curves,
        )
        # These labels were already revealed for fitting: this is explicitly
        # in-sample calibration after the whole prefix ended, not a blind test.
        disruptive, _ = io23.read_outcome("EAST", sid)
        if not disruptive:
            safe_ids.append(sid)
            for mode in MODES:
                maxima[mode][sid] = float(curves[mode].max())
    thresholds = choose_thresholds(maxima, safe_ids)
    durable_write(
        output / "thresholds.json",
        dict(safe_ids=safe_ids, thresholds=thresholds, empirical_safe_maxima=maxima),
    )
    calibration_seal = dict(
        threshold_sha256=sha(output / "thresholds.json"),
        suffix_labels_or_scores_used=False,
    )
    durable_write(output / "calibration_seal.json", calibration_seal)
    per = {cap: {mode: [] for mode in MODES} for cap in thresholds}
    for index, sid in enumerate(ids, 1):
        if sha(output / "thresholds.json") != calibration_seal["threshold_sha256"]:
            raise ValueError("sealed calibration changed")
        prefix = REFERENCE / "shots" / f"{index:04d}_{sid}"
        seal = read(prefix.with_name(prefix.name + "_seal.json"))
        if (
            seal != done["seals"][index - 1]
            or sha(prefix.with_suffix(".npz")) != seal["curve_sha256"]
        ):
            raise ValueError("original complete-stream prediction seal changed")
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as curve:
            if index <= STARTUP and any(
                not np.array_equal(curve[m], curve["native"]) for m in MODES
            ):
                raise ValueError(
                    "prefix actual predictions must be common and untrained"
                )
            flag, down = io23.read_outcome("EAST", sid)
            for cap, cutoffs in thresholds.items():
                for mode in MODES:
                    cutoff = threshold_at(index, source_threshold, cutoffs[mode])
                    per[cap][mode].append(
                        dict(
                            shot=sid,
                            disruptive=flag,
                            **score_curve(
                                curve["times"], curve[mode], flag, down, cutoff
                            ),
                        )
                    )
    if calls != ids[:STARTUP] or state_digest(base) != original:
        raise AssertionError("calibration signal scope or source changed")
    if any(
        state_digest(m) != fitted[name]["fitted_state_sha256"]
        for name, m in models.items()
    ):
        raise AssertionError("calibration changed model parameters")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound calibration producer/input changed")
    summary = {
        cap: {
            mode: {
                name: timing_decomposition(rows, deadline_ms=30)
                for name, rows in (
                    ("full360", values),
                    ("startup20", values[:STARTUP]),
                    ("suffix340", values[STARTUP:]),
                )
            }
            for mode, values in methods.items()
        }
        for cap, methods in per.items()
    }
    bias_parity = {
        cap: sum(a != b for a, b in zip(rows["bias"], rows["native"], strict=True))
        for cap, rows in per.items()
    }
    write(output / "per_shot.json", per)
    write(output / "summary.json", summary)
    result = dict(
        status="COMPLETED_PAST20_THRESHOLD_CONTROL_NOT_NEURAL_GAIN",
        seconds=time.monotonic() - started,
        calibration_safe_shots=len(safe_ids),
        calibration_signal_reads=len(calls),
        neural_updates=0,
        bias_vs_native_per_shot_disagreements=bias_parity,
        artifact_sha256={
            str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
        },
    )
    write(output / "completed.json", result)
    print(
        dict(
            status=result["status"],
            seconds=result["seconds"],
            bias_parity=bias_parity,
            full360={
                c: {m: v["full360"] for m, v in ms.items()} for c, ms in summary.items()
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
