"""Same-budget ended20 readout/temporal controls, scored on the entire flow.

All four adaptations occur once after shot20. They are not intra-shot learning.
"""

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
from ..mask_repair_20261104.east_evaluation import (
    check_reference,
    configure,
    load_inputs,
    prepare_actual,
)
from ..mask_repair_20261104.evaluation import _write as durable_write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_anticipation_20261104.static_stream import timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..history_gradient_metric_20261102.run import sha, write
from .adapters import make_adapter
from .data import build_pools, make_schedule
from .training import fit_model, make_readout

MODES = ("native", "bias", "head", "norm", "qv_lora")
COUNTS = dict(bias=1, head=65, norm=256, qv_lora=512)
STEPS, BATCH_SIZE, STARTUP = 256, 128, 20
PLAN = io23.ROOT / "COLD20_COMPARISON_PLAN_20261104.md"


def predict_curves(base, models, shot, *, device, batch_size=256):
    if models is not None and set(models) != set(MODES[1:]):
        raise ValueError("all four explicitly frozen target models required")
    if type(batch_size) is not int or batch_size < 1 or not len(shot["values"]):
        raise ValueError("nonempty shot and positive prediction batch required")
    curves = {mode: np.empty(len(shot["values"]), np.float32) for mode in MODES}
    for lo in range(0, len(shot["values"]), batch_size):
        hi = min(lo + batch_size, len(shot["values"]))
        windows = [simple_data.window(shot, q, 128) for q in range(lo, hi)]
        x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        a = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        with torch.no_grad():
            for mode in MODES:
                if mode != "native" and models is None:
                    curves[mode][lo:hi] = curves["native"][lo:hi]
                    continue
                model = base if mode == "native" else models[mode]
                z = model(x, a)["logit"]
                if not torch.isfinite(z).all():
                    raise FloatingPointError("nonfinite target risk")
                curves[mode][lo:hi] = z.sigmoid().cpu().numpy()
    return curves


def fit_prefix(base, past, truth, ids, mapping, output, *, device):
    """Only ended prefix arrays/labels are accepted; no future I/O occurs."""
    if len(ids) != STARTUP or set(past) != set(ids) or set(truth) != set(ids):
        raise ValueError("exactly the ended20 data and labels required")
    output.mkdir()
    mapped = {}
    for sid in ids:
        # At this point all20 have ended: their map is now available for
        # training, but it was never used to rewrite their sealed predictions.
        prepared = prepare_actual(past[sid], STARTUP + 1, mapping)
        mapped[sid] = dict(prepared, device="EAST", times=past[sid]["times"])
    records, pools = build_pools(mapped, truth, ids)
    schedule = make_schedule(pools, steps=STEPS, batch_size=BATCH_SIZE)
    write(output / "records.json", records)
    write(output / "schedule.json", schedule)
    write(
        output / "scope.json",
        dict(
            label_shots=ids,
            n_disruptive=sum(truth[s][0] for s in ids),
            n_safe=sum(not truth[s][0] for s in ids),
            positive_horizon_ms=125,
            group_windows={
                g: sum(map(len, rows.values())) for g, rows in pools.items()
            },
            group_shots={g: len(rows) for g, rows in pools.items()},
            target_labels_from_postseal_hdf=True,
            prefix_predictions_not_recomputed=True,
            steps=STEPS,
            batch_size=BATCH_SIZE,
            future_training_shots=0,
        ),
    )
    models = {
        mode: (
            make_readout(base, mode)
            if mode in ("bias", "head")
            else make_adapter(base, mode)
        )
        for mode in MODES[1:]
    }
    with torch.no_grad():
        first = records[0]
        xx, aa = simple_data.window(mapped[first["shot"]], first["query"], 128)
        x = torch.as_tensor(xx[None], device=device)
        a = torch.as_tensor(aa[None], device=device)
        expected = base(x, a)
        for mode, model in models.items():
            actual = model(x, a)
            if not all(
                torch.equal(actual[k], expected[k]) for k in ("logit", "forecast")
            ):
                raise AssertionError(f"initial target clone differs: {mode}")
    details = {}
    for mode, model in models.items():
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != COUNTS[mode]:
            raise AssertionError("trainable scope count changed")
        before = state_digest(model)
        history = fit_model(model, mapped, records, schedule, device=device, mode=mode)
        write(output / f"{mode}_training.json", history)
        torch.save(
            dict(
                mode=mode,
                source_sha256=PARENT_SHA256,
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            ),
            output / f"{mode}.pt",
        )
        details[mode] = dict(
            trainable_names=names,
            trained_parameters=count,
            initial_state_sha256=before,
            fitted_state_sha256=state_digest(model),
            steps=len(history),
            label_shots=ids,
        )
    write(output / "fitted.json", details)
    return models, details


def paired_counts(per):
    result = {}
    for mode in MODES[1:]:
        result[mode] = {}
        for name, sl in (
            ("full360", slice(None)),
            ("startup20", slice(0, STARTUP)),
            ("suffix340", slice(STARTUP, None)),
        ):
            rows = list(zip(per[mode][sl], per["native"][sl], strict=True))
            result[mode][name] = dict(
                timely_gained=sum(
                    a["disruptive"] and a["timely"] and not b["timely"] for a, b in rows
                ),
                timely_lost=sum(
                    a["disruptive"] and not a["timely"] and b["timely"] for a, b in rows
                ),
                safe_alarms_gained=sum(
                    not a["disruptive"] and a["alarm"] and not b["alarm"]
                    for a, b in rows
                ),
                safe_alarms_removed=sum(
                    not a["disruptive"] and not a["alarm"] and b["alarm"]
                    for a, b in rows
                ),
            )
    return result


def run(output, *, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit CPU or CUDA required")
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs["ids"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.to(device)
    original = state_digest(base)
    threshold = float(parent["threshold"])
    if (
        inputs["baseline_protocol"]["threshold"] != threshold
        or inputs["baseline_protocol"]["source_state_sha256"] != original
    ):
        raise ValueError("fixed qualified source and baseline threshold required")
    code = imported_code_hashes()
    bound = {**inputs["bound"], **{str(p): sha(p) for p in (PLAN, DEFAULT_PARENT)}}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="ENDED20_BUDGET_READOUT_VS_TEMPORAL_NOT_INTRA_OR_CONTINUAL_GAIN",
            source_sha256=PARENT_SHA256,
            source_state_sha256=original,
            threshold=threshold,
            order=inputs["order"],
            modes=list(MODES),
            steps=STEPS,
            batch_size=BATCH_SIZE,
            startup_budget=STARTUP,
            trained_parameter_counts=COUNTS,
            first20_unmapped_and_untrained=True,
            suffix_fixed_map=True,
            point_label_horizon_ms=125,
            timely_deadline_ms=30,
            suffix_labels_never_used_for_training=True,
            source_checkpoint_immutable=True,
            independent_future_validation=False,
            whole_shot_io_but_causal_model_windows=True,
            input_sha256=bound,
            code_sha256=code,
            compute_device=device,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    per, seals = {mode: [] for mode in MODES}, []
    past, truth, models, fit_details = {}, {}, None, None
    reader, calls = io23.read_signals, []
    for index, sid in enumerate(ids, 1):
        path = io23.path_for("EAST", sid)
        if sha(path) != inputs["hashes"][str(sid)]:
            raise ValueError("raw EAST hash changed")

        def checked(domain, shot, *args, **kwargs):
            if domain != "EAST" or shot != sid:
                raise ValueError("only current EAST signal allowed during prediction")
            calls.append(sid)
            return reader(domain, shot, *args, **kwargs)

        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23,
                "read_outcome",
                side_effect=RuntimeError("current outcome forbidden before seal"),
            ),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = prepare_actual(raw, index, inputs["mapping"])
            curves = predict_curves(base, models, signal, device=device)
            prefix = f"{index:04d}_{sid}"
            reference = inputs["baseline_shots"] / prefix
            check_reference(
                reference.with_suffix(".npz"),
                reference.with_name(prefix + "_seal.json"),
                inputs["seals"][index - 1],
                raw["times"],
                curves["native"],
            )
            path_out = output / "shots" / (prefix + ".npz")
            with path_out.open("xb") as stream:
                np.savez_compressed(stream, times=raw["times"], **curves)
                stream.flush()
                os.fsync(stream.fileno())
            seal = dict(
                shot=sid,
                curve_sha256=sha(path_out),
                current_outcome_used_in_prediction=False,
                target_fit_completed=index > STARTUP,
            )
            durable_write(output / "shots" / (prefix + "_seal.json"), seal)
            directory = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            seals.append(seal)
        if sha(path_out) != seal["curve_sha256"]:
            raise RuntimeError("prediction changed before truth revelation")
        with np.load(path_out, allow_pickle=False) as stored:
            sealed_times = stored["times"].copy()
            sealed_scores = {mode: stored[mode].copy() for mode in MODES}
        flag, down = io23.read_outcome("EAST", sid)
        for mode in MODES:
            per[mode].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    down_s=down,
                    **score_curve(
                        sealed_times, sealed_scores[mode], flag, down, threshold
                    ),
                )
            )
        if index <= STARTUP:
            past[sid], truth[sid] = raw, (flag, down)
        if index == STARTUP:
            models, fit_details = fit_prefix(
                base,
                past,
                truth,
                ids[:STARTUP],
                inputs["mapping"],
                output / "fit",
                device=device,
            )
            past.clear()  # Discard only in-memory ended signals; no files removed.
        if (
            sha(path) != inputs["hashes"][str(sid)]
            or sha(path_out) != seal["curve_sha256"]
        ):
            raise RuntimeError("raw or sealed prediction changed")
        if index % 20 == 0 or index == len(ids):
            print(
                dict(
                    stage="cold20_comparison",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        calls != ids
        or set(truth) != set(ids[:STARTUP])
        or state_digest(base) != original
        or any(p.grad is not None for p in base.parameters())
        or models is None
    ):
        raise AssertionError("source, label, signal or fitting boundary changed")
    for mode, model in models.items():
        if state_digest(model) != fit_details[mode]["fitted_state_sha256"] or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise AssertionError("model changed after ended20 fitting")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound inputs or producer changed")
    summary = {
        mode: {
            name: timing_decomposition(rows, deadline_ms=30)
            for name, rows in (
                ("full360", per[mode]),
                ("startup20", per[mode][:STARTUP]),
                ("suffix340", per[mode][STARTUP:]),
            )
        }
        for mode in MODES
    }
    write(output / "per_shot.json", per)
    write(output / "summary.json", summary)
    write(output / "paired.json", paired_counts(per))
    done = dict(
        status="COMPLETED_ENDED20_READOUT_TEMPORAL_CONTROL_NOT_ONLINE_GAIN",
        seconds=time.monotonic() - started,
        native_exact_baseline=True,
        source_unchanged=True,
        fit_label_shots=ids[:STARTUP],
        source_training_shots=0,
        suffix_training_shots=0,
        postseal_outcome_reads=len(seals),
        fitted_parameters=COUNTS,
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
            full360={m: s["full360"] for m, s in summary.items()},
        ),
        flush=True,
    )
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    run(args.output, device=args.device)
