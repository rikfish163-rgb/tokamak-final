"""Qualified-source EAST360 baseline, with numeric preparation kept explicit."""

import argparse
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import configure
from ..hdf5_score_recheck_20261103 import score_curve
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import predict_shot
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform

PREPARATION = io23.ROOT / "probe_outputs/stratified_cold20_context360_v1"
ORDER = io23.ROOT / "probe_outputs/east_decoder_complete360_v1/protocol.json"
PLAN = io23.ROOT / "QUALIFIED_SOURCE_STATIC360_PLAN_20261104.md"
ARMS = ("unmapped", "map_after20", "retrospective_map_all")


def operational_curve(index, raw, mapped):
    if type(index) is not int or index < 1 or np.shape(raw) != np.shape(mapped):
        raise ValueError("positive one-based shot index and matched curves required")
    return raw if index <= 20 else mapped


def timing_decomposition(rows, *, deadline_ms=30):
    if deadline_ms not in (10, 30):
        raise ValueError("only declared J-TEXT10 or EAST30 deadlines supported")
    if len({r["shot"] for r in rows}) != len(rows):
        raise ValueError("duplicate shot in scoring")
    for row in rows:
        if type(row["disruptive"]) is not bool or type(row["alarm"]) is not bool:
            raise ValueError("boolean HDF outcome and sealed alarm required")
        lead = row["lead_ms"]
        expected = bool(
            row["disruptive"]
            and row["alarm"]
            and lead is not None
            and lead > deadline_ms
        )
        if row["timely"] != expected:
            raise ValueError("timely must match the strict device deadline")
        if (row["disruptive"] and row["alarm"]) != (lead is not None):
            raise ValueError("lead only exists for an alarming disruption")
        if lead is not None and (not math.isfinite(lead) or lead < -1e-6):
            raise ValueError("finite pre-disruption first alarm required")
    ds = [r for r in rows if r["disruptive"]]
    ss = [r for r in rows if not r["disruptive"]]
    tp, fp = sum(r["timely"] for r in ds), sum(r["alarm"] for r in ss)
    tpr, fpr = (tp / len(ds) if ds else None), (fp / len(ss) if ss else None)
    return dict(
        shots=len(rows),
        n_disruptive=len(ds),
        n_safe=len(ss),
        tp=tp,
        fp=fp,
        tardy=sum(r["alarm"] and not r["timely"] for r in ds),
        miss=sum(not r["alarm"] for r in ds),
        tpr=tpr,
        fpr=fpr,
        J=tpr - fpr if tpr is not None and fpr is not None else None,
        early_over3000=sum(r["timely"] and r["lead_ms"] > 3000 for r in ds),
        diagnostic_oracle_advance_ms={
            str(h): sum(r["alarm"] and r["lead_ms"] + h > deadline_ms for r in ds)
            for h in (0, 5, 10, 15, 20, 25, 30, 50)
        },
        oracle_is_deployable=False,
    )


def main(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    prior = read(ORDER)
    order = prior["order"]
    ids = [r["shot"] for r in order]
    mapping_path = PREPARATION / "input_map.json"
    hashes_path = PREPARATION / "input_hashes.json"
    mapping, hashes = read(mapping_path), read(hashes_path)
    if (
        len(ids) != 360
        or set(ids) != set(io23.cohort_ids("EAST"))
        or mapping["calibration_ids"] != ids[:20]
        or sha(mapping_path) != prior["input_map_sha256"]
        or set(hashes) != {str(sid) for sid in ids}
    ):
        raise ValueError("locked360 order, raw hashes and fixed20 map required")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    original = state_digest(base)
    cutoff = float(parent["threshold"])
    code = imported_code_hashes()
    bound = {
        str(p): sha(p) for p in (PLAN, ORDER, mapping_path, hashes_path, DEFAULT_PARENT)
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="QUALIFIED_SOURCE_STATIC360_NUMERIC_PREPARATION_DIAGNOSTIC_NOT_NEURAL_ONLINE_GAIN",
            source_sha256=PARENT_SHA256,
            source_state_sha256=original,
            cold_checkpoint_loaded=False,
            source_parameter_updates=0,
            threshold=cutoff,
            order=order,
            arms=list(ARMS),
            numerical_preparation_budget=20,
            map_after20="first20 unmapped; fixed prepared map used only from shot21",
            retrospective_map_all="nondeployable on first20; comparison diagnostic only",
            map_fitting_reexecuted=False,
            map_physical_units_certified=False,
            cohort_and_order_retrospectively_developed=True,
            checkpoint_selection_or_training=False,
            independent_future_validation=False,
            signal_scope="whole ended-shot files read, prefix-only model windows; no outcome before each seal",
            source_native_fast_origin=True,
            batch_size=256,
            source_risk_head_frozen=True,
            raw_sha256=hashes,
            input_sha256=bound,
            code_sha256=code,
            torch_version=torch.__version__,
            device=torch.cuda.get_device_name(),
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            archive.write(name, str(Path(name).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    per = {arm: [] for arm in ARMS}
    seals = []
    reader = io23.read_signals
    calls = []

    def checked(device, sid, *args, **kwargs):
        if device != "EAST" or sid not in ids:
            raise ValueError("only the locked EAST360 signals are allowed")
        calls.append(sid)
        return reader(device, sid, *args, **kwargs)

    for index, sid in enumerate(ids, 1):
        path = io23.path_for("EAST", sid)
        stat = path.stat()
        if sha(path) != hashes[str(sid)]:
            raise ValueError("raw EAST input changed")
        with (
            patch.object(io23, "read_signals", side_effect=checked),
            patch.object(
                io23,
                "read_outcome",
                side_effect=RuntimeError(
                    "current outcome is unavailable before prediction seal"
                ),
            ),
        ):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = {k: raw[k] for k in ("values", "available")}
            mapped = transform(signal, mapping)
            if not np.array_equal(signal["available"], mapped["available"]):
                raise AssertionError("numeric map changed true availability")
            original_curve, n0 = predict_shot(
                base, dict(signal, shot=sid), intra=False, device="cuda"
            )
            mapped_curve, n1 = predict_shot(
                base, dict(mapped, shot=sid), intra=False, device="cuda"
            )
            if n0 or n1:
                raise AssertionError("static baseline performed neural updates")
            curves = dict(
                unmapped=original_curve,
                map_after20=operational_curve(index, original_curve, mapped_curve),
                retrospective_map_all=mapped_curve,
            )
            prefix = output / "shots" / f"{index:04d}_{sid}"
            npz = prefix.with_suffix(".npz")
            np.savez_compressed(npz, times=raw["times"], **curves)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                current_outcome_used_in_prediction=False,
            )
            write(prefix.with_name(prefix.name + "_seal.json"), seal)
            seals.append(seal)
        disruptive, down = io23.read_outcome("EAST", sid)
        for arm, scores in curves.items():
            per[arm].append(
                dict(
                    shot=sid,
                    disruptive=disruptive,
                    **score_curve(raw["times"], scores, disruptive, down, cutoff),
                )
            )
        if (path.stat().st_size, path.stat().st_mtime_ns) != (
            stat.st_size,
            stat.st_mtime_ns,
        ):
            raise RuntimeError("raw input changed during inference")
        if index % 20 == 0 or index == len(ids):
            print(
                dict(
                    stage="static360", shots=index, seconds=time.monotonic() - started
                ),
                flush=True,
            )
    if (
        calls != ids
        or state_digest(base) != original
        or any(p.grad is not None for p in base.parameters())
    ):
        raise AssertionError("source immutability or signal scope failed")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound producer or input changed")
    write(output / "per_shot.json", per)
    summary = {
        arm: {
            name: timing_decomposition(rows)
            for name, rows in (
                ("full360", per[arm]),
                ("startup20", per[arm][:20]),
                ("suffix340", per[arm][20:]),
            )
        }
        for arm in ARMS
    }
    write(output / "summary.json", summary)
    done = dict(
        status="COMPLETED_STATIC_BASELINE_DIAGNOSTIC",
        seconds=time.monotonic() - started,
        source_and_native_heads_unchanged=True,
        source_state_sha256=original,
        neural_updates=0,
        source_signal_reads=0,
        east_signal_reads=len(calls),
        postseal_outcome_reads=len(ids),
        curves=360 * len(ARMS),
        seals=seals,
        independent_future_validation=False,
        operational_neural_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
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


def source_mask_probe(output):
    """Static TEST220 availability intervention; not a source online stream."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    plan = io23.ROOT / "SOURCE_MISSING_DIAGNOSTIC_PLAN_20261104.md"
    ids = io23.source_roles()["test"]
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    before = state_digest(base)
    threshold = float(parent["threshold"])
    views = {"native": (), "no_MNM": (2,), "no_MNM_CIII": (1, 2)}
    per = {name: [] for name in views}
    raw_hashes, seals = {}, []
    code = imported_code_hashes()
    bound = {str(p): sha(p) for p in (plan, DEFAULT_PARENT, io23.JTEXT_INDEX)}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            scope="STATIC_SOURCE_AVAILABILITY_COUNTERFACTUAL_NOT_JTEXT_ONLINE",
            source_sha256=PARENT_SHA256,
            source_state_sha256=before,
            test_ids=ids,
            views=views,
            threshold=threshold,
            timely_deadline_ms=10,
            source_labels_unchanged=True,
            neural_updates=0,
            east_signal_reads=0,
            input_rule="original source finite masks; interventions only remove CIII/MNM observations",
            causal_prefix_windows=True,
            whole_shot_io=True,
            outcomes_read_after_each_seal=True,
            independent_future_validation=False,
            code_sha256=code,
            input_sha256=bound,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(plan, plan.name)
    for index, sid in enumerate(ids, 1):
        path = io23.path_for("J-TEXT", sid)
        raw_hashes[str(sid)] = sha(path)
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcome forbidden before source prediction seal"),
        ):
            raw = simple_data.read_shot("J-TEXT", sid)
            curves = {}
            for name, columns in views.items():
                mask = raw["available"].copy()
                if columns:
                    mask[:, columns] = False
                curve, updates = predict_shot(
                    base,
                    dict(values=raw["values"], available=mask, shot=sid),
                    intra=False,
                    device="cuda",
                )
                if updates:
                    raise AssertionError("static source control performed updates")
                curves[name] = curve
            prefix = output / "shots" / f"{index:04d}_{sid}"
            npz = prefix.with_suffix(".npz")
            np.savez_compressed(npz, times=raw["times"], **curves)
            seal = dict(
                shot=sid,
                curve_sha256=sha(npz),
                current_outcome_used_in_prediction=False,
            )
            write(prefix.with_name(prefix.name + "_seal.json"), seal)
            seals.append(seal)
        disruptive, down = io23.read_outcome("J-TEXT", sid)
        for name, curve in curves.items():
            scored = score_curve(raw["times"], curve, disruptive, down, threshold)
            scored["timely"] = bool(
                scored["lead_ms"] is not None and scored["lead_ms"] > 10
            )
            per[name].append(dict(shot=sid, disruptive=disruptive, **scored))
        if sha(path) != raw_hashes[str(sid)]:
            raise RuntimeError("raw source shot changed during static control")
        if index % 40 == 0 or index == len(ids):
            print(
                dict(
                    stage="source_missing",
                    shots=index,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if state_digest(base) != before or any(
        p.grad is not None for p in base.parameters()
    ):
        raise AssertionError("qualified source changed")
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound producer changed")
    summary = {
        name: timing_decomposition(rows, deadline_ms=10) for name, rows in per.items()
    }
    write(output / "per_shot.json", per)
    write(output / "summary.json", summary)
    write(output / "raw_hashes.json", raw_hashes)
    done = dict(
        status="COMPLETED_STATIC_SOURCE_MISSING_DIAGNOSTIC",
        seconds=time.monotonic() - started,
        source_unchanged=True,
        source_state_sha256=before,
        source_signal_reads=len(ids),
        east_signal_reads=0,
        neural_updates=0,
        seals=seals,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", done)
    print(
        dict(status=done["status"], seconds=done["seconds"], summary=summary),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-mask-probe", action="store_true")
    args = parser.parse_args()
    if args.source_mask_probe:
        source_mask_probe(args.output)
    else:
        main(args.output)
