"""COUNTERFACTUAL_NOT_DEPLOYABLE: frozen-source, TEST220 value/mask audit.

Run only as a module, on the main agent's authorized GPU, with a new output
directory. The five comparison arms are reused sealed reference curves, not
newly reproduced baselines. No source learning or threshold selection occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .context_vector_20261030.source_run import source_gate
from .east_context_operator_20261031.run import configure
from .history_gradient_metric_20261102.run import read, sha, write
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_event_tasks_20261102.context_pair import imported_code_hashes
from .source_missing_diagnostics_20261103 import metrics, score_shot

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "probe_outputs/source_missing_diagnostics_test220_v1"
REFERENCE_COMPLETED_SHA256 = (
    "cf809ad49c96f12a0ffc472229e2cdfec0f1fe716874616cc2f9038f1577822b"
)
REFERENCE_MANIFEST_SHA256 = (
    "f2705c7b8512152ef6defe05dfb874a38c4383910c8fa399ce5193d606d4cb86"
)
CAPTION = "COUNTERFACTUAL_NOT_DEPLOYABLE"
ARM_CHANNELS = {
    "center_CIII_observed": (1,),
    "center_MNM_observed": (2,),
    "center_both_observed": (1, 2),
}
REFERENCE_ARMS = (
    "qualified_native",
    "contract_native",
    "no_CIII",
    "no_MNM",
    "no_MNM_CIII",
)


def neutralize_observed(signal, center23, channels):
    """Center only finite, observed values after the pointwise CIII contract.

    The result's availability is exactly the contract-masked input availability.
    No unavailable value, nonfinite value, other channel, or input is mutated.
    This intentionally artificial input is not a deployable missing-data method.
    """
    if not isinstance(signal, dict) or set(signal) != {"values", "available"}:
        raise ValueError("signal-only values/available dict required")
    if (
        not isinstance(channels, tuple)
        or not channels
        or any(
            type(channel) is not int or channel not in (1, 2) for channel in channels
        )
        or len(set(channels)) != len(channels)
    ):
        raise ValueError("unique CIII/MNM channel indices in a nonempty tuple required")
    center = np.asarray(center23)
    if (
        center.shape != (23,)
        or not np.issubdtype(center.dtype, np.number)
        or np.issubdtype(center.dtype, np.complexfloating)
        or not np.isfinite(center).all()
    ):
        raise ValueError("finite real source center with exactly 23 entries required")
    view = apply_input_contract(signal)
    # Preserve exact source-center values even for an integer-valued input.
    dtype = np.result_type(view["values"].dtype, center.dtype, np.float32)
    view["values"] = view["values"].astype(dtype, copy=False)
    for channel in channels:
        valid = view["available"][:, channel] & np.isfinite(view["values"][:, channel])
        view["values"][valid, channel] = center[channel]
    return view


def predict_curves(model, signal, center23):
    """Run only past 128-point windows, with the same 256-window batches."""
    curves = {}
    n = len(signal["values"])
    if n == 0:
        raise ValueError("nonempty shot required")
    with torch.no_grad():
        for arm, channels in ARM_CHANNELS.items():
            view = neutralize_observed(signal, center23, channels)
            batches = []
            for lo in range(0, n, 256):
                windows = [
                    simple_data.window(view, end, 128)
                    for end in range(lo, min(lo + 256, n))
                ]
                values, available = zip(*windows, strict=True)
                xx = torch.as_tensor(np.stack(values), device=model.center.device)
                aa = torch.as_tensor(np.stack(available), device=model.center.device)
                batches.append(model(xx, aa)["logit"].sigmoid().cpu().numpy())
            curves[arm] = np.concatenate(batches)
            if curves[arm].shape != (n,) or not np.isfinite(curves[arm]).all():
                raise FloatingPointError("invalid frozen-source prediction")
    return curves


def compare_curves(sid, times, new, reference, flag, down, threshold):
    """Pair all new/reference curves on one verified shot and its time grid."""
    times = np.asarray(times)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or set(new) != set(ARM_CHANNELS)
        or set(reference) != set(REFERENCE_ARMS)
    ):
        raise ValueError(
            "one nonempty time grid and exactly the declared arms required"
        )
    for curve in (*new.values(), *reference.values()):
        curve = np.asarray(curve)
        if curve.shape != times.shape or not np.isfinite(curve).all():
            raise ValueError("curve shape or finiteness differs from paired time grid")
        if np.any((curve < 0) | (curve > 1)):
            raise ValueError("probability curve outside [0, 1]")
    new_rows = {
        arm: score_shot(sid, times, curve, flag, down, threshold)
        for arm, curve in new.items()
    }
    reference_rows = {
        arm: score_shot(sid, times, curve, flag, down, threshold)
        for arm, curve in reference.items()
    }
    comparisons = {}
    for arm, curve in new.items():
        comparisons[arm] = {}
        for ref_arm, ref_curve in reference.items():
            delta = np.asarray(curve, np.float64) - np.asarray(ref_curve, np.float64)
            left, right = new_rows[arm], reference_rows[ref_arm]
            lead_delta = (
                left["lead_ms"] - right["lead_ms"]
                if left["lead_ms"] is not None and right["lead_ms"] is not None
                else None
            )
            comparisons[arm][ref_arm] = dict(
                mean_signed_score_difference=float(delta.mean()),
                max_abs_score_difference=float(np.abs(delta).max()),
                alarm_changed=left["alarm"] != right["alarm"],
                timely_changed=left["timely"] != right["timely"],
                lead_difference_ms=lead_delta,
            )
    return dict(
        shot=int(sid), new=new_rows, reference=reference_rows, comparisons=comparisons
    )


def reference_manifest(root):
    """Hash every reference file, including every shot NPZ and its seal."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError("reference directory missing")
    manifest = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("reference symlinks are not allowed")
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = sha(path)
    return dict(sorted(manifest.items()))


def manifest_digest(manifest):
    # Matches: LC_ALL=C sort paths, then sha256sum for each relative path.
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def check_raw_hash(sid, actual, expected):
    if expected.get(str(sid)) != actual:
        raise ValueError(f"raw HDF5 hash mismatch for shot {sid}")


def load_reference(ids, threshold):
    manifest = reference_manifest(REFERENCE)
    if (
        manifest.get("completed.json") != REFERENCE_COMPLETED_SHA256
        or manifest_digest(manifest) != REFERENCE_MANIFEST_SHA256
    ):
        raise ValueError("completed reference or its full artifact manifest changed")
    completed = read(REFERENCE / "completed.json")
    protocol = read(REFERENCE / "protocol.json")
    expected_top = completed["artifact_sha256"]
    actual_top = {
        k: v for k, v in manifest.items() if "/" not in k and k != "completed.json"
    }
    if expected_top != actual_top:
        raise ValueError("reference completion artifact hashes mismatch")
    if (
        completed["status"] != "COMPLETED_STATIC_SOURCE_AUDIT"
        or completed["shots"] != 220
        or completed["arms"] != len(REFERENCE_ARMS)
        or not completed["source_unchanged"]
        or completed["jtext_online_run"]
        or completed["optimizer_steps"] != 0
        or completed["historical_mismatches"]
        or protocol["ids"] != ids
        or len(ids) != 220
        or len(set(ids)) != 220
        or set(protocol["arms"]) != set(REFERENCE_ARMS)
        or protocol["source_sha256"] != PARENT_SHA256
        or protocol["threshold"] != threshold
        or protocol["batch_size"] != 256
    ):
        raise ValueError(
            "reference completion, parent, threshold or locked TEST220 mismatch"
        )
    for index, sid in enumerate(ids, 1):
        stem = f"shots/{index:04d}_{sid}"
        if stem + ".npz" not in manifest or stem + "_seal.json" not in manifest:
            raise ValueError(f"reference curve or seal missing for shot {sid}")
    expected_inputs = read(REFERENCE / "input_hashes.json")
    if set(expected_inputs) != {str(sid) for sid in ids}:
        raise ValueError("reference raw HDF5 identity mismatch")
    return manifest, protocol, expected_inputs


def load_reference_curves(index, sid, times, manifest):
    """Read an already-sealed reference only after the new shot is sealed."""
    stem = f"shots/{index:04d}_{sid}"
    curve_path = REFERENCE / (stem + ".npz")
    seal_path = REFERENCE / (stem + "_seal.json")
    if (
        sha(curve_path) != manifest[stem + ".npz"]
        or sha(seal_path) != manifest[stem + "_seal.json"]
    ):
        raise ValueError(f"reference shot artifact changed for {sid}")
    seal = read(seal_path)
    if (
        seal["shot"] != sid
        or seal["curve_sha256"] != manifest[stem + ".npz"]
        or seal["current_outcome_used_in_prediction"] is not False
    ):
        raise ValueError(f"invalid reference shot seal for {sid}")
    with np.load(curve_path, allow_pickle=False) as archive:
        if set(archive.files) != {"times", *REFERENCE_ARMS}:
            raise ValueError(f"unexpected reference curve arms for {sid}")
        if not np.array_equal(archive["times"], times):
            raise ValueError(f"reference time grid mismatch for {sid}")
        return {arm: archive[arm] for arm in REFERENCE_ARMS}


def save_new_curves(path, times, curves):
    if set(curves) != set(ARM_CHANNELS):
        raise ValueError("only new counterfactual curves may be saved")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, times=times, **curves)


def verify_output_seals(output, ids, input_hashes):
    for index, sid in enumerate(ids, 1):
        path = Path(output) / "shots" / f"{index:04d}_{sid}.npz"
        seal = read(path.with_name(path.stem + "_seal.json"))
        if (
            seal["caption"] != CAPTION
            or seal["shot"] != sid
            or seal["curve_sha256"] != sha(path)
            or seal["raw_hdf5_sha256"] != input_hashes.get(str(sid))
            or seal["current_outcome_used_in_prediction"] is not False
        ):
            raise ValueError(f"output curve seal changed for shot {sid}")


def snapshot_source(model):
    return dict(
        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        parameters={k: v.detach().cpu().clone() for k, v in model.named_parameters()},
        buffers={k: v.detach().cpu().clone() for k, v in model.named_buffers()},
        training={k: v.training for k, v in model.named_modules()},
        requires_grad={k: v.requires_grad for k, v in model.named_parameters()},
    )


def source_snapshot_hash(snapshot):
    """Hash all persistent and nonpersistent tensor bytes and model flags."""
    groups = {}
    for group in ("state", "parameters", "buffers"):
        groups[group] = {}
        for name, tensor in sorted(snapshot[group].items()):
            data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            groups[group][name] = dict(
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                sha256=hashlib.sha256(data).hexdigest(),
            )
    groups["training"] = snapshot["training"]
    groups["requires_grad"] = snapshot["requires_grad"]
    encoded = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_source_unchanged(model, frozen):
    current = snapshot_source(model)
    if source_snapshot_hash(current) != source_snapshot_hash(frozen):
        raise AssertionError(
            "qualified source tensor state, mode or gradient flags changed"
        )


def summarize(records):
    paired = {}
    for arm in ARM_CHANNELS:
        paired[arm] = {}
        for ref_arm in REFERENCE_ARMS:
            comparisons = [row["comparisons"][arm][ref_arm] for row in records]
            paired[arm][ref_arm] = dict(
                mean_of_shot_mean_signed_score_differences=float(
                    np.mean(
                        [row["mean_signed_score_difference"] for row in comparisons]
                    )
                ),
                max_abs_score_difference=max(
                    row["max_abs_score_difference"] for row in comparisons
                ),
                alarm_changed_shots=sum(row["alarm_changed"] for row in comparisons),
                timely_changed_shots=sum(row["timely_changed"] for row in comparisons),
            )
    return dict(
        caption=CAPTION,
        new_counterfactual_arms={
            arm: metrics([record["new"][arm] for record in records])
            for arm in ARM_CHANNELS
        },
        reference_reused_not_new_baseline={
            arm: metrics([record["reference"][arm] for record in records])
            for arm in REFERENCE_ARMS
        },
        paired_comparisons=paired,
    )


def main(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(REFERENCE.resolve()):
        raise ValueError("output must not be inside the sealed reference")
    started = time.monotonic()
    configure()
    model, parent = load_parent(DEFAULT_PARENT)
    threshold = float(parent["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid frozen source threshold")
    ids = io23.source_roles()["test"]
    reference_files, reference_protocol, input_hashes = load_reference(ids, threshold)
    code = imported_code_hashes()
    code[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    # The signal, model, input-contract and score implementations must match
    # those that produced the reused reference, not merely their current hash.
    for name in (
        "io23.py",
        "simple_data.py",
        "simple23.py",
        "causal_input.py",
        "source_missing_diagnostics_20261103.py",
        "redesign_20261004/prepare_prior.py",
        "context_vector_20261030/source_run.py",
        "east_context_operator_20261031/run.py",
        "history_gradient_metric_20261102/run.py",
        "source_event_tasks_20261102/context_pair.py",
    ):
        path = str((Path(__file__).parent / name).resolve())
        if reference_protocol["code_sha256"].get(path) != code.get(path):
            raise ValueError(f"reference implementation changed: {name}")
    model = model.cuda().eval().requires_grad_(False)
    frozen = snapshot_source(model)
    frozen_state_hash = source_snapshot_hash(frozen)
    center = model.center.detach().cpu().numpy().copy()
    if not np.array_equal(
        center, np.asarray(parent["source_center"], dtype=center.dtype)
    ):
        raise ValueError("parent source center disagrees with loaded model")
    scale = model.scale.detach().cpu().numpy()
    if not np.array_equal(scale, np.asarray(parent["source_scale"], dtype=scale.dtype)):
        raise ValueError("parent source scale disagrees with loaded model")
    output.mkdir()
    (output / "shots").mkdir()
    write(
        output / "protocol.json",
        dict(
            caption=CAPTION,
            scope="FROZEN_SOURCE_TEST220_DIAGNOSTIC_VALUE_VS_MASK_FACTORIAL",
            ids=ids,
            new_arms={k: list(v) for k, v in ARM_CHANNELS.items()},
            reference_arms=list(REFERENCE_ARMS),
            reference_use="saved reference curves explicitly reused; no fresh baseline",
            reference_path=str(REFERENCE),
            reference_completed_sha256=REFERENCE_COMPLETED_SHA256,
            reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
            reference_artifact_sha256=reference_files,
            source_sha256=PARENT_SHA256,
            source_full_state_sha256=frozen_state_hash,
            threshold=threshold,
            code_sha256=code,
            input_contract="pointwise CIII sentinel first; preserve post-contract availability",
            intervention="replace only finite available selected-channel values with the source center",
            unit_contract="J-TEXT ip multiplied1000 once; no EAST map",
            batch_size=256,
            window=128,
            hard_outcome_authority="HDF5 meta/IsDisrupt and meta/DownTime only after new curve seal",
            strict_timely_lead_ms=10,
            source_parameter_updates=0,
            jtext_online_updates=0,
            east_reads=0,
            source_scaler_changes=0,
            threshold_changes=0,
            no_test_threshold_selection=True,
            independent_confirmation=False,
        ),
    )
    records, verified_inputs = [], {}
    with source_gate(ids):
        for index, sid in enumerate(ids, 1):
            raw_path = io23.path_for("J-TEXT", sid)
            raw_hash = sha(raw_path)
            check_raw_hash(sid, raw_hash, input_hashes)
            # Neither the signal read, window construction, nor model inference
            # may access this shot's outcome before its new curves are sealed.
            with patch.object(
                io23, "read_outcome", side_effect=RuntimeError("no truth in prediction")
            ):
                raw = simple_data.read_shot("J-TEXT", sid)
                if raw["device"] != "J-TEXT" or int(raw["shot"]) != sid:
                    raise ValueError("source shot identity mismatch")
                signal = {name: raw[name] for name in ("values", "available")}
                curves = predict_curves(model, signal, center)
                if len(raw["times"]) != len(signal["values"]):
                    raise ValueError("signal/time grid length mismatch")
                contract = apply_input_contract(signal)
                observed = {
                    str(channel): int(
                        (
                            contract["available"][:, channel]
                            & np.isfinite(contract["values"][:, channel])
                        ).sum()
                    )
                    for channel in (1, 2)
                }
                check_raw_hash(sid, sha(raw_path), input_hashes)
                path = output / "shots" / f"{index:04d}_{sid}.npz"
                save_new_curves(path, raw["times"], curves)
                write(
                    path.with_name(path.stem + "_seal.json"),
                    dict(
                        caption=CAPTION,
                        shot=int(sid),
                        curve_sha256=sha(path),
                        raw_hdf5_sha256=raw_hash,
                        current_outcome_used_in_prediction=False,
                    ),
                )
            flag, down = io23.read_outcome("J-TEXT", sid)
            check_raw_hash(sid, sha(raw_path), input_hashes)
            references = load_reference_curves(
                index, sid, raw["times"], reference_files
            )
            record = compare_curves(
                sid, raw["times"], curves, references, flag, down, threshold
            )
            record["contract_finite_observed_counts"] = observed
            records.append(record)
            verified_inputs[str(sid)] = raw_hash
            if index % 40 == 0:
                print(
                    dict(
                        stage=CAPTION,
                        shots=index,
                        total=len(ids),
                        seconds=time.monotonic() - started,
                    ),
                    flush=True,
                )

    # Reference per-shot rows are read only after all new shots are sealed.
    old_rows = read(REFERENCE / "per_shot.json")
    for arm in REFERENCE_ARMS:
        if old_rows[arm] != [record["reference"][arm] for record in records]:
            raise ValueError(f"reference scored rows disagree with HDF5 truth: {arm}")
    assert_source_unchanged(model, frozen)
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError("qualified source checkpoint changed")
    if any(sha(Path(path)) != digest for path, digest in code.items()):
        raise AssertionError("bound code changed")
    if reference_manifest(REFERENCE) != reference_files:
        raise AssertionError("bound reference artifacts changed")
    for sid in ids:
        check_raw_hash(sid, sha(io23.path_for("J-TEXT", sid)), input_hashes)
    verify_output_seals(output, ids, input_hashes)
    summary = summarize(records)
    write(output / "per_shot.json", dict(caption=CAPTION, shots=records))
    write(output / "summary.json", summary)
    write(output / "input_hashes.json", verified_inputs)
    artifacts = {
        path.relative_to(output).as_posix(): sha(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    completed = dict(
        caption=CAPTION,
        status="COMPLETED_COUNTERFACTUAL_NOT_DEPLOYABLE",
        shots=len(ids),
        new_arms=list(ARM_CHANNELS),
        reference_arms_reused=list(REFERENCE_ARMS),
        source_unchanged=True,
        source_checkpoint_sha256=PARENT_SHA256,
        source_full_state_sha256=frozen_state_hash,
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
        jtext_online_run=False,
        optimizer_steps=0,
        source_scaler_changes=0,
        threshold_changes=0,
        seconds=time.monotonic() - started,
        artifact_sha256=artifacts,
    )
    write(output / "completed.json", completed)
    print(dict(caption=CAPTION, status=completed["status"], shots=len(ids)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
