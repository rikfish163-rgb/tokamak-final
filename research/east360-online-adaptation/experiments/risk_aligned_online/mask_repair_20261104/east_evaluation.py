"""Frozen four-arm EAST actual-mask static replay; no online learning."""
import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..hdf5_score_recheck_20261103 import score_curve
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_anticipation_20261104.static_stream import ORDER, PREPARATION, timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform
from .evaluation import (ARMS, FIT, configure, evaluate_batch, load_fit, sha,
                         _read, _verify_files, _write)

BASELINE = io23.ROOT / "probe_outputs/qualified_source_static360_20261104_v1"
BASELINE_DONE_SHA = "ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f"
PLAN = io23.ROOT / "EAST_MASK_REPAIR_STATIC_PLAN_20261104.md"


def load_inputs():
    """Bind the exact previously sealed baseline, ordered cohort and fixed map."""
    done_path = BASELINE / "completed.json"
    if sha(done_path) != BASELINE_DONE_SHA:
        raise ValueError("baseline completed SHA256 mismatch")
    done = _read(done_path)
    if done.get("status") != "COMPLETED_STATIC_BASELINE_DIAGNOSTIC" or done.get("neural_updates") != 0:
        raise ValueError("completed frozen static baseline required")
    artifact_hashes = done.get("artifact_sha256", {})
    if "protocol.json" not in artifact_hashes or any(Path(name).name != name for name in artifact_hashes):
        raise ValueError("baseline artifact hashes required")
    bound = {str(BASELINE / name): value for name, value in artifact_hashes.items()}
    bound[str(done_path)] = BASELINE_DONE_SHA
    _verify_files(bound)
    baseline = _read(BASELINE / "protocol.json")
    prior = _read(ORDER)
    order = prior["order"]
    ids = [r["shot"] for r in order]
    map_path, hashes_path = PREPARATION / "input_map.json", PREPARATION / "input_hashes.json"
    mapping, hashes = _read(map_path), _read(hashes_path)
    if (len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids("EAST"))
            or baseline["order"] != order or baseline["batch_size"] != 256
            or baseline["source_sha256"] != PARENT_SHA256
            or mapping["calibration_ids"] != ids[:20] or sha(map_path) != prior["input_map_sha256"]
            or set(hashes) != {str(sid) for sid in ids} or baseline["raw_sha256"] != hashes
            or len(done["seals"]) != 360 or [s["shot"] for s in done["seals"]] != ids):
        raise ValueError("locked baseline same360 order, map and raw hashes required")
    bound.update({str(path): sha(path) for path in (ORDER, map_path, hashes_path, io23.EAST_COHORT)})
    _verify_files(baseline["input_sha256"])
    bound.update(baseline["input_sha256"])
    return dict(ids=ids, order=order, mapping=mapping, hashes=hashes, seals=done["seals"],
                baseline_shots=BASELINE / "shots", baseline_protocol=baseline, bound=bound)


def prepare_actual(shot, index, mapping, *, report_coverage=False):
    if shot["device"] != "EAST" or type(index) is not int or index < 1 or shot["available"][:, 2].any():
        raise ValueError("actual EAST mask with unobserved MNM and positive index required")
    raw = apply_input_contract(shot)
    coverage = None
    if report_coverage:
        mask = raw["available"]
        total = len(mask)
        if not total:
            raise ValueError("nonempty EAST signal required for coverage")
        coverage = dict(shot=shot["shot"], total_points=total)
        for name, column in (("CIII", 1), ("MNM", 2)):
            count = int(mask[:, column].sum())
            coverage[name] = dict(available_points=count, available_fraction=count / total)
    signal = {k: raw[k] for k in ("values", "available")}
    result = signal if index <= 20 else transform(signal, mapping)
    if not np.array_equal(result["available"], raw["available"]):
        raise AssertionError("fixed numerical map changed actual availability")
    prepared = dict(result, shot=shot["shot"])
    return (prepared, coverage) if report_coverage else prepared


def predict_actual_curves(base, repair, constants, shot, device, *, batch_size=256):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch size required")
    curves = {arm: np.empty(len(shot["values"]), np.float32) for arm in ARMS}
    if not len(shot["values"]):
        raise ValueError("nonempty EAST signal required")
    for start in range(0, len(shot["values"]), batch_size):
        stop = min(start + batch_size, len(shot["values"]))
        windows = [simple_data.window(shot, end, 128) for end in range(start, stop)]
        x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        available = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        for arm in ARMS:
            out = evaluate_batch(base, repair, constants, x, available, view="native", arm=arm)
            curves[arm][start:stop] = out["logit"].sigmoid().cpu().numpy()
    return curves


def check_reference(path, receipt_path, expected, times, native):
    if (_read(receipt_path) != expected or expected.get("current_outcome_used_in_prediction") is not False
            or sha(path) != expected.get("curve_sha256")):
        raise ValueError("baseline seal or curve SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["times"], times):
            raise ValueError("baseline clock parity mismatch")
        if not np.array_equal(archive["map_after20"], native):
            raise ValueError("baseline native parity mismatch: map_after20 differs")


def _paired(per):
    result = {}
    for candidate, reference in (("constant", "native"), ("conditional", "native"),
                                 ("wrong_context", "native"), ("conditional", "constant"),
                                 ("conditional", "wrong_context")):
        rows = []
        for a, b in zip(per[candidate], per[reference], strict=True):
            if a["shot"] != b["shot"] or a["disruptive"] != b["disruptive"]:
                raise ValueError("paired HDF outcomes and shot order must match")
            rows.append(dict(shot=a["shot"], disruptive=a["disruptive"],
                             candidate_alarm=a["alarm"], reference_alarm=b["alarm"],
                             candidate_timely=a["timely"], reference_timely=b["timely"]))
        groups = {}
        for name, subset in (("full360", rows), ("startup20", rows[:20]), ("suffix340", rows[20:])):
            groups[name] = dict(shots=len(subset),
                timely_gained=sum(r["disruptive"] and r["candidate_timely"] and not r["reference_timely"] for r in subset),
                timely_lost=sum(r["disruptive"] and not r["candidate_timely"] and r["reference_timely"] for r in subset),
                safe_alarms_gained=sum(not r["disruptive"] and r["candidate_alarm"] and not r["reference_alarm"] for r in subset),
                safe_alarms_removed=sum(not r["disruptive"] and not r["candidate_alarm"] and r["reference_alarm"] for r in subset))
        result[f"{candidate}_vs_{reference}"] = dict(rows=rows, groups=groups)
    return result


def run_east(output, *, compute_device="cpu"):
    if compute_device not in ("cpu", "cuda"):
        raise ValueError("compute device must be cpu or cuda")
    if compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(FIT.resolve()) or output.is_relative_to(BASELINE.resolve()):
        raise FileExistsError("new output outside fit and baseline required")
    started = time.monotonic()
    inputs = load_inputs()
    ids, mapping, hashes = inputs["ids"], inputs["mapping"], inputs["hashes"]
    repair, constants, fit_protocol = load_fit(FIT, expected_source_sha=PARENT_SHA256, device=compute_device)
    base, parent = load_parent(DEFAULT_PARENT)
    base = base.to(compute_device).eval().requires_grad_(False)
    before, repair_before = state_digest(base), state_digest(repair)
    threshold = float(parent["threshold"])
    baseline = inputs["baseline_protocol"]
    if (fit_protocol["threshold"] != threshold or baseline["threshold"] != threshold
            or fit_protocol["source_state_sha256"] != before or baseline["source_state_sha256"] != before):
        raise ValueError("fit, baseline and fixed parent must match")
    code = imported_code_hashes()
    bound = {**inputs["bound"], **{str(p): sha(p) for p in
        (PLAN, DEFAULT_PARENT, FIT / "completed.json", FIT / "protocol.json", FIT / "repair.pt")}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "shots").mkdir()
    _write(output / "protocol.json", dict(scope="EAST_ACTUAL_MASK_STATIC_NOT_INTRA_OR_INDEPENDENT",
        order=inputs["order"], arms=ARMS, threshold=threshold, timely_deadline_ms=30,
        source_sha256=PARENT_SHA256, source_state_sha256=before, repair_state_sha256=repair_before,
        baseline_completed_sha256=BASELINE_DONE_SHA, map_after20="first20 unmapped, suffix340 fixed map",
        actual_mask="EAST CIII sentinel and MNM unavailable; no counterfactual views",
        wrong_context="within current window reversal including padding; nondeployable negative control",
        native_exact_baseline_required=True, neural_updates=0, independent_future_validation=False,
        input_sha256=bound, code_sha256=code, batch_size=256, compute_device=compute_device))
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    per, seals, calls, coverages = {arm: [] for arm in ARMS}, [], [], []
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        path = io23.path_for("EAST", sid)
        if sha(path) != hashes[str(sid)]:
            raise ValueError("EAST raw SHA256 mismatch")
        def checked(device, requested, *args, **kwargs):
            if device != "EAST" or requested != sid:
                raise ValueError("only current locked EAST shot allowed")
            calls.append(sid)
            return reader(device, requested, *args, **kwargs)
        with (patch.object(io23, "read_signals", side_effect=checked),
              patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome forbidden before seal"))):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            if raw["shot"] != sid:
                raise ValueError("EAST shot identity mismatch")
            prepared, coverage = prepare_actual(raw, index, mapping, report_coverage=True)
            curves = predict_actual_curves(base, repair, constants, prepared, compute_device)
            times = np.asarray(raw["times"])
            if times.shape != (len(raw["values"]),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("finite increasing EAST clock required")
            prefix = f"{index:04d}_{sid}"
            reference = inputs["baseline_shots"] / prefix
            check_reference(reference.with_suffix(".npz"), reference.with_name(prefix + "_seal.json"),
                            inputs["seals"][index - 1], times, curves["native"])
            npz = output / "shots" / (prefix + ".npz")
            with npz.open("xb") as stream:
                np.savez_compressed(stream, times=times, **curves)
                stream.flush()
                os.fsync(stream.fileno())
            seal = dict(shot=sid, curve_sha256=sha(npz), current_outcome_used_in_prediction=False,
                        reference_curve_sha256=inputs["seals"][index - 1]["curve_sha256"], coverage=coverage)
            _write(output / "shots" / (prefix + "_seal.json"), seal)
            fd = os.open(output / "shots", os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            seals.append(seal)
            coverages.append(coverage)
        if sha(npz) != seal["curve_sha256"]:
            raise ValueError("prediction seal changed")
        with np.load(npz, allow_pickle=False) as stored:
            sealed_times = stored["times"].copy()
            sealed_curves = {arm: stored[arm].copy() for arm in ARMS}
        disruptive, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            per[arm].append(dict(shot=sid, disruptive=disruptive, down_s=down,
                                 **score_curve(sealed_times, sealed_curves[arm], disruptive, down, threshold)))
        if sha(path) != hashes[str(sid)] or sha(npz) != seal["curve_sha256"]:
            raise ValueError("raw file or sealed predictions changed")
        if index % 20 == 0 or index == len(ids):
            print(dict(stage="east_static_repair", shots=index, seconds=time.monotonic() - started), flush=True)
    if (calls != ids or state_digest(base) != before or state_digest(repair) != repair_before
            or any(p.grad is not None for p in (*base.parameters(), *repair.parameters()))):
        raise AssertionError("EAST read scope or frozen model changed")
    _verify_files({**bound, **code})
    _write(output / "per_shot.json", per)
    _write(output / "summary.json", {arm: {name: timing_decomposition(rows, deadline_ms=30)
        for name, rows in (("full360", per[arm]), ("startup20", per[arm][:20]), ("suffix340", per[arm][20:]))}
        for arm in ARMS})
    _write(output / "paired.json", _paired(per))
    _write(output / "raw_hashes.json", hashes)
    _write(output / "coverage.json", coverages)
    result = dict(status="COMPLETED_EAST_STATIC_REPAIR_DIAGNOSTIC", neural_updates=0,
                  source_unchanged=True, repair_unchanged=True, east_signal_reads=len(calls),
                  postseal_outcome_reads=len(seals), native_exact_baseline=True, seals=seals,
                  independent_future_validation=False, seconds=time.monotonic() - started,
                  artifact_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()})
    _write(output / "completed.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.compute_device == "cuda":
        configure()
    else:
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
    result = run_east(args.output, compute_device=args.compute_device)
    print(dict(status=result["status"], seconds=result["seconds"]), flush=True)


if __name__ == "__main__":
    main()
