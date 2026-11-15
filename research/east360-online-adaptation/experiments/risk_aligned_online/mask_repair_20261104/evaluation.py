"""Offline, frozen source representation-repair controls; not an online replay.

Run with ``python -m experiments.risk_aligned_online.mask_repair_20261104.evaluation``.
Wrong-context is a deliberately non-deployable, within-window temporal control.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..east_context_operator_20261031.run import configure
from ..hdf5_score_recheck_20261103 import score_curve
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_anticipation_20261104.static_stream import timing_decomposition
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from . import core


FIT = io23.ROOT / "probe_outputs/source_mask_repair_train_20261104_v1"
FIT_COMPLETED_SHA256 = "1a7304a619414508bfecb84dc710dbd5b6a15f46265de66f829914efe4a0bc85"
PLAN = io23.ROOT / "SOURCE_MASK_REPAIR_PLAN_20261104.md"
VIEWS = {"native": (), "no_MNM": (2,), "no_MNM_CIII": (1, 2)}
ARMS = ("native", "constant", "conditional", "wrong_context")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verify_files(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"bound file SHA256 mismatch: {path}")


def load_fit(fit_dir, *, expected_source_sha, device):
    """Verify the completed fit and all its receipts before weights-only load."""
    fit_dir = Path(fit_dir).resolve()
    completed_path = fit_dir / "completed.json"
    if sha(completed_path) != FIT_COMPLETED_SHA256:
        raise ValueError("fit completed SHA256 mismatch")
    done = _read(completed_path)
    if done.get("status") != "TRAINED_EMBEDDING_REPAIR_NOT_VALIDATED":
        raise ValueError("completed TRAIN-only fit required")
    artifacts = done.get("artifact_sha256", {})
    if not {"protocol.json", "repair.pt"} <= set(artifacts):
        raise ValueError("fit protocol and repair artifact hashes required")
    if any(Path(name).name != name for name in artifacts):
        raise ValueError("fit artifact names must be local filenames")
    _verify_files({fit_dir / name: digest for name, digest in artifacts.items()})
    protocol = _read(fit_dir / "protocol.json")
    if protocol.get("source_sha256") != expected_source_sha:
        raise ValueError("fit source SHA256 mismatch")
    if (done.get("steps") != 512 or protocol.get("steps") != 512
            or protocol.get("fixed_last_iterate") is not True
            or protocol.get("source_parameter_updates") != 0):
        raise ValueError("fixed last-iterate 512-step source-frozen fit required")
    if any(protocol.get(key) != 0 for key in
           ("hard_outcome_reads", "val_signal_reads", "test_signal_reads", "east_signal_reads")):
        raise ValueError("fit must not read holdout signals or hard outcomes")
    _verify_files({**protocol.get("input_sha256", {}), **protocol.get("code_sha256", {})})
    checkpoint = torch.load(fit_dir / "repair.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("source_sha256") != expected_source_sha or checkpoint.get("trained_not_validated") is not True:
        raise ValueError("repair checkpoint source or fit status mismatch")
    repair = core.TokenRepair()
    repair.load_state_dict(checkpoint["repair"], strict=True)
    repair = repair.to(device).eval().requires_grad_(False)
    raw_constants = checkpoint["constant_delta"]
    if set(raw_constants) != {"no_MNM", "no_MNM_CIII"}:
        raise ValueError("both fixed TRAIN constant controls required")
    constants = {}
    for name, value in raw_constants.items():
        if not isinstance(value, torch.Tensor) or value.shape != (64,) or not torch.isfinite(value).all():
            raise ValueError("finite 64-dimensional TRAIN constant required")
        constants[name] = value.detach().to(device=device, dtype=next(repair.parameters()).dtype).clone()
    return repair, constants, protocol


class ConstantRepair(core.TokenRepair):
    """Core enforces eligibility and clips the fixed TRAIN vector per token."""

    def __init__(self, constants, view):
        super().__init__()
        self.register_buffer("no_mnm", constants["no_MNM"].clone())
        self.register_buffer("no_both", constants["no_MNM_CIII"].clone())
        self.view = view

    def forward(self, context):
        if self.view == "no_MNM":
            return self.no_mnm.expand(len(context), -1)
        if self.view == "no_MNM_CIII":
            return self.no_both.expand(len(context), -1)
        # Native missingness was not a training view: use the two-channel
        # constant wherever CIII is missing, otherwise the MNM-only constant.
        return torch.where((context[:, -2] < 1)[:, None], self.no_both, self.no_mnm)


class WrongContextRepair(core.TokenRepair):
    """Same repair weights on reversed tokens of this *same* past window."""

    def __init__(self, trained):
        super().__init__()
        self.load_state_dict(trained.state_dict(), strict=True)
        self.context = None

    def forward(self, context):
        if self.context is None or self.context.shape != context.shape:
            raise RuntimeError("one reversed context per eligible current token required")
        return self.net(self.context)


def evaluate_batch(base, repair, constants, x, available, *, view, arm):
    if view not in VIEWS or arm not in ARMS:
        raise ValueError("unknown view or arm")
    if any(p.requires_grad for p in repair.parameters()) or any(m.training for m in repair.modules()):
        raise ValueError("evaluation repair must be frozen and in eval mode")
    if arm in ("native", "conditional"):
        with torch.no_grad():
            return core.forward(base, repair, x, available, enabled=arm == "conditional")
    if arm == "constant":
        control = ConstantRepair(constants, view).to(device=x.device, dtype=x.dtype).eval().requires_grad_(False)
        with torch.no_grad():
            return core.forward(base, control, x, available)
    control = WrongContextRepair(repair).to(device=x.device, dtype=x.dtype).eval().requires_grad_(False)

    def capture(_module, _inputs, embedding):
        grouped = available.reshape(len(x), 16, 8, 23)
        valid = grouped.any(-1)
        count = valid.sum(-1)
        fractions = grouped[..., (1, 2)].sum(-2).to(x.dtype) / count.clamp_min(1)[..., None]
        eligible = (count > 0) & (fractions < 1).any(-1)
        control.context = torch.cat((embedding, fractions), -1).flip(1)[eligible]

    hook = base.embedding.register_forward_hook(capture)
    try:
        with torch.no_grad():
            return core.forward(base, control, x, available)
    finally:
        hook.remove()
        control.context = None


def predict_curves(base, repair, constants, shot, device, *, batch_size=256):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch size required")
    values, available = np.asarray(shot["values"]), np.asarray(shot["available"])
    if (values.ndim != 2 or values.shape[1:] != (23,) or not len(values)
            or values.dtype != np.float32 or available.shape != values.shape
            or available.dtype != np.bool_ or not np.isfinite(values[available]).all()):
        raise ValueError("nonempty finite-observed source flat23 signal and mask required")
    curves = {f"{view}__{arm}": np.empty(len(values), np.float32)
              for view in VIEWS for arm in ARMS}
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        windows = [simple_data.window(shot, end, 128) for end in range(start, stop)]
        x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        actual = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        for view, columns in VIEWS.items():
            mask = actual.clone()
            if columns:
                mask[:, :, columns] = False
            for arm in ARMS:
                out = evaluate_batch(base, repair, constants, x, mask, view=view, arm=arm)
                curves[f"{view}__{arm}"][start:stop] = out["logit"].sigmoid().cpu().numpy()
    return curves


def score_source(times, scores, disruptive, down, threshold):
    if type(disruptive) is not bool or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("HDF boolean outcome and fixed finite threshold required")
    if disruptive and (down is None or not np.isfinite(down)):
        raise ValueError("finite HDF disruption time required")
    result = score_curve(times, scores, disruptive, down, threshold)
    result["timely"] = bool(result["lead_ms"] is not None and result["lead_ms"] > 10)
    return result


def _paired(per):
    result = {}
    comparisons = (("conditional", "native"), ("conditional", "constant"),
                   ("conditional", "wrong_context"), ("constant", "native"),
                   ("wrong_context", "native"))
    for view in VIEWS:
        for candidate, reference in comparisons:
            left, right = per[f"{view}__{candidate}"], per[f"{view}__{reference}"]
            rows = []
            for a, b in zip(left, right, strict=True):
                if a["shot"] != b["shot"] or a["disruptive"] != b["disruptive"]:
                    raise AssertionError("paired shots and HDF outcomes must match")
                rows.append(dict(shot=a["shot"], disruptive=a["disruptive"],
                                 candidate_alarm=a["alarm"], reference_alarm=b["alarm"],
                                 candidate_timely=a["timely"], reference_timely=b["timely"],
                                 candidate_lead_ms=a["lead_ms"], reference_lead_ms=b["lead_ms"]))
            result[f"{view}__{candidate}_vs_{reference}"] = dict(
                rows=rows,
                disruptive_timely_gained=sum(r["disruptive"] and r["candidate_timely"] and not r["reference_timely"] for r in rows),
                disruptive_timely_lost=sum(r["disruptive"] and not r["candidate_timely"] and r["reference_timely"] for r in rows),
                safe_alarms_gained=sum(not r["disruptive"] and r["candidate_alarm"] and not r["reference_alarm"] for r in rows),
                safe_alarms_removed=sum(not r["disruptive"] and not r["candidate_alarm"] and r["reference_alarm"] for r in rows),
            )
    return result


def run_source(output, *, role="val", fit_dir=FIT, compute_device="cpu", batch_size=256):
    if role not in ("val", "test"):
        raise ValueError("source role must be val or test")
    if compute_device not in ("cpu", "cuda"):
        raise ValueError("compute device must be cpu or cuda")
    if compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch size required")
    output, fit_dir = Path(output).resolve(), Path(fit_dir).resolve()
    if output.exists() or output.is_relative_to(fit_dir):
        raise FileExistsError("output must be a new directory outside the fit")
    started = time.monotonic()
    roles = io23.source_roles()
    ids = roles[role]
    if (not ids or len(ids) != len(set(ids)) or any(type(sid) is not int for sid in ids)
            or set(ids) & (set(roles["train"]) | set(roles["test" if role == "val" else "val"]))):
        raise ValueError("distinct non-overlapping locked source role IDs required")
    repair, constants, fit_protocol = load_fit(fit_dir, expected_source_sha=PARENT_SHA256, device=compute_device)
    base, parent = load_parent(DEFAULT_PARENT)
    base = base.to(compute_device).eval().requires_grad_(False)
    base_digest, repair_digest = state_digest(base), state_digest(repair)
    threshold = float(parent["threshold"])
    if (not 0 < threshold <= 1 or fit_protocol.get("threshold") != threshold
            or fit_protocol.get("source_state_sha256") != base_digest):
        raise ValueError("fit source state or fixed parent threshold mismatch")
    code = imported_code_hashes()
    bound = {str(path): sha(path) for path in
             (PLAN, DEFAULT_PARENT, io23.JTEXT_INDEX, fit_dir / "completed.json",
              fit_dir / "protocol.json", fit_dir / "repair.pt")}
    output.mkdir(parents=True, exist_ok=False)
    shots_dir = output / "shots"
    shots_dir.mkdir()
    _write(output / "protocol.json", dict(
        scope="OFFLINE_SOURCE_REPRESENTATION_REPAIR_NOT_JTEXT_ONLINE_OR_EAST_GAIN",
        role=role, shot_ids=ids, views=VIEWS, arms=ARMS, threshold=threshold,
        threshold_source="fixed qualified parent; no threshold selection", timely_deadline_ms=10,
        source_sha256=PARENT_SHA256, source_state_sha256=base_digest, repair_state_sha256=repair_digest,
        fit_completed_sha256=FIT_COMPLETED_SHA256, input_sha256=bound, code_sha256=code,
        original_source_mask="finite source observations; no EAST CIII sentinel",
        constant_native_policy="CIII missing: no_MNM_CIII TRAIN constant; otherwise no_MNM TRAIN constant",
        wrong_context="reverse all 16 token contexts within same current past window; negative control, not deployable",
        correction="core eligibility, all-padding and fully observed tokens unchanged, per-token RMS <= 0.25",
        causal_prefix_windows=True, whole_shot_io=True, outcomes_read_after_each_seal=True,
        hard_outcome="HDF5 meta/IsDisrupt and meta/DownTime only", neural_updates=0,
        independent_future_validation=False, data_previously_used_in_development=True,
        batch_size=batch_size, compute_device=compute_device, torch_version=str(torch.__version__),
    ))
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    per = {f"{view}__{arm}": [] for view in VIEWS for arm in ARMS}
    raw_hashes, seals, signal_reads, outcome_reads = {}, [], [], []
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        path = io23.path_for("J-TEXT", sid)
        raw_hashes[str(sid)] = sha(path)

        def checked_signal(device, requested, *args, **kwargs):
            if device != "J-TEXT" or requested != sid:
                raise ValueError("only current locked source role signal allowed")
            signal_reads.append(requested)
            return reader(device, requested, *args, **kwargs)

        with (patch.object(io23, "read_signals", side_effect=checked_signal),
              patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome forbidden before seal"))):
            raw = simple_data.read_shot("J-TEXT", sid)
            if raw["shot"] != sid or raw["device"] != "J-TEXT":
                raise ValueError("source signal identity mismatch")
            shot = {key: raw[key] for key in ("shot", "values", "available")}
            curves = predict_curves(base, repair, constants, shot, compute_device, batch_size=batch_size)
            times = np.asarray(raw["times"])
            if times.shape != (len(raw["values"]),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("finite strictly increasing source clock required")
            prefix = shots_dir / f"{index:04d}_{sid}"
            npz_path = prefix.with_suffix(".npz")
            with npz_path.open("xb") as stream:
                np.savez_compressed(stream, times=times, **curves)
                stream.flush()
                os.fsync(stream.fileno())
            seal = dict(shot=sid, curve_sha256=sha(npz_path), current_outcome_used_in_prediction=False)
            _write(prefix.with_name(prefix.name + "_seal.json"), seal)
            fd = os.open(shots_dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            seals.append(seal)
        if sha(npz_path) != seal["curve_sha256"]:
            raise RuntimeError("sealed predictions changed before outcome")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"times", *per}:
                raise ValueError("sealed archive must contain all twelve curves")
            sealed_times = archive["times"].copy()
            sealed_curves = {key: archive[key].copy() for key in per}
        disruptive, down = io23.read_outcome("J-TEXT", sid)
        outcome_reads.append(sid)
        for key, curve in sealed_curves.items():
            per[key].append(dict(shot=sid, disruptive=disruptive, down_s=down,
                                 **score_source(sealed_times, curve, disruptive, down, threshold)))
        if sha(path) != raw_hashes[str(sid)] or sha(npz_path) != seal["curve_sha256"]:
            raise RuntimeError("source input or sealed curve changed during evaluation")
        if index % 40 == 0 or index == len(ids):
            print(dict(stage="source_repair", role=role, shots=index, seconds=time.monotonic() - started), flush=True)
    if (signal_reads != ids or outcome_reads != ids
            or state_digest(base) != base_digest or state_digest(repair) != repair_digest
            or any(p.grad is not None for p in (*base.parameters(), *repair.parameters()))):
        raise AssertionError("signal/outcome scope or frozen source/repair changed")
    _verify_files({**bound, **code})
    _write(output / "per_shot.json", per)
    _write(output / "summary.json", {
        name: {group: timing_decomposition(rows, deadline_ms=10)
               for group, rows in (("all", rows),
                                   ("disruptive", [r for r in rows if r["disruptive"]]),
                                   ("safe", [r for r in rows if not r["disruptive"]]))}
        for name, rows in per.items()
    })
    _write(output / "paired.json", _paired(per))
    _write(output / "raw_hashes.json", raw_hashes)
    artifacts = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    result = dict(status="COMPLETED_OFFLINE_SOURCE_REPAIR_DIAGNOSTIC", role=role,
                  seconds=time.monotonic() - started, source_unchanged=True, repair_unchanged=True,
                  source_state_sha256=base_digest, repair_state_sha256=repair_digest,
                  source_signal_reads=len(signal_reads), east_signal_reads=0, postseal_outcome_reads=len(outcome_reads),
                  neural_updates=0, seals=seals, independent_future_validation=False,
                  online_or_east_gain_measured=False, artifact_sha256=artifacts)
    _write(output / "completed.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--fit-dir", type=Path, default=FIT)
    parser.add_argument("--compute-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.compute_device == "cuda":
        configure()
    else:
        # The shared configure() deliberately requires CUDA. Keep its numerical
        # settings and seed on CPU without making CPU evaluation depend on GPU.
        torch.set_num_threads(2)
        torch.manual_seed(131)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
    result = run_source(args.output, role=args.role, fit_dir=args.fit_dir,
                        compute_device=args.compute_device, batch_size=args.batch_size)
    print(dict(status=result["status"], role=result["role"], seconds=result["seconds"]), flush=True)


if __name__ == "__main__":
    main()
