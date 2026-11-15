"""Frozen JointLoRA features for the previously sealed, trained warm20 cases.

This is an offline feature producer, not an independent evaluation. It reads
only each origin's 128-point signal window, never HDF5 outcomes or a future
risk window. All inputs are bound to the earlier span/probe receipts.
"""

from __future__ import annotations

import os
import sys

# Set these before importing NumPy/PyTorch, including for standalone runs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import prepare_actual
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
)
from joint_lora_gradient_probe_20261104 import MAP_SHA256, sha, state_digest


ROOT = io23.ROOT
SPAN = ROOT / "probe_outputs/joint_lora_gradient_span_20261104_v1"
PROBE = ROOT / "probe_outputs/joint_lora_gradient_probe_20261104_v1"
WARM = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_gate_features_20261104_v1"
SPAN_COMPLETED_SHA = "cc91a701507aa6181757018f3d96fae29d692d263d20f5ccc41241ed7f62acfe"
PROBE_COMPLETED_SHA = "5d9ef8ff763a6f501ebe513806e2c37df402290c89164407484061b8cabc36b5"

CODE_PATHS = (
    "joint_lora_gate_features_20261104.py",
    "joint_lora_gradient_span_20261104.py",
    "joint_lora_gradient_probe_20261104.py",
    "experiments/risk_aligned_online/io23.py",
    "experiments/risk_aligned_online/simple_data.py",
    "experiments/risk_aligned_online/causal_input.py",
    "experiments/risk_aligned_online/simple23.py",
    "experiments/risk_aligned_online/joint_lora_20261104/model.py",
    "experiments/risk_aligned_online/cold20_compare_20261104/adapters.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/model.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/memory.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/timeline.py",
    "experiments/risk_aligned_online/mask_repair_20261104/east_evaluation.py",
    "experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py",
    "experiments/risk_aligned_online/redesign_20261004/prepare_prior.py",
)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def case_order(cases, prior_cases, ids):
    """Read only shot/origin from case records, preserving sealed row order."""
    if len(ids) != 20 or len(set(ids)) != 20 or any(type(s) is not int for s in ids):
        raise ValueError("20 distinct integer warm shots required")

    def coordinates(rows):
        result = []
        for row in rows:
            sid, origin = row["shot"], row["origin"]
            if type(sid) is not int or type(origin) is not int or origin < 135 or (origin - 135) % 32:
                raise ValueError("integer shot and valid origin required")
            result.append((sid, origin))
        return result

    result = coordinates(cases)
    if result != coordinates(prior_cases):
        raise ValueError("span/probe case order mismatch")
    if len(result) != 60 or len(set(result)) != 60:
        raise ValueError("60 cases without duplicate coordinates required")
    if [sid for sid, _ in result] != [sid for sid in ids for _ in range(3)]:
        raise ValueError("warm-shot grouping/order mismatch")
    return result


def require_hdf_stat(sid, stat, old):
    if (old["shot"], old["size"], old["mtime_ns"]) != (sid, stat.st_size, stat.st_mtime_ns):
        raise ValueError(f"ended-shot HDF stat mismatch: {sid}")


def read_causal_window(handle, sid, origin, mapping):
    """Read exactly [origin-127, origin] from the 23 signal datasets."""
    if type(origin) is not int or origin < 127:
        raise ValueError("origin must have a full 128-point window")
    data = handle["data"]  # Deliberately never touch /meta.
    datasets = [data[name] for name in io23.RAW23]
    length = len(datasets[io23.RAW23.index("ip")])
    if origin >= length:
        raise ValueError("origin outside signal record")
    ip = datasets[io23.RAW23.index("ip")]
    rate = float(np.asarray(ip.attrs["SampleRate"]).reshape(-1)[0])
    start = float(np.asarray(ip.attrs["StartTime"]).reshape(-1)[0])
    tolerance = io23.rate_tolerance_hz("EAST", "nominal")
    if not math.isfinite(rate) or not math.isclose(rate, 1000.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("nominal EAST sample rate required")
    for dataset in datasets:
        if len(dataset) != length:
            raise ValueError("unaligned signal length")
        for name, expected in (("SampleRate", rate), ("StartTime", start)):
            if name not in dataset.attrs or not np.isclose(float(np.asarray(dataset.attrs[name]).reshape(-1)[0]), expected):
                raise ValueError(f"unaligned signal {name}")
    values = np.stack([np.asarray(dataset[origin - 127 : origin + 1], dtype=np.float32) for dataset in datasets], axis=1)
    if values.shape != (128, 23):
        raise ValueError("complete 128-by-23 window required")
    available = np.isfinite(values)
    values = np.where(available, values, 0.0).astype(np.float32)
    available[:, io23.RAW23.index("MNM")] = False
    values[~available] = 0.0
    # The span used index 21 for every case: apply the saved pointwise map.
    shot = dict(device="EAST", shot=sid, values=values, available=available)
    prepared = prepare_actual(shot, 21, mapping)
    return prepared["values"], prepared["available"]


def extract_latent(model, values, available):
    if model.training or any(p.requires_grad or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("frozen CPU eval model required")
    if values.shape != (128, 23) or values.dtype != np.float32 or available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("float32 values and boolean mask for one 128-by-23 window required")
    with torch.no_grad():
        result = model(torch.from_numpy(values[None]), torch.from_numpy(available[None]))["latent"]
    if result.shape != (1, 192) or result.dtype != torch.float32 or not torch.isfinite(result).all():
        raise ValueError("finite float32 192-coordinate latent required")
    return result[0].cpu().numpy().copy()


def load_bound_inputs():
    paths = {
        "span_completed": SPAN / "completed.json",
        "span_cases": SPAN / "cases.json",
        "span_summary": SPAN / "summary.json",
        "span_gradients": SPAN / "gradients.npz",
        "probe_completed": PROBE / "completed.json",
        "probe_cases": PROBE / "cases.json",
        "probe_summary": PROBE / "summary.json",
        "warm_json": WARM / "warm.json",
        "warm_pt": WARM / "warm.pt",
        "density_map": MAP,
        "source_checkpoint": DEFAULT_PARENT,
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    code = {path: sha(ROOT / path) for path in CODE_PATHS}
    if hashes["span_completed"] != SPAN_COMPLETED_SHA or hashes["probe_completed"] != PROBE_COMPLETED_SHA:
        raise ValueError("sealed span/probe completed receipt mismatch")
    span_done, probe_done = read_json(paths["span_completed"]), read_json(paths["probe_completed"])
    for prefix, done, names in (
        ("span", span_done, ("cases", "summary", "gradients")),
        ("probe", probe_done, ("cases", "summary")),
    ):
        for name in names:
            suffix = "npz" if name == "gradients" else "json"
            if hashes[f"{prefix}_{name}"] != done["artifact_sha256"][f"{name}.{suffix}"]:
                raise ValueError(f"sealed {prefix} {name} SHA mismatch")
    if (code["joint_lora_gradient_span_20261104.py"] != span_done["script_sha256"]
            or code["joint_lora_gradient_probe_20261104.py"] != probe_done["script_sha256"]
            or hashes["source_checkpoint"] != PARENT_SHA256 or hashes["density_map"] != MAP_SHA256):
        raise ValueError("sealed code/source/map SHA mismatch")
    span, prior = read_json(paths["span_summary"]), read_json(paths["probe_summary"])
    for name in ("warm_json", "warm_pt", "density_map", "source_checkpoint"):
        if hashes[name] != span["input_sha256"][name] or hashes[name] != prior["input_sha256"][name]:
            raise ValueError(f"prior input changed: {name}")
    if (span["input_sha256"]["prior_script"] != probe_done["script_sha256"]
            or prior["input_sha256"]["script"] != probe_done["script_sha256"]
            or span["prior_completed_sha256"] != PROBE_COMPLETED_SHA):
        raise ValueError("span/probe provenance mismatch")
    warm, mapping = read_json(paths["warm_json"]), read_json(paths["density_map"])
    ids = warm["label_shots"]
    if ids != prior["label_shots"] or ids != span["shots"] or ids != mapping["calibration_ids"]:
        raise ValueError("warm/map/prior shot identities differ")
    order = case_order(read_json(paths["span_cases"]), read_json(paths["probe_cases"]), ids)
    stats = prior["shots"]
    if [row["shot"] for row in stats] != ids:
        raise ValueError("prior HDF identity order mismatch")
    return paths, hashes, code, warm, mapping, span, prior, order, stats


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite or reuse an output directory")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths, hashes, code, warm, mapping, span, prior, order, stats = load_bound_inputs()
    base, _ = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = state_digest(base)
    if source_before != prior["source_state_sha256"] or source_before != span["source_state_sha256"]:
        raise ValueError("source state differs from prior probes")
    state = torch.load(paths["warm_pt"], map_location="cpu", weights_only=True)
    model = JointLoRA(base, state["W0"])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    warm_before = state_digest(model)
    if warm_before != warm["fitted_state_sha256"] or warm_before != span["warm_state_sha256"] or warm_before != prior["warm_state_sha256"]:
        raise ValueError("frozen warm state differs from prior receipts")
    latents = []
    hdf_stats = []
    with (patch.object(io23, "read_outcome", side_effect=RuntimeError("HDF outcomes forbidden")),
          patch.object(io23, "read_signals", side_effect=RuntimeError("whole-shot reads forbidden"))):
        for index, old in enumerate(stats):
            sid = old["shot"]
            path = io23.path_for("EAST", sid)
            before = path.stat()
            require_hdf_stat(sid, before, old)
            with h5py.File(path, "r") as handle:
                for case_sid, origin in order[index * 3 : index * 3 + 3]:
                    if case_sid != sid:
                        raise AssertionError("case order changed")
                    values, available = read_causal_window(handle, sid, origin, mapping)
                    latents.append(extract_latent(model, values, available))
            require_hdf_stat(sid, path.stat(), old)
            hdf_stats.append(dict(shot=sid, size=before.st_size, mtime_ns=before.st_mtime_ns))
            print(f"ended shots {index + 1}/20; origin windows {len(latents)}/60", flush=True)
    if (len(latents) != 60 or state_digest(base) != source_before or state_digest(model) != warm_before
            or any(p.grad is not None for p in model.parameters())):
        raise AssertionError("scope, frozen state, or gradient isolation failed")
    for old in stats:
        require_hdf_stat(old["shot"], io23.path_for("EAST", old["shot"]).stat(), old)
    if any(sha(path) != hashes[name] for name, path in paths.items()) or any(sha(ROOT / path) != digest for path, digest in code.items()):
        raise ValueError("bound input or code changed during feature extraction")
    latent = np.stack(latents).astype(np.float32, copy=False)
    shots = np.asarray([sid for sid, _ in order], dtype=np.int64)
    origins = np.asarray([origin for _, origin in order], dtype=np.int64)
    if latent.shape != (60, 192) or not np.isfinite(latent).all():
        raise AssertionError("complete finite feature array required")
    order_digest = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
    protocol = dict(
        scope="TRAINED_ENDED20_60_CASE_ORIGIN_FEATURES_NOT_INDEPENDENT_VALIDATION",
        description="These ended shots participated in the common warm fit; no independent validation or online policy evaluation.",
        input_sha256=hashes,
        input_paths={name: str(path) for name, path in paths.items()},
        code_sha256=code,
        case_order_sha256=order_digest,
        case_order=[dict(shot=sid, origin=origin) for sid, origin in order],
        hdf_stats=hdf_stats,
        warm_state_sha256=warm_before,
        source_state_sha256=source_before,
        device="cpu", torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        window="128 samples inclusive [origin-127, origin]; 23 channels; pointwise index-21 saved input map",
        latent="JointLoRA latent: concatenated final CLS, last token, and last-minus-previous token",
        hdf_outcomes_read=0, whole_shot_reads=0, future_risk_windows_read=0,
        updated_parameters=0, independent_validation=False, online_policy_tested=False,
    )
    OUTPUT.mkdir(parents=False, exist_ok=False)
    with (OUTPUT / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, latent=latent, shot=shots, origin=origins)
    write_json(OUTPUT / "protocol.json", protocol)
    completed = dict(
        status="COMPLETED_TRAINED_ENDED20_CPU_ORIGIN_FEATURES",
        input_sha256=hashes, code_sha256=code, case_order_sha256=order_digest,
        warm_state_sha256=warm_before,
        artifact_sha256={name: sha(OUTPUT / name) for name in ("features.npz", "protocol.json")},
    )
    write_json(OUTPUT / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    run()
