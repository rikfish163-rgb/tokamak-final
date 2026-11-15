"""Fixed, bounded, noncausal current-shot hindsight LoRA capacity diagnostic.

Run ``--select`` before ``--run``. Selection reads only sealed source curves
and v5 metadata; only the latter phase opens the selected raw HDF files.
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.cold20_compare_20261104.adapters import make_adapter
from experiments.risk_aligned_online.joint_lora_20261104.timeline import _project
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import prepare_actual
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import state_digest
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "probe_outputs/qualified_source_static360_20261104_v1"
INVENTORY = ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v5"
OUTPUT = ROOT / "probe_outputs/joint_lora_hindsight_capacity_20261104_v1"
PLAN = ROOT / "JOINT_LORA_HINDSIGHT_CAPACITY_20261104.md"
SOURCE_DONE_SHA = "ef41bc2db11c19d12f7fd2f6b2fea24a5208d0daa4cf5b818c996bf80d66d78f"
V5_DONE_SHA = "5d0e33682a1cd62ced8b39b428a1b0da885b93b49672ef6aa63b173cf6ae8222"
V5_METADATA_SHA = "286f838542f5186be61696485bd3952fda66bcacc72a32a27f72a96971ea79b1"
THRESHOLD = 0.7258946895599365
FIRST_ADAPTED = 135
PARITY_ATOL = 3e-6
STEPS = 60
LR = 0.01
RADIUS = 0.5
CLIP = 1.0
BATCH = 256


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def hindsight_labels(times, disruptive, down):
    times = np.asarray(times, dtype=np.float64)
    if not disruptive:
        return np.zeros(len(times), dtype=np.float32), np.ones(len(times), dtype=bool)
    if down is None or not math.isfinite(down):
        raise ValueError("a disruptive HDF outcome requires a finite DownTime")
    raw = (down - times) * 1000
    valid = raw >= -1e-6
    # Match the existing EAST label contract's microsecond-of-ms tolerance.
    return (valid & (raw <= 125 + 1e-6)).astype(np.float32), valid


def timeline(times, values, disruptive, down, threshold):
    times, values = np.asarray(times), np.asarray(values)
    if times.ndim != 1 or values.shape != times.shape or not np.isfinite(values).all():
        raise ValueError("finite aligned one-dimensional curve required")
    _, valid = hindsight_labels(times, disruptive, down)
    hits = np.flatnonzero(valid & (values >= threshold))
    first = int(hits[0]) if len(hits) else None
    lead = round(float((down - times[first]) * 1000), 9) if disruptive and first is not None else None
    status = ("tp" if lead > 30 else "tardy") if lead is not None else ("miss" if disruptive else ("fp" if first is not None else "tn"))
    return dict(status=status, first_index=first, lead_ms=lead)


def _four(indices):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 4:
        raise ValueError("four distinct support endpoints required")
    result = [int(indices[(len(indices) * k) // 5]) for k in (1, 2, 3, 4)]
    if len(set(result)) != 4:
        raise ValueError("support quantiles not distinct")
    return result


def fixed_support(times, disruptive, down, probabilities, threshold):
    times = np.asarray(times)
    if len(times) != len(probabilities):
        raise ValueError("support curve and clock differ")
    indices = np.arange(len(times))
    if disruptive:
        lead = np.round((down - times) * 1000, 9)
        neg = indices[(indices >= FIRST_ADAPTED) & (lead > 125)]
        pos = indices[(indices >= FIRST_ADAPTED) & (lead > 30) & (lead <= 125)]
        result = _four(neg) + _four(pos)
    else:
        first = timeline(times, probabilities, False, None, threshold)["first_index"]
        if first is None or first < 151 or first + 48 >= len(times):
            raise ValueError("safe first-alarm support boundary is unavailable")
        result = _four(np.arange(FIRST_ADAPTED, first)) + [first + x for x in (0, 16, 32, 48)]
    if len(result) != len(set(result)) or min(result) < FIRST_ADAPTED:
        raise ValueError("eight distinct post-prefix support endpoints required")
    return result


def choose_cases(records):
    result = []
    for kind in ("close", "deep", "safe"):
        eligible = [r for r in records if r["index"] > 20 and r["kind"] == kind]
        eligible.sort(key=lambda r: r["index"])
        if not eligible:
            raise ValueError(f"design gate: no {kind} stratum")
        result.extend(eligible[:2])
    return sorted(result, key=lambda r: r["index"])


def held_splice(baseline, adapted, support):
    baseline, adapted = np.asarray(baseline), np.asarray(adapted)
    if baseline.ndim != 1 or baseline.shape != adapted.shape:
        raise ValueError("aligned one-dimensional curves required")
    if len(set(support)) != len(support) or any(i < FIRST_ADAPTED or i >= len(baseline) for i in support):
        raise ValueError("distinct post-prefix support indices required")
    result = adapted.copy()
    result[:FIRST_ADAPTED] = baseline[:FIRST_ADAPTED]
    result[support] = baseline[support]
    return result


def collateral(times, baseline, adapted, disruptive, down, threshold, support):
    times, baseline, adapted = np.asarray(times), np.asarray(baseline), np.asarray(adapted)
    y, valid = hindsight_labels(times, disruptive, down)
    mask = valid & (y == 0) & (np.arange(len(times)) >= FIRST_ADAPTED)
    mask[np.asarray(support, dtype=int)] = False
    new = np.flatnonzero(mask & (baseline < threshold) & (adapted >= threshold))
    delta = (adapted - baseline)[mask]
    return dict(held_earlier_negative_points=int(mask.sum()), new_crossings=int(len(new)),
                first_new_crossing_index=int(new[0]) if len(new) else None,
                max_increase=float(delta.max()) if len(delta) else None,
                mean_change=float(delta.mean()) if len(delta) else None)


def project_joint(parameters, initial, radius):
    return _project(parameters, initial, radius)


def verify_inputs():
    if sha(SOURCE / "completed.json") != SOURCE_DONE_SHA or sha(INVENTORY / "completed.json") != V5_DONE_SHA:
        raise ValueError("completion receipt anchor changed")
    done, v5 = read(SOURCE / "completed.json"), read(INVENTORY / "completed.json")
    if done["status"] != "COMPLETED_STATIC_BASELINE_DIAGNOSTIC" or done["neural_updates"] != 0 or v5["status"] != "PASS":
        raise ValueError("completed static and PASS v5 receipts required")
    for directory, receipt in ((SOURCE, done), (INVENTORY, v5)):
        for name, expected in receipt["artifact_sha256"].items():
            if Path(name).name != name or sha(directory / name) != expected:
                raise ValueError(f"artifact seal failed: {name}")
    if sha(INVENTORY / "hdf5_metadata.json") != V5_METADATA_SHA:
        raise ValueError("v5 inventory anchor changed")
    protocol = read(SOURCE / "protocol.json")
    order = [int(r["shot"]) for r in protocol["order"]]
    if len(order) != 360 or len(set(order)) != 360 or protocol["threshold"] != THRESHOLD:
        raise ValueError("locked 360 source order or threshold changed")
    if protocol["source_sha256"] != PARENT_SHA256 or len(done["seals"]) != 360:
        raise ValueError("qualified source identity or seals changed")
    metadata = read(INVENTORY / "hdf5_metadata.json")["EAST"]
    meta = {int(row["shot"]): row for row in metadata}
    if len(meta) != len(metadata) or not set(order) <= set(meta):
        raise ValueError("v5 EAST metadata incomplete or duplicated")
    return protocol, done, order, meta


def sealed_curve(index, shot, done, meta):
    stem = f"{index:04d}_{shot}"
    path = SOURCE / "shots" / (stem + ".npz")
    seal = read(SOURCE / "shots" / (stem + "_seal.json"))
    if (seal != done["seals"][index - 1] or seal["shot"] != shot
            or seal["current_outcome_used_in_prediction"] is not False
            or sha(path) != seal["curve_sha256"]):
        raise ValueError(f"sealed source curve failed for {shot}")
    with np.load(path, allow_pickle=False) as archive:
        times = archive["times"].copy()
        scores = archive["map_after20"].copy()
    truth = meta[shot]
    if (times.ndim != 1 or len(times) != truth["samples"] or len(times) < FIRST_ADAPTED + 1
            or not np.array_equal(times, truth["start"] + np.arange(len(times)) / truth["rate"])
            or scores.shape != times.shape or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1)) or type(truth["disruptive"]) is not bool):
        raise ValueError(f"sealed curve/v5 clock contract failed for {shot}")
    return times, scores, seal


def select():
    start = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError("precommit output already exists")
    protocol, done, order, meta = verify_inputs()
    candidates = []
    for index, shot in enumerate(order, 1):
        if index <= 20:
            continue
        times, scores, seal = sealed_curve(index, shot, done, meta)
        truth = meta[shot]
        disruptive, down = truth["disruptive"], truth["down"]
        scored = timeline(times, scores, disruptive, down, THRESHOLD)
        kind, margin = None, None
        if disruptive and scored["status"] == "miss":
            lead = np.round((down - times) * 1000, 9)
            eligible = (np.arange(len(times)) >= FIRST_ADAPTED) & (lead > 30) & (lead <= 3000)
            if eligible.any():
                margin = THRESHOLD - float(scores[eligible].max())
                if 0 < margin <= 0.1:
                    kind = "close"
                elif margin >= 0.4:
                    kind = "deep"
        elif not disruptive and scored["status"] == "fp":
            first = scored["first_index"]
            if first >= 151 and first + 48 < len(times):
                kind = "safe"
        if kind:
            candidates.append(dict(index=index, shot=shot, kind=kind, margin=margin,
                                   source_first_index=scored["first_index"], source_curve_sha256=seal["curve_sha256"]))
    selected = choose_cases(candidates)
    for row in selected:
        times, scores, _ = sealed_curve(row["index"], row["shot"], done, meta)
        truth = meta[row["shot"]]
        support = fixed_support(times, truth["disruptive"], truth["down"], scores, THRESHOLD)
        labels, valid = hindsight_labels(times, truth["disruptive"], truth["down"])
        if not valid[support].all():
            raise ValueError("invalid preterminal support")
        row["support_indices"] = support
        row["support_labels"] = [int(i) for i in labels[support]]
        row["down_s"] = truth["down"]
        row["disruptive"] = truth["disruptive"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="PRECOMMITTED_BEFORE_RAW_SIGNAL_ACCESS", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="HINDSIGHT_ORACLE_NOT_ONLINE_NOT_UPPER_BOUND", source_completed_sha256=SOURCE_DONE_SHA,
        v5_completed_sha256=V5_DONE_SHA, v5_metadata_sha256=V5_METADATA_SHA,
        script_sha256=sha(Path(__file__)), plan_sha256=sha(PLAN),
        settings=dict(threshold=THRESHOLD, first_adapted=FIRST_ADAPTED, max_per_stratum=2,
                      support_count=8, steps=STEPS, learning_rate=LR, grad_clip=CLIP,
                      joint_radius=RADIUS, optimizer="Adam, zero weight decay", seed=20261104,
                      trainable="last-layer rank-2 Q/V LoRA, 512", inference_batch=BATCH,
                      cpu_threads=1, device="cpu", reference_parity_atol=PARITY_ATOL),
        eligible_stratum_counts={kind: sum(row["kind"] == kind for row in candidates) for kind in ("close", "deep", "safe")},
        shots=selected, seconds=time.monotonic() - start,
    )
    write_new(OUTPUT / "precommit.json", result)
    print(json.dumps(dict(stage="PRECOMMITTED", precommit_sha256=sha(OUTPUT / "precommit.json"),
                          shots=[(r["index"], r["shot"], r["kind"], r["margin"]) for r in selected],
                          seconds=result["seconds"])), flush=True)


def windows(shot, indices):
    rows = [simple_data.window(shot, int(i), 128) for i in indices]
    return (torch.from_numpy(np.stack([r[0] for r in rows])),
            torch.from_numpy(np.stack([r[1] for r in rows])))


@torch.no_grad()
def curve(model, signal):
    scores = np.empty(len(signal["values"]), dtype=np.float32)
    for lo in range(0, len(scores), BATCH):
        hi = min(lo + BATCH, len(scores))
        x, available = windows(signal, range(lo, hi))
        scores[lo:hi] = model(x, available)["logit"].sigmoid().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite prediction")
    return scores


def frozen_digest(model, parameters):
    allowed = {id(p) for p in parameters}
    result = hashlib.sha256()
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if id(tensor) not in allowed:
            result.update(name.encode())
            result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def band_metrics(times, baseline, adapted, down, support, lower, upper):
    lead = np.round((down - times) * 1000, 9)
    held = (np.arange(len(times)) >= FIRST_ADAPTED) & (lead > lower) & (lead <= upper)
    held[np.asarray(support, dtype=int)] = False
    if not held.any():
        raise ValueError("no held endpoints in required timely band")
    before, after = baseline[held], adapted[held]
    return dict(n=int(held.sum()), baseline_max=float(before.max()), oracle_max=float(after.max()),
                baseline_margin=float(THRESHOLD - before.max()), oracle_margin=float(THRESHOLD - after.max()),
                max_increase=float((after - before).max()), max_abs_change=float(np.abs(after - before).max()),
                upward_crossings=int(((before < THRESHOLD) & (after >= THRESHOLD)).sum()))


def run(expected_precommit_sha):
    started = time.monotonic()
    if sha(OUTPUT / "precommit.json") != expected_precommit_sha:
        raise ValueError("precommit SHA256 differs from caller-supplied commitment")
    pre = read(OUTPUT / "precommit.json")
    if (pre["status"] != "PRECOMMITTED_BEFORE_RAW_SIGNAL_ACCESS" or pre["script_sha256"] != sha(Path(__file__))
            or pre["plan_sha256"] != sha(PLAN)):
        raise ValueError("precommit, code or plan changed")
    protocol, done, order, meta = verify_inputs()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    original = state_digest(base)
    if parent["threshold"] != THRESHOLD or original != protocol["source_state_sha256"]:
        raise ValueError("risk-neutral source or threshold differs from sealed reference")
    map_path = next(Path(p) for p in protocol["input_sha256"] if p.endswith("/input_map.json"))
    if sha(map_path) != protocol["input_sha256"][str(map_path)]:
        raise ValueError("fixed numerical map SHA changed")
    mapping = read(map_path)
    if mapping["calibration_ids"] != order[:20]:
        raise ValueError("fixed map does not match the source startup")
    initial_digest = None
    rows = []
    for case in pre["shots"]:
        step_start = time.monotonic()
        index, shot = case["index"], case["shot"]
        if index <= 20 or order[index - 1] != shot:
            raise ValueError("selected shot not in locked suffix")
        truth = meta[shot]
        times, baseline, seal = sealed_curve(index, shot, done, meta)
        if seal["curve_sha256"] != case["source_curve_sha256"]:
            raise ValueError("selected seal differs from precommit")
        path = io23.path_for("EAST", shot)
        expected_raw = protocol["raw_sha256"][str(shot)]
        if sha(path) != expected_raw:
            raise ValueError(f"raw SHA changed before reading {shot}")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (truth["size"], truth["mtime_ns"]):
            raise ValueError("raw file stat differs from v5 inventory")
        observed_outcome = io23.read_outcome("EAST", shot)
        if observed_outcome != (case["disruptive"], case["down_s"]):
            raise ValueError("selected raw HDF outcome differs from v5")
        raw = simple_data.read_shot("EAST", shot, rate_contract="nominal")
        if not np.array_equal(times, raw["times"]):
            raise ValueError("selected raw clock differs from sealed curve")
        signal = prepare_actual(raw, index, mapping)
        if sha(path) != expected_raw:
            raise ValueError("raw SHA changed after reading")
        support = case["support_indices"]
        if fixed_support(times, case["disruptive"], case["down_s"], baseline, THRESHOLD) != support:
            raise ValueError("support does not replay from precommit")
        labels, valid = hindsight_labels(times, case["disruptive"], case["down_s"])
        if labels[support].astype(int).tolist() != case["support_labels"] or not valid[support].all():
            raise ValueError("support labels differ from precommit")
        model = make_adapter(base, "qv_lora")
        params = [p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in params) != 512 or len(params) != 4:
            raise ValueError("only four Q/V rank-2 tensors (512 scalars) may train")
        initial = [p.detach().clone() for p in params]
        init_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in initial)).hexdigest()
        if initial_digest is None:
            initial_digest = init_hash
        if init_hash != initial_digest:
            raise ValueError("initial Q/V state changed between shots")
        frozen_before = frozen_digest(model, params)
        neutral = curve(model, signal)
        parity_gap = float(np.max(np.abs(neutral - baseline)))
        if parity_gap > PARITY_ATOL:
            raise ValueError(f"neutral source parity failed: {shot}, {parity_gap}")
        x, available = windows(signal, support)
        target = torch.from_numpy(labels[support])
        opt = torch.optim.Adam(params, lr=LR, weight_decay=0)
        losses, max_grad, max_distance = [], 0.0, 0.0
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, available)["logit"], target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite oracle loss")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, CLIP, error_if_nonfinite=True))
            opt.step()
            distance = project_joint(params, initial, RADIUS)
            losses.append(float(loss.detach()))
            max_grad, max_distance = max(max_grad, grad), max(max_distance, distance)
        with torch.no_grad():
            final_loss = float(F.binary_cross_entropy_with_logits(model(x, available)["logit"], target))
        if frozen_digest(model, params) != frozen_before or state_digest(base) != original:
            raise AssertionError("frozen parent state changed")
        if max_distance > RADIUS or any(not bool(torch.isfinite(p).all()) for p in params):
            raise AssertionError("LoRA radius or finiteness violated")
        adapted = curve(model, signal)
        full = adapted.copy()
        full[:FIRST_ADAPTED] = baseline[:FIRST_ADAPTED]
        held = held_splice(baseline, adapted, support)
        held_mask = np.arange(len(times)) >= FIRST_ADAPTED
        held_mask[np.asarray(support)] = False
        delta = (adapted - baseline)[held_mask]
        if not np.array_equal(full[:FIRST_ADAPTED], baseline[:FIRST_ADAPTED]) or not np.array_equal(held[support], baseline[support]):
            raise AssertionError("prefix or held splice failed")
        path_out = OUTPUT / f"{index:04d}_{shot}_oracle.npz"
        with path_out.open("xb") as handle:
            np.savez_compressed(handle, adapted_full=full, held_splice=held)
        detail = dict(**case, baseline=timeline(times, baseline, case["disruptive"], case["down_s"], THRESHOLD),
                      oracle_full=timeline(times, full, case["disruptive"], case["down_s"], THRESHOLD),
                      oracle_held_splice=timeline(times, held, case["disruptive"], case["down_s"], THRESHOLD),
                      held_max_abs_probability_change=float(np.abs(delta).max()),
                      held_max_probability_increase=float(delta.max()), held_max_probability_decrease=float(delta.min()),
                      collateral=collateral(times, baseline, adapted, case["disruptive"], case["down_s"], THRESHOLD, support),
                      support_bce_initial=losses[0], support_bce_final=final_loss,
                      maximum_gradient_norm_preclip=max_grad, maximum_joint_displacement=max_distance,
                      final_joint_displacement=float(torch.linalg.vector_norm(torch.cat([(p.detach().double()-p0.double()).flatten() for p, p0 in zip(params, initial, strict=True)]))),
                      neutral_vs_sealed_max_abs=parity_gap, raw_sha256=expected_raw,
                      raw_sha_checked_before_after=True, frozen_clone_unchanged=True,
                      oracle_npz=path_out.name, oracle_sha256=sha(path_out), seconds=time.monotonic() - step_start)
        if case["disruptive"]:
            for upper in (125, 3000):
                detail[f"held_band_30_{upper}ms"] = band_metrics(times, baseline, adapted, case["down_s"], support, 30, upper)
        rows.append(detail)
        print(json.dumps(dict(stage="SHOT_DONE", shot=shot, kind=case["kind"], seconds=detail["seconds"],
                              full=detail["oracle_full"]["status"], held=detail["oracle_held_splice"]["status"])), flush=True)
    if state_digest(base) != original or sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(map_path) != protocol["input_sha256"][str(map_path)]:
        raise AssertionError("source parent or map changed")
    write_new(OUTPUT / "results.json", dict(status="COMPLETED_HINDSIGHT_ORACLE_DIAGNOSTIC", shots=rows))
    completed = dict(status="COMPLETED_HINDSIGHT_ORACLE_DIAGNOSTIC", seconds=time.monotonic() - started,
                     source_state_sha256=original, source_checkpoint_sha256=PARENT_SHA256,
                     identical_initial_lora_sha256=initial_digest, raw_files_read=len(rows),
                     raw_reads_outside_selected=0, cpu_threads=torch.get_num_threads(), device="cpu",
                     precommit_sha256=expected_precommit_sha, threshold=THRESHOLD,
                     artifact_sha256={p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    write_new(OUTPUT / "completed.json", completed)
    print(json.dumps(dict(stage="COMPLETED", seconds=completed["seconds"], shots=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select", action="store_true")
    group.add_argument("--run", metavar="PRECOMMIT_SHA256")
    args = parser.parse_args()
    if args.select:
        select()
    else:
        run(args.run)


if __name__ == "__main__":
    main()
