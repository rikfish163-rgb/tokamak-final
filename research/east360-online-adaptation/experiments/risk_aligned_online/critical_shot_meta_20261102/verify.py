"""CPU-only consistency audit of completed critical-shot records, not neural replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "probe_outputs/critical_shot_meta360_v1"
REFERENCE = ROOT / "probe_outputs/lora_stream_v1"
TRAINING = ROOT / "probe_outputs/critical_shot_heads20_v1"
TRAINING_SOURCE = ROOT / "probe_outputs/difference_feedback_probe20_v1"
# The worktree's source checkpoint is an authorized read-only shared symlink.
SHARED_SOURCE = Path("/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt")
EXPECTED_COLD = ROOT / "probe_outputs/stratified_cold20_context360_v1/models/cold.pt"
PLAN = "CRITICAL_SHOT_META_PLAN_20261102.md"
REFERENCE_SHA = "76ebb4035873ba3c6b85da9d89b995233d2ce36222237bf1733058d4c9482f4c"
TRAINING_SOURCE_SHA = "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
COLD_SHA = "a06377a218a0f06def1eb3652cc091b5c514653681bffcf5dbfda7906f9db344"
THRESHOLD = 0.7258946895599365
ARMS = ("cold", "original_difference", "forecast", "point", "operational", "reverse", "hold")
LEARNERS = ARMS[1:]
HEADS = ("forecast", "point", "operational")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float)), name)
    require(math.isfinite(value), name)
    return float(value)


def check_model_files(protocol, reference):
    for key, expected_path, hash_key, expected_hash in (
        ("source_checkpoint", SHARED_SOURCE, "source_sha256", SOURCE_SHA),
        ("cold_state", EXPECTED_COLD, "cold_state_sha256", COLD_SHA),
    ):
        path = Path(protocol[key]).resolve(strict=True)
        reference_path = Path(reference[key]).resolve(strict=True)
        require(path == reference_path == expected_path, "unexpected model path: " + key)
        require(protocol[hash_key] == reference[hash_key] == expected_hash == sha_file(path), "model hash: " + key)


def classify_curve(times, scores, disruptive, down, threshold):
    """Reconstruct the first preterminal alarm without the runner's classifier."""
    times, scores = np.asarray(times), np.asarray(scores)
    require(times.ndim == 1 and len(times) > 0 and scores.shape == times.shape, "curve shape")
    require(np.isfinite(times).all() and np.isfinite(scores).all(), "nonfinite curve")
    require(np.all(np.diff(times) > 0) and np.all((0 <= scores) & (scores <= 1)), "invalid curve")
    require(type(disruptive) is bool and 0 < finite(threshold, "threshold") < 1, "outcome/threshold")
    valid = (finite(down, "DownTime") - times) * 1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    crossing = np.flatnonzero(valid & (scores >= threshold))
    lead = float(round((down - times[crossing[0]]) * 1000, 9)) if disruptive and len(crossing) else None
    return dict(alarm=bool(len(crossing)), lead_ms=lead, timely=bool(lead is not None and lead > 30))


def recount(rows):
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    tp = sum(r["timely"] for r in rows)
    fp = sum(not r["disruptive"] and r["alarm"] for r in rows)
    tpr, fpr = (tp / nd if nd else None), (fp / ns if ns else None)
    leads = [r["lead_ms"] for r in rows if r["timely"]]
    return dict(n=len(rows), n_disrupt=nd, n_safe=ns, tp=tp, fp=fp, tpr=tpr, fpr=fpr,
                J=tpr - fpr if nd and ns else None,
                early_over3000=sum(lead > 3000 for lead in leads),
                median_timely_lead_ms=float(np.median(leads)) if leads else None)


def paired_flips(reference, candidate):
    require(len(reference) == len(candidate), "flip length")
    result = {k: [] for k in ("rescued_D", "lost_D", "removed_FP", "added_FP")}
    for before, after in zip(reference, candidate, strict=True):
        require((before["shot"], before["disruptive"]) == (after["shot"], after["disruptive"]), "flip identity")
        if before["disruptive"]:
            if after["timely"] and not before["timely"]:
                result["rescued_D"].append(before["shot"])
            if before["timely"] and not after["timely"]:
                result["lost_D"].append(before["shot"])
        else:
            if before["alarm"] and not after["alarm"]:
                result["removed_FP"].append(before["shot"])
            if after["alarm"] and not before["alarm"]:
                result["added_FP"].append(before["shot"])
    return result


def posthoc_caps(maxima, disruptive):
    """Development-set operating points; these thresholds are never deployed."""
    values, flags = np.asarray(maxima, dtype=np.float64), np.asarray(disruptive, dtype=bool)
    require(values.shape == flags.shape and flags.any() and (~flags).any(), "posthoc cohort")
    require(np.all(np.isfinite(values) | np.isneginf(values)), "posthoc maxima")
    thresholds = np.unique(np.r_[values[np.isfinite(values)], np.nextafter(1.0, np.inf)])
    result = []
    for cap in (0.10, 0.15, 0.20):
        choices = []
        for threshold in thresholds:
            hits = values >= threshold
            tpr, fpr = float(hits[flags].mean()), float(hits[~flags].mean())
            if fpr <= cap:
                choices.append((tpr, -fpr, float(threshold)))
        tpr, neg_fpr, threshold = max(choices)
        result.append(dict(fpr_cap=cap, tpr=tpr, fpr=-neg_fpr, threshold=threshold,
                           scope="POSTHOC_DEVELOPMENT_NOT_DEPLOYED"))
    return result


def check_snapshot(root, protocol, *, training=False):
    declared = protocol["code_sha256"]
    module = "experiments/risk_aligned_online/critical_shot_meta_20261102/"
    required = {PLAN, module + "stream.py", module + "replay.py", module + "objectives.py", module + "prepare.py"}
    if not training:
        required.add(module + "run.py")
    require(isinstance(declared, dict) and required <= set(declared), "snapshot entries")
    require(declared[PLAN] == protocol["plan_sha256"], "snapshot plan hash")
    current = {}
    archive_path = root / "code_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) and set(names) == set(declared), "archive names")
        require(sum(entry.file_size for entry in entries) <= 512 * 1024 * 1024, "archive size")
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                    and not entry.is_dir() and entry.file_size <= 64 * 1024 * 1024, "archive path")
            require(name == PLAN or (path.parts[:2] == ("experiments", "risk_aligned_online")
                    and path.suffix == ".py"), "unexpected non-code snapshot entry")
            expected = declared[name]
            require(isinstance(expected, str) and HEX.fullmatch(expected), "archive declared hash")
            require(hashlib.sha256(archive.read(entry)).hexdigest() == expected, "archive entry hash: " + name)
            current_path = ROOT / name
            if current_path.is_file():
                resolved = current_path.resolve()
                require(resolved.is_relative_to(ROOT.resolve()) and resolved.suffix == path.suffix,
                        "current code path is not a workspace code file")
                current[name] = sha_file(resolved)
            else:
                current[name] = None
    drift = {name: {"snapshot": declared[name], "actual_current": actual}
             for name, actual in current.items() if actual != declared[name]}
    return dict(archive_bytes_sha256=sha_file(archive_path), archive_entries_verified=len(declared),
                actual_current_sha256=current, current_drift=drift)


def read_outcome(shot):
    """Open only two metadata datasets; never read or hash HDF5 signal data."""
    import h5py

    path = ROOT / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    with h5py.File(path, "r") as handle:
        raw_flag = np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)
        raw_down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    require(len(raw_flag) == 1 and raw_flag[0] in (0, 1), "invalid HDF outcome")
    flag = bool(raw_flag[0])
    down = float(raw_down[0]) if flag and len(raw_down) else None
    require(not flag or (down is not None and math.isfinite(down)), "invalid HDF DownTime")
    return flag, down


def check_trace(info, arrays, length, warmup):
    require(info.get("reset_each_shot") is True and info.get("temporary_parameters") == 256, "trace protocol")
    updates, actions = info["updates"], info["actions"]
    require(set(updates) == set(ARMS) and set(actions) == set(ARMS), "trace arms")
    require(set(arrays) == {a + suffix for a in ARMS for suffix in ("_before", "_after", "_gradient")}, "trace arrays")
    totals = {}
    for arm in ARMS:
        records = updates[arm]
        ends = [] if warmup or arm == "cold" else list(range(159, length, 128))
        require([r["maturity"] for r in records] == ends, "clock: " + arm)
        before, after, gradient = (np.asarray(arrays[arm + suffix]) for suffix in ("_before", "_after", "_gradient"))
        require(before.shape == after.shape == (len(ends), 4, 64) and gradient.shape == (len(ends), 256), "state shape")
        require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in (before, after, gradient)), "state finite/dtype")
        if ends:
            require(not np.any(before[0]) and np.array_equal(after[:-1], before[1:]), "state reset/continuity")
        totals[arm] = dict(events=len(ends), accepted=0, nonzero_gradients=0, observed_scalars=0)
        for i, (record, end) in enumerate(zip(records, ends, strict=True)):
            require(record["origin"] == end - 25 and record["first_affected"] == end + 1, "causal clock")
            for key in ("forecast_sha256", "mask_sha256", "target_sha256"):
                require(isinstance(record.get(key), str) and HEX.fullmatch(record[key]), "invalid " + key)
            observed, accepted, computed = record["observed"], record["accepted"], record["gradient_computed"]
            require(type(observed) is int and 0 <= observed <= 550 and type(accepted) is bool
                    and type(computed) is bool, "receipt types")
            grad_norm, step_norm = finite(record["gradient_norm"], "gradient norm"), finite(record["step_norm"], "step norm")
            actual_grad = float(np.linalg.norm(gradient[i].astype(np.float64)))
            actual_step = float(np.linalg.norm((after[i] - before[i]).astype(np.float64)))
            require(grad_norm >= 0 and math.isclose(grad_norm, actual_grad, rel_tol=1e-9, abs_tol=1e-14), "gradient norm")
            require(0 <= step_norm <= 0.100001 and math.isclose(step_norm, actual_step, rel_tol=1e-9, abs_tol=1e-14), "step norm")
            if observed:
                old = finite(record["loss_before"], "loss before")
                proposed = finite(record["loss_proposal"], "loss proposal")
                require(computed and accepted == (proposed < old)
                        and record["reason"] == ("improved" if accepted else "not_improved"), "acceptance")
            else:
                require(not computed and not accepted and record["reason"] == "no_target" and grad_norm == 0
                        and step_norm == 0 and not np.any(gradient[i])
                        and "loss_before" not in record and "loss_proposal" not in record, "no-target sentinel")
            if not accepted:
                require(np.array_equal(before[i], after[i]), "rejected state changed")
            totals[arm]["accepted"] += int(accepted)
            totals[arm]["nonzero_gradients"] += int(grad_norm > 0)
            totals[arm]["observed_scalars"] += observed
    for group in zip(*(updates[a] for a in LEARNERS), strict=True):
        require(len({r["mask_sha256"] for r in group}) == 1 and len({r["observed"] for r in group}) == 1, "learner mask")
        require(len({r["target_sha256"] for r in group[:4]}) == 1, "real target hash")
    for arm, stats in actions.items():
        endpoints, capped = stats["endpoints"], stats["capped_endpoints"]
        rms = finite(stats["max_rms"], "action RMS")
        require(type(endpoints) is int and endpoints == (0 if warmup else length)
                and type(capped) is int and 0 <= capped <= endpoints and 0 <= rms <= 0.250001
                and (endpoints > 0 or rms == 0), "action summary")
    return totals, actions


def check_training(protocol, training_done, histories, cohort, warmup):
    require([c["shot"] for c in cohort] == protocol["ids"] and len(cohort) == 20, "training cohort")
    require(sum(c["disruptive"] for c in cohort) == 8, "training outcome balance")
    known = {c["shot"]: c for c in cohort}
    for shot, c in known.items():
        times, flag, down = warmup[shot]
        require(c["disruptive"] is flag and c["down"] == down and c["points"] == len(times), "training HDF metadata")
    require(len(histories) == 128, "training step count")
    support = {a: 0 for a in HEADS}
    for step, row in enumerate(histories, 1):
        ids = row["sample_shots"]
        require(row["step"] == step and len(ids) == 2 and all(s in known for s in ids)
                and known[ids[0]]["disruptive"] is True and known[ids[1]]["disruptive"] is False, "training sample")
        require(set(row["episodes"]) == set(HEADS) and set(row["gradient_norms"]) == set(HEADS), "training heads")
        for arm in HEADS:
            require(finite(row["gradient_norms"][arm], "head gradient") >= 0, "head gradient")
            episodes = row["episodes"][arm]
            require(len(episodes) == 2 and [e["shot"] for e in episodes] == ids, "episode shots")
            for event in episodes:
                times, flag, down = warmup[event["shot"]]
                q = event["query"]
                require(type(q) is int and 0 <= q < len(times), "training query")
                lead = (down - times[q]) * 1000 if flag else None
                require(not flag or round(lead, 9) > 30, "training timely query")
                supported = q >= 160
                maturity = 159 + ((q - 160) // 128) * 128 if supported else None
                require(event["supported"] is supported and event["maturity"] == maturity, "training support clock")
                expected_label = int(flag and (arm != "point" or lead <= 125 + 1e-6))
                require(event["label"] == expected_label, "training task label")
                for key in ("risk_before", "risk_after", "signal", "objective"):
                    finite(event[key], "training " + key)
                if not supported:
                    require(event["risk_before"] == event["risk_after"], "unsupported risk changed")
                support[arm] += int(supported)
    require(training_done["support_counts"] == support, "training support totals")
    return support


def verify(output):
    output = Path(output)
    done = read_json(output / "completed.json")
    require(done.get("status") == "COMPLETED", "completed 360 run required before reading records")
    protocol = read_json(output / "protocol.json")
    fixed = dict(scope="DEVELOPMENT_COMPLETE360_CRITICAL_ENDPOINT_NEURAL_META", arms=list(ARMS),
                 threshold=THRESHOLD, first_update=159, first_possible_effect=160, update_interval=128,
                 parameter_step=0.1, hidden_rms_cap=0.25, temporary_parameters=256, reset_each_shot=True,
                 head_fitted_ended20_only=True, suffix_head_updates=0, suffix_inter_supervision=0,
                 warmup_cost_included=True, outcome_authority="HDF5", lead_strictly_greater_ms=30,
                 no_outcome_input_to_simulate=True, independent_future_confirmation=False, real_time_latency_verified=False)
    for key, expected in fixed.items():
        require(protocol.get(key) == expected, "protocol: " + key)
    require(Path(protocol["reference"]).resolve() == REFERENCE.resolve()
            and Path(protocol["training"]).resolve() == TRAINING.resolve(), "reference/training paths")
    require(protocol["reference_completed_sha256"] == REFERENCE_SHA == sha_file(REFERENCE / "completed.json"), "reference completion hash")
    ref = read_json(REFERENCE / "protocol.json")
    ref_check = read_json(REFERENCE / "verification_records.json")
    require(ref_check["status"] == "PASS" and ref_check["completed_sha256"] == REFERENCE_SHA
            and ref_check["protocol_sha256"] == sha_file(REFERENCE / "protocol.json"), "reference verification binding")
    order = protocol["order"]
    ids = [entry["shot"] for entry in order]
    require(len(ids) == len(set(ids)) == 360 and all(type(s) is int and s > 0 for s in ids)
            and order == ref["order"], "360 distinct ordered shots")
    for key in ("source_sha256", "cold_state_sha256", "input_map_sha256", "threshold"):
        require(protocol[key] == ref[key], "reference protocol: " + key)
    check_model_files(protocol, ref)
    calibration = Path(protocol["calibration_reference"])
    require(calibration.resolve() == Path(ref["calibration_reference"]).resolve(), "calibration path")
    require(sha_file(calibration / "input_map.json") == protocol["input_map_sha256"]
            and read_json(calibration / "input_map.json")["calibration_ids"] == ids[:20], "input map")
    require(sha_file(calibration / "input_hashes.json") == ref_check["input_hashes_sha256"], "stored input hash list")
    require(sha_file(ROOT / PLAN) == protocol["plan_sha256"], "current plan hash")
    training_done = read_json(TRAINING / "completed.json")
    training_protocol = read_json(TRAINING / "protocol.json")
    require(training_done["status"] == "COMPLETED" and training_done["head_optimizer_steps"] == 128
            and training_done["number_heads"] == 3 and training_done["source_tensors_unchanged"] is True
            and training_done["outcome_reads"] == 20 and training_done["suffix_shots_read"] == 0, "training completion")
    require(sha_file(TRAINING / "completed.json") == protocol["training_completed_sha256"]
            and sha_file(TRAINING / "heads.json") == training_done["heads_sha256"] == protocol["heads_sha256"]
            and sha_file(TRAINING / "history.json") == training_done["history_sha256"], "training hashes")
    require(sha_file(TRAINING_SOURCE / "completed.json") == training_protocol["source_completed_sha256"] == TRAINING_SOURCE_SHA, "training source hash")
    train_fixed = dict(scope="ENDED20_COMPLETE_SHOT_CRITICAL_REPLAY_HEAD_FIT", ids=ids[:20], heads=list(HEADS),
                       forecast_parameters_per_head=575, optimizer="Adam", learning_rate=0.001, steps=128,
                       gradient_clip=1.0, seed=131, temporal_gradient_truncation_updates=1,
                       first_update=159, update_interval=128, warmup_cost_retained=True,
                       independent_confirmation=False, source_sha256=SOURCE_SHA, cold_state_sha256=COLD_SHA,
                       input_map_sha256=protocol["input_map_sha256"], plan_sha256=protocol["plan_sha256"])
    for key, expected in train_fixed.items():
        require(training_protocol.get(key) == expected, "training protocol: " + key)
    require(training_done["ids"] == ids[:20] and training_done["plan_sha256"] == protocol["plan_sha256"], "training IDs/plan")
    heads = read_json(TRAINING / "heads.json")
    require(set(heads) == set(HEADS), "training head names")
    for head in heads.values():
        table = np.asarray(head, dtype=np.float64)
        require(table.shape == (25, 23) and np.isfinite(table).all(), "training head shape/finite")
    snapshots = dict(run=check_snapshot(output, protocol), training=check_snapshot(TRAINING, training_protocol, training=True))
    require(sha_file(output / "per_shot.json") == done["per_shot_sha256"], "per-shot hash")
    saved = read_json(output / "per_shot.json")
    require(set(saved) == set(ARMS) and all(len(saved[a]) == 360 for a in ARMS), "per-shot rows")
    rows, maxima, warmup, shot_hashes = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, []
    totals = {a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0) for a in ARMS}
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, shot in enumerate(ids, 1):
        stem = f"{index:04d}_{shot}"
        base, refbase = output / "shots" / stem, REFERENCE / "shots" / stem
        curve, state, info_path = (Path(str(base) + suffix) for suffix in (".npz", "_states.npz", "_updates.json"))
        seal_path, refseal_path = Path(str(base) + "_seal.json"), Path(str(refbase) + "_seal.json")
        seal, refseal = read_json(seal_path), read_json(refseal_path)
        require(seal["shot"] == shot and seal["index"] == index and seal["warmup_curves_reused"] is (index <= 20)
                and seal["warmup_outcome_already_known_for_head_fit"] is (index <= 20)
                and seal["current_outcome_used_in_prediction"] is False, "seal identity/flags")
        for path, key in ((curve, "sha256"), (state, "states_sha256"), (info_path, "updates_sha256")):
            require(sha_file(path) == seal[key], "seal hash: " + str(path))
        refcurve = Path(str(refbase) + ".npz")
        require(refseal["shot"] == shot and refseal["index"] == index and refseal["outcome_available"] is False
                and sha_file(refcurve) == refseal["sha256"], "reference seal")
        shot_hashes.append(dict(shot=shot, index=index, curve_sha256=seal["sha256"],
                                states_sha256=seal["states_sha256"], updates_sha256=seal["updates_sha256"],
                                seal_sha256_at_verification=sha_file(seal_path),
                                reference_curve_sha256=refseal["sha256"],
                                reference_seal_sha256_at_verification=sha_file(refseal_path)))
        with np.load(curve, allow_pickle=False) as archive:
            require(set(archive.files) == {"times", *ARMS}, "curve keys")
            data = {name: archive[name] for name in archive.files}
        with np.load(refcurve, allow_pickle=False) as archive:
            ref_times, ref_cold = archive["times"], archive["cold"]
        times = data["times"]
        require(times.dtype == ref_times.dtype and data["cold"].dtype == ref_cold.dtype
                and np.array_equal(times, ref_times) and np.array_equal(data["cold"], ref_cold), "reference cold/times")
        flag, down = read_outcome(shot)
        if index <= 20:
            warmup[shot] = (times, flag, down)
        with np.load(state, allow_pickle=False) as archive:
            state_arrays = {name: archive[name] for name in archive.files}
        shot_totals, shot_actions = check_trace(read_json(info_path), state_arrays, len(times), index <= 20)
        eligible = np.round((down - times) * 1000, 9) > 30 if flag else np.ones(len(times), bool)
        for arm in ARMS:
            values = data[arm]
            require(values.dtype == data["cold"].dtype and np.array_equal(values[:160], data["cold"][:160]), "first 160")
            if index <= 20:
                require(np.array_equal(values, ref_cold), "warmup curve copy")
            row = dict(shot=shot, disruptive=flag, **classify_curve(times, values, flag, down, THRESHOLD))
            require(row == saved[arm][index - 1], "per-shot scoring: " + stem + " " + arm)
            rows[arm].append(row)
            maxima[arm].append(float(values[eligible].max()) if eligible.any() else -np.inf)
            for key in totals[arm]:
                totals[arm][key] += shot_totals[arm][key]
            for key in ("endpoints", "capped_endpoints"):
                actions[arm][key] += shot_actions[arm][key]
            actions[arm]["max_rms"] = max(actions[arm]["max_rms"], shot_actions[arm]["max_rms"])
    support = check_training(training_protocol, training_done, read_json(TRAINING / "history.json"),
                             read_json(TRAINING / "training_cohort.json"), warmup)
    metrics = {a: {name: recount(part) for name, part in
                   (("full360", rows[a]), ("warmup20", rows[a][:20]), ("suffix340", rows[a][20:]))} for a in ARMS}
    flips = {a: paired_flips(rows["cold"], rows[a]) for a in LEARNERS}
    require(metrics == done["metrics"] and flips == done["flips"], "metrics or paired flips")
    require(totals == done["update_statistics"] and actions == done["action_statistics"], "update/action totals")
    base, candidate = metrics["cold"]["full360"], metrics["operational"]["full360"]
    require(all(candidate[k] is not None for k in ("fpr", "J")) and base["J"] is not None, "full cohort classes")
    stage = candidate["fpr"] <= 0.20 and candidate["tp"] >= base["tp"] and candidate["fp"] <= base["fp"] and candidate["J"] > base["J"]
    separation = all(candidate["J"] > metrics[a]["full360"]["J"] for a in ("reverse", "hold"))
    comparisons = {a: candidate["J"] > metrics[a]["full360"]["J"] for a in ("original_difference", "forecast", "point")}
    require(done["operational_gate"] is stage and done["control_separation_gate"] is separation
            and done["task_comparisons"] == comparisons and done["stage_success_pending_verification"] is (stage and separation), "gates")
    require(done["fresh_curves"] == 340 * len(ARMS) and done["reused_warmup_curves"] == 20 * len(ARMS)
            and done["source_tensors_unchanged"] is True and done["heads_unchanged"] is True
            and done["method_promoted"] is False and done["scientific_goal_achieved"] is False
            and done["independent_future_confirmation"] is False, "completion flags")
    flags = [r["disruptive"] for r in rows["cold"]]
    return dict(status="PASS", scope="RECORDED_DATA_CONSISTENCY_ONLY_NOT_INDEPENDENT_NEURAL_REPLAY",
                cases=360, curves=360 * len(ARMS), metadata_outcomes_read=360, raw_signal_datasets_read=0,
                raw_hdf5_file_hashes_recomputed=False, threshold=THRESHOLD, metrics=metrics, flips=flips,
                update_statistics=totals, action_statistics=actions, training_supported_episodes=support,
                operational_gate=stage, control_separation_gate=separation, task_comparisons=comparisons,
                posthoc_workpoints={a: posthoc_caps(maxima[a], flags) for a in ARMS},
                hashes=dict(completed_sha256=sha_file(output / "completed.json"),
                            protocol_sha256_at_verification=sha_file(output / "protocol.json"),
                            per_shot_sha256=done["per_shot_sha256"], source_sha256=SOURCE_SHA,
                            cold_state_sha256=COLD_SHA, reference_completed_sha256=REFERENCE_SHA,
                            training_completed_sha256=protocol["training_completed_sha256"],
                            heads_sha256=protocol["heads_sha256"], history_sha256=training_done["history_sha256"],
                            input_map_sha256=protocol["input_map_sha256"], plan_sha256=protocol["plan_sha256"],
                            verifier_sha256=sha_file(Path(__file__)), snapshots=snapshots, shot_files=shot_hashes),
                limitations=["No neural replay, signal reconstruction, independently recomputed masks or actions.",
                             "Training critical maxima, neural derivatives and seal-before-outcome timing are recorded claims, not independently reproduced here.",
                             "Current run completion does not anchor protocol, archive, or per-shot seal files; hashes above are consistency checks, not authenticated provenance.",
                             "Raw HDF5 hashes were not recomputed because raw signal access was forbidden.",
                             "Posthoc operating points use the development cohort and are NOT_DEPLOYED; the fixed source threshold is unchanged.",
                             "No independent future validation, real-time latency or physical contribution is established."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report_path = args.output / "verification_records.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = verify(args.output)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(json.dumps({key: report[key] for key in ("status", "cases", "curves")}))


if __name__ == "__main__":
    main()
