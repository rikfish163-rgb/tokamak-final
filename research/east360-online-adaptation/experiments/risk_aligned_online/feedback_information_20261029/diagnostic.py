"""Read-only, CPU-only analysis of the sealed H25 factorial feedback."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "probe_outputs/local_factorial_east360_v1"
OUTPUT = ROOT / "probe_outputs/feedback_information_v1"
PROTOCOL = Path(__file__).with_name("PROTOCOL.md")
REVIEW = ROOT / "FACTORIAL_TASK_INTERFACE_REVIEW.md"
EXPECTED_COMPLETE = "8898224f755c1129411de2942789f32c0a9e979edc7f418f20b92593a23d56e2"
EXPECTED_AUDIT = "7cf661f42b89d2454a757521bd891ca0364af5362b97197cd159ed87e4e02fca"
EXPECTED_REVIEW = "54028b334ba70070a326d36f7c0ba7ae80ed93d7a01978801ce43c1b175b5164"
EXPECTED_PROTOCOL = "c234e6209441829ed3a2eb7b33ffd7b3b2f22bc1b6bb42cb5e61f5918242e050"
HISTORIES = ("true", "reverse", "hold")
FACTORIAL_ARMS = tuple(f"{h}_{t}" for h in HISTORIES for t in HISTORIES)
ADAPTIVE_ARMS = ("prior_true", *FACTORIAL_ARMS)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def choose_disruptive_clock(clocks, leads, first_alarm):
    if len(clocks) != len(leads):
        raise ValueError("clock/lead lengths differ")
    allowed = []
    for clock, lead in zip(clocks, leads):
        lead = finite_number(lead, "lead")
        if np.round(lead, 9) > 30.0 and lead <= 125.0:
            allowed.append(clock)
    if not allowed:
        return None, "no_time_window"
    before_alarm = [clock for clock in allowed if first_alarm >= clock]
    if not before_alarm:
        return None, "cold_already_alarmed"
    return before_alarm[-1], "selected"


def reconstruct_masks(available, active, clocks):
    available, active = np.asarray(available), np.asarray(active)
    if (
        available.ndim != 2
        or available.shape[1] != 23
        or available.dtype != np.bool_
        or active.shape != (23,)
        or active.dtype != np.bool_
    ):
        raise ValueError("expected boolean availability N x 23 and active 23")
    masks = np.zeros((len(clocks), 2, 23), dtype=bool)
    for i, c in enumerate(clocks):
        if not isinstance(c, (int, np.integer)) or c < 256 or c > len(available):
            raise ValueError("feedback clock outside arrived prefix")
        masks[i, 0] = available[c - 51 : c - 25].all(axis=0) & active
        masks[i, 1] = available[c - 26 : c].all(axis=0) & active
    return masks


def validate_feedback(record, mask):
    mask = np.asarray(mask)
    if mask.shape != (2, 23) or mask.dtype != np.bool_:
        raise ValueError("complete two-segment 23-channel mask required")
    arms = record.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ADAPTIVE_ARMS):
        raise ValueError("missing or unexpected feedback arm")
    expected = int(mask.sum()) * 25
    for arm in ADAPTIVE_ARMS:
        item = arms[arm]
        if not isinstance(item, dict) or "loss_before" not in item:
            raise ValueError(f"{arm}: missing feedback loss")
        observed = item.get("observed")
        if type(observed) is not int or observed != expected:
            raise ValueError(f"{arm}: mask and observed count differ")
        norm = finite_number(item.get("accepted_norm"), f"{arm} accepted norm")
        reason = item.get("reason")
        if norm < 0 or not isinstance(reason, str) or not reason:
            raise ValueError(f"{arm}: invalid acceptance receipt")
        if expected:
            if finite_number(item["loss_before"], f"{arm} loss_before") < 0:
                raise ValueError(f"{arm}: negative loss")
            if (norm > 0) != (reason == "accepted"):
                raise ValueError(f"{arm}: acceptance reason disagrees")
            if norm > 0:
                if finite_number(item.get("factor"), f"{arm} factor") <= 0:
                    raise ValueError(f"{arm}: invalid accepted factor")
        elif item["loss_before"] is not None or norm != 0 or reason != "no_observed_target":
            raise ValueError(f"{arm}: unobserved feedback must abstain")


def greedy_pairs(disruptive, safe):
    used, pairs, unmatched = set(), [], []
    for d in sorted(disruptive, key=lambda p: p["shot"]):
        choices = []
        for s in safe:
            if s["shot"] in used or not np.array_equal(d["mask"], s["mask"]):
                continue
            probability_gap = abs(s["probability"] - d["probability"])
            clock_gap = abs(s["clock"] - d["clock"])
            if probability_gap <= 0.05 and clock_gap <= 128:
                choices.append(((probability_gap, clock_gap, s["shot"], s["clock"]), s))
        if not choices:
            unmatched.append(d["shot"])
            continue
        _, chosen = min(choices, key=lambda choice: choice[0])
        used.add(chosen["shot"])
        pairs.append((d, chosen))
    return pairs, unmatched


def bootstrap_summary(q, hold):
    q, hold = np.asarray(q, dtype=np.float64), np.asarray(hold, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or hold.shape != q.shape:
        raise ValueError("paired contrasts must have shape n x 3")
    if not np.isfinite(q).all() or not np.isfinite(hold).all():
        raise ValueError("paired contrasts must be finite")
    n = len(q)
    names = ("D_true", "abs_D_true_minus_abs_D_reverse", "abs_D_true_minus_abs_D_hold")
    if not n:
        empty = dict(estimate=None, ci95=None)
        return dict(
            q={h: dict(empty) for h in HISTORIES},
            hold_target={h: dict(empty) for h in HISTORIES},
            screen_contrasts={k: dict(empty) for k in names},
            screen_status="insufficient_evidence",
        )
    indices = np.random.default_rng(131).integers(0, n, size=(2000, n))
    draws, hold_draws = q[indices].mean(axis=1), hold[indices].mean(axis=1)
    means, hold_means = q.mean(axis=0), hold.mean(axis=0)

    def report(estimate, samples):
        return dict(
            estimate=float(estimate),
            ci95=np.percentile(samples, [2.5, 97.5], method="linear").tolist(),
        )

    contrast_estimates = (means[0], abs(means[0]) - abs(means[1]), abs(means[0]) - abs(means[2]))
    contrast_draws = (
        draws[:, 0],
        np.abs(draws[:, 0]) - np.abs(draws[:, 1]),
        np.abs(draws[:, 0]) - np.abs(draws[:, 2]),
    )
    contrast = {k: report(v, sample) for k, v, sample in zip(names, contrast_estimates, contrast_draws)}
    ci = contrast["D_true"]["ci95"]
    supported = (
        (ci[0] > 0 or ci[1] < 0)
        and contrast[names[1]]["ci95"][0] > 0
        and contrast[names[2]]["ci95"][0] > 0
    )
    status = "insufficient_evidence" if n < 20 else ("supported" if supported else "stop_scalar_route")
    return dict(
        q={h: report(means[i], draws[:, i]) for i, h in enumerate(HISTORIES)},
        hold_target={h: report(hold_means[i], hold_draws[:, i]) for i, h in enumerate(HISTORIES)},
        screen_contrasts=contrast,
        screen_status=status,
    )


def verify_inputs():
    require(sha(REVIEW) == EXPECTED_REVIEW, "review document changed")
    require(sha(PROTOCOL) == EXPECTED_PROTOCOL, "fixed diagnostic protocol changed")
    require(sha(SOURCE / "complete.json") == EXPECTED_COMPLETE, "completion hash differs")
    require(sha(SOURCE / "audit.json") == EXPECTED_AUDIT, "audit hash differs")
    done, audit = load(SOURCE / "complete.json"), load(SOURCE / "audit.json")
    require(done.get("complete") is True and audit.get("complete") is True, "run/audit incomplete")
    require(audit.get("complete_sha256") == EXPECTED_COMPLETE, "audit does not bind completion")
    require(audit.get("all_first_passing_verified") is True, "acceptance audit incomplete")
    require(audit.get("scope") == "all360 sealed first-alarm decisions and raw masks, factorial acceptance receipts, and changed-shot neural replay", "audit scope differs")
    audit_code = ROOT / "experiments/risk_aligned_online/local_factorial_20261029/audit.py"
    require(sha(audit_code) == audit.get("code_sha256"), "audit code changed")
    artifacts = {}
    for name in ("protocol", "per_shot", "receipts", "auxiliary_summary", "auxiliary_per_shot"):
        digest = sha(SOURCE / f"{name}.json")
        require(digest == done.get(f"{name}_sha256"), f"changed sealed artifact: {name}")
        artifacts[name] = digest
    protocol = load(SOURCE / "protocol.json")
    require(protocol.get("arms") == ["cold", *ADAPTIVE_ARMS], "eleven arms differ")
    expected_modes = {"prior_true": ["prior", "true"]}
    expected_modes.update({f"{h}_{t}": [h, t] for h in HISTORIES for t in HISTORIES})
    require(protocol.get("modes") == expected_modes, "factorial modes differ")
    order = protocol.get("order")
    require(isinstance(order, list) and len(order) == len(set(order)) == 360 and protocol.get("warmup") == 20, "invalid 360-shot order or warmup")
    require(protocol.get("reset_each_feedback") is True and protocol.get("alarm_updates") == 0, "feedback contract differs")
    code = protocol["code_sha256"]
    with zipfile.ZipFile(SOURCE / "code_snapshot.zip") as archive:
        for name, digest in code.items():
            require(sha(ROOT / name) == digest and hashlib.sha256(archive.read(name)).hexdigest() == digest, f"changed production code: {name}")
        plan = "LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md"
        require(sha(ROOT / plan) == protocol["plan_sha256"] and hashlib.sha256(archive.read(plan)).hexdigest() == protocol["plan_sha256"], "factorial plan changed")
    reference = ROOT / protocol["reference"]
    for name, digest in protocol["reference_hashes"].items():
        require(sha(reference / name) == digest, f"changed cold reference: {name}")
    readout = Path(protocol["readout"])
    require(sha(readout / "cold_readouts.npz") == protocol["readout_sha256"], "readout prior changed")
    require(sha(readout / "audit_with_fold_gap.json") == protocol["readout_audit_sha256"], "readout audit changed")
    with np.load(readout / "cold_readouts.npz", allow_pickle=False) as prior:
        active = np.array([f"last_token_{c}_coef" in prior.files for c in range(23)], dtype=bool)
    require(active.any() and not active[2] and not active[10], "unexpected prior active channels")
    mapping = load(reference / "input_map.json")
    gain, offset = np.asarray(mapping["gain"], dtype=np.float64), np.asarray(mapping["offset"], dtype=np.float64)
    require(gain.shape == offset.shape == (23,) and np.isfinite(gain).all() and np.isfinite(offset).all() and (gain > 0).all(), "invalid fixed input mapping")
    require(mapping["calibration_ids"] == order[:20] and mapping["availability_changes"] == 0, "mapping changed cold20 or availability")
    hashes = load(reference / "input_hashes.json")
    receipts, decisions = load(SOURCE / "receipts.json"), load(SOURCE / "per_shot.json")
    require(len(receipts) == 360 and set(decisions) == {"cold", *ADAPTIVE_ARMS}, "receipt/decision cardinality differs")
    require(all(len(decisions[a]) == 360 and [row["shot"] for row in decisions[a]] == order for a in decisions), "arm decision order differs")
    provenance = dict(
        review_sha256=EXPECTED_REVIEW,
        protocol_sha256=sha(PROTOCOL),
        diagnostic_sha256=sha(Path(__file__)),
        test_sha256=sha(ROOT / "tests/test_feedback_information.py"),
        factorial_complete_sha256=EXPECTED_COMPLETE,
        factorial_audit_sha256=EXPECTED_AUDIT,
        factorial_artifact_sha256=artifacts,
        factorial_code_snapshot_sha256=sha(SOURCE / "code_snapshot.zip"),
        reference_hashes=protocol["reference_hashes"],
        prior_sha256=protocol["readout_sha256"],
        prior_audit_sha256=protocol["readout_audit_sha256"],
        shots=[],
    )
    return protocol, active, gain, offset, hashes, receipts, decisions, provenance


def read_arrived_masks(path, times, clocks, active, gain, offset):
    limit = max(clocks, default=0)
    available = np.zeros((limit, 23), dtype=bool)
    with h5py.File(path, "r") as handle:
        flag = bool(np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)[0])
        down = float(np.asarray(handle["meta/DownTime"][()]).reshape(-1)[0]) if flag else None
        require(not flag or math.isfinite(down), "invalid disruption time")
        ip = handle["data/ip"]
        rate = finite_number(np.asarray(ip.attrs["SampleRate"]).reshape(-1)[0], "sample rate")
        start = finite_number(np.asarray(ip.attrs["StartTime"]).reshape(-1)[0], "start time")
        require(math.isclose(rate, 1000.0, rel_tol=0, abs_tol=0.5), "non-nominal sample rate")
        require(np.array_equal(times, start + np.arange(len(times)) / rate), "sealed timestamps differ from raw grid")
        for j, channel in enumerate(io23.RAW23):
            dataset = handle["data"][channel]
            require(len(dataset) == len(times), "raw channel length differs")
            if not active[j] or not limit:
                continue
            values = np.asarray(dataset[:limit], dtype=np.float32)
            valid = np.isfinite(values)
            if j == 1:
                valid &= ~np.isclose(values, -10.0, rtol=0.0, atol=0.01)
            mapped = values[valid].astype(np.float64) * gain[j] + offset[j]
            require(np.isfinite(mapped).all() and (np.abs(mapped) <= np.finfo(np.float32).max).all(), "mapped prefix not finite/representable")
            available[:, j] = valid
    return flag, down, reconstruct_masks(available, active, clocks)


def acceptance_counter():
    return {arm: dict(accepted=0, abstained=0, reasons=Counter(), accepted_factors=Counter()) for arm in ADAPTIVE_ARMS}


def add_acceptance(counts, record):
    for arm in ADAPTIVE_ARMS:
        item, row = record["arms"][arm], counts[arm]
        accepted = item["accepted_norm"] > 0
        row["accepted" if accepted else "abstained"] += 1
        row["reasons"][item["reason"]] += 1
        if accepted:
            row["accepted_factors"][str(item["factor"])] += 1


def cold_decision(sid, times, cold, flag, down, threshold):
    valid = np.round((down - times) * 1000.0, 9) >= -1e-6 if flag else np.ones(len(times), dtype=bool)
    hit = np.flatnonzero((cold >= threshold) & valid)
    lead = float(round((down - times[hit[0]]) * 1000.0, 9)) if len(hit) and flag else None
    return dict(shot=sid, is_disrupt=flag, alarm=bool(len(hit)), lead_ms=lead, timely=bool(lead is not None and lead > 30))


def point_json(p):
    record = p["record"]
    losses = {arm: float(record["arms"][arm]["loss_before"]) for arm in FACTORIAL_ARMS}
    q = {h: losses[f"{h}_reverse"] - losses[f"{h}_true"] for h in HISTORIES}
    r = {h: losses[f"{h}_hold"] - losses[f"{h}_true"] for h in HISTORIES}
    result = dict(
        shot=p["shot"], clock=p["clock"], cold_probability=p["probability"],
        observed=int(p["mask"].sum()) * 25, mask=p["mask"].tolist(),
        loss_before=losses, q_reverse_minus_true=q, hold_target_minus_true=r,
    )
    if "lead_ms" in p:
        result.update(lead_ms=p["lead_ms"], cold_timely=p["cold_timely"])
    return result


def analyze(protocol, active, gain, offset, hashes, receipts, decisions, provenance):
    order, threshold = protocol["order"], finite_number(protocol["threshold"], "threshold")
    positives, negatives = [], []
    exclusions = dict(disruptive=[], safe=[], safe_clock_no_observed=0)
    counts = Counter()
    acceptance = acceptance_counter()
    for index, sid in enumerate(order[20:], start=20):
        name = f"{index + 1:04d}_{sid}"
        seal_path, curve_path = SOURCE / "shots" / f"{name}_seal.json", SOURCE / "shots" / f"{name}.npz"
        seal = load(seal_path)
        raw_path = io23.path_for("EAST", sid)
        raw_hash, curve_hash = sha(raw_path), sha(curve_path)
        require(raw_hash == hashes[str(sid)], f"{sid}: original HDF5 hash differs")
        require(seal == receipts[index] and seal.get("shot") == sid and seal.get("index") == index + 1 and seal.get("adaptation_active") is True and seal.get("outcome_used") is False and seal.get("sha256") == curve_hash, f"{sid}: sealed receipt differs")
        with np.load(curve_path, allow_pickle=False) as z:
            times, cold = z["times"], z["cold"]
        require(times.ndim == 1 and cold.shape == times.shape and len(times) > 0 and np.isfinite(times).all() and np.isfinite(cold).all() and (np.diff(times) > 0).all() and (cold >= 0).all() and (cold <= 1).all(), f"{sid}: invalid cold curve")
        records = seal["updates"].get("feedback")
        require(isinstance(records, list), f"{sid}: missing feedback")
        clocks = list(range(256, len(times), 128))
        require([r.get("clock") for r in records] == clocks, f"{sid}: feedback clocks differ")
        flag, down, masks = read_arrived_masks(raw_path, times, clocks, active, gain, offset)
        decision = cold_decision(sid, times, cold, flag, down, threshold)
        require(decision == decisions["cold"][index], f"{sid}: cold decision differs")
        require(all(row[index]["is_disrupt"] == flag for row in decisions.values()), f"{sid}: arm outcome differs")
        for record, mask in zip(records, masks):
            validate_feedback(record, mask)
            add_acceptance(acceptance, record)
        provenance["shots"].append(dict(shot=sid, index=index + 1, raw_sha256=raw_hash, curve_sha256=curve_hash, seal_sha256=sha(seal_path)))
        hit = np.flatnonzero(cold >= threshold)
        first = int(hit[0]) if len(hit) else len(times)
        by_clock = {c: (mask, record) for c, mask, record in zip(clocks, masks, records)}
        if flag:
            counts["disruptive"] += 1
            leads = [(down - times[c]) * 1000.0 for c in clocks]
            clock, status = choose_disruptive_clock(clocks, leads, first)
            if status != "no_time_window":
                counts["d_time_window"] += 1
            if status != "selected":
                exclusions["disruptive"].append(dict(shot=sid, reason=status))
                continue
            counts["d_after_no_alarm"] += 1
            mask, record = by_clock[clock]
            if not mask.any():
                counts["d_no_observed"] += 1
                exclusions["disruptive"].append(dict(shot=sid, clock=clock, reason="no_observed_target"))
                continue
            positives.append(dict(shot=sid, clock=clock, probability=float(cold[clock - 1]), mask=mask, record=record, lead_ms=float((down - times[clock]) * 1000.0), cold_timely=decision["timely"]))
        else:
            counts["safe"] += 1
            eligible = [c for c in clocks if first >= c]
            kept = [c for c in eligible if by_clock[c][0].any()]
            exclusions["safe_clock_no_observed"] += len(eligible) - len(kept)
            if not kept:
                exclusions["safe"].append(dict(shot=sid, reason="no_clock_before_alarm" if not eligible else "no_observed_target"))
            else:
                counts["safe_with_candidate"] += 1
                negatives.extend(dict(shot=sid, clock=c, probability=float(cold[c - 1]), mask=by_clock[c][0], record=by_clock[c][1]) for c in kept)
        if (index + 1) % 50 == 0:
            print(f"verified {index + 1 - 20}/340 suffix shots", flush=True)
    pairs, unmatched = greedy_pairs(positives, negatives)
    exclusions["disruptive"].extend(dict(shot=sid, reason="unmatched") for sid in unmatched)
    counts["eligible_disruptive"] = len(positives)
    counts["eligible_cold_not_timely"] = sum(not d["cold_timely"] for d in positives)
    counts["matched_pairs"] = len(pairs)
    counts["matched_cold_not_timely"] = sum(not d["cold_timely"] for d, _ in pairs)
    counts["safe_candidate_clocks"] = len(negatives)
    selected_acceptance = acceptance_counter()
    output_pairs, qdiff, rdiff = [], [], []
    for d, s in pairs:
        add_acceptance(selected_acceptance, d["record"])
        add_acceptance(selected_acceptance, s["record"])
        dj, sj = point_json(d), point_json(s)
        qdiff.append([dj["q_reverse_minus_true"][h] - sj["q_reverse_minus_true"][h] for h in HISTORIES])
        rdiff.append([dj["hold_target_minus_true"][h] - sj["hold_target_minus_true"][h] for h in HISTORIES])
        output_pairs.append(dict(disruptive=dj, safe=sj, probability_gap=abs(d["probability"] - s["probability"]), clock_gap=abs(d["clock"] - s["clock"])))
    stats = bootstrap_summary(np.asarray(qdiff, dtype=float).reshape(-1, 3), np.asarray(rdiff, dtype=float).reshape(-1, 3))
    result = dict(
        status=stats["screen_status"], counts=dict(counts), statistics=stats,
        bootstrap=dict(replicates=2000, seed=131, confidence=0.95, method="percentile linear", paired=True, rematched=False, shared_draws_for_all_contrasts=True),
        acceptance_all_suffix_feedback=acceptance,
        acceptance_matched_feedback=selected_acceptance,
        old_feasibility_reference=dict(eligible_disruptive=45, matched_pairs=41, matched_cold_not_timely=29),
        limitations=["outcome-selected clocks are retrospective only", "confidence intervals describe the fixed matched set, not full-stream significance or rematching/development uncertainty", "cold probability and clock do not match the complete prefix", "no causal physics or alarm benefit follows", "hold target is zero increment, not a hold-last predictor benchmark", "original data and experiment were previously exposed to development"],
    )
    return result, output_pairs, exclusions


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    require(not OUTPUT.exists(), f"fresh output required: {OUTPUT}")
    OUTPUT.mkdir()
    try:
        inputs = verify_inputs()
        result, pairs, exclusions = analyze(*inputs)
        provenance = inputs[-1]
        write_json(OUTPUT / "result.json", result)
        write_json(OUTPUT / "pairs.json", pairs)
        write_json(OUTPUT / "exclusions.json", exclusions)
        provenance["output_sha256"] = {name: sha(OUTPUT / name) for name in ("result.json", "pairs.json", "exclusions.json")}
        write_json(OUTPUT / "hashes.json", provenance)
        print(json.dumps(dict(status=result["status"], counts=result["counts"], statistics=result["statistics"]), indent=2), flush=True)
    except Exception as exc:
        write_json(OUTPUT / "failure.json", dict(status="insufficient_evidence", error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("No options: only the frozen input and fresh fixed output are supported.")
    main()
