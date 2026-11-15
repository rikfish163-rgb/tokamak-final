"""CPU-only audit of a completed diagnostic stream, not a neural replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .. import io23

REFERENCE = io23.ROOT / "probe_outputs/innovation_metric360_v2"
CV = io23.ROOT / "probe_outputs/east_decoder_calibration20_v1"
REFERENCE_SHA = "f558ee30e33192da29daa7863f58267d25913a9c8dc25a645188ab32cac11014"
ARMS = ("static", "prior_real", "forecast_real", "worst_real", "worst_reverse", "worst_hold")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite receipt")
    return float(value)


def classify_curve(times, scores, disruptive, down, threshold):
    valid = (down - times) * 1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    hits = np.flatnonzero(valid & (scores >= threshold))
    lead = float(round((down - times[hits[0]]) * 1000, 9)) if disruptive and len(hits) else None
    return dict(alarm=bool(len(hits)), lead_ms=lead, timely=bool(lead is not None and lead > 30))


def recount(rows):
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    tp = sum(r["timely"] for r in rows)
    fp = sum(r["alarm"] and not r["disruptive"] for r in rows)
    tpr, fpr = (tp / nd if nd else None), (fp / ns if ns else None)
    leads = [r["lead_ms"] for r in rows if r["timely"]]
    return dict(n=len(rows), n_disrupt=nd, n_safe=ns, tp=tp, fp=fp, tpr=tpr, fpr=fpr,
                J=tpr - fpr if nd and ns else None,
                early_over3000=sum(v > 3000 for v in leads),
                median_timely_lead_ms=float(np.median(leads)) if leads else None)


def paired_flips(before, after):
    require(len(before) == len(after), "flip length")
    flips = {k: [] for k in ("rescued_D", "lost_D", "removed_FP", "added_FP")}
    for a, b in zip(before, after, strict=True):
        require((a["shot"], a["disruptive"]) == (b["shot"], b["disruptive"]), "flip identity")
        if a["disruptive"]:
            if b["timely"] and not a["timely"]:
                flips["rescued_D"].append(a["shot"])
            if a["timely"] and not b["timely"]:
                flips["lost_D"].append(a["shot"])
        else:
            if a["alarm"] and not b["alarm"]:
                flips["removed_FP"].append(a["shot"])
            if b["alarm"] and not a["alarm"]:
                flips["added_FP"].append(a["shot"])
    return flips


def validate_curves(times, curves, ref_times, ref_cold, warmup):
    require(set(curves) == set(ARMS), "curve arms")
    require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0), "invalid times")
    require(np.array_equal(times, ref_times), "reference times")
    require(np.array_equal(curves["static"], ref_cold), "reference cold")
    for arm, values in curves.items():
        require(values.shape == times.shape and np.isfinite(values).all()
                and np.all((values >= 0) & (values <= 1)), "invalid curve: " + arm)
        require(np.array_equal(values[:160], curves["static"][:160]), "first 160: " + arm)
        if warmup:
            require(np.array_equal(values, curves["static"]), "warmup: " + arm)


def vector(value):
    raw = np.asarray(value, dtype=np.float64)
    require(raw.shape == (256,) and np.isfinite(raw).all(), "256-vector")
    v = raw.astype(np.float32)
    require(np.isfinite(v).all() and np.array_equal(v.astype(np.float64), raw), "float32 vector")
    return v


def empty_totals():
    return dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0,
                max_state_norm=0.0, max_gradient_norm=0.0, max_step_norm=0.0)


def check_updates(updates, length, warmup):
    require(set(updates) == set(ARMS[1:]), "update arms")
    ends = [] if warmup else list(range(159, length, 128))
    totals = {}
    for arm in ARMS[1:]:
        events = updates[arm]
        require(isinstance(events, list) and [r["maturity"] for r in events] == ends, "update clock")
        previous = np.zeros(256, dtype=np.float32)
        totals[arm] = empty_totals()
        totals[arm]["events"] = len(events)
        for end, r in zip(ends, events, strict=True):
            require(type(r["maturity"]) is int and type(r["origin"]) is int
                    and type(r["sealed_through"]) is int and type(r["first_affected"]) is int
                    and r["origin"] == end - 25 and r["sealed_through"] == end
                    and r["first_affected"] == end + 1, "causal clock")
            for key in ("forecast_sha256", "mask_sha256", "target_sha256"):
                require(isinstance(r[key], str) and HEX.fullmatch(r[key]), "receipt hash")
            n, accepted = r["observed"], r["accepted"]
            require(type(n) is int and 0 <= n <= 550 and type(accepted) is bool, "receipt types")
            gradient, state = vector(r["gradient256"]), vector(r["state256_after"])
            gn, sn = finite(r["gradient_norm"]), finite(r["step_norm"])
            expected_gn = float(np.linalg.norm(gradient.astype(np.float64)))
            expected_sn = float(np.linalg.norm((state - previous).astype(np.float64)))
            require(gn >= 0 and math.isclose(gn, expected_gn, rel_tol=1e-9, abs_tol=1e-14), "gradient norm")
            require(sn >= 0 and math.isclose(sn, expected_sn, rel_tol=1e-9, abs_tol=1e-14), "state step norm")
            if n:
                before, proposal = finite(r["loss_before"]), finite(r["loss_proposal"])
                require(accepted == (proposal < before) and r["reason"] ==
                        ("improved" if accepted else "not_improved"), "strict acceptance")
            else:
                require(not accepted and r["reason"] == "no_target" and r["loss_before"] is None
                        and r["loss_proposal"] is None and gn == 0 and sn == 0
                        and not np.any(gradient), "no target")
            require(accepted or np.array_equal(previous, state), "rejected state changed")
            previous = state
            totals[arm]["accepted"] += int(accepted)
            totals[arm]["nonzero_gradients"] += int(gn > 0)
            totals[arm]["observed_scalars"] += n
            for key, value in (("max_state_norm", float(np.linalg.norm(state.astype(np.float64)))),
                               ("max_gradient_norm", gn), ("max_step_norm", sn)):
                totals[arm][key] = max(totals[arm][key], value)
    for group in zip(*(updates[a] for a in ARMS[1:]), strict=True):
        require(len({r["mask_sha256"] for r in group}) == 1
                and len({r["observed"] for r in group}) == 1, "mask mismatch")
        require(len({r["target_sha256"] for r in group[:3]}) == 1, "real target mismatch")
    return totals


def check_seal(seal, index, shot, curve, updates, warmup):
    require(seal["index"] == index and seal["shot"] == shot
            and seal["current_outcome_used_in_prediction"] is False
            and seal["reused_warmup"] is warmup and sha(curve) == seal["curve_sha256"]
            and sha(updates) == seal["updates_sha256"], "shot seal")


def main(root):
    root = Path(root)
    done = read(root / "completed.json")
    require(done.get("status") == "COMPLETED", "COMPLETED stream required")
    completed_sha = sha(root / "completed.json")
    output = root / "audit_cpu.json"
    if output.exists():
        raise FileExistsError(output)
    protocol = read(root / "protocol.json")
    require(sha(root / "protocol.json") == done["protocol_sha256"]
            and sha(root / "per_shot.json") == done["per_shot_sha256"], "completion hashes")
    reference_done = read(REFERENCE / "completed.json")
    reference = read(REFERENCE / "protocol.json")
    verified = read(REFERENCE / "verification_records.json")
    neural = read(REFERENCE / "verification_neural.json")
    require(reference_done["status"] == "COMPLETED" and sha(REFERENCE / "completed.json") ==
            protocol["reference_completed_sha256"] == REFERENCE_SHA
            and sha(REFERENCE / "protocol.json") == reference_done["protocol_sha256"]
            == verified["hashes"]["protocol_sha256"]
            and sha(REFERENCE / "shot_seals.json") == reference_done["shot_seals_sha256"]
            == verified["hashes"]["shot_seals_sha256"]
            and verified["status"] == neural["status"] == "PASS"
            and verified["hashes"]["completed_sha256"] == neural["completed_sha256"]
            == REFERENCE_SHA, "reference binding")
    cv = read(CV / "completed.json")
    require(cv["status"] == "COMPLETED" and sha(CV / "completed.json") ==
            protocol["local_cv_completed_sha256"] and cv["gate"] == protocol["local_cv_gate"]
            and cv["gate"]["passed"] is False and done["local_gate_remains_failed"] is True,
            "local CV gate must remain failed")
    order = protocol["order"]
    ids = [r["shot"] for r in order]
    require(len(ids) == len(set(ids)) == 360 and all(type(s) is int and s > 0 for s in ids)
            and order == reference["order"] and protocol["head_fit_ids"] == ids[:20]
            and protocol["arms"] == list(ARMS) and protocol["threshold"] == reference["threshold"]
            and protocol["input_map_sha256"] == reference["input_map_sha256"]
            and protocol["cold_sha256"] == reference["cold_state_sha256"], "cohort or reference")
    require(protocol["scope"] == "DIAGNOSTIC_COMPLETE360_AFTER_LOCAL_SURROGATE_GATE_FAILED"
            and protocol["warmup20_cost_included"] is True
            and protocol["suffix_supervised_updates"] == protocol["suffix_head_updates"] == 0
            and protocol["method_promoted"] is False and protocol["independent_confirmation"] is False
            and done["independent_confirmation"] is False, "diagnostic scope")
    threshold = finite(protocol["threshold"])
    require(0 < threshold < 1, "threshold")
    stems = [f"{i:04d}_{sid}" for i, sid in enumerate(ids, 1)]
    for suffix in (".npz", "_seal.json", "_updates.json"):
        require({p.name for p in (root / "shots").glob("*" + suffix)} ==
                {s + suffix for s in stems}, "shot files: " + suffix)
    entries = verified["hashes"]["shot_files"]
    manifests = read(REFERENCE / "shot_seals.json")
    require(len(entries) == len(manifests) == 360, "reference shot count")
    rows = {a: [] for a in ARMS}
    totals = {a: empty_totals() for a in ARMS[1:]}
    saved = read(root / "per_shot.json")
    for index, (sid, stem, entry, manifest) in enumerate(zip(ids, stems, entries, manifests, strict=True), 1):
        base, refbase = root / "shots" / stem, REFERENCE / "shots" / stem
        curve, updates, sealpath = (Path(str(base) + x) for x in (".npz", "_updates.json", "_seal.json"))
        refcurve, refsealpath = (Path(str(refbase) + x) for x in (".npz", "_seal.json"))
        seal, refseal = read(sealpath), read(refsealpath)
        check_seal(seal, index, sid, curve, updates, index <= 20)
        require(entry["index"] == refseal["index"] == index and entry["shot"] == refseal["shot"] == sid
                and manifest == {k: entry[k] for k in ("index", "shot", "seal_sha256")}
                and refseal["current_outcome_used_in_prediction"] is False
                and sha(refsealpath) == entry["seal_sha256"] and sha(refcurve) ==
                refseal["sha256"] == entry["curve_sha256"], "reference seal")
        with np.load(curve, allow_pickle=False) as data:
            require(set(data.files) == {"times", *ARMS}, "curve keys")
            times, curves = data["times"], {a: data[a] for a in ARMS}
        with np.load(refcurve, allow_pickle=False) as refdata:
            validate_curves(times, curves, refdata["times"], refdata["cold"], index <= 20)
        per_arm = check_updates(read(updates), len(times), index <= 20)
        for arm in totals:
            for key in totals[arm]:
                value = per_arm[arm][key]
                totals[arm][key] = (max(totals[arm][key], value) if key.startswith("max_")
                                    else totals[arm][key] + value)
        flag, down = io23.read_outcome("EAST", sid)
        require(type(flag) is bool and (not flag or (down is not None and math.isfinite(down))), "outcome")
        for arm in ARMS:
            rows[arm].append(dict(shot=sid, disruptive=flag,
                                  **classify_curve(times, curves[arm], flag, down, threshold)))
    parts = {"full360": slice(None), "warmup20": slice(None, 20), "suffix340": slice(20, None)}
    metrics = {a: {name: recount(rows[a][part]) for name, part in parts.items()} for a in ARMS}
    flips = {a: {name: paired_flips(rows["static"][part], rows[a][part])
                 for name, part in parts.items()} for a in ARMS[1:]}
    require(metrics["static"]["full360"]["n_disrupt"] == 180
            and metrics["static"]["full360"]["n_safe"] == 180, "180D/180S cohort")
    require(rows == saved and metrics == done["metrics"]
            and {a: flips[a]["full360"] for a in flips} == done["flips"], "reported scoring mismatch")
    base, candidate = metrics["static"]["full360"], metrics["worst_real"]["full360"]
    def net(m):
        return m["tp"] - m["fp"]

    controls = {a: net(candidate) > net(metrics[a]["full360"]) for a in
                ("prior_real", "forecast_real", "worst_reverse", "worst_hold")}
    operational = candidate["fp"] <= 36 and candidate["tp"] > base["tp"] and net(candidate) > net(base)
    gate = dict(operational=operational, controls=controls, passed=operational and all(controls.values()))
    require(gate == done["gate"], "diagnostic gate mismatch")
    require(sha(root / "completed.json") == completed_sha, "completion changed during audit")
    report = dict(status="PASS", audit_type="RECORDED_DATA_CONSISTENCY_ONLY",
                  scope=protocol["scope"], threshold=threshold,
                  local_cv_gate=cv["gate"], diagnostic_gate=gate, metrics=metrics, flips=flips,
                  update_statistics=totals, neural_replay_performed=False, hidden_rms_verified=False,
                  raw_signal_datasets_read=0, independent_future_confirmation=False,
                  evidence=dict(shot_seals_verified=360, reference_seals_verified=360,
                                first160_matches=360, warmup_curves_match=20,
                                reference_cold_curves_equal=360, completed_sha256=completed_sha,
                                protocol_sha256=done["protocol_sha256"],
                                reference_completed_sha256=REFERENCE_SHA),
                  limitations=["Receipt losses, gradients, targets and seal timing are recorded claims; "
                               "neural computation was not replayed.",
                               "Neither actual hidden RMS nor the metric-direction bound can be checked "
                               "without hidden features.",
                               "File hashes establish consistency, not external provenance; "
                               "raw HDF files were not hashed."])
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    result = main(parser.parse_args().root)
    print(json.dumps({"status": result["status"], "scope": result["scope"]}))
