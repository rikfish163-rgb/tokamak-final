"""CPU audit of a COMPLETED functional stream; optional exclusive report output.

Usage from the worktree root (limit BLAS threads as appropriate):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m \
    experiments.risk_aligned_online.east_decoder_calibration_20261102.functional_audit \
    probe_outputs/east_functional_protection360_v1

This checks recorded consistency, not neural execution, hidden RMS, or the
historical time at which an unanchored seal was written. It never reads raw
signal datasets, trains a model, or imports a runner. With --output it writes
only the requested new report after successful verification, never overwrites.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .. import io23
from .audit360 import (
    classify_curve,
    finite,
    paired_flips,
    read,
    recount,
    require,
    sha,
    vector,
)

ROOT = io23.ROOT
OLD = ROOT / "probe_outputs/east_decoder_complete360_v1"
CV = ROOT / "probe_outputs/east_decoder_calibration20_v1"
ANCESTOR = ROOT / "probe_outputs/innovation_metric360_v2"
OLD_SHA = "7de4d8999fbf3c12706c3bbcbf6218c0b04e8e4629f9a45769ed603990f7f48a"
ARMS = (
    "static",
    "old_real",
    "parameter_real",
    "hidden_real",
    "hidden_reverse",
    "hidden_hold",
)
FRACTIONS = (1.0, 0.25, 0.0625, 0.015625)
PARTS = {
    "full360": slice(None),
    "warmup20": slice(None, 20),
    "suffix340": slice(20, None),
}
HEX_DIGITS = frozenset("0123456789abcdef")

# Parameter protection can add two float32 scalars; hidden protection may
# promote to float64. One float32 rounding unit is allowed here only. The
# model's acceptance and candidate ordering below always use strict <, no tol.
OBJECTIVE_REL_TOL = 1e-7
OBJECTIVE_ABS_TOL = 1e-15


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX_DIGITS


def validate_curves(times, curves, ref_times, ref_cold, ref_old, warmup):
    require(set(curves) == set(ARMS), "curve arms")
    require(
        times.ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    require(np.array_equal(times, ref_times), "reference times")
    for arm, curve in curves.items():
        require(
            curve.shape == times.shape
            and np.isfinite(curve).all()
            and np.all((curve >= 0) & (curve <= 1)),
            "invalid curve: " + arm,
        )
    require(np.array_equal(curves["static"], ref_cold), "reference static curve")
    require(np.array_equal(curves["old_real"], ref_old), "reference old_real curve")
    for arm, curve in curves.items():
        require(
            np.array_equal(curve[:160], curves["static"][:160]), "first 160: " + arm
        )
        if warmup:
            require(np.array_equal(curve, curves["static"]), "warmup: " + arm)


def empty_totals():
    return dict(
        events=0,
        accepted=0,
        all_rejected=0,
        no_target=0,
        tested_steps=0,
        nonzero_gradients=0,
        observed_scalars=0,
        max_state_norm=0.0,
        max_gradient_norm=0.0,
        max_step_norm=0.0,
        tested_by_fraction={str(f): 0 for f in FRACTIONS},
        accepted_by_fraction={str(f): 0 for f in FRACTIONS},
    )


def check_trial_steps(r, accepted, before, stats):
    steps = r["tested_steps"]
    require(
        isinstance(steps, list) and 1 <= len(steps) <= len(FRACTIONS),
        "tested_steps must be a nonempty fraction prefix",
    )
    values = []
    for step, fraction in zip(steps, FRACTIONS, strict=False):
        require(
            isinstance(step, dict)
            and set(step)
            == {"fraction", "diagnostic_loss", "protection_loss", "objective"},
            "tested step fields",
        )
        actual_fraction = finite(step["fraction"])
        require(actual_fraction == fraction, "fraction prefix")
        diagnostic = finite(step["diagnostic_loss"])
        protection = finite(step["protection_loss"])
        objective = finite(step["objective"])
        require(
            diagnostic >= 0
            and protection >= 0
            and objective >= 0
            and math.isclose(
                objective,
                diagnostic + protection,
                rel_tol=OBJECTIVE_REL_TOL,
                abs_tol=OBJECTIVE_ABS_TOL,
            ),
            "objective != diagnostic + protection",
        )
        values.append(objective)
        stats["tested_by_fraction"][str(fraction)] += 1
    stats["tested_steps"] += len(steps)
    require(
        finite(r["loss_proposal"]) == values[-1],
        "proposal differs from last tested objective",
    )
    require(
        all(value >= before for value in values[:-1]),
        "earlier fraction already improved",
    )
    if accepted:
        require(
            values[-1] < before
            and finite(r["chosen_fraction"]) == FRACTIONS[len(steps) - 1],
            "chosen fraction or strict improvement",
        )
        stats["accepted_by_fraction"][str(FRACTIONS[len(steps) - 1])] += 1
    else:
        require(
            len(steps) == len(FRACTIONS)
            and values[-1] >= before
            and r["chosen_fraction"] is None,
            "all rejected fractions or chosen fraction",
        )
        stats["all_rejected"] += 1


def check_updates(updates, length, warmup):
    require(isinstance(updates, dict) and set(updates) == set(ARMS[1:]), "update arms")
    ends = [] if warmup else list(range(159, length, 128))
    totals = {}
    for arm in ARMS[1:]:
        events = updates[arm]
        require(
            isinstance(events, list) and [r["maturity"] for r in events] == ends,
            "update clock",
        )
        previous = np.zeros(256, dtype=np.float32)
        stats = totals[arm] = empty_totals()
        stats["events"] = len(events)
        for end, r in zip(ends, events, strict=True):
            require(
                all(
                    type(r[key]) is int
                    for key in (
                        "origin",
                        "maturity",
                        "sealed_through",
                        "first_affected",
                    )
                )
                and r["origin"] == end - 25
                and r["maturity"] == end
                and r["sealed_through"] == end
                and r["first_affected"] == end + 1,
                "causal clock",
            )
            require(
                all(
                    valid_hash(r[key])
                    for key in ("forecast_sha256", "mask_sha256", "target_sha256")
                ),
                "receipt hash",
            )
            n, accepted = r["observed"], r["accepted"]
            require(
                type(n) is int and 0 <= n <= 550 and type(accepted) is bool,
                "receipt types",
            )
            gradient, state = vector(r["gradient256"]), vector(r["state256_after"])
            gradient_norm, step_norm = (
                finite(r["gradient_norm"]),
                finite(r["step_norm"]),
            )
            expected_gradient = float(np.linalg.norm(gradient.astype(np.float64)))
            # Match the producer: subtract in float32 before converting to float64.
            expected_step = float(np.linalg.norm((state - previous).astype(np.float64)))
            require(
                gradient_norm >= 0
                and math.isclose(
                    gradient_norm, expected_gradient, rel_tol=1e-9, abs_tol=1e-14
                ),
                "gradient norm",
            )
            require(
                step_norm >= 0
                and math.isclose(step_norm, expected_step, rel_tol=1e-9, abs_tol=1e-14),
                "step norm",
            )
            if arm == "old_real":
                require("tested_steps" not in r, "old_real must not have tested_steps")
            else:
                require("tested_steps" in r, "new receipt needs tested_steps")
            if n:
                before, proposal = finite(r["loss_before"]), finite(r["loss_proposal"])
                require(before >= 0 and proposal >= 0, "negative receipt loss")
                require(
                    accepted == (proposal < before)
                    and r["reason"] == ("improved" if accepted else "not_improved"),
                    "strict acceptance",
                )
                if arm == "old_real":
                    chosen_valid = (
                        finite(r["chosen_fraction"]) == 1.0
                        if accepted
                        else r["chosen_fraction"] is None
                    )
                    require(chosen_valid, "old chosen fraction")
                    if accepted:
                        stats["accepted_by_fraction"]["1.0"] += 1
                    else:
                        stats["all_rejected"] += 1
                else:
                    check_trial_steps(r, accepted, before, stats)
            else:
                require(
                    not accepted
                    and r["reason"] == "no_target"
                    and r["loss_before"] is None
                    and r["loss_proposal"] is None
                    and r["chosen_fraction"] is None
                    and gradient_norm == step_norm == 0
                    and not np.any(gradient),
                    "no target loss or gradient",
                )
                if arm != "old_real":
                    require(r["tested_steps"] == [], "no target tested_steps")
                stats["no_target"] += 1
            require(
                accepted or np.array_equal(previous, state), "rejected state changed"
            )
            previous = state
            stats["accepted"] += int(accepted)
            stats["nonzero_gradients"] += int(gradient_norm > 0)
            stats["observed_scalars"] += n
            stats["max_state_norm"] = max(
                stats["max_state_norm"], float(np.linalg.norm(state.astype(np.float64)))
            )
            stats["max_gradient_norm"] = max(stats["max_gradient_norm"], gradient_norm)
            stats["max_step_norm"] = max(stats["max_step_norm"], step_norm)
    for group in zip(*(updates[a] for a in ARMS[1:]), strict=True):
        require(
            len({r["mask_sha256"] for r in group}) == 1
            and len({r["observed"] for r in group}) == 1,
            "mask mismatch",
        )
        require(
            len({r["target_sha256"] for r in group[:3]}) == 1, "real target mismatch"
        )
    return totals


def check_seal(seal, index, shot, curve_hash, updates_hash, warmup):
    require(
        type(seal["index"]) is int
        and seal["index"] == index
        and type(seal["shot"]) is int
        and seal["shot"] == shot
        and seal["current_outcome_used_in_prediction"] is False
        and seal["reused_warmup"] is warmup
        and valid_hash(seal["curve_sha256"])
        and valid_hash(seal["updates_sha256"])
        and seal["curve_sha256"] == curve_hash
        and seal["updates_sha256"] == updates_hash,
        "shot seal",
    )


def verify_snapshot(archive, bound, root):
    require(isinstance(bound, dict) and bool(bound), "snapshot empty bindings")
    expected = {}
    for raw_path, digest in bound.items():
        path = Path(raw_path)
        require(
            path.is_absolute()
            and path.is_relative_to(root)
            and ".." not in path.parts
            and valid_hash(digest),
            "snapshot binding",
        )
        name = path.relative_to(root).as_posix()
        require(name not in expected, "snapshot duplicate binding")
        expected[name] = digest
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(
            len(names) == len(set(names)) and set(names) == set(expected),
            "snapshot members",
        )
        for name, digest in expected.items():
            require(
                hashlib.sha256(z.read(name)).hexdigest() == digest,
                "snapshot hash: " + name,
            )


def diagnostic_gate(scores):
    base = scores["static"]["full360"]
    candidate = scores["hidden_real"]["full360"]
    net = lambda score: score["tp"] - score["fp"]  # noqa: E731
    operational = (
        candidate["fp"] <= 36
        and candidate["tp"] > 125
        and net(candidate) > 90
        and candidate["tp"] > base["tp"]
        and net(candidate) > net(base)
    )
    controls = {
        arm: net(candidate) > net(scores[arm]["full360"])
        for arm in ("old_real", "parameter_real", "hidden_reverse", "hidden_hold")
    }
    return dict(
        operational=operational,
        controls=controls,
        passed=operational and all(controls.values()),
    )


def summarize(rows):
    require(set(rows) == set(ARMS), "score arms")
    metrics = {
        arm: {name: recount(rows[arm][part]) for name, part in PARTS.items()}
        for arm in ARMS
    }
    flips = {
        arm: {
            name: paired_flips(rows["static"][part], rows[arm][part])
            for name, part in PARTS.items()
        }
        for arm in ARMS[1:]
    }
    return metrics, flips, diagnostic_gate(metrics)


def compare_report(rows, saved, done):
    metrics, flips, gate = summarize(rows)
    require(
        rows == saved
        and metrics == done["metrics"]
        and {a: flips[a]["full360"] for a in flips} == done["flips"]
        and gate == done["gate"],
        "report rows, metrics, flips or gate mismatch",
    )
    return metrics, flips, gate


def check_header(root):
    # This status check must precede every HDF read, including failed audits.
    done = read(root / "completed.json")
    require(done.get("status") == "COMPLETED", "COMPLETED stream required")
    completed_sha = sha(root / "completed.json")
    protocol = read(root / "protocol.json")
    require(
        sha(root / "protocol.json") == done["protocol_sha256"]
        and sha(root / "per_shot.json") == done["per_shot_sha256"],
        "completion hashes",
    )
    old_done, old_protocol = read(OLD / "completed.json"), read(OLD / "protocol.json")
    cpu, neural = read(OLD / "audit_cpu.json"), read(OLD / "verification_neural.json")
    require(
        old_done["status"] == "COMPLETED"
        and sha(OLD / "completed.json") == OLD_SHA == protocol["old_completed_sha256"]
        and sha(OLD / "protocol.json") == old_done["protocol_sha256"]
        and sha(OLD / "per_shot.json") == old_done["per_shot_sha256"]
        and cpu["status"] == neural["status"] == "PASS"
        and cpu["evidence"]["completed_sha256"] == neural["completed_sha256"] == OLD_SHA
        and cpu["evidence"]["protocol_sha256"] == old_done["protocol_sha256"],
        "old completed/CPU/neural reference binding",
    )
    cv = read(CV / "completed.json")
    require(
        cv["status"] == "COMPLETED"
        and sha(CV / "completed.json") == old_protocol["local_cv_completed_sha256"]
        and cv["gate"] == old_protocol["local_cv_gate"] == cpu["local_cv_gate"]
        and cv["gate"]["passed"] is False
        and old_done["local_gate_remains_failed"] is True
        and done["old_local_gate_remains_failed"] is True,
        "old local CV gate",
    )
    order = protocol["order"]
    ids = [r["shot"] for r in order]
    require(
        len(ids) == 360
        and len(set(ids)) == 360
        and all(type(sid) is int and sid > 0 for sid in ids)
        and order == old_protocol["order"]
        and old_protocol["head_fit_ids"] == ids[:20]
        and protocol["arms"] == list(ARMS)
        and protocol["fractions"] == list(FRACTIONS),
        "cohort, arms or fractions",
    )
    require(
        all(
            protocol[k] == old_protocol[k]
            for k in (
                "threshold",
                "source_sha256",
                "cold_sha256",
                "input_map_sha256",
                "prior_sha256",
            )
        ),
        "old protocol binding",
    )
    threshold = finite(protocol["threshold"])
    require(0 < threshold < 1, "threshold")
    require(
        protocol["scope"] == "DIAGNOSTIC_FUNCTIONAL_PROTECTION_SAME_FROZEN_HEAD"
        and protocol["head_preselected"] == "worst_meta"
        and protocol["candidate_preselected"] == "hidden_real"
        and protocol["head_training_steps"]
        == protocol["suffix_supervised_updates"]
        == 0
        and protocol["warmup20_cost_included"] is True
        and protocol["method_promoted"] is False
        and protocol["independent_confirmation"] is False
        and done["independent_confirmation"] is False
        and done["optimizer_steps"] == 0
        and done["source_unchanged"] is True
        and done["cold_unchanged"] is True
        and done["heads_unchanged"] is True,
        "diagnostic scope or frozen state",
    )
    require(
        old_done["metrics"]["static"]["full360"]["tp"] == 125
        and old_done["metrics"]["static"]["full360"]["fp"] == 35,
        "fixed predeclared static baseline",
    )
    calibration = read(OLD / "calibration.json")
    require(
        sha(OLD / "heads.pt") == calibration["heads_sha256"] == protocol["heads_sha256"]
        and sha(CV / "diagnostic_basis.pt")
        == cv["artifact_sha256"]["diagnostic_basis.pt"]
        == protocol["basis_sha256"],
        "frozen heads or basis hash",
    )
    reference = read(ANCESTOR / "protocol.json")
    mapping_path = Path(reference["calibration_reference"]) / "input_map.json"
    mapping = read(mapping_path)
    require(
        sha(mapping_path) == protocol["input_map_sha256"]
        and mapping["calibration_ids"] == ids[:20],
        "input mapping hash or roster",
    )
    files = {
        ROOT
        / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt": protocol[
            "source_sha256"
        ],
        ROOT / "probe_outputs/stratified_cold20_context360_v1/models/cold.pt": protocol[
            "cold_sha256"
        ],
        ROOT / "probe_outputs/direct_decoder_source_v1/decoders.pt": protocol[
            "prior_sha256"
        ],
    }
    require(
        all(sha(path) == digest for path, digest in files.items()),
        "frozen checkpoint hash",
    )
    bound = protocol["bound_code_sha256"]
    required_paths = (
        ROOT / "EAST_FUNCTIONAL_PROTECTION360_PLAN_20261102.md",
        Path(__file__).with_name("functional_stream.py"),
        Path(__file__).with_name("functional_update.py"),
    )
    require(
        all(str(path) in bound for path in required_paths),
        "required code/plan bindings",
    )
    verify_snapshot(root / "code_snapshot.zip", bound, ROOT)
    require(
        all(sha(path) == digest for path, digest in bound.items()), "live code hash"
    )
    return done, protocol, old_done, cv, ids, completed_sha


def accumulate(totals, per_shot):
    for arm, stats in totals.items():
        for key, value in stats.items():
            new = per_shot[arm][key]
            if isinstance(value, dict):
                for fraction in value:
                    value[fraction] += new[fraction]
            elif key.startswith("max_"):
                stats[key] = max(value, new)
            else:
                stats[key] += new


def main(root):
    root = Path(root)
    done, protocol, old_done, cv, ids, completed_sha = check_header(root)
    stems = [f"{i:04d}_{sid}" for i, sid in enumerate(ids, 1)]
    for directory in (root / "shots", OLD / "shots"):
        for suffix in (".npz", "_seal.json", "_updates.json"):
            require(
                {p.name for p in directory.glob("*" + suffix)}
                == {stem + suffix for stem in stems},
                "shot file set: " + suffix,
            )
    saved, old_rows = read(root / "per_shot.json"), read(OLD / "per_shot.json")
    totals = {arm: empty_totals() for arm in ARMS[1:]}
    rows = {arm: [] for arm in ARMS}
    seal_digest = hashlib.sha256()
    for index, (sid, stem) in enumerate(zip(ids, stems, strict=True), 1):
        paths = [
            root / "shots" / (stem + ext)
            for ext in (".npz", "_updates.json", "_seal.json")
        ]
        old_paths = [
            OLD / "shots" / (stem + ext)
            for ext in (".npz", "_updates.json", "_seal.json")
        ]
        curve, updates, sealpath = paths
        refcurve, refupdates, refsealpath = old_paths
        seal, old_seal = read(sealpath), read(refsealpath)
        check_seal(seal, index, sid, sha(curve), sha(updates), index <= 20)
        check_seal(old_seal, index, sid, sha(refcurve), sha(refupdates), index <= 20)
        seal_digest.update(f"{stem} {sha(sealpath)} {sha(refsealpath)}\n".encode())
        with np.load(curve, allow_pickle=False) as data:
            require(set(data.files) == {"times", *ARMS}, "curve keys")
            times = data["times"]
            curves = {arm: data[arm] for arm in ARMS}
        with np.load(refcurve, allow_pickle=False) as ref:
            require(
                {"times", "static", "worst_real"} <= set(ref.files), "old curve keys"
            )
            validate_curves(
                times,
                curves,
                ref["times"],
                ref["static"],
                ref["worst_real"],
                index <= 20,
            )
        accumulate(totals, check_updates(read(updates), len(times), index <= 20))
        # Only meta/IsDisrupt and meta/DownTime via io23; no signal data are accessed.
        flag, down = io23.read_outcome("EAST", sid)
        require(
            type(flag) is bool
            and (not flag or (down is not None and math.isfinite(down))),
            "outcome metadata",
        )
        for arm in ARMS:
            rows[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify_curve(
                        times, curves[arm], flag, down, protocol["threshold"]
                    ),
                )
            )
    metrics, flips, gate = compare_report(rows, saved, done)
    require(
        metrics["static"]["full360"]["n_disrupt"] == 180
        and metrics["static"]["full360"]["n_safe"] == 180,
        "180D/180S cohort",
    )
    require(
        rows["static"] == old_rows["static"]
        and rows["old_real"] == old_rows["worst_real"]
        and metrics["static"] == old_done["metrics"]["static"]
        and metrics["old_real"] == old_done["metrics"]["worst_real"],
        "old score mismatch",
    )
    require(
        sha(root / "completed.json") == completed_sha
        and sha(root / "protocol.json") == done["protocol_sha256"]
        and sha(root / "per_shot.json") == done["per_shot_sha256"]
        and sha(OLD / "completed.json") == OLD_SHA,
        "completion changed during audit",
    )
    return dict(
        status="PASS",
        audit_type="RECORDED_DATA_CONSISTENCY_ONLY",
        scope=protocol["scope"],
        threshold=protocol["threshold"],
        metrics=metrics,
        flips=flips,
        diagnostic_gate=gate,
        local_cv_gate=cv["gate"],
        old_local_gate_remains_failed=True,
        method_promoted=False,
        independent_future_confirmation=False,
        update_statistics=totals,
        objective_addition_tolerance=dict(
            relative=OBJECTIVE_REL_TOL,
            absolute=OBJECTIVE_ABS_TOL,
            acceptance_comparison="strict <; no tolerance",
        ),
        neural_replay_performed=False,
        hidden_rms_verified=False,
        raw_signal_datasets_read=0,
        evidence=dict(
            shots=360,
            old_shots=360,
            first160_matches=360,
            warmup_curves_match=20,
            static_curves_equal_old=360,
            old_real_curves_equal_old_worst=360,
            seal_pairs_sha256=seal_digest.hexdigest(),
            completed_sha256=completed_sha,
            old_completed_sha256=OLD_SHA,
            protocol_sha256=done["protocol_sha256"],
            per_shot_sha256=done["per_shot_sha256"],
            code_snapshot_sha256=sha(root / "code_snapshot.zip"),
        ),
        limitations=[
            "Receipt values and seal ordering are recorded claims, not a full-model replay.",
            "State is reconstructed from zero per shot; actual hidden RMS and metric bounds were not rerun.",
            "Per-shot seal hashes are self-consistency checks, not externally anchored provenance or timestamps.",
            "No raw signal datasets were read or hashed; old CV failure and promotion status are unchanged.",
        ],
    )


def save_report(report, output):
    if report.get("status") != "PASS":
        raise ValueError("successful audit required before report publication")
    with Path(output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="completed functional stream output directory"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = main(args.root)
    result["verifier_sha256"] = sha(Path(__file__))
    if args.output:
        save_report(result, args.output)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
