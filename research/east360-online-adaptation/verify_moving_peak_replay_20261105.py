"""CPU-only, post-completion audit of the fixed-versus-moving peak replay.

This recounts saved predictions, not model training or its internal causality.
The first 20 shots are training data; their metrics are not future validation.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    RAW_ROOT,
    SOURCE,
    SOURCE_DONE_SHA,
    Hashes,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)
from verify_latent_trajectory_adapt_20261105 import expected_queries


ROOT = Path(__file__).resolve().parent
PRIOR_RUN = ROOT / "probe_outputs/latent_trajectory_adapt360_20261105_v1"
PRIOR_AUDIT = (
    ROOT / "probe_outputs/latent_trajectory_adapt360_verification_20261105_v1.json"
)
PRIOR_AUDIT_SHA = "0c5c748d879d4739e19538064acf35b4020ccbb7c1727bf34bf4e5cae597b92d"
N_SHOTS, LABEL_BUDGET, FIRST, STRIDE = 360, 20, 135, 32
ROUNDS, STEPS_PER_ROUND, WINDOWS_PER_SHOT = 3, 12, 8
ATOL = 4e-6
TRAIN_ARMS = ("fixed_peak", "moving_peak")
ARMS = ["static", "history20_reference", *TRAIN_ARMS, "fixed_stale", "moving_stale"]
ARTIFACTS = {
    "protocol.json",
    "per_shot.json",
    "summary.json",
    "fit_receipt.json",
    "mining_curves.npz",
    "fit_fixed_peak.npz",
    "fit_moving_peak.npz",
}


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_curve(p, times, label):
    need(
        p.ndim == 1
        and p.shape == times.shape
        and np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all(),
        f"invalid probability curve: {label}",
    )


def parity(p, reference, threshold, label, *, require_crossings=True):
    """Use bounded numerical error and, for online references, exact crossings."""
    need(
        p.shape == reference.shape and np.isfinite(reference).all(),
        f"{label} parity shape",
    )
    error = float(np.max(np.abs(p.astype(float) - reference.astype(float)), initial=0))
    need(
        error <= ATOL
        and (
            not require_crossings
            or np.array_equal(p >= threshold, reference >= threshold)
        ),
        f"{label} parity",
    )
    return error


def check_input_hashes(cfg, hashes):
    inputs = cfg["input_sha256"]
    need(isinstance(inputs, dict) and inputs, "missing input hash bindings")
    for name, digest in inputs.items():
        need(
            isinstance(name, str) and Path(name).is_absolute() and valid_sha(digest),
            "invalid input hash binding",
        )
        need(hashes(name) == digest, f"input hash changed: {name}")
    bound = {Path(name).resolve() for name in inputs}
    needed = {
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
        PRIOR_RUN / "protocol.json",
        PRIOR_RUN / "completed.json",
        PRIOR_RUN / "fit20.npz",
    }
    need(
        {p.resolve() for p in needed} <= bound,
        "locked source or prior fit not bound as input",
    )


def check_fit_receipt(fit, order, run, old_done, hashes):
    need(
        fit["ids"] == order[:LABEL_BUDGET],
        "fit ids are not the ordered historical prefix",
    )
    initial = fit["initial_fit_sha256"]
    need(
        valid_sha(initial)
        and initial == hashes(PRIOR_RUN / "fit20.npz")
        and initial == old_done["artifact_sha256"]["fit20.npz"],
        "initial fit is not the audited previous fit20",
    )
    with np.load(PRIOR_RUN / "fit20.npz", allow_pickle=False) as archive:
        initial_state = {
            key.removeprefix("history."): archive[key]
            for key in archive.files
            if key.startswith("history.")
        }
    need(
        initial_state
        and all(
            v.dtype.kind == "f" and np.isfinite(v).all() for v in initial_state.values()
        ),
        "invalid initial fit state",
    )
    need(set(fit["arms"]) == set(TRAIN_ARMS), "fit arm population")
    for arm in TRAIN_ARMS:
        record = fit["arms"][arm]
        need(record["fit_sha256"] == hashes(run / f"fit_{arm}.npz"), f"fit hash: {arm}")
        with np.load(run / f"fit_{arm}.npz", allow_pickle=False) as archive:
            need(
                set(archive.files) == set(initial_state)
                and len(archive.files) == len(initial_state),
                f"fit state keys: {arm}",
            )
            for key, original in initial_state.items():
                value = archive[key]
                need(
                    value.shape == original.shape
                    and value.dtype == original.dtype
                    and np.isfinite(value).all(),
                    f"invalid fit tensor: {arm}/{key}",
                )
                if key in ("mean", "scale"):
                    need(
                        np.array_equal(value, original),
                        f"fit normalization changed: {arm}/{key}",
                    )
        need(
            type(record["steps"]) is int
            and record["steps"] == ROUNDS * STEPS_PER_ROUND,
            f"fit steps: {arm}",
        )
        need(
            isinstance(record["rounds"], list) and len(record["rounds"]) == ROUNDS,
            f"fit rounds: {arm}",
        )
        for r, item in enumerate(record["rounds"], 1):
            need(
                type(item["round"]) is int
                and item["round"] == r
                and type(item["steps"]) is int
                and item["steps"] == STEPS_PER_ROUND,
                f"fit round/steps: {arm}/{r}",
            )
            selection = item["selection"]
            need(
                isinstance(selection, dict)
                and set(selection) == {str(s) for s in order[:LABEL_BUDGET]},
                f"fit selection population: {arm}/{r}",
            )
            for q in selection.values():
                need(
                    isinstance(q, list)
                    and len(q) == WINDOWS_PER_SHOT
                    and all(type(i) is int for i in q),
                    f"fit selection window population: {arm}/{r}",
                )
    return {a: fit["arms"][a]["fit_sha256"] for a in TRAIN_ARMS}


def verify_mining(run, fit, order, truth_by_id, times_by_id, source_done, tau, hashes):
    """Reconstruct each positive/negative query from the pre-round saved curve."""
    population = order[:LABEL_BUDGET]
    expected_keys = {f"{sid}_native" for sid in population}
    expected_keys |= {
        f"{sid}_{arm}_{stage}"
        for sid in population
        for arm in TRAIN_ARMS
        for stage in range(1, ROUNDS + 2)
    }
    stage_rows = {
        stage: {arm: [] for arm in TRAIN_ARMS} for stage in range(1, ROUNDS + 2)
    }
    native_rows, selected, mapped_max_error, initial_max_error = [], {}, 0.0, 0.0
    mapped_crossing_differences = initial_crossing_differences = 0
    with np.load(run / "mining_curves.npz", allow_pickle=False) as data:
        need(
            set(data.files) == expected_keys and len(data.files) == len(expected_keys),
            "mining curve roster",
        )
        for rank, sid in enumerate(population, 1):
            times, truth = times_by_id[sid], truth_by_id[sid]
            native = data[f"{sid}_native"]
            check_curve(native, times, f"{sid}/native mining")
            source_path = SOURCE / "shots" / f"{rank:04d}_{sid}.npz"
            source_seal = source_done["seals"][rank - 1]
            need(
                source_seal["shot"] == sid
                and source_seal["current_outcome_used_in_prediction"] is False
                and hashes(source_path) == source_seal["curve_sha256"],
                "mapped native source seal",
            )
            with np.load(source_path, allow_pickle=False) as archive:
                need(
                    np.array_equal(archive["times"], times),
                    "mapped native source clock",
                )
                old_native = archive["retrospective_map_all"]
                mapped_max_error = max(
                    mapped_max_error,
                    parity(
                        native,
                        old_native,
                        tau,
                        "mapped native",
                        require_crossings=False,
                    ),
                )
                mapped_crossing_differences += int(
                    np.count_nonzero((native >= tau) != (old_native >= tau))
                )
            native_rows.append(score(times, native, truth, tau))
            initial = [data[f"{sid}_{arm}_1"] for arm in TRAIN_ARMS]
            initial_max_error = max(
                initial_max_error,
                parity(
                    initial[0],
                    initial[1],
                    tau,
                    "initial round-one",
                    require_crossings=False,
                ),
            )
            initial_crossing_differences += int(
                np.count_nonzero((initial[0] >= tau) != (initial[1] >= tau))
            )
            for arm in TRAIN_ARMS:
                for stage in range(1, ROUNDS + 2):
                    curve = data[f"{sid}_{arm}_{stage}"]
                    check_curve(curve, times, f"{sid}/{arm}/{stage}")
                    need(
                        np.array_equal(curve[:FIRST], native[:FIRST]),
                        "mining curve changed before active prefix",
                    )
                    stage_rows[stage][arm].append(score(times, curve, truth, tau))
                    if stage <= ROUNDS:
                        target = native if arm == "fixed_peak" else curve
                        q, _ = expected_queries(
                            times, target, truth["disruptive"], truth["down"], FIRST
                        )
                        saved = fit["arms"][arm]["rounds"][stage - 1]["selection"][
                            str(sid)
                        ]
                        need(
                            q.tolist() == saved
                            and (q >= FIRST).all()
                            and (q < len(times)).all(),
                            f"selection query mismatch: {sid}/{arm}/{stage}",
                        )
                        selected[(sid, arm, stage)] = q
            for stage in range(1, ROUNDS + 1):
                fixed, moving = (selected[(sid, arm, stage)] for arm in TRAIN_ARMS)
                need(
                    len(fixed) == len(moving) == WINDOWS_PER_SHOT,
                    "unequal fit selection population",
                )
                if truth["disruptive"]:
                    need(
                        np.array_equal(fixed, moving),
                        "positive selection differs between arms",
                    )
    metrics = {
        f"round_{stage}": {arm: counts(rows) for arm, rows in stage_rows[stage].items()}
        for stage in range(1, ROUNDS + 1)
    }
    metrics["final"] = {
        arm: counts(rows) for arm, rows in stage_rows[ROUNDS + 1].items()
    }
    metrics["native"] = counts(native_rows)
    need(
        metrics["native"]["n_disruptive"] == 8 and metrics["native"]["n_safe"] == 12,
        "historical HDF fit population",
    )
    parity_detail = dict(
        mapped_native_max_abs_error=mapped_max_error,
        mapped_native_crossing_differences=mapped_crossing_differences,
        initial_curve_max_abs_error=initial_max_error,
        initial_curve_crossing_differences=initial_crossing_differences,
    )
    return metrics, parity_detail


def verify(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    hashes = Hashes()
    completion = run / "completed.json"
    done = read(completion)
    need(done["status"] == "COMPLETE_MOVING_PEAK_REPLAY", "COMPLETE replay required")
    completion_sha = hashes(completion)
    artifacts = done["artifact_sha256"]
    need(
        isinstance(artifacts, dict) and ARTIFACTS <= set(artifacts),
        "artifact seal roster",
    )
    for name, expected in artifacts.items():
        need(
            isinstance(name, str) and Path(name).name == name and valid_sha(expected),
            "invalid artifact seal",
        )
        need(hashes(run / name) == expected, f"artifact hash: {name}")
        need(
            (run / name).stat().st_mtime_ns <= completion.stat().st_mtime_ns,
            "artifact after completion",
        )
    cfg, saved, summary, fit = [
        read(run / name)
        for name in (
            "protocol.json",
            "per_shot.json",
            "summary.json",
            "fit_receipt.json",
        )
    ]
    check_input_hashes(cfg, hashes)
    need(
        hashes(SOURCE / "completed.json") == SOURCE_DONE_SHA,
        "locked source receipt changed",
    )
    source_done, source = (
        read(SOURCE / "completed.json"),
        read(SOURCE / "protocol.json"),
    )
    need(
        hashes(SOURCE / "protocol.json")
        == source_done["artifact_sha256"]["protocol.json"],
        "locked source protocol changed",
    )
    need(hashes(PRIOR_AUDIT) == PRIOR_AUDIT_SHA, "previous independent audit changed")
    old_audit = read(PRIOR_AUDIT)
    old_done, old_cfg = (
        read(PRIOR_RUN / "completed.json"),
        read(PRIOR_RUN / "protocol.json"),
    )
    need(
        old_audit["status"] == "PASS_LATENT_ADAPTATION_INDEPENDENT_RECOUNT"
        and old_audit["n_shots"] == N_SHOTS
        and old_audit["independent_future_validation"] is False
        and hashes(PRIOR_RUN / "completed.json")
        == old_audit["producer_completion_sha256"]
        and old_done["status"] == "COMPLETE_LATENT_TRAJECTORY_ADAPT_DEVELOPMENT"
        and old_done["n_shots"] == N_SHOTS
        and old_done["n_curves"] == N_SHOTS * 7
        and hashes(PRIOR_RUN / "protocol.json")
        == old_done["artifact_sha256"]["protocol.json"],
        "prior independent replay audit binding",
    )
    order, tau = cfg["order"], cfg["threshold"]
    need(
        isinstance(order, list)
        and len(order) == N_SHOTS
        and all(type(s) is int and s > 0 for s in order)
        and len(set(order)) == N_SHOTS,
        "invalid fixed order",
    )
    need(
        order == [r["shot"] for r in source["order"]] == old_cfg["order"],
        "locked fixed order changed",
    )
    need(
        type(tau) in (int, float)
        and np.isfinite(tau)
        and 0 < tau < 1
        and tau == source["threshold"] == old_cfg["threshold"],
        "locked threshold",
    )
    need(
        cfg["raw_sha256"] == source["raw_sha256"] == old_cfg["raw_sha256"]
        and set(cfg["raw_sha256"]) == {str(s) for s in order},
        "locked raw hash roster",
    )
    need(
        cfg["arms"] == ARMS
        and cfg["first_anchor"] == FIRST
        and cfg["stride"] == STRIDE
        and cfg["label_budget"] == LABEL_BUDGET
        and cfg["rounds"] == ROUNDS
        and cfg["steps_per_arm"] == ROUNDS * STEPS_PER_ROUND
        and cfg["independent_future_validation"] is False,
        "protocol scope or maturity",
    )
    need(
        done["n_shots"] == N_SHOTS
        and done["n_curves"] == N_SHOTS * len(ARMS)
        and len(done["sealed"])
        == len(old_done["sealed"])
        == len(source_done["seals"])
        == N_SHOTS,
        "completion population",
    )
    need(
        set(saved) == set(ARMS) and all(len(saved[a]) == N_SHOTS for a in ARMS),
        "per-shot population",
    )
    fit_hashes = check_fit_receipt(fit, order, run, old_done, hashes)
    if "initial_fit_sha256" in cfg:
        need(
            cfg["initial_fit_sha256"] == fit["initial_fit_sha256"],
            "protocol initial fit mismatch",
        )
    observed, truth_by_id, times_by_id = {a: [] for a in ARMS}, {}, {}
    max_error = {"static": 0.0, "history20_reference": 0.0}
    previous_reveal = None
    for rank, (sid, binding) in enumerate(zip(order, done["sealed"], strict=True), 1):
        stem = run / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        seal, reveal = (
            stem.with_name(stem.name + suffix)
            for suffix in ("_seal.json", "_revealed.json")
        )
        need(
            binding
            == dict(
                shot=sid,
                curve_sha256=hashes(curve),
                seal_sha256=hashes(seal),
                revealed_sha256=hashes(reveal),
            ),
            "shot seal hash",
        )
        note, label = read(seal), read(reveal)
        need(
            (note["rank"], note["shot"], label["shot"]) == (rank, sid, sid)
            and note["curve_sha256"] == binding["curve_sha256"]
            and label["after_seal_sha256"] == binding["seal_sha256"]
            and note["current_outcome_used_in_prediction"] is False,
            "seal/reveal identity or boundary",
        )
        need(
            curve.stat().st_mtime_ns
            <= seal.stat().st_mtime_ns
            <= reveal.stat().st_mtime_ns
            <= completion.stat().st_mtime_ns,
            "seal/reveal order",
        )
        if previous_reveal is not None:
            need(
                previous_reveal.stat().st_mtime_ns <= curve.stat().st_mtime_ns,
                "shot reveal/next prediction order",
            )
        need(
            note["active_fit_sha256"] == ({} if rank <= LABEL_BUDGET else fit_hashes),
            "active fit mismatch",
        )
        if rank == LABEL_BUDGET:
            next_curve = run / "shots" / f"{rank + 1:04d}_{order[rank]}.npz"
            for name in (
                "mining_curves.npz",
                "fit_fixed_peak.npz",
                "fit_moving_peak.npz",
                "fit_receipt.json",
            ):
                need(
                    reveal.stat().st_mtime_ns
                    <= (run / name).stat().st_mtime_ns
                    <= next_curve.stat().st_mtime_ns,
                    f"fit/reveal order: {name}",
                )
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(
            hashes(raw) == note["raw_sha256"] == cfg["raw_sha256"][str(sid)], "raw hash"
        )
        truth, times = raw_truth(raw, sid)
        hashes(raw)
        if rank <= LABEL_BUDGET:
            truth_by_id[sid], times_by_id[sid] = truth, times
        old = PRIOR_RUN / "shots" / curve.name
        old_binding = old_done["sealed"][rank - 1]
        need(
            old_binding["shot"] == sid and hashes(old) == old_binding["curve_sha256"],
            "prior curve seal",
        )
        with (
            np.load(curve, allow_pickle=False) as f,
            np.load(old, allow_pickle=False) as previous,
        ):
            need(
                set(f.files) == {"times", *ARMS} and len(f.files) == 1 + len(ARMS),
                "shot curve keys",
            )
            need(
                np.array_equal(f["times"], times)
                and np.array_equal(previous["times"], times),
                "HDF signal clock",
            )
            for arm in ARMS:
                p = f[arm]
                check_curve(p, times, f"{sid}/{arm}")
                if arm != "static":
                    end = len(times) if rank <= LABEL_BUDGET else FIRST
                    need(
                        np.array_equal(p[:end], f["static"][:end]),
                        "inactive or pre-maturity static prefix",
                    )
                row = score(times, p, truth, tau)
                need(
                    row == label["rows"][arm] == saved[arm][rank - 1],
                    f"HDF row recount: {sid}/{arm}",
                )
                observed[arm].append(row)
            need(set(label["rows"]) == set(ARMS), "reveal arm roster")
            for arm, old_arm in (
                ("static", "static"),
                ("history20_reference", "history20"),
            ):
                max_error[arm] = max(
                    max_error[arm], parity(f[arm], previous[old_arm], tau, arm)
                )
        previous_reveal = reveal
        if rank % 25 == 0:
            print("verified", rank, "of", N_SHOTS, flush=True)
    for name, subset in (
        ("full360", slice(None)),
        ("after20", slice(LABEL_BUDGET, None)),
    ):
        group = {a: rows[subset] for a, rows in observed.items()}
        expected = dict(
            metrics={a: counts(rows) for a, rows in group.items()},
            pairs={
                a: pair(rows, group["static"])
                for a, rows in group.items()
                if a != "static"
            },
            pairs_capped={
                a: pair(rows, group["static"], True)
                for a, rows in group.items()
                if a != "static"
            },
        )
        need(summary[name] == expected, f"summary recount: {name}")
    need(set(summary) == {"full360", "after20"}, "summary sections")
    training, mining_parity = verify_mining(
        run, fit, order, truth_by_id, times_by_id, source_done, tau, hashes
    )
    compared, harmful = {}, {}
    for name, subset in (
        ("full360", slice(None)),
        ("after20", slice(LABEL_BUDGET, None)),
    ):
        moving, fixed = observed["moving_peak"][subset], observed["fixed_peak"][subset]
        compared[name] = {
            "full": pair(moving, fixed),
            "capped": pair(moving, fixed, True),
        }
        harmful_ids = set(
            compared[name]["full"]["fp_added"] + compared[name]["full"]["tp_lost"]
        )
        harmful[name] = [
            dict(
                shot=m["shot"],
                fixed_status=f["status"],
                moving_status=m["status"],
                fixed_first_index=f["first_index"],
                moving_first_index=m["first_index"],
                fixed_lead_ms=f["lead_ms"],
                moving_lead_ms=m["lead_ms"],
            )
            for m, f in zip(moving, fixed, strict=True)
            if m["shot"] in harmful_ids
        ]
    verifier_sha = hashes(Path(__file__))
    hashes.check()
    report = dict(
        status="PASS_MOVING_PEAK_INDEPENDENT_RECOUNT",
        n_shots=N_SHOTS,
        n_curves=N_SHOTS * len(ARMS),
        producer_completion_sha256=completion_sha,
        verifier_sha256=verifier_sha,
        prior_audit_sha256=PRIOR_AUDIT_SHA,
        independent_future_validation=False,
        summary=summary,
        parity_max_abs_error=max_error,
        mining_parity=mining_parity,
        training_curve_metrics=training,
        moving_vs_fixed=compared,
        harmful_cases=harmful,
        training_metric_scope="same first-20 fitting shots; round curves are pre-round, final is after all three rounds",
        limitations=[
            "File hashes and timestamps support ordering but cannot prove internal prediction causality or actual optimizer execution.",
            "Saved curves are not independently recomputed from model weights; initial parameters and training selections are established from receipts and curve evidence, not a training replay.",
            "First-20 training metrics are in-sample; after20 is a fixed-order development comparison, not independent future validation.",
        ],
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(report["status"], N_SHOTS, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    verify(args.run, args.output)


if __name__ == "__main__":
    main()
