"""Independent CPU records audit; explicit --output execution only.

Uses saved curves, arrays and receipts, plus HDF5 IsDisrupt/DownTime metadata.
No runner classifier/gates, neural model, or raw signal loader is imported.
Armijo scalar/state receipt semantics are delegated to receipt_verify.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from ..innovation_metric_20261102 import verify as cpu

ROOT = cpu.ROOT
REFERENCE = ROOT / "probe_outputs/full_path_forecast_controls360_v1"
TRAINING = ROOT / "probe_outputs/armijo_heads20_v1"
REFERENCE_SHA = "e63f2a10d4f917279f0e47d38d3851b5cb93bba5b1184e5529a2cf41207b4587"
RECEIPT_VERIFIER_SHA = (
    "9e93cc46a15859cf755e0d79830315993bf6a3a68be0367e2a9f08e449f34c0c"
)
PLAN = "ARMIJO_FEEDBACK_PLAN_20261102.md"
EVALUATION = "ARMIJO_CONTROL_EVALUATION_20261102.md"
MODULE = "experiments/risk_aligned_online/armijo_feedback_20261102/"
HEADS = ("meta", "forecast")
REUSED = dict(cold="cold", fixed_full="full_real")
FRESH = tuple(
    p + "_" + m for p in ("forecast", "meta") for m in ("real", "reverse", "hold")
)
ARMS = (*REUSED, *FRESH)
STATISTICS_SCOPE = "fresh_recomputed_only_not_historical_activity"
THRESHOLD = cpu.THRESHOLD
require, finite = cpu.require, cpu.finite
read_json, sha_file = cpu.read_json, cpu.sha_file
bitwise_equal, read_outcome = cpu.bitwise_equal, cpu.read_outcome
classify_curve, recount, paired_flips = (
    cpu.classify_curve,
    cpu.recount,
    cpu.paired_flips,
)
posthoc_caps = cpu.posthoc_caps


def verify_receipt(before, after, gradient, receipt, observed):
    """Lazy pure-CPU dependency: no duplicate Armijo implementation here."""
    from .receipt_verify import verify_receipt as audit

    return audit(before, after, gradient, receipt, observed)


def score_gates(scores):
    require(set(scores) == set(ARMS), "score arms")
    metrics = [scores[a]["full360"] for a in ARMS]
    require(
        all((m["n_disrupt"], m["n_safe"]) == (180, 180) for m in metrics),
        "balanced180/180",
    )
    require(
        all(
            type(m[k]) is int and 0 <= m[k] <= 180
            for m in metrics
            for k in ("tp", "fp")
        ),
        "integer TP/FP counts",
    )

    def net(m):
        return m["tp"] - m["fp"]

    base, real = scores["cold"]["full360"], scores["meta_real"]["full360"]
    stage = real["fp"] <= 36 and real["tp"] >= base["tp"] and net(real) > net(base)
    controls = {
        a: net(real) > net(scores[a]["full360"])
        for a in ("meta_reverse", "meta_hold", "forecast_real", "fixed_full")
    }
    forecast = {
        a: net(scores["forecast_real"]["full360"]) > net(scores[a]["full360"])
        for a in ("forecast_reverse", "forecast_hold")
    }
    return dict(
        operational_gate=stage,
        control_comparisons=controls,
        control_separation_gate=all(controls.values()),
        forecast_control_comparisons=forecast,
        forecast_control_separation_gate=all(forecast.values()),
        stage_success_pending_verification=stage and all(controls.values()),
    )


def summarize(rows):
    scores = {
        a: {
            name: recount(part)
            for name, part in (
                ("full360", rows[a]),
                ("warmup20", rows[a][:20]),
                ("suffix340", rows[a][20:]),
            )
        }
        for a in ARMS
    }
    return dict(
        metrics=scores,
        flips={a: paired_flips(rows["cold"], rows[a]) for a in ARMS[1:]},
        full360_integer_net={
            a: scores[a]["full360"]["tp"] - scores[a]["full360"]["fp"] for a in ARMS
        },
        **score_gates(scores),
    )


def validate_curves(times, curves, reference, *, warmup):
    require(
        set(curves) == set(ARMS) and bitwise_equal(times, reference["times"]),
        "curve/time roster",
    )
    require(
        times.ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    for a, values in curves.items():
        require(
            values.shape == times.shape
            and values.dtype.str == reference["cold"].dtype.str
            and np.isfinite(values).all()
            and np.all((values >= 0) & (values <= 1)),
            "curve: " + a,
        )
        require(bitwise_equal(values[:160], reference["cold"][:160]), "first160: " + a)
        if warmup:
            require(bitwise_equal(values, reference["cold"]), "warmup: " + a)
        elif a in REUSED:
            require(
                bitwise_equal(values, reference[REUSED[a]]), "reference curve: " + a
            )


def check_trace(info, arrays, length, *, index, reference_path, reference_sha):
    require(
        info.get("reset_each_shot") is True
        and info.get("temporary_parameters") == 256
        and info.get("statistics_scope") == STATISTICS_SCOPE
        and info.get("reference_curve_sha256") == reference_sha,
        "trace contract",
    )
    require(
        all(set(info[k]) == set(ARMS) for k in ("updates", "actions", "execution")),
        "trace arms",
    )
    require(
        set(arrays)
        == {a + s for a in ARMS for s in ("_before", "_after", "_gradient")},
        "array roster",
    )
    totals = {}
    for a in ARMS:
        fresh = index > 20 and a in FRESH
        expected = dict(
            kind="reused_warmup"
            if index <= 20
            else ("fresh_uncached_gpu" if fresh else "reused_reference"),
            reference=None if fresh else str(reference_path),
            reference_arm="cold" if index <= 20 else REUSED.get(a),
        )
        require(info["execution"][a] == expected, "execution/reference: " + a)
        ends = list(range(159, length, 128)) if fresh else []
        es = info["updates"][a]
        require([e["maturity"] for e in es] == ends, "update clock")
        before, after, gradient = (
            arrays[a + s] for s in ("_before", "_after", "_gradient")
        )
        require(
            before.shape == after.shape == (len(ends), 4, 64)
            and gradient.shape == (len(ends), 256),
            "state shapes",
        )
        require(
            all(
                x.dtype == np.float32 and np.isfinite(x).all()
                for x in (before, after, gradient)
            ),
            "state dtype/finite",
        )
        if ends:
            require(
                bitwise_equal(before[0], np.zeros((4, 64), np.float32))
                and bitwise_equal(after[:-1], before[1:]),
                "reset/chain",
            )
        totals[a] = dict(
            events=0, accepted=0, evaluations=0, nonzero_gradients=0, observed_scalars=0
        )
        for i, end in enumerate(ends):
            e = es[i]
            require(
                e["origin"] == end - 25 and e["first_affected"] == end + 1,
                "causal receipt clock",
            )
            for k in ("forecast_sha256", "mask_sha256", "target_sha256"):
                require(
                    isinstance(e.get(k), str) and cpu.HEX.fullmatch(e[k]),
                    "receipt hash: " + k,
                )
            require(
                type(e["observed"]) is int
                and 0 <= e["observed"] <= 550
                and type(e["gradient_computed"]) is bool,
                "observed/gradient flag",
            )
            # The independently written receipt helper owns all Armijo, norm,
            # budget, first-acceptable and rejection-state checks.
            summary = verify_receipt(
                before[i].reshape(256),
                after[i].reshape(256),
                gradient[i],
                e,
                e["observed"],
            )
            for key in totals[a]:
                totals[a][key] += summary[key]
        action = info["actions"][a]
        n, capped, rms = (
            action["endpoints"],
            action["capped_endpoints"],
            finite(action["max_rms"], "recorded RMS"),
        )
        require(
            type(n) is int
            and n == (length if fresh else 0)
            and type(capped) is int
            and 0 <= capped <= n
            and 0 <= rms <= 0.250001
            and (fresh or (capped == 0 and rms == 0)),
            "action execution",
        )
    for group in zip(*(info["updates"][a] for a in FRESH), strict=True):
        require(
            len({e["mask_sha256"] for e in group}) == 1
            and len({e["observed"] for e in group}) == 1,
            "shared mask",
        )
    for f, m in zip(
        info["updates"]["forecast_real"], info["updates"]["meta_real"], strict=True
    ):
        require(f["target_sha256"] == m["target_sha256"], "real target equality")
    return totals, info["actions"]


def check_training(protocol, done, history, cohort, warmup):
    require(
        len(cohort) == 20
        and [c["shot"] for c in cohort] == protocol["ids"]
        and sum(c["disruptive"] for c in cohort) == 8,
        "training cohort",
    )
    for c in cohort:
        times, flag, down = warmup[c["shot"]]
        require(
            c["disruptive"] is flag and c["down"] == down and c["points"] == len(times),
            "training HDF metadata",
        )
    groups = {
        flag: [c["shot"] for c in cohort if c["disruptive"] is flag]
        for flag in (True, False)
    }
    rng, support = np.random.default_rng(131), {a: 0 for a in HEADS}
    require(len(history) == 128, "128 training steps")
    for step, row in enumerate(history, 1):
        pair = [groups[b][int(rng.integers(len(groups[b])))] for b in (True, False)]
        require(
            row["step"] == step and row["sample_shots"] == pair, "seeded paired sample"
        )
        require(
            set(row["episodes"]) == set(row["gradient_norms"]) == set(HEADS),
            "two heads",
        )
        for a in HEADS:
            require(
                finite(row["gradient_norms"][a], "head gradient") >= 0, "head gradient"
            )
            es = row["episodes"][a]
            require(len(es) == 2 and [e["shot"] for e in es] == pair, "episode samples")
            for e in es:
                times, flag, down = warmup[e["shot"]]
                q = e["query"]
                require(
                    type(q) is int
                    and 0 <= q < len(times)
                    and (not flag or round((down - times[q]) * 1000, 9) > 30),
                    "training query",
                )
                maturity = 159 + ((q - 160) // 128) * 128 if q >= 160 else None
                require(
                    e["supported"] is (q >= 160) and e["maturity"] == maturity,
                    "training clock",
                )
                count = 1 + (maturity - 159) // 128 if maturity is not None else 0
                require(
                    type(e["replay_updates"]) is int and e["replay_updates"] == count,
                    "full replay",
                )
                accepted = e["accepted_path_updates"]
                require(
                    type(accepted) is int and 0 <= accepted <= count, "accepted path"
                )
                neval = e["proposal_evaluations"]
                require(
                    type(neval) is int and accepted <= neval <= 11 * count,
                    "proposal evaluations",
                )
                require(
                    type(e["label"]) is int and e["label"] == int(flag),
                    "historical shot label",
                )
                for k in ("risk_before", "risk_after", "signal", "objective"):
                    finite(e[k], "training " + k)
                if maturity is None:
                    require(
                        e["risk_before"] == e["risk_after"], "unsupported risk changed"
                    )
                support[a] += int(q >= 160)
    require(done["support_counts"] == support, "training support totals")
    return support


def check_snapshot(root, p, *, training=False):
    required = {
        PLAN,
        *(MODULE + n for n in ("__init__.py", "prepare.py", "core.py", "stream.py")),
    }
    if not training:
        required |= {EVALUATION, MODULE + "run.py"}
    declared = p["code_sha256"]
    require(
        required <= set(declared) and declared[PLAN] == p["plan_sha256"],
        "snapshot roster/plan",
    )
    if not training:
        require(declared[EVALUATION] == p["evaluation_plan_sha256"], "evaluation plan")
    plans, current = {PLAN} if training else {PLAN, EVALUATION}, {}
    archive_path = root / "code_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        es = archive.infolist()
        names = [e.filename for e in es]
        require(
            len(names) == len(set(names)) and set(names) == set(declared),
            "archive roster",
        )
        require(sum(e.file_size for e in es) <= 512 * 1024 * 1024, "archive size")
        for e in es:
            name, path = e.filename, PurePosixPath(e.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and not e.is_dir()
                and e.file_size <= 64 * 1024 * 1024,
                "archive path",
            )
            require(
                name in plans
                or (
                    path.parts[:2] == ("experiments", "risk_aligned_online")
                    and path.suffix == ".py"
                ),
                "snapshot non-code",
            )
            require(
                isinstance(declared[name], str)
                and cpu.HEX.fullmatch(declared[name])
                and hashlib.sha256(archive.read(e)).hexdigest() == declared[name],
                "archive hash",
            )
            actual = ROOT / name
            if actual.is_file():
                require(
                    actual.resolve().is_relative_to(ROOT.resolve()), "current code path"
                )
                current[name] = sha_file(actual)
            else:
                current[name] = None
    return dict(
        archive_bytes_sha256=sha_file(archive_path),
        archive_entries_verified=len(es),
        actual_current_sha256=current,
        current_drift={
            n: dict(snapshot=declared[n], actual_current=h)
            for n, h in current.items()
            if h != declared[n]
        },
    )


def check_protocol(p, reference):
    fixed = dict(
        scope="DEVELOPMENT_COMPLETE360_BOUNDED_ARMIJO_MATCHED_FEEDBACK",
        arms=list(ARMS),
        reused_arms=REUSED,
        fresh_arms=list(FRESH),
        reference_completed_sha256=REFERENCE_SHA,
        threshold=THRESHOLD,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        temporary_parameters=256,
        reset_each_shot=True,
        max_proposal_evaluations=11,
        armijo_c=1e-4,
        metric_increment_hidden_rms_cap=0.05,
        hidden_rms_cap=0.25,
        head_training_shots=20,
        parameters_per_head=575,
        head_optimizer_steps=128,
        suffix_head_updates=0,
        suffix_inter_supervision=0,
        warmup_cost_included=True,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        J_comparison_basis="integer_tp_minus_fp_fixed_180_180",
        no_outcome_input_to_simulate=True,
        independent_future_confirmation=False,
        real_time_latency_verified=False,
        equal_floating_point_compute_budget_claimed=False,
    )
    for k, value in fixed.items():
        require(p.get(k) == value, "run contract: " + k)
    require(
        Path(p["reference"]).resolve() == REFERENCE.resolve()
        and Path(p["training"]).resolve() == TRAINING.resolve(),
        "reference/training paths",
    )
    ids = [e["shot"] for e in p["order"]]
    require(
        p["order"] == reference["order"]
        and len(ids) == len(set(ids)) == 360
        and all(type(s) is int and s > 0 for s in ids),
        "ordered360 roster",
    )
    for k in (
        "source_sha256",
        "cold_state_sha256",
        "input_map_sha256",
        "input_hashes_sha256",
        "threshold",
    ):
        require(p[k] == reference[k], "reference contract: " + k)
    require(
        Path(p["calibration_reference"]).resolve()
        == Path(reference["calibration_reference"]).resolve(),
        "calibration path",
    )
    return ids


def check_training_contract(done, p, ids, run):
    completion = dict(
        status="COMPLETED",
        head_optimizer_steps=128,
        number_heads=2,
        ids=ids,
        outcome_reads=20,
        suffix_shots_read=0,
        source_tensors_unchanged=True,
        online_gain_measured=False,
        independent_future_confirmation=False,
    )
    for k, value in completion.items():
        require(done.get(k) == value, "training completion: " + k)
    fixed = dict(
        scope="ENDED20_ARMIJO_FULL_PATH_MATCHED_TRAINING",
        heads=list(HEADS),
        ids=ids,
        forecast_parameters_per_head=575,
        steps=128,
        optimizer="Adam",
        learning_rate=0.001,
        gradient_clip=1.0,
        seed=131,
        initial_head="zero",
        risk_meta_gradient_horizon="all_matured_updates_before_fixed_query",
        diagnostic_incoming_state_detached=True,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        diagnostic_forecast_points=25,
        metric_learning_rate=1.0,
        relative_metric_damping=0.01,
        absolute_metric_damping=1e-8,
        metric_increment_hidden_rms_cap=0.05,
        total_hidden_rms_cap=0.25,
        state_anchor=0.01,
        armijo_c=1e-4,
        alpha_reduction=0.5,
        max_proposal_evaluations=11,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        reset_each_shot=True,
        warmup_cost_retained=True,
        independent_confirmation=False,
    )
    for k, value in fixed.items():
        require(p.get(k) == value, "training protocol: " + k)
    require(
        all(
            p[k] == run[k]
            for k in ("source_sha256", "cold_state_sha256", "input_map_sha256")
        )
        and done["plan_sha256"] == p["plan_sha256"] == run["plan_sha256"],
        "training parents/plan",
    )


def check_bound_files(p):
    paths = [
        ROOT / PLAN,
        ROOT / EVALUATION,
        Path(p["source_checkpoint"]),
        Path(p["cold_state"]),
    ]
    paths += [
        REFERENCE / n
        for n in (
            "completed.json",
            "protocol.json",
            "verification_records.json",
            "verification_neural.json",
        )
    ]
    paths += [
        TRAINING / n
        for n in (
            "completed.json",
            "protocol.json",
            "heads.json",
            "history.json",
            "training_cohort.json",
        )
    ]
    paths += [
        Path(p["calibration_reference"]) / n
        for n in ("input_map.json", "input_hashes.json")
    ]
    declared = p["bound_files_sha256"]
    require(set(declared) == {str(path) for path in paths}, "bound file roster")
    for path, digest in declared.items():
        require(
            isinstance(digest, str)
            and cpu.HEX.fullmatch(digest)
            and sha_file(Path(path)) == digest,
            "bound file hash: " + path,
        )
    require(
        declared[str(ROOT / PLAN)] == p["plan_sha256"]
        and declared[str(ROOT / EVALUATION)] == p["evaluation_plan_sha256"],
        "bound plans",
    )
    return dict(declared)


def verify(output):
    """Read-only until returning; PASS is record consistency, not a science gate."""
    output = Path(output)
    done = read_json(output / "completed.json")
    require(
        done.get("status") == "COMPLETED", "COMPLETED run required before other reads"
    )
    for name, key in (
        ("protocol.json", "protocol_sha256"),
        ("code_snapshot.zip", "code_snapshot_sha256"),
        ("per_shot.json", "per_shot_sha256"),
        ("shot_seals.json", "shot_seals_sha256"),
    ):
        require(sha_file(output / name) == done[key], "completion hash: " + name)
    p = read_json(output / "protocol.json")
    reference, rd, records, neural = (
        read_json(REFERENCE / n)
        for n in (
            "protocol.json",
            "completed.json",
            "verification_records.json",
            "verification_neural.json",
        )
    )
    require(
        sha_file(REFERENCE / "completed.json") == REFERENCE_SHA
        and rd["status"] == "COMPLETED"
        and records["status"] == neural["status"] == "PASS"
        and records["hashes"]["completed_sha256"]
        == neural["completed_sha256"]
        == REFERENCE_SHA
        and sha_file(REFERENCE / "protocol.json")
        == rd["protocol_sha256"]
        == records["hashes"]["protocol_sha256"],
        "reference verification binding",
    )
    require(
        records["cases"] == 360
        and records["curves"] == 2880
        and neural["fresh_curves_replayed"] == 24
        and neural["referenced_curves_checked"] == 8
        and neural["actual_layer_hook_batches"] == 96
        and neural["fixed_preselected_shots"] == [74087, 73403, 63656, 62573]
        and records["metrics"] == rd["metrics"],
        "reference verification scope",
    )
    ids = check_protocol(p, reference)
    cpu.check_model_files(p, reference)
    approved = (
        ROOT.parent
        / "Fuison/experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
    )
    require(
        Path(p["source_checkpoint"]).resolve() == approved.resolve(),
        "approved shared source",
    )
    bound = check_bound_files(p)
    calibration = Path(p["calibration_reference"])
    require(
        bound[str(calibration / "input_map.json")] == p["input_map_sha256"]
        and bound[str(calibration / "input_hashes.json")] == p["input_hashes_sha256"]
        and read_json(calibration / "input_map.json")["calibration_ids"] == ids[:20]
        and set(read_json(calibration / "input_hashes.json")) == {str(s) for s in ids},
        "input contract and known20",
    )
    td, tp = (read_json(TRAINING / n) for n in ("completed.json", "protocol.json"))
    require(
        bound[str(TRAINING / "completed.json")] == p["training_completed_sha256"]
        and bound[str(TRAINING / "heads.json")] == td["heads_sha256"]
        and bound[str(TRAINING / "history.json")] == td["history_sha256"],
        "training artifact hashes",
    )
    check_training_contract(td, tp, ids[:20], p)
    cpu.check_heads(TRAINING / "heads.json", HEADS)
    snapshots = dict(
        run=check_snapshot(output, p),
        training=check_snapshot(TRAINING, tp, training=True),
    )
    receipt_path = Path(__file__).with_name("receipt_verify.py")
    receipt_hash = sha_file(receipt_path)
    require(receipt_hash == RECEIPT_VERIFIER_SHA, "approved CPU receipt verifier hash")
    helper_hash = sha_file(Path(cpu.__file__))
    saved, seals = (read_json(output / n) for n in ("per_shot.json", "shot_seals.json"))
    rr = records["hashes"]["shot_files"]
    require(
        set(saved) == set(ARMS)
        and all(len(saved[a]) == 360 for a in ARMS)
        and len(seals) == len(rr) == 360,
        "per-shot roster",
    )
    rows, maxima, warmup, shot_hashes = (
        {a: [] for a in ARMS},
        {a: [] for a in ARMS},
        {},
        [],
    )
    totals = {
        a: dict(
            events=0, accepted=0, evaluations=0, nonzero_gradients=0, observed_scalars=0
        )
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, sid in enumerate(ids, 1):
        stem = f"{index:04d}_{sid}"
        base, refbase = output / "shots" / stem, REFERENCE / "shots" / stem
        curve, state, meta, sealpath = (
            Path(str(base) + s)
            for s in (".npz", "_states.npz", "_updates.json", "_seal.json")
        )
        refcurve, refsealpath = (
            Path(str(refbase) + ".npz"),
            Path(str(refbase) + "_seal.json"),
        )
        seal, refseal, record = (
            read_json(sealpath),
            read_json(refsealpath),
            rr[index - 1],
        )
        ch, sh, mh, seh = (sha_file(path) for path in (curve, state, meta, sealpath))
        cpu.check_shot_seal(index, sid, seal, seals[index - 1], seh, ch, sh, mh)
        require(
            record["shot"] == refseal["shot"] == sid
            and record["index"] == refseal["index"] == index
            and sha_file(refsealpath) == record["seal_sha256"]
            and refseal["current_outcome_used_in_prediction"] is False
            and sha_file(refcurve) == refseal["sha256"] == record["curve_sha256"],
            "reference shot seal",
        )
        with np.load(curve, allow_pickle=False) as z:
            require(set(z.files) == {"times", *ARMS}, "curve keys")
            data = {k: z[k] for k in z.files}
        with np.load(refcurve, allow_pickle=False) as z:
            ref = {k: z[k] for k in ("times", "cold", "full_real")}
        times = data["times"]
        validate_curves(times, {a: data[a] for a in ARMS}, ref, warmup=index <= 20)
        with np.load(state, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        ts, ac = check_trace(
            read_json(meta),
            arrays,
            len(times),
            index=index,
            reference_path=refcurve,
            reference_sha=record["curve_sha256"],
        )
        flag, down = read_outcome(sid)
        if index <= 20:
            warmup[sid] = (times, flag, down)
        eligible = (
            np.round((down - times) * 1000, 9) > 30
            if flag
            else np.ones(len(times), bool)
        )
        for a in ARMS:
            scores = data[a]
            row = dict(
                shot=sid,
                disruptive=flag,
                **classify_curve(times, scores, flag, down, THRESHOLD),
            )
            require(
                row == saved[a][index - 1], "independent scoring: " + stem + " " + a
            )
            rows[a].append(row)
            maxima[a].append(
                float(scores[eligible].max()) if eligible.any() else -np.inf
            )
            for key in totals[a]:
                totals[a][key] += ts[a][key]
            for key in ("endpoints", "capped_endpoints"):
                actions[a][key] += ac[a][key]
            actions[a]["max_rms"] = max(actions[a]["max_rms"], ac[a]["max_rms"])
        shot_hashes.append(
            dict(
                shot=sid,
                index=index,
                curve_sha256=ch,
                states_sha256=sh,
                updates_sha256=mh,
                seal_sha256=seh,
                reference_curve_sha256=record["curve_sha256"],
            )
        )
    support = check_training(
        tp,
        td,
        read_json(TRAINING / "history.json"),
        read_json(TRAINING / "training_cohort.json"),
        warmup,
    )
    summary = summarize(rows)
    require(
        all(done.get(k) == value for k, value in summary.items()),
        "24-part metrics/flips/integer gates",
    )
    require(
        done["recomputed_update_statistics"] == totals
        and done["recomputed_action_statistics"] == actions,
        "fresh receipt/action totals",
    )
    fixed = dict(
        statistics_scope=STATISTICS_SCOPE,
        total_curves=2880,
        curve_execution_counts=dict(
            fresh_suffix=2040, copied_preparation=120, referenced_parent=720
        ),
        source_tensors_unchanged=True,
        heads_unchanged=True,
        method_promoted=False,
        scientific_goal_achieved=False,
        independent_future_confirmation=False,
    )
    require(
        all(done.get(k) == value for k, value in fixed.items()),
        "completion flags/execution counts",
    )
    require(
        sha_file(receipt_path) == receipt_hash
        and sha_file(Path(cpu.__file__)) == helper_hash,
        "audit dependency changed during verification",
    )
    flags = [r["disruptive"] for r in rows["cold"]]
    return dict(
        status="PASS",
        scope="RECORDED_DATA_CONSISTENCY_ONLY_NOT_INDEPENDENT_NEURAL_REPLAY",
        cases=360,
        curves=2880,
        metadata_outcomes_read=360,
        raw_signal_datasets_read=0,
        raw_hdf5_file_hashes_recomputed=False,
        threshold=THRESHOLD,
        **summary,
        **fixed,
        recomputed_update_statistics=totals,
        recomputed_action_statistics=actions,
        training_supported_episodes=support,
        posthoc_scope="ALL360_REVEALED_OUTCOMES_DEVELOPMENT_FRONTIER_NOT_DEPLOYED",
        posthoc_workpoints={a: posthoc_caps(maxima[a], flags) for a in ARMS},
        hashes=dict(
            completed_sha256=sha_file(output / "completed.json"),
            protocol_sha256=done["protocol_sha256"],
            per_shot_sha256=done["per_shot_sha256"],
            shot_seals_sha256=done["shot_seals_sha256"],
            reference_completed_sha256=REFERENCE_SHA,
            training_completed_sha256=p["training_completed_sha256"],
            heads_sha256=td["heads_sha256"],
            history_sha256=td["history_sha256"],
            verifier_sha256=sha_file(Path(__file__)),
            pure_helper_sha256=helper_hash,
            receipt_verifier_sha256=receipt_hash,
            bound_files_sha256=bound,
            snapshots=snapshots,
            shot_files=shot_hashes,
        ),
        limitations=[
            "Archived-arm empty new receipts/actions mean no execution this run, not no historical adaptation.",
            "Receipt loss values and Armijo decisions are checked algebraically, not recomputed through the neural model.",
            "The unrecorded raw metric direction prevents independently recovering rejected-step directional derivatives.",
            "No raw signals, current-mask reconstruction, independent neural/training replay, or critical-maximum verification.",
            "Shapes, norms and receipts cannot independently certify actual total/incremental hidden RMS.",
            "HDF5 outcome metadata is authoritative; raw HDF5 hashes are not independently authenticated here.",
            "Posthoc frontiers use all 360 revealed outcomes and do not change the deployed threshold.",
            "Matched label/parameter/optimizer-step budgets do not imply matched floating-point compute.",
            "No physical mechanism, independent future validation, real-time latency, or scientific-goal success follows from PASS.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output / "verification_records.json"
    if path.exists():
        raise FileExistsError(path)
    result = verify(args.output)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({k: result[k] for k in ("status", "cases", "curves")}))


if __name__ == "__main__":
    main()
