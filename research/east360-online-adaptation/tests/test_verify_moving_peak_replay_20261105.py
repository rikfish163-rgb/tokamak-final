"""Synthetic CPU-only tests. No research-cohort HDF5 files are opened."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import verify_moving_peak_replay_20261105 as audit


ARMS = [
    "static",
    "history20_reference",
    "fixed_peak",
    "moving_peak",
    "fixed_stale",
    "moving_stale",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def get(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shot_row(sid, flag, down, t, p):
    hits = [
        i
        for i, v in enumerate(p)
        if v >= 0.5 and (not flag or (down - t[i]) * 1000 >= -1e-6)
    ]
    first = hits[0] if hits else None
    lead = round(float((down - t[first]) * 1000), 9) if flag and hits else None
    status = (
        ("miss" if first is None else "tp" if lead > 30 else "tardy")
        if flag
        else ("fp" if hits else "tn")
    )
    return dict(
        shot=sid,
        disruptive=flag,
        down=down,
        status=status,
        first_index=first,
        lead_ms=lead,
    )


def summary_for(rows):
    # Producer-compatible independent expected aggregates for the synthetic fixture.
    result = {}
    for name, sl in (("full360", slice(None)), ("after20", slice(20, None))):
        group = {k: v[sl] for k, v in rows.items()}
        result[name] = {
            "metrics": {k: audit.counts(v) for k, v in group.items()},
            "pairs": {
                k: audit.pair(v, group["static"])
                for k, v in group.items()
                if k != "static"
            },
            "pairs_capped": {
                k: audit.pair(v, group["static"], True)
                for k, v in group.items()
                if k != "static"
            },
        }
    return result


def fixture_run(tmp_path, monkeypatch, n_shots=22):
    """22-shot miniature preserves the real 20-label transition and all schemas."""
    run, source, prior, rawroot = (
        tmp_path / n for n in ("run", "source", "prior", "raw")
    )
    for root in (run, source, prior):
        (root / "shots").mkdir(parents=True)
    monkeypatch.setattr(audit, "SOURCE", source)
    monkeypatch.setattr(audit, "PRIOR_RUN", prior)
    monkeypatch.setattr(audit, "PRIOR_AUDIT", tmp_path / "prior_audit.json")
    monkeypatch.setattr(audit, "RAW_ROOT", rawroot)
    monkeypatch.setattr(audit, "N_SHOTS", n_shots)
    ids, tau = list(range(1001, 1001 + n_shots)), 0.5
    np.savez(prior / "fit20.npz", **{"history.weights": np.array([0.0])})
    rows, sealed, old_sealed, source_seals, rawhash = (
        {a: [] for a in ARMS},
        [],
        [],
        [],
        {},
    )
    mining, selections = (
        {},
        {a: [dict() for _ in range(3)] for a in ("fixed_peak", "moving_peak")},
    )
    t = np.arange(215, dtype=float) / 1000
    for rank, sid in enumerate(ids, 1):
        flag = (rank <= 16 or rank > 20) and rank % 2 == 1
        down = 0.2 if flag else None
        raw = rawroot / str(sid // 100 * 100) / f"{sid}.hdf5"
        raw.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(raw, "w") as f:
            ip = f.create_dataset("data/ip", data=np.zeros(len(t)))
            ip.attrs["StartTime"], ip.attrs["SampleRate"] = 0.0, 1000.0
            f.create_dataset("meta/IsDisrupt", data=int(flag))
            f.create_dataset("meta/DownTime", data=down if flag else 999.0)
        rawhash[str(sid)] = digest(raw)
        native = np.full(len(t), 0.1)
        mapped = np.full(len(t), 0.2)
        if flag:
            mapped[150] = 0.8
        else:
            mapped[139] = 0.4
        curves = {a: native.copy() for a in ARMS}
        if rank > 20:
            if flag:
                for a in ARMS:
                    curves[a][150] = 0.9
                curves["moving_peak"][150] = 0.1
            else:
                curves["moving_peak"][145] = 0.9
        stem = run / "shots" / f"{rank:04d}_{sid}"
        path = stem.with_suffix(".npz")
        np.savez(path, times=t, **curves)
        seal = stem.with_name(stem.name + "_seal.json")
        active = (
            {}
            if rank <= 20
            else {
                a: digest(run / f"fit_{a}.npz") for a in ("fixed_peak", "moving_peak")
            }
        )
        put(
            seal,
            dict(
                shot=sid,
                rank=rank,
                raw_sha256=rawhash[str(sid)],
                curve_sha256=digest(path),
                current_outcome_used_in_prediction=False,
                active_fit_sha256=active,
            ),
        )
        reveal = stem.with_name(stem.name + "_revealed.json")
        shotrows = {a: shot_row(sid, flag, down, t, curves[a]) for a in ARMS}
        for a in ARMS:
            rows[a].append(shotrows[a])
        put(reveal, dict(shot=sid, after_seal_sha256=digest(seal), rows=shotrows))
        sealed.append(
            dict(
                shot=sid,
                curve_sha256=digest(path),
                seal_sha256=digest(seal),
                revealed_sha256=digest(reveal),
            )
        )
        oldpath = prior / "shots" / path.name
        np.savez(
            oldpath,
            times=t,
            static=curves["static"],
            history20=curves["history20_reference"],
        )
        old_sealed.append(dict(shot=sid, curve_sha256=digest(oldpath)))
        sourcepath = source / "shots" / path.name
        np.savez(sourcepath, times=t, map_after20=native, retrospective_map_all=mapped)
        source_seals.append(
            dict(
                shot=sid,
                curve_sha256=digest(sourcepath),
                current_outcome_used_in_prediction=False,
            )
        )
        if rank <= 20:
            mining[f"{sid}_native"] = mapped
            for arm in ("fixed_peak", "moving_peak"):
                for r in range(1, 5):
                    p = mapped.copy()
                    if not flag and arm == "moving_peak" and r > 1:
                        p[141] = 0.9
                    mining[f"{sid}_{arm}_{r}"] = p
                    if r <= 3:
                        if flag:
                            valid = [
                                i
                                for i, ti in enumerate(t)
                                if i >= 135 and 30 < round((down - ti) * 1000, 9) <= 125
                            ]
                            q = [
                                valid[i]
                                for i in np.linspace(
                                    0, len(valid) - 1, min(8, len(valid)), dtype=int
                                )
                            ]
                        else:
                            target = mapped if arm == "fixed_peak" else p
                            q = [
                                int(chunk[np.argmax(target[chunk])])
                                for chunk in np.array_split(np.arange(135, len(t)), 8)
                                if len(chunk)
                            ]
                        selections[arm][r - 1][str(sid)] = q
        if rank == 20:
            np.savez(run / "mining_curves.npz", **mining)
            np.savez(run / "fit_fixed_peak.npz", weights=np.array([1.0]))
            np.savez(run / "fit_moving_peak.npz", weights=np.array([2.0]))
            receipt = dict(
                ids=ids[:20],
                initial_fit_sha256=digest(prior / "fit20.npz"),
                arms={
                    a: dict(
                        fit_sha256=digest(run / f"fit_{a}.npz"),
                        steps=36,
                        rounds=[
                            dict(round=r, steps=12, selection=selections[a][r - 1])
                            for r in range(1, 4)
                        ],
                    )
                    for a in ("fixed_peak", "moving_peak")
                },
            )
            put(run / "fit_receipt.json", receipt)
    source_protocol = source / "protocol.json"
    put(
        source_protocol,
        dict(order=[dict(shot=s) for s in ids], threshold=tau, raw_sha256=rawhash),
    )
    source_done = source / "completed.json"
    put(
        source_done,
        dict(
            artifact_sha256={"protocol.json": digest(source_protocol)},
            seals=source_seals,
        ),
    )
    monkeypatch.setattr(audit, "SOURCE_DONE_SHA", digest(source_done))
    put(prior / "protocol.json", dict(order=ids, threshold=tau, raw_sha256=rawhash))
    put(
        prior / "completed.json",
        dict(
            status="COMPLETE_LATENT_TRAJECTORY_ADAPT_DEVELOPMENT",
            n_shots=len(ids),
            n_curves=len(ids) * 7,
            artifact_sha256={
                "protocol.json": digest(prior / "protocol.json"),
                "fit20.npz": digest(prior / "fit20.npz"),
            },
            sealed=old_sealed,
        ),
    )
    put(
        audit.PRIOR_AUDIT,
        dict(
            status="PASS_LATENT_ADAPTATION_INDEPENDENT_RECOUNT",
            n_shots=len(ids),
            producer_completion_sha256=digest(prior / "completed.json"),
            independent_future_validation=False,
        ),
    )
    monkeypatch.setattr(audit, "PRIOR_AUDIT_SHA", digest(audit.PRIOR_AUDIT))
    put(
        run / "protocol.json",
        dict(
            order=ids,
            threshold=tau,
            first_anchor=135,
            stride=32,
            arms=ARMS,
            input_sha256={
                str(p.resolve()): digest(p)
                for p in (
                    source_protocol,
                    source_done,
                    prior / "protocol.json",
                    prior / "completed.json",
                    prior / "fit20.npz",
                    audit.PRIOR_AUDIT,
                )
            },
            raw_sha256=rawhash,
            label_budget=20,
            rounds=3,
            steps_per_arm=36,
            independent_future_validation=False,
        ),
    )
    put(run / "per_shot.json", rows)
    put(run / "summary.json", summary_for(rows))
    files = (
        "protocol.json",
        "per_shot.json",
        "summary.json",
        "fit_receipt.json",
        "mining_curves.npz",
        "fit_fixed_peak.npz",
        "fit_moving_peak.npz",
    )
    put(
        run / "completed.json",
        dict(
            status="COMPLETE_MOVING_PEAK_REPLAY",
            n_shots=len(ids),
            n_curves=len(ids) * len(ARMS),
            sealed=sealed,
            artifact_sha256={name: digest(run / name) for name in files},
        ),
    )
    return run


def reseal_artifact(run, name):
    done = get(run / "completed.json")
    done["artifact_sha256"][name] = digest(run / name)
    put(run / "completed.json", done)


def restore_fit_window_timestamp(run, name):
    # A forged timestamp must not hide a forged selection or probability curve.
    value = (run / "shots/0020_1020_revealed.json").stat().st_mtime_ns
    os.utime(run / name, ns=(value, value))


def reseal_shot(run, rank):
    done = get(run / "completed.json")
    binding = done["sealed"][rank - 1]
    stem = run / "shots" / f"{rank:04d}_{binding['shot']}"
    curve, seal, reveal = (
        stem.with_suffix(".npz"),
        stem.with_name(stem.name + "_seal.json"),
        stem.with_name(stem.name + "_revealed.json"),
    )
    note = get(seal)
    note["curve_sha256"] = binding["curve_sha256"] = digest(curve)
    put(seal, note)
    label = get(reveal)
    label["after_seal_sha256"] = binding["seal_sha256"] = digest(seal)
    put(reveal, label)
    binding["revealed_sha256"] = digest(reveal)
    put(run / "completed.json", done)


def test_complete_replay_reports_independent_diagnostics(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    out = tmp_path / "audit.json"
    audit.verify(run, out)
    report = get(out)
    assert report["status"] == "PASS_MOVING_PEAK_INDEPENDENT_RECOUNT"
    assert report["independent_future_validation"] is False
    assert report["training_curve_metrics"]["round_1"]["fixed_peak"]["n_safe"] == 12
    assert report["training_curve_metrics"]["final"]["moving_peak"]["fp"] == 12
    assert report["training_curve_metrics"]["final"]["fixed_peak"]["fp"] == 0
    assert report["training_curve_metrics"]["final"]["moving_peak"]["tpr"] == 1.0
    assert report["moving_vs_fixed"]["after20"]["full"]["fp_added"] == [1022]
    assert report["moving_vs_fixed"]["after20"]["full"]["tp_lost"] == [1021]
    assert report["moving_vs_fixed"]["after20"]["capped"]["tp_lost"] == [1021]
    assert [row["shot"] for row in report["harmful_cases"]["after20"]] == [1021, 1022]
    with pytest.raises(FileExistsError):
        audit.verify(run, out)


def test_synthetic_full_360_and_all_2160_curves(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch, n_shots=360)
    out = tmp_path / "audit.json"
    audit.main([str(run), str(out)])
    report = get(out)
    assert report["n_shots"] == 360
    assert report["n_curves"] == 2160
    assert (
        report["summary"]["full360"]["metrics"]["static"]["n_safe"]
        + report["summary"]["full360"]["metrics"]["static"]["n_disruptive"]
        == 360
    )


def test_incomplete_does_not_open_hdf(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    done = get(run / "completed.json")
    done["status"] = "RUNNING"
    put(run / "completed.json", done)
    monkeypatch.setattr(
        h5py, "File", lambda *_a, **_k: pytest.fail("opened HDF before completion")
    )
    with pytest.raises(ValueError, match="COMPLETE|complete"):
        audit.verify(run, tmp_path / "audit.json")


@pytest.mark.parametrize(
    "arm,round_index,sid",
    [("fixed_peak", 2, "1002"), ("moving_peak", 2, "1002"), ("moving_peak", 1, "1001")],
)
def test_resealed_forged_query_rejected(tmp_path, monkeypatch, arm, round_index, sid):
    run = fixture_run(tmp_path, monkeypatch)
    receipt = get(run / "fit_receipt.json")
    receipt["arms"][arm]["rounds"][round_index - 1]["selection"][sid][0] += 1
    put(run / "fit_receipt.json", receipt)
    restore_fit_window_timestamp(run, "fit_receipt.json")
    reseal_artifact(run, "fit_receipt.json")
    with pytest.raises(ValueError, match="selection|quer"):
        audit.verify(run, tmp_path / "audit.json")


@pytest.mark.parametrize("field", ["steps", "ids", "round", "population", "initial"])
def test_forged_fit_contract_rejected(tmp_path, monkeypatch, field):
    run = fixture_run(tmp_path, monkeypatch)
    receipt = get(run / "fit_receipt.json")
    if field == "steps":
        receipt["arms"]["moving_peak"]["steps"] = 35
    elif field == "ids":
        receipt["ids"] = receipt["ids"][::-1]
    elif field == "round":
        receipt["arms"]["fixed_peak"]["rounds"][1]["steps"] = 13
    elif field == "population":
        del receipt["arms"]["moving_peak"]["rounds"][0]["selection"]["1001"]
    else:
        receipt["initial_fit_sha256"] = "not-a-sha256"
    put(run / "fit_receipt.json", receipt)
    reseal_artifact(run, "fit_receipt.json")
    with pytest.raises(ValueError, match="fit|round|population|initial|selection"):
        audit.verify(run, tmp_path / "audit.json")


def test_bad_fit_time_and_wrong_active_fit_are_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    fit = run / "fit_fixed_peak.npz"
    earlier = (run / "shots/0020_1020_revealed.json").stat().st_mtime_ns - 1_000_000_000
    os.utime(fit, ns=(earlier, earlier))
    with pytest.raises(ValueError, match="fit.*(order|reveal)|fit.*time"):
        audit.verify(run, tmp_path / "audit.json")
    later = (run / "shots/0021_1021.npz").stat().st_mtime_ns
    os.utime(fit, ns=(later, later))
    seal = run / "shots/0021_1021_seal.json"
    note = get(seal)
    note["active_fit_sha256"] = {}
    put(seal, note)
    reseal_shot(run, 21)
    with pytest.raises(ValueError, match="active.*fit"):
        audit.verify(run, tmp_path / "audit.json")


@pytest.mark.parametrize("kind", ["static", "history", "prefix", "inactive"])
def test_resealed_curve_tampering_rejected(tmp_path, monkeypatch, kind):
    run = fixture_run(tmp_path, monkeypatch)
    rank = 1 if kind == "inactive" else 21
    path = run / "shots" / f"{rank:04d}_{1000 + rank}.npz"
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    key, index = {
        "static": ("static", 40),
        "history": ("history20_reference", 140),
        "prefix": ("fixed_peak", 40),
        "inactive": ("moving_peak", 140),
    }[kind]
    arrays[key][index] = 0.2
    np.savez(path, **arrays)
    reseal_shot(run, rank)
    with pytest.raises(ValueError, match="parity|prefix|inactive|static"):
        audit.verify(run, tmp_path / "audit.json")


def test_raw_hdf_tampering_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    raw = audit.RAW_ROOT / "1000/1001.hdf5"
    with h5py.File(raw, "r+") as f:
        f["meta/IsDisrupt"][()] = 0
    with pytest.raises(ValueError, match="raw|hash"):
        audit.verify(run, tmp_path / "audit.json")


def test_resealed_wrong_clock_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "shots/0001_1001.npz"
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["times"][140] += 0.0001
    np.savez(path, **arrays)
    reseal_shot(run, 1)
    out = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="HDF signal clock"):
        audit.verify(run, out)
    assert not out.exists()


def test_resealed_wrong_revealed_outcome_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "shots/0001_1001_revealed.json"
    reveal = get(path)
    reveal["rows"]["static"]["disruptive"] = False
    put(path, reveal)
    done = get(run / "completed.json")
    done["sealed"][0]["revealed_sha256"] = digest(path)
    put(run / "completed.json", done)
    with pytest.raises(ValueError, match="HDF row recount"):
        audit.verify(run, tmp_path / "audit.json")


def test_resealed_reordered_protocol_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    cfg = get(run / "protocol.json")
    cfg["order"][0], cfg["order"][1] = cfg["order"][1], cfg["order"][0]
    put(run / "protocol.json", cfg)
    reseal_artifact(run, "protocol.json")
    with pytest.raises(ValueError, match="locked fixed order"):
        audit.verify(run, tmp_path / "audit.json")


def test_round_one_initial_curves_and_mapped_native_are_bound(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "mining_curves.npz"
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["1001_moving_peak_1"][150] = 0.3
    np.savez(path, **arrays)
    restore_fit_window_timestamp(run, "mining_curves.npz")
    reseal_artifact(run, "mining_curves.npz")
    with pytest.raises(ValueError, match="initial|round.one"):
        audit.verify(run, tmp_path / "audit.json")


def test_mapped_native_cannot_be_replaced_after_receipt_resealing(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "mining_curves.npz"
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["1002_native"][145] = 0.9
    np.savez(path, **arrays)
    restore_fit_window_timestamp(run, "mining_curves.npz")
    reseal_artifact(run, "mining_curves.npz")
    with pytest.raises(ValueError, match="mapped native.*parity"):
        audit.verify(run, tmp_path / "audit.json")


def test_resealed_fit_tensor_shape_is_checked(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    name = "fit_fixed_peak.npz"
    np.savez(run / name, weights=np.array([1.0, 2.0]))
    restore_fit_window_timestamp(run, name)
    receipt = get(run / "fit_receipt.json")
    receipt["arms"]["fixed_peak"]["fit_sha256"] = digest(run / name)
    put(run / "fit_receipt.json", receipt)
    restore_fit_window_timestamp(run, "fit_receipt.json")
    reseal_artifact(run, name)
    reseal_artifact(run, "fit_receipt.json")
    with pytest.raises(ValueError, match="invalid fit tensor"):
        audit.verify(run, tmp_path / "audit.json")


def test_resealed_wrong_summary_is_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    summary = get(run / "summary.json")
    summary["after20"]["metrics"]["moving_peak"]["fp"] = 0
    put(run / "summary.json", summary)
    reseal_artifact(run, "summary.json")
    with pytest.raises(ValueError, match="summary recount"):
        audit.verify(run, tmp_path / "audit.json")


def test_probability_parity_permits_roundoff_but_not_changed_crossings():
    reference = np.array([0.1, 0.5])
    assert audit.parity(np.array([0.100001, 0.500001]), reference, 0.5, "test") < 4e-6
    with pytest.raises(ValueError, match="parity"):
        audit.parity(np.array([0.1, 0.499999]), reference, 0.5, "test")
    assert (
        audit.parity(
            np.array([0.1, 0.499999]), reference, 0.5, "test", require_crossings=False
        )
        < 4e-6
    )


def test_positive_query_window_and_safe_dynamic_peak():
    t = np.arange(1000) / 1000
    p = np.zeros(len(t))
    q, _ = audit.expected_queries(t, p, True, 1.0, 135)
    assert (q[0], q[-1], len(q)) == (875, 969, 8)
    t = t[:215]
    native, current = np.zeros(len(t)), np.zeros(len(t))
    native[139], current[141] = 0.8, 0.9
    q_fixed, _ = audit.expected_queries(t, native, False, None, 135)
    q_moving, _ = audit.expected_queries(t, current, False, None, 135)
    assert q_fixed.tolist() == [139, 145, 155, 165, 175, 185, 195, 205]
    assert q_moving.tolist() == [141, 145, 155, 165, 175, 185, 195, 205]
