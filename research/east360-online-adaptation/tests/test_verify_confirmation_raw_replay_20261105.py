"""Synthetic, CPU-only checks; never open the research cohort or its labels."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import verify_confirmation_raw_replay_20261105 as audit


ARMS = [
    "static",
    "point_real",
    "point_lagged",
    "previous_clock",
    "first_full",
    "intra_confirm_past32",
    "donor_confirm_past32",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")


def get(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_row(sid, disruptive, down, times, score):
    indices = [
        i
        for i, (t, p) in enumerate(zip(times, score))
        if p >= 0.5 and (not disruptive or (down - t) * 1000 >= -1e-6)
    ]
    index = indices[0] if indices else None
    lead = (
        round(float((down - times[index]) * 1000), 9)
        if disruptive and indices
        else None
    )
    status = (
        ("miss" if index is None else "tp" if lead > 30 else "tardy")
        if disruptive
        else ("fp" if indices else "tn")
    )
    return dict(
        shot=sid,
        disruptive=disruptive,
        down=down,
        status=status,
        first_index=index,
        lead_ms=lead,
    )


def expected_counts(rows):
    d = sum(r["disruptive"] for r in rows)
    safe = len(rows) - d
    tp = sum(r["status"] == "tp" for r in rows)
    fp = sum(r["status"] == "fp" for r in rows)
    early = sum(r["status"] == "tp" and r["lead_ms"] > 3000 for r in rows)
    tpr, fpr = (tp / d if d else None), (fp / safe if safe else None)
    return dict(
        n_disruptive=d,
        n_safe=safe,
        tp=tp,
        fp=fp,
        tpr=tpr,
        fpr=fpr,
        j=tpr - fpr if tpr is not None and fpr is not None else None,
        early_over3000=early,
        tp_30_to_3000=tp - early,
    )


def expected_pair(a, b, capped=False):
    def good(r):
        return r["status"] == "tp" and (not capped or r["lead_ms"] <= 3000)

    return dict(
        tp_gained=[x["shot"] for x, y in zip(a, b) if good(x) and not good(y)],
        tp_lost=[x["shot"] for x, y in zip(a, b) if good(y) and not good(x)],
        fp_added=[
            x["shot"]
            for x, y in zip(a, b)
            if x["status"] == "fp" and y["status"] != "fp"
        ],
        fp_removed=[
            x["shot"]
            for x, y in zip(a, b)
            if y["status"] == "fp" and x["status"] != "fp"
        ],
    )


def fixture_run(tmp_path, monkeypatch, mode="conditional"):
    run = tmp_path / "run"
    (run / "shots").mkdir(parents=True)
    monkeypatch.setattr(audit, "SOURCE", tmp_path / "source")
    monkeypatch.setattr(audit, "PARENT", audit.SOURCE / "protocol.json")
    monkeypatch.setattr(audit, "POINT", tmp_path / "point")
    monkeypatch.setattr(audit, "DONOR", tmp_path / "donor")
    monkeypatch.setattr(audit, "GUARD", tmp_path / "guard")
    monkeypatch.setattr(audit, "COHORT", tmp_path / "cohort.json")
    monkeypatch.setattr(audit, "ROSTER", tmp_path / "roster.json")
    monkeypatch.setattr(audit, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(audit, "FIXED_COUNT", 101)
    ids = list(range(1001, 1102)) if mode == "fixed" else [1001, 1003, 1004]
    parent_ids = ids if mode == "fixed" else [1002] + list(range(1, 101))
    (audit.SOURCE / "shots").mkdir(parents=True)
    for root in (audit.POINT, audit.DONOR, audit.GUARD):
        (root / "shots").mkdir(parents=True)
    put(
        audit.PARENT, {"order": [{"shot": sid} for sid in parent_ids], "threshold": 0.5}
    )
    put(
        audit.COHORT,
        {
            "test": {
                "disruption_shots": parent_ids[::2],
                "non_disruption_shots": parent_ids[1::2],
            }
        },
    )
    rows = {a: [] for a in ARMS}
    raw_hashes, receipts, source_seals = {}, [], []
    archive_seals = {p: [] for p in (audit.POINT, audit.DONOR, audit.GUARD)}
    previous_n = 0
    for rank, sid in enumerate(ids, 1):
        n = 9 if mode == "conditional" and rank == 2 else 170
        raw = audit.RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        raw.parent.mkdir(parents=True, exist_ok=True)
        disruptive = rank % 2 == 1
        down = 0.3 if disruptive else None
        with h5py.File(raw, "w") as f:
            ip = f.create_dataset("data/ip", data=np.zeros(n))
            ip.attrs["StartTime"], ip.attrs["SampleRate"] = 0.0, 1000.0
            f.create_dataset("meta/IsDisrupt", data=int(disruptive))
            f.create_dataset("meta/DownTime", data=down if disruptive else 12.0)
        raw_hashes[str(sid)] = digest(raw)
        donor = (
            ids[rank - 2]
            if rank > (100 if mode == "fixed" else 0) and rank > 1 and previous_n > 135
            else None
        )
        times = np.arange(n) / 1000.0
        curves = {a: np.full(n, 0.1) for a in ARMS}
        if n > 135 and not (mode == "fixed" and rank <= 100):
            curves["point_real"][135] = 0.9
            curves["point_lagged"][135] = 0.8
            curves["first_full"][135] = 0.7
            if donor is not None:
                curves["previous_clock"][135] = 0.8
            for target, shadow in (
                ("intra_confirm_past32", "point_lagged"),
                ("donor_confirm_past32", "previous_clock"),
            ):
                for q in range(n):
                    if curves["point_real"][q] >= 0.5 and any(
                        curves["static"][i] < 0.5 and curves[shadow][i] >= 0.5
                        for i in range(max(0, q - 32), q + 1)
                    ):
                        curves[target][q] = curves["point_real"][q]
        stem = run / "shots" / f"{rank:04d}_{sid}"
        path = stem.with_suffix(".npz")
        np.savez(path, times=times, **curves)
        seal = stem.with_name(stem.name + "_seal.json")
        put(
            seal,
            dict(
                rank=rank,
                shot=sid,
                curve_sha256=digest(path),
                raw_sha256=raw_hashes[str(sid)],
                current_outcome_used_in_prediction=False,
                donor=donor,
            ),
        )
        reveal = stem.with_name(stem.name + "_revealed.json")
        shot_rows = {
            a: expected_row(sid, disruptive, down, times, curves[a]) for a in ARMS
        }
        for a in ARMS:
            rows[a].append(shot_rows[a])
        put(reveal, dict(shot=sid, after_seal_sha256=digest(seal), rows=shot_rows))
        receipts.append(
            dict(
                shot=sid,
                curve_sha256=digest(path),
                seal_sha256=digest(seal),
                revealed_sha256=digest(reveal),
            )
        )
        source_curve = audit.SOURCE / "shots" / path.name
        np.savez(source_curve, times=times, map_after20=curves["static"])
        source_seal = dict(
            shot=sid,
            curve_sha256=digest(source_curve),
            current_outcome_used_in_prediction=False,
        )
        put(audit.SOURCE / "shots" / seal.name, source_seal)
        source_seals.append(source_seal)
        if mode == "fixed" and rank > 100:
            for root, names in (
                (audit.POINT, ("static", "point_real", "point_lagged")),
                (audit.DONOR, ("previous_clock", "first_full")),
                (audit.GUARD, ("intra_confirm_past32", "donor_confirm_past32")),
            ):
                ref = root / "shots" / path.name
                np.savez(ref, times=times, **{a: curves[a] for a in names})
                archive_seals[root].append(dict(shot=sid, curve_sha256=digest(ref)))
        previous_n = n
    later = ids[1:] if mode == "conditional" else []
    put(
        audit.ROSTER,
        dict(
            status="LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER",
            ids=ids,
            later_proxy_ids=later,
            raw_sha256=raw_hashes,
            independent_validation=False,
        ),
    )
    if mode == "conditional":
        source_seals = [{"shot": sid} for sid in parent_ids]
    put(
        audit.SOURCE / "completed.json",
        dict(
            artifact_sha256={"protocol.json": digest(audit.PARENT)}, seals=source_seals
        ),
    )
    monkeypatch.setattr(
        audit, "SOURCE_DONE_SHA", digest(audit.SOURCE / "completed.json")
    )
    for root, seals in archive_seals.items():
        put(root / "completed.json", dict(sealed=seals))
    paths = (
        (
            audit.PARENT,
            audit.SOURCE / "completed.json",
            *(p / "completed.json" for p in archive_seals),
        )
        if mode == "fixed"
        else (audit.PARENT, audit.ROSTER)
    )
    protocol = dict(
        mode=mode,
        order=ids,
        threshold=0.5,
        arms=ARMS,
        input_sha256={str(p.resolve()): digest(p) for p in paths},
        later_proxy_ids=later,
        independent_validation=False,
        fixed_prefix_shots=100 if mode == "fixed" else 0,
        raw_sha256=raw_hashes,
    )
    put(run / "protocol.json", protocol)
    put(run / "per_shot.json", rows)
    put(
        run / "summary.json",
        dict(
            full={a: expected_counts(r) for a, r in rows.items()},
            later_proxy={
                a: expected_counts([r for r in rs if r["shot"] in later])
                for a, rs in rows.items()
            }
            if later
            else None,
            paired_to_static={
                a: expected_pair(r, rows["static"])
                for a, r in rows.items()
                if a != "static"
            },
            paired_capped_to_static={
                a: expected_pair(r, rows["static"], True)
                for a, r in rows.items()
                if a != "static"
            },
        ),
    )
    put(
        run / "completed.json",
        dict(
            status="COMPLETE_RAW_CONFIRMATION_REPLAY",
            n_shots=len(ids),
            n_curves=len(ids) * len(ARMS),
            sealed=receipts,
            artifact_sha256={
                name: digest(run / name)
                for name in ("protocol.json", "per_shot.json", "summary.json")
            },
        ),
    )
    return run


def reseal_artifact(run, name):
    done = get(run / "completed.json")
    done["artifact_sha256"][name] = digest(run / name)
    put(run / "completed.json", done)


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


def test_first_crossing_boundaries_and_absent_class_rates():
    t = np.array([0.0, 0.97, 1.0, 1.0000000005, 1.001])
    truth = dict(shot=1, disruptive=True, down=1.0)
    assert (
        audit.score(t, np.array([0.0, 0.5, 0.0, 0.0, 0.0]), truth, 0.5)["status"]
        == "tardy"
    )
    assert (
        audit.score(t, np.array([0.0, 0.0, 0.0, 0.5, 0.0]), truth, 0.5)["status"]
        == "tardy"
    )
    assert (
        audit.score(t, np.array([0.0, 0.0, 0.0, 0.0, 0.5]), truth, 0.5)["status"]
        == "miss"
    )
    assert (
        audit.score(t, np.array([0.5, 0.9, 0.0, 0.0, 0.0]), truth, 0.5)["first_index"]
        == 0
    )
    almost = np.array([0.9699999999996])
    beyond = np.array([0.9699999999994])
    assert audit.score(almost, np.ones(1), truth, 0.5)["status"] == "tardy"
    assert audit.score(beyond, np.ones(1), truth, 0.5)["status"] == "tp"
    safe = audit.score(
        t,
        np.array([0.0, 0.0, 0.0, 0.0, 0.5]),
        dict(shot=2, disruptive=False, down=None),
        0.5,
    )
    assert safe["status"] == "fp" and audit.counts([safe])["fpr"] == 1
    one = audit.score(t, np.ones(5), truth, 0.5)
    assert audit.counts([one])["fpr"] is None and audit.counts([one])["j"] is None
    assert audit.counts([])["tpr"] is None and audit.counts([])["fpr"] is None
    early = dict(one, lead_ms=3000.000000001)
    exact = dict(one, lead_ms=3000.0)
    assert audit.counts([early, exact])["early_over3000"] == 1
    assert audit.pair([early], [exact], capped=True)["tp_lost"] == [1]


def test_hash_cache_reads_file_once_and_detects_later_change(tmp_path, monkeypatch):
    path = tmp_path / "raw.hdf5"
    path.write_bytes(b"before")
    link = tmp_path / "alias.hdf5"
    link.symlink_to(path)
    original_open = Path.open
    calls = []

    def counted(p, *args, **kwargs):
        if p.resolve() == path:
            calls.append(p)
        return original_open(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    hashes = audit.Hashes()
    assert hashes(path) == hashes(link) == hashes(path)
    assert len(calls) == 1
    path.write_bytes(b"after!")  # same length, different file content
    with pytest.raises(ValueError, match="changed"):
        hashes.check()


def test_confirmation_window_is_inclusive_past_only_and_static_filtered():
    base = np.full(80, 0.1)
    real = np.full(80, 0.1)
    shadow = np.full(80, 0.1)
    real[[31, 32, 33, 64, 65]] = 0.9
    shadow[[0, 33, 66]] = 0.8
    base[33] = 0.7  # shadow at a static crossing cannot authorize future points
    actual = audit.confirmed(base, real, shadow, 0.5)
    assert actual[[31, 32, 33, 64, 65, 66]].tolist() == [0.9, 0.9, 0.7, 0.1, 0.1, 0.1]


@pytest.mark.parametrize("mode", ["fixed", "conditional"])
def test_valid_complete_replay_and_existing_output_refusal(tmp_path, monkeypatch, mode):
    run = fixture_run(tmp_path, monkeypatch, mode)
    out = tmp_path / "verified.json"
    audit.verify(run, out)
    assert get(out)["status"].startswith("PASS")
    assert get(out)["independent_validation"] is False
    if mode == "fixed":
        assert get(out)["fixed_crossing_parity_curves"] == 101 * len(ARMS)
    with pytest.raises(FileExistsError):
        audit.verify(run, out)


def test_incomplete_run_never_opens_raw_or_roster(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    done = get(run / "completed.json")
    done["status"] = "RUNNING"
    put(run / "completed.json", done)
    monkeypatch.setattr(
        h5py, "File", lambda *_a, **_k: pytest.fail("raw opened before completion")
    )
    monkeypatch.setattr(audit, "ROSTER", tmp_path / "missing")
    with pytest.raises((AssertionError, ValueError), match="COMPLETE|complet"):
        audit.verify(run, tmp_path / "out.json")


def test_resealed_forged_gate_is_rejected(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "shots/0001_1001.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["intra_confirm_past32"][136] = 0.9
    np.savez(path, **arrays)
    reseal_shot(run, 1)
    with pytest.raises((AssertionError, ValueError), match="gate|confirm"):
        audit.verify(run, tmp_path / "out.json")


def test_resealed_wrong_donor_and_raw_tamper_fail(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    seal = run / "shots/0001_1001_seal.json"
    note = get(seal)
    note["donor"] = 1001
    put(seal, note)
    reseal_shot(run, 1)
    with pytest.raises((AssertionError, ValueError), match="donor"):
        audit.verify(run, tmp_path / "out.json")
    note["donor"] = None
    put(seal, note)
    reseal_shot(run, 1)
    raw = audit.RAW_ROOT / "1000/1001.hdf5"
    with h5py.File(raw, "r+") as f:
        f["meta/IsDisrupt"][()] = 0
    with pytest.raises((AssertionError, ValueError), match="raw|hash"):
        audit.verify(run, tmp_path / "out.json")


def test_disabled_first_donor_curve_cannot_change_below_threshold(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path, monkeypatch)
    path = run / "shots/0001_1001.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["previous_clock"][136] = 0.2
    np.savez(path, **arrays)
    reseal_shot(run, 1)
    with pytest.raises(ValueError, match="disabled donor"):
        audit.verify(run, tmp_path / "out.json")


def test_resealed_roster_order_and_summary_mismatches_fail(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    protocol = get(run / "protocol.json")
    protocol["order"] = protocol["order"][::-1]
    put(run / "protocol.json", protocol)
    reseal_artifact(run, "protocol.json")
    with pytest.raises((AssertionError, ValueError), match="order|roster"):
        audit.verify(run, tmp_path / "out.json")
    protocol["order"] = protocol["order"][::-1]
    put(run / "protocol.json", protocol)
    reseal_artifact(run, "protocol.json")
    summary = get(run / "summary.json")
    summary["full"]["static"]["fp"] += 1
    put(run / "summary.json", summary)
    reseal_artifact(run, "summary.json")
    with pytest.raises((AssertionError, ValueError), match="summary|count"):
        audit.verify(run, tmp_path / "out.json")


def test_resealed_fixed_crossing_change_fails_independent_reference(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path, monkeypatch, "fixed")
    path = run / "shots/0101_1101.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for arm in ("point_real", "intra_confirm_past32", "donor_confirm_past32"):
        arrays[arm][136] = (
            0.9  # first crossing and all summary statistics are unchanged
        )
    np.savez(path, **arrays)
    reseal_shot(run, 101)
    with pytest.raises((AssertionError, ValueError), match="crossing parity"):
        audit.verify(run, tmp_path / "out.json")


@pytest.mark.parametrize("field", ["IsDisrupt", "DownTime", "clock"])
def test_resealed_live_hdf_tamper_is_detected(tmp_path, monkeypatch, field):
    run = fixture_run(tmp_path, monkeypatch)
    raw = audit.RAW_ROOT / "1000/1001.hdf5"
    with h5py.File(raw, "r+") as f:
        if field == "clock":
            f["data/ip"].attrs["SampleRate"] = 999.75
        elif field == "DownTime":
            f["meta/DownTime"][()] = 0.4
        else:
            f["meta/IsDisrupt"][()] = 0
    updated = digest(raw)
    roster = get(audit.ROSTER)
    roster["raw_sha256"]["1001"] = updated
    put(audit.ROSTER, roster)
    cfg = get(run / "protocol.json")
    cfg["raw_sha256"]["1001"] = updated
    cfg["input_sha256"][str(audit.ROSTER.resolve())] = digest(audit.ROSTER)
    put(run / "protocol.json", cfg)
    reseal_artifact(run, "protocol.json")
    seal = run / "shots/0001_1001_seal.json"
    note = get(seal)
    note["raw_sha256"] = updated
    put(seal, note)
    reseal_shot(run, 1)
    with pytest.raises((AssertionError, ValueError), match="row recount|signal clock"):
        audit.verify(run, tmp_path / "out.json")
