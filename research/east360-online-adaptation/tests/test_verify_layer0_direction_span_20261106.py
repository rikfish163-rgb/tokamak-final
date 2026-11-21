"""Synthetic-only tests for the sealed layer-0 direction-span verifier."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import verify_layer0_direction_span_20261106 as verifier


def anchor_example():
    n = 200
    curves = {arm: np.full(n, 0.5, dtype=np.float32) for arm in verifier.ARMS}
    curves["full_pos"][135] = 0.4
    curves["full_neg"][135] = 0.6
    curves["random_pos"][167] = 0.4
    curves["random_neg"][167] = 0.6
    curves.update(
        anchor_norms=np.array(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
        anchor_gradient_norms=np.array([[5, 3, 4], [0, 0, 0], [2, 2, 0]], dtype=np.float64),
        anchor_valid=np.array(
            [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=bool
        ),
    )
    return curves


def test_anchor_receipts_random_independent_of_zero_gradient_and_actual_anchor():
    result = verifier.audit_anchors(anchor_example(), 200)
    assert result["n_anchors"] == 3
    assert result["sign_inversions"]["full"] == {
        "denominator": 3, "positive": 1, "negative": 1,
        "valid_denominator": 1, "valid_positive": 1, "valid_negative": 1,
    }
    assert result["sign_inversions"]["random"] == {
        "denominator": 3, "positive": 1, "negative": 1,
        "valid_denominator": 2, "valid_positive": 1, "valid_negative": 1,
    }
    assert result["gradient_energy"]["full_squared_sum"] == 29.0
    assert result["gradient_energy"]["in_span_squared_sum"] == 13.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["anchor_valid"].__setitem__((1, 3), False),
        lambda a: a["anchor_valid"].__setitem__((0, 1), False),
        lambda a: a["anchor_norms"].__setitem__((0, 1), 0.99),
        lambda a: a["anchor_norms"].__setitem__((1, 1), 1e-8),
        lambda a: a["anchor_gradient_norms"].__setitem__((0, 1), 3.1),
        lambda a: a["anchor_gradient_norms"].__setitem__((1, 0), np.nan),
        lambda a: a.__setitem__("anchor_valid", a["anchor_valid"].astype(np.int8)),
        lambda a: a.__setitem__("anchor_norms", a["anchor_norms"].astype(np.float64)),
        lambda a: a.__setitem__("anchor_gradient_norms", a["anchor_gradient_norms"].astype(np.float32)),
    ],
)
def test_bad_anchor_receipts_are_rejected(mutation):
    a = anchor_example()
    mutation(a)
    with pytest.raises(ValueError):
        verifier.audit_anchors(a, 200)


def test_relative_direction_epsilon_boundary_and_zero_original():
    a = anchor_example()
    a["anchor_gradient_norms"][0] = [1.0, 1e-10, 1.0]
    a["anchor_valid"][0] = [True, False, True, True]
    a["anchor_norms"][0] = [1, 1, 0, 1, 1]
    result = verifier.audit_anchors(a, 200)
    assert result["sign_inversions"]["in_span"]["valid_denominator"] == 0
    a["anchor_valid"][0, 1] = True
    with pytest.raises(ValueError):
        verifier.audit_anchors(a, 200)


def test_later_eligible_crossing_cannot_displace_immutable_early_first_alarm():
    times = np.array([-4.0, -2.0])
    truth = {"shot": 1, "disruptive": True, "down": 0.0}
    static = np.array([0.9, 0.0])
    candidate = np.array([0.9, 0.9])
    result = verifier.menu_result(times, np.array([static, candidate]), truth, first=1, stride=1)
    assert result["strict"]["lead_ms"] == 4000
    assert result["capped"]["lead_ms"] == 4000
    assert result["capped_relative_to_static_miss"] == "eligible_crossing_cannot_be_first"


@pytest.mark.parametrize(
    ("lead", "strict_good", "capped_good"),
    [(30.0, False, False), (30.0000000006, True, True),
     (3000.0, True, True), (3000.0000000006, True, False)],
)
def test_menu_lead_boundaries_match_raw_scorer(lead, strict_good, capped_good):
    times = np.array([-lead / 1000])
    truth = {"shot": 2, "disruptive": True, "down": 0.0}
    result = verifier.menu_result(times, np.array([[0.0], [0.9]]), truth, first=0, stride=1)
    assert verifier.is_good(result["strict"], capped=False) is strict_good
    assert verifier.is_good(result["capped"], capped=True) is capped_good


def test_safe_menu_minimizes_false_positives_and_does_not_hide_forced_hit():
    truth = {"shot": 3, "disruptive": False, "down": None}
    times = np.arange(2, dtype=float)
    optional = verifier.menu_result(times, np.array([[0.0, 0.9], [0.0, 0.0]]), truth, first=0, stride=1)
    forced = verifier.menu_result(times, np.array([[0.9, 0.0], [0.9, 0.9]]), truth, first=1, stride=1)
    assert optional["strict"]["status"] == "tn"
    assert forced["strict"]["status"] == "fp"
    assert forced["strict"]["first_index"] == 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Write only artificial HDF5, curves, provenance, and oracle to pytest tmp."""
    run, ref, raw_root = (tmp_path / name for name in ("run", "signed", "raw"))
    (run / "shots").mkdir(parents=True)
    (ref / "shots").mkdir(parents=True)
    checked, oracle_path = tmp_path / "checked.json", tmp_path / "oracle.json"
    fit, mapping = tmp_path / "fit.npz", tmp_path / "map.bin"
    # The synthetic generator's 33 output columns span the first 33 axes.
    weights = np.zeros((256, 32), dtype=np.float64)
    weights[:32] = np.eye(32)
    bias = np.zeros(256, dtype=np.float64)
    bias[32] = 1.0
    np.savez(fit, **{"layers.2.weight": weights, "layers.2.bias": bias})
    mapping.write_bytes(b"synthetic map")
    monkeypatch.setattr(verifier, "SOURCE", "a" * 64)
    monkeypatch.setattr(verifier, "FIT", sha(fit))
    monkeypatch.setattr(verifier, "MAP", sha(mapping))
    monkeypatch.setattr(verifier, "RAW_ROOT", raw_root)
    monkeypatch.setattr(verifier, "ORACLE", oracle_path)
    monkeypatch.setattr(verifier, "REFS", {"fixed": (ref, checked, 21, 21, (1, 20), "fixed360")})
    order = list(range(100001, 100022))
    raw_hashes, old_sealed, new_sealed, oracle_rows = {}, [], [], []
    times = np.arange(140, dtype=np.float64) / 100
    for rank, sid in enumerate(order, 1):
        raw = raw_root / str(sid // 100 * 100) / f"{sid}.hdf5"
        raw.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(raw, "w") as f:
            f.create_dataset("meta/IsDisrupt", data=int(rank == 21))
            f.create_dataset("meta/DownTime", data=1.39)
            ip = f.create_dataset("data/ip", data=np.zeros(len(times)))
            ip.attrs["StartTime"], ip.attrs["SampleRate"] = 0.0, 100.0
        raw_hashes[str(sid)] = sha(raw)
        static = np.full(len(times), 0.1, dtype=np.float32)
        if rank == 1:
            static[1] = 0.9
        curves = {arm: static.copy() for arm in verifier.ARMS}
        if rank == 21:
            curves["full_pos"][135] = 0.9
            evidence = dict(
                anchor_norms=np.ones((1, 5), dtype=np.float32),
                anchor_gradient_norms=np.array([[5.0, 3.0, 4.0]], dtype=np.float64),
                anchor_valid=np.ones((1, 4), dtype=bool),
            )
        else:
            evidence = {}
        stem = f"{rank:04d}_{sid}"
        new_curve, old_curve = run / "shots" / f"{stem}.npz", ref / "shots" / f"{stem}.npz"
        np.savez(new_curve, times=times, **curves, **evidence)
        np.savez(old_curve, times=times, static=static, intra=static,
                 signed=static, blind_signed=static)
        old_note = dict(rank=rank, shot=sid, raw_sha256=raw_hashes[str(sid)],
                        curve_sha256=sha(old_curve), previous_seal_sha256=old_sealed[-1]["seal_sha256"] if old_sealed else None,
                        bank_enabled=rank >= 21, seal_ns=100 + rank)
        new_note = dict(rank=rank, shot=sid, raw_sha256=raw_hashes[str(sid)],
                        curve_sha256=sha(new_curve), previous_seal_sha256=new_sealed[-1]["seal_sha256"] if new_sealed else None,
                        active=rank >= 21, seal_ns=200 + rank)
        for root, note, seals in ((ref, old_note, old_sealed), (run, new_note, new_sealed)):
            path = root / "shots" / f"{stem}_seal.json"
            put_json(path, note)
            seals.append(dict(note, seal_sha256=sha(path)))
        truth = {"shot": sid, "disruptive": rank == 21, "down": 1.39 if rank == 21 else None}
        menus = {"extended": {}}
        if rank == 21:
            menus["extended"]["capped_relative_to_static_miss"] = "no_eligible_candidate_crossing"
        oracle_rows.append(dict(rank=rank, **truth,
                                static=verifier.score(times, static, truth, verifier.TAU), menus=menus))
    ref_cfg = dict(mode="fixed", order=order, raw_sha256=raw_hashes, threshold=verifier.TAU,
                   first=135, stride=32, initial_ns=100, source_state_sha256=verifier.SOURCE,
                   source_fit_sha256=verifier.FIT)
    put_json(ref / "protocol.json", ref_cfg)
    ref_done = dict(status="COMPLETE_LAYER0_SIGNED_PROJECTION", mode="fixed", n_shots=21,
                    n_curves=126, sealed=old_sealed, completed_ns=130,
                    protocol_sha256=sha(ref / "protocol.json"))
    put_json(ref / "completed.json", ref_done)
    put_json(checked, dict(status="PASS_SIGNED_PROJECTION_HDF5_RECOUNT", producer_sha256=sha(ref / "completed.json")))
    put_json(oracle_path, dict(status="PASS_POSTHOC_ACTION_MENU_HDF5", cohorts={
        "fixed360": dict(parent_sha256=sha(ref / "completed.json"), n_shots=21, per_shot=oracle_rows)}))
    np.savez(run / "basis.npz", basis=np.eye(128, dtype=np.float64)[:, :33],
             singular_values=np.ones(33, dtype=np.float64))
    cfg = dict(mode="fixed", order=order, arms=list(verifier.ARMS), raw_sha256=raw_hashes,
               threshold=verifier.TAU, initial_ns=200, first=135, stride=32, active_rank=21,
               basis_rank=33, basis_sha256=sha(run / "basis.npz"),
               source_state_sha256=verifier.SOURCE, source_fit_sha256=verifier.FIT,
               source_map_sha256=verifier.MAP, input_sha256={
                   str(fit): sha(fit), str(mapping): sha(mapping),
                   str(ref / "protocol.json"): sha(ref / "protocol.json"),
                   str(ref / "completed.json"): sha(ref / "completed.json")},
               development_only=True, new_fit_labels=0, inter_updates=0,
               outcome_reads_during_prediction=0, random_seed=131, relative_direction_epsilon=1e-10)
    put_json(run / "protocol.json", cfg)
    done = dict(status="COMPLETE_LAYER0_DIRECTION_SPAN", mode="fixed", n_shots=21,
                n_curves=210, sealed=new_sealed, max_reference_error={"static": 0.0, "intra": 0.0},
                protocol_sha256=sha(run / "protocol.json"), completed_ns=230, seconds=1.0)
    put_json(run / "completed.json", done)
    return dict(run=run, ref=ref, oracle=oracle_path, output=tmp_path / "result.json", fit=fit)


def test_synthetic_full_cli_recounts_arms_and_conditional_oracle(synthetic_run, monkeypatch):
    case = synthetic_run
    monkeypatch.setattr(sys, "argv", ["verify", str(case["run"]), str(case["output"])])
    verifier.main()
    out = json.loads(case["output"].read_text())
    assert out["status"] == "PASS_LAYER0_DIRECTION_SPAN_HDF5_RECOUNT"
    assert out["actual_arms"]["static"]["fp"] == 1
    assert out["actual_arms"]["full_pos"]["tp_30_to_3000"] == 1
    assert out["actual_arms"]["full_pos"]["tpr_30_to_3000"] == 1.0
    assert out["actual_arms_active"]["static"]["fp"] == 0
    cohort = out["prior_extended_no_eligible_capped"]
    assert cohort["n_shots"] == 1
    assert cohort["menus"]["binary"]["newly_reachable_capped_first_alarms"] == 0
    assert cohort["menus"]["full"]["newly_reachable_capped_first_alarms"] == 1
    assert cohort["menus"]["random"]["newly_reachable_capped_first_alarms"] == 0
    assert out["per_shot"][-1]["menus"]["full"]["capped"]["first_index"] == 135
    assert out["menus"]["full"]["capped"]["max_tp"] == 1
    assert out["menus"]["binary"]["capped"]["min_fp"] == 1
    assert out["menus_active"]["binary"]["capped"]["min_fp"] == 0
    assert out["menu_comparisons"]["full_vs_random"]["capped"] == {
        "max_tp_difference": 1, "min_fp_difference": 0,
    }


@pytest.mark.parametrize("attack", ["curve", "seal", "raw", "basis", "input", "oracle"])
def test_corruption_fails_closed_without_output(synthetic_run, attack):
    c = synthetic_run
    targets = {
        "curve": c["run"] / "shots/0021_100021.npz",
        "seal": c["run"] / "shots/0021_100021_seal.json",
        "raw": c["run"].parent / "raw/100000/100021.hdf5",
        "basis": c["run"] / "basis.npz",
        "input": c["fit"],
        "oracle": c["oracle"],
    }
    path = targets[attack]
    if attack == "oracle":
        data = json.loads(path.read_text())
        data["cohorts"]["fixed360"]["per_shot"][-1]["menus"]["extended"]["capped_relative_to_static_miss"] = "selectable_rescue"
        put_json(path, data)
    elif attack == "seal":
        data = json.loads(path.read_text())
        data["previous_seal_sha256"] = "b" * 64
        put_json(path, data)
    else:
        with path.open("ab") as f:
            f.write(b"synthetic corruption")
    with pytest.raises(ValueError):
        verifier.verify(c["run"], c["output"])
    assert not c["output"].exists()


def test_output_cannot_overwrite_or_live_inside_run(synthetic_run):
    c = synthetic_run
    with pytest.raises((FileExistsError, ValueError)):
        verifier.verify(c["run"], c["run"] / "protocol.json")
    with pytest.raises(ValueError):
        verifier.verify(c["run"], c["run"] / "new_report.json")


def reseal_run(run):
    cfg = json.loads((run / "protocol.json").read_text())
    done = json.loads((run / "completed.json").read_text())
    previous, rows = None, []
    for rank, sid in enumerate(cfg["order"], 1):
        stem = f"{rank:04d}_{sid}"
        seal_path = run / "shots" / f"{stem}_seal.json"
        row = json.loads(seal_path.read_text())
        row["previous_seal_sha256"] = previous
        row["curve_sha256"] = sha(run / "shots" / f"{stem}.npz")
        put_json(seal_path, row)
        previous = sha(seal_path)
        rows.append(dict(row, seal_sha256=previous))
    done["sealed"] = rows
    done["protocol_sha256"] = sha(run / "protocol.json")
    put_json(run / "completed.json", done)


@pytest.mark.parametrize("where", ["inactive", "prefix"])
def test_resealed_pre_action_changes_are_rejected(synthetic_run, where):
    c = synthetic_run
    path = c["run"] / ("shots/0001_100001.npz" if where == "inactive" else "shots/0021_100021.npz")
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["full_pos"][139 if where == "inactive" else 100] = 0.99
    np.savez(path, **values)
    reseal_run(c["run"])
    with pytest.raises(ValueError, match="pre-action or inactive"):
        verifier.verify(c["run"], c["output"])


def test_wrong_but_resealed_orthonormal_basis_is_not_the_frozen_fit_span(synthetic_run):
    c = synthetic_run
    path = c["run"] / "basis.npz"
    # Also orthonormal and rank 33, but e_33 replaces the fit's e_32.
    basis = np.eye(128, dtype=np.float64)[:, list(range(32)) + [33]]
    np.savez(path, basis=basis, singular_values=np.ones(33, dtype=np.float64))
    cfg = json.loads((c["run"] / "protocol.json").read_text())
    cfg["basis_sha256"] = sha(path)
    put_json(c["run"] / "protocol.json", cfg)
    reseal_run(c["run"])
    with pytest.raises(ValueError, match="fit.*span|span.*fit"):
        verifier.verify(c["run"], c["output"])


def test_prior_no_crossing_cohort_counts_blocked_hits_and_paired_controls():
    def row(sid, full, random):
        reasons = {"binary": "no_eligible_candidate_crossing", "full": full,
                   "in_span": "no_eligible_candidate_crossing",
                   "out_span": "no_eligible_candidate_crossing", "random": random}
        return {"shot": sid, "prior_extended_capped_reason": "no_eligible_candidate_crossing",
                "menus": {name: {"capped_relative_to_static_miss": why} for name, why in reasons.items()}}
    records = [row(1, "eligible_crossing_cannot_be_first", "no_eligible_candidate_crossing"),
               row(2, "selectable_rescue", "selectable_rescue")]
    result = verifier.cohort_summary(records)
    assert result["menus"]["full"]["newly_reachable_capped_first_alarms"] == 1
    assert result["menus"]["full"]["eligible_crossing_cannot_be_first"] == 1
    assert result["comparisons"]["full_vs_binary"]["gained_shots"] == [2]
    assert result["comparisons"]["full_vs_random"]["newly_reachable_difference"] == 0
    assert result["comparisons"]["full_vs_random"]["cannot_be_first_difference"] == 1


def test_missing_reference_input_binding_fails_even_if_resealed(synthetic_run):
    c = synthetic_run
    cfg = json.loads((c["run"] / "protocol.json").read_text())
    del cfg["input_sha256"][str(c["ref"] / "protocol.json")]
    put_json(c["run"] / "protocol.json", cfg)
    reseal_run(c["run"])
    with pytest.raises(ValueError, match="reference.*bound"):
        verifier.verify(c["run"], c["output"])


def test_noninteger_protocol_clock_is_rejected_even_when_numerically_equal(synthetic_run):
    c = synthetic_run
    cfg = json.loads((c["run"] / "protocol.json").read_text())
    cfg["first"] = 135.0
    put_json(c["run"] / "protocol.json", cfg)
    reseal_run(c["run"])
    with pytest.raises(ValueError, match="invalid protocol"):
        verifier.verify(c["run"], c["output"])


def test_all_seals_are_checked_before_any_hdf5_outcome_is_revealed(synthetic_run, monkeypatch):
    c = synthetic_run
    with (c["run"] / "shots/0021_100021.npz").open("ab") as f:
        f.write(b"synthetic corruption")
    def must_not_reveal(*args):
        pytest.fail("HDF5 outcome accessed before all seals were validated")
    monkeypatch.setattr(verifier, "raw_truth", must_not_reveal)
    with pytest.raises(ValueError, match="seal"):
        verifier.verify(c["run"], c["output"])


def test_conditional_mode_is_active_from_the_first_shot(synthetic_run, monkeypatch):
    c = synthetic_run
    run, ref = c["run"], c["ref"]
    checked = run.parent / "checked.json"
    monkeypatch.setattr(verifier, "REFS", {"conditional": (ref, checked, 21, 1, (1, 20), "conditional835")})
    ref_cfg = json.loads((ref / "protocol.json").read_text())
    ref_cfg["mode"] = "conditional"
    put_json(ref / "protocol.json", ref_cfg)
    old_done = json.loads((ref / "completed.json").read_text())
    old_done["mode"] = "conditional"
    old_done["protocol_sha256"] = sha(ref / "protocol.json")
    previous, rows = None, []
    for rank, sid in enumerate(ref_cfg["order"], 1):
        stem = f"{rank:04d}_{sid}"
        seal_path = ref / "shots" / f"{stem}_seal.json"
        note = json.loads(seal_path.read_text())
        note.update(previous_seal_sha256=previous, bank_enabled=True)
        put_json(seal_path, note)
        previous = sha(seal_path)
        rows.append(dict(note, seal_sha256=previous))
        if rank < 21:
            curve_path = run / "shots" / f"{stem}.npz"
            with np.load(curve_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays.update(anchor_norms=np.ones((1, 5), dtype=np.float32),
                          anchor_gradient_norms=np.array([[5., 3., 4.]], dtype=np.float64),
                          anchor_valid=np.ones((1, 4), dtype=bool))
            np.savez(curve_path, **arrays)
        new_seal = run / "shots" / f"{stem}_seal.json"
        note = json.loads(new_seal.read_text())
        note["active"] = True
        put_json(new_seal, note)
    old_done["sealed"] = rows
    put_json(ref / "completed.json", old_done)
    audit = json.loads(checked.read_text())
    audit["producer_sha256"] = sha(ref / "completed.json")
    put_json(checked, audit)
    oracle = json.loads(c["oracle"].read_text())
    cohort = oracle["cohorts"].pop("fixed360")
    cohort["parent_sha256"] = sha(ref / "completed.json")
    oracle["cohorts"]["conditional835"] = cohort
    put_json(c["oracle"], oracle)
    cfg = json.loads((run / "protocol.json").read_text())
    cfg.update(mode="conditional", active_rank=1)
    cfg["input_sha256"][str(ref / "protocol.json")] = sha(ref / "protocol.json")
    cfg["input_sha256"][str(ref / "completed.json")] = sha(ref / "completed.json")
    put_json(run / "protocol.json", cfg)
    done = json.loads((run / "completed.json").read_text())
    done["mode"] = "conditional"
    put_json(run / "completed.json", done)
    reseal_run(run)
    result = verifier.verify(run, c["output"])
    assert result["mode"] == "conditional"
    assert result["actual_arms_active"] == result["actual_arms"]
    assert result["anchor_sign_inversions"]["full"]["denominator"] == 21
