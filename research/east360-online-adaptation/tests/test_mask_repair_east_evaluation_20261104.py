import json

import h5py
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104 import core, east_evaluation as east
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import state_digest
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.simple_online import predict_shot


@pytest.fixture
def models():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(31)
    model = Simple23(Config(fast_site="tokens", fast_rule="normalized", input_view="source"),
                     torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].weight.normal_(0, .03)
        repair.net[-1].bias.fill_(.03)
    repair.eval().requires_grad_(False)
    constants = {"no_MNM": torch.ones(64), "no_MNM_CIII": -torch.ones(64)}
    return model, repair, constants


def _shot():
    values = np.random.default_rng(9).normal(size=(12, 23)).astype(np.float32)
    values[2, 1] = -10.0
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2] = False
    mask[4, 1] = False
    return dict(shot=101, device="EAST", values=values, available=mask,
                times=np.arange(len(values)) / 1000.)


def _map():
    return dict(gain=[2.0] * 23, offset=[1.0] * 23, calibration_ids=list(range(20)))


def test_actual_contract_and_map_after20_keep_mask_unchanged():
    shot, mapping = _shot(), _map()
    values, original = shot["values"].copy(), shot["available"].copy()
    first = east.prepare_actual(shot, 20, mapping)
    later = east.prepare_actual(shot, 21, mapping)
    expected = original.copy()
    expected[2, 1] = False  # EAST-specific pointwise sentinel
    np.testing.assert_array_equal(first["available"], expected)
    np.testing.assert_array_equal(later["available"], expected)
    np.testing.assert_array_equal(first["values"], values)
    np.testing.assert_array_equal(later["values"][expected], (2 * values + 1)[expected])
    np.testing.assert_array_equal(shot["available"], original)
    np.testing.assert_array_equal(shot["values"], values)
    assert not later["available"][:, 2].any()


def test_coverage_is_post_contract_premap_and_keeps_zero_support():
    shot = _shot()
    first, coverage = east.prepare_actual(shot, 20, _map(), report_coverage=True)
    later, later_coverage = east.prepare_actual(shot, 21, _map(), report_coverage=True)
    assert coverage == later_coverage == dict(
        shot=101, total_points=12,
        CIII=dict(available_points=10, available_fraction=10 / 12),
        MNM=dict(available_points=0, available_fraction=0.0),
    )
    np.testing.assert_array_equal(first["available"], later["available"])
    shot["available"][:, 1] = False
    _, absent = east.prepare_actual(shot, 21, _map(), report_coverage=True)
    assert absent["CIII"] == dict(available_points=0, available_fraction=0.0)


def test_four_actual_arms_are_causal_and_native_matches_baseline(models):
    base, repair, constants = models
    signal = east.prepare_actual(_shot(), 1, _map())
    original_state, repair_state = state_digest(base), state_digest(repair)
    curves = east.predict_actual_curves(base, repair, constants, signal, "cpu")
    baseline, updates = predict_shot(base, signal, intra=False, device="cpu")
    assert set(curves) == {"native", "constant", "conditional", "wrong_context"}
    assert updates == 0
    np.testing.assert_array_equal(curves["native"], baseline)
    changed = dict(signal, values=signal["values"].copy(), available=signal["available"].copy())
    changed["values"][7:] += 500
    changed["available"][7:, 1:] = False
    altered = east.predict_actual_curves(base, repair, constants, changed, "cpu", batch_size=3)
    for arm in curves:
        np.testing.assert_allclose(curves[arm][:7], altered[arm][:7], rtol=1e-6, atol=1e-7)
    assert state_digest(base) == original_state and state_digest(repair) == repair_state
    assert all(p.grad is None for p in (*base.parameters(), *repair.parameters()))


def test_reference_mismatch_and_tampered_seal_fail_before_scoring(tmp_path):
    path = tmp_path / "curve.npz"
    times = np.arange(3) / 1000.
    curve = np.array([.1, .2, .3], dtype=np.float32)
    np.savez(path, times=times, map_after20=curve)
    seal = dict(shot=1, curve_sha256=east.sha(path), current_outcome_used_in_prediction=False)
    receipt = tmp_path / "seal.json"
    receipt.write_text(json.dumps(seal))
    east.check_reference(path, receipt, seal, times, curve)
    with pytest.raises(ValueError, match="native|parity"):
        east.check_reference(path, receipt, seal, times, curve + 1e-6)
    with pytest.raises(ValueError, match="time|clock|parity"):
        east.check_reference(path, receipt, seal, times + 1e-9, curve)
    seal["curve_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal|hash|SHA"):
        east.check_reference(path, receipt, seal, times, curve)


def test_thirty_ms_is_strict():
    times, scores = np.array([0.0]), np.array([.9])
    assert not east.score_curve(times, scores, True, .030, .5)["timely"]
    assert east.score_curve(times, scores, True, .031, .5)["timely"]


def test_paired_counts_preserve_twenty_shot_boundary():
    per = {arm: [] for arm in ("native", "constant", "conditional", "wrong_context")}
    for sid in range(22):
        for arm in per:
            per[arm].append(dict(shot=sid, disruptive=sid % 2 == 0,
                                 timely=(arm == "conditional" and sid % 2 == 0),
                                 alarm=(arm == "conditional")))
    groups = east._paired(per)["conditional_vs_native"]["groups"]
    assert groups["startup20"]["shots"] == 20
    assert groups["startup20"]["timely_gained"] == 10
    assert groups["suffix340"]["shots"] == 2
    assert groups["suffix340"]["timely_gained"] == 1
    assert groups["suffix340"]["safe_alarms_gained"] == 1
    assert groups["full360"]["shots"] == 22


def test_cpu_end_to_end_all_sealed_before_outcome(tmp_path, monkeypatch, models):
    base, repair, constants = models
    base_before, repair_before = state_digest(base), state_digest(repair)
    path, ref = tmp_path / "101.hdf5", tmp_path / "reference"
    ref.mkdir()
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for channel in io23.RAW23:
            values = np.linspace(.1, .2, 6, dtype=np.float32)
            if channel == "CIII":
                values[1] = -10
            ds = data.create_dataset(channel, data=values)
            ds.attrs["SampleRate"], ds.attrs["StartTime"] = 1000., 0.
        meta = f.create_group("meta")
        meta.create_dataset("IsDisrupt", data=1)
        meta.create_dataset("DownTime", data=.030)
    monkeypatch.setattr(io23, "path_for", lambda device, sid: path)
    shot = simple_data.read_shot("EAST", 101, rate_contract="nominal")
    actual = east.prepare_actual(shot, 1, _map())
    reference, _ = predict_shot(base, actual, intra=False, device="cpu")
    archive = ref / "0001_101.npz"
    np.savez_compressed(archive, times=shot["times"], map_after20=reference)
    seal = dict(shot=101, curve_sha256=east.sha(archive), current_outcome_used_in_prediction=False)
    (ref / "0001_101_seal.json").write_text(json.dumps(seal))
    plan = tmp_path / "plan.md"
    plan.write_text("fixture")
    monkeypatch.setattr(east, "PLAN", plan)
    monkeypatch.setattr(east, "load_inputs", lambda: dict(ids=[101], order=[dict(shot=101)],
        mapping=_map(), hashes={"101": east.sha(path)}, seals=[seal], baseline_shots=ref, bound={},
        baseline_protocol={"threshold": .5, "source_state_sha256": base_before}))
    monkeypatch.setattr(east, "load_parent", lambda _p: (base, {"threshold": .5}))
    monkeypatch.setattr(east, "load_fit", lambda *_a, **_k: (repair, constants,
        {"threshold": .5, "source_state_sha256": base_before}))
    output, outcomes = tmp_path / "out", []
    signal_reader, signal_reads = io23.read_signals, []
    def checked_signal(device, sid, *args, **kwargs):
        signal_reads.append((device, sid))
        return signal_reader(device, sid, *args, **kwargs)
    monkeypatch.setattr(io23, "read_signals", checked_signal)
    read_outcome = io23.read_outcome
    def checked_outcome(device, sid):
        receipt = json.loads((output / "shots/0001_101_seal.json").read_text())
        sealed = output / "shots/0001_101.npz"
        assert receipt["curve_sha256"] == east.sha(sealed)
        with np.load(sealed, allow_pickle=False) as f:
            assert set(f.files) == {"times", "native", "constant", "conditional", "wrong_context"}
        outcomes.append((device, sid))
        return read_outcome(device, sid)
    monkeypatch.setattr(io23, "read_outcome", checked_outcome)
    result = east.run_east(output, compute_device="cpu")
    assert outcomes == [("EAST", 101)] and result["neural_updates"] == 0
    assert signal_reads == [("EAST", 101)]
    assert state_digest(base) == base_before and state_digest(repair) == repair_before
    assert (output / "paired.json").exists() and (output / "summary.json").exists()
    assert json.loads((output / "raw_hashes.json").read_text())["101"] == east.sha(path)
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage == [dict(shot=101, total_points=6,
        CIII=dict(available_points=5, available_fraction=5 / 6),
        MNM=dict(available_points=0, available_fraction=0.0))]
    seal_coverage = json.loads((output / "shots/0001_101_seal.json").read_text())["coverage"]
    assert seal_coverage == coverage[0]
    assert result["artifact_sha256"]["coverage.json"] == east.sha(output / "coverage.json")
    np.savez_compressed(archive, times=shot["times"], map_after20=reference + 1e-5)
    seal["curve_sha256"] = east.sha(archive)
    (ref / "0001_101_seal.json").write_text(json.dumps(seal))
    outcomes.clear()
    with pytest.raises(ValueError, match="native parity"):
        east.run_east(tmp_path / "bad_parity", compute_device="cpu")
    assert outcomes == []
