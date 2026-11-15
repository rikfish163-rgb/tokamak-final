"""CPU-only pure-forecast reference tests using synthetic data and source checkpoint."""

from copy import deepcopy
import hashlib
import json
import zipfile

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.native_token_probe_20261104.core import forward
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
    sha256,
)
from experiments.risk_aligned_online.source_anticipation_20261104 import (
    fidelity,
    forecast_reference as ref,
)


@pytest.fixture(scope="module")
def source():
    assert sha256(DEFAULT_PARENT) == PARENT_SHA256
    torch.set_num_threads(1)
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    assert base.center.device.type == "cpu"
    yield base
    torch.backends.mha.set_fastpath_enabled(old)


@pytest.fixture(autouse=True)
def source_unchanged(source):
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    hooks = {n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for n, m in source.named_modules()}
    yield
    assert all(not m.training for m in source.modules())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in state.items())
    assert hooks == {n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for n, m in source.named_modules()}


def test_single_score_matches_each_real_reference_column():
    rng = np.random.default_rng(13)
    forecasts = rng.normal(size=(4, 25, 23))
    hold = rng.normal(size=(25, 23))
    truth = rng.normal(size=(25, 23))
    mask = rng.random((25, 23)) > 0.3
    mask[:, 10] = False
    center = rng.normal(size=23)
    scale = np.exp(rng.normal(size=23))
    old = fidelity.score_window(forecasts, hold, truth, mask, center, scale)
    for i, name in enumerate(fidelity.HEADS):
        assert ref.single_score(forecasts[i], truth, mask, center, scale) == pytest.approx(old["loss"][name], abs=1e-15)
    assert ref.single_score(hold, truth, mask, center, scale) == pytest.approx(old["hold_loss"], abs=1e-15)
    assert ref.single_score(hold, truth, np.zeros_like(mask), center, scale) is None
    forecasts[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        ref.single_score(forecasts[0], truth, mask, center, scale)


def test_single_head_query_forecast_has_no_placeholder_heads_or_future_input(source):
    rng = torch.Generator(device="cpu").manual_seed(44)
    xq = source.center + source.scale * 0.2 * torch.randn(2, 128, 23, generator=rng)
    aq = torch.rand(2, 128, 23, generator=rng) > 0.1
    head = deepcopy(source.forecast_head).requires_grad_(False).eval()
    with torch.no_grad():
        head.bias.add_(0.1)
    head_before = {k: v.clone() for k, v in head.state_dict().items()}
    reference = np.empty((2, 4, 1, 25, 23), np.float32)
    with torch.no_grad():
        for vi, view in enumerate(fidelity.VIEWS):
            mask = aq[:1].clone()
            if view == "no_MNM_CIII":
                mask[:, :, [1, 2]] = False
            native = forward(source, xq[:1], mask, torch.zeros(1, 128))["forecast"][0].numpy()
            reference[vi, :, 0] = native
    pack = {"xq": xq, "aq": aq, "truth": torch.full((2, 25, 23), float("nan"))}
    bundle, parity = ref.forecast_pack(source, head, pack, {0: 0}, np.array([0], dtype=np.int64), reference)
    changed_future = dict(pack, truth=torch.randn(2, 25, 23))
    second, _ = ref.forecast_pack(source, head, changed_future, {0: 0}, np.array([0], dtype=np.int64), reference)
    assert set(bundle) == {"indices", "forecast_only"}
    assert bundle["forecast_only"].shape == (2, 1, 25, 23)
    np.testing.assert_array_equal(bundle["forecast_only"], second["forecast_only"])
    assert parity == 0.0
    mismatch = reference.copy()
    mismatch[0, 0, 0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="zero-fast"):
        ref.forecast_pack(source, head, pack, {0: 0}, np.array([0], dtype=np.int64), mismatch)
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in head_before.items())
    assert all(p.grad is None for p in head.parameters())


def _paired_fixture():
    rng = np.random.default_rng(21)
    values = rng.normal(size=(170, 23)).astype(np.float32)
    available = np.ones_like(values, dtype=bool)
    available[142, 5] = False
    signal = dict(values=values, available=available)
    center, scale = np.zeros(23), np.ones(23)
    predictions = rng.normal(size=(2, 4, 1, 25, 23)).astype(np.float32)
    holds = np.broadcast_to(np.arcsinh(values[140].astype(np.float64)), (2, 1, 25, 23)).copy()
    rows = [dict(index=0, shot=1, origin=127, query=140, stratum="safe", hdf_disruptive=False,
                 eligible=True, missing_future_points=0, prediction_index=0, views={}),
            dict(index=1, shot=1, origin=127, query=151, stratum="safe", hdf_disruptive=False,
                 eligible=False, missing_future_points=7, prediction_index=None, views=None)]
    for vi, view in enumerate(fidelity.VIEWS):
        t = fidelity.future_target(signal, 140, view)
        rows[0]["views"][view] = fidelity.score_window(predictions[vi, :, 0], holds[vi, 0], t["truth"], t["mask"], center, scale)
    records = [dict(shot=1, origin=127, query=140, anticipation_stratum="safe", hdf5_disruptive=False),
               dict(shot=1, origin=127, query=151, anticipation_stratum="safe", hdf5_disruptive=False)]
    old = dict(indices=np.array([0], dtype=np.int64), forecasts=predictions, hold=holds)
    return records, rows, old, signal, center, scale


def test_sealed_single_head_paired_score_and_missing_coverage(tmp_path):
    records, old_rows, old, signal, center, scale = _paired_fixture()
    bundle = dict(indices=np.array([0], dtype=np.int64), forecast_only=np.full((2, 1, 25, 23), 0.25, np.float32))
    ref.seal_forecasts(tmp_path, bundle)
    with np.load(tmp_path / "forecasts.npz", allow_pickle=False) as z:
        assert set(z.files) == {"indices", "forecast_only"}
    calls = []
    def reader(sid):
        assert (tmp_path / "prediction_seal.json").is_file()
        calls.append(sid)
        return signal
    rows, errors = ref.score_sealed(tmp_path, records, old_rows, old, {1: 170}, center, scale, reader)
    assert calls == [1] and rows[0]["eligible"] and rows[0]["prediction_index"] == 0
    assert not rows[1]["eligible"] and rows[1]["missing_future_points"] == 7
    assert rows[1]["views"] is None and rows[1]["prediction_index"] is None
    for view in fidelity.VIEWS:
        cell = rows[0]["views"][view]
        assert set(cell) >= {"mean_loss", "delta_vs_native", "delta_vs_risk_taught", "delta_vs_hold"}
        assert cell["delta_vs_native"] == pytest.approx(cell["mean_loss"] - old_rows[0]["views"][view]["loss"]["native_reference"])
    assert errors["single_vs_existing_score_window"] < 1e-14
    assert errors["reference_recalculation"] < 1e-14


def test_seal_and_reference_mask_disagreement_fail_before_false_pairing(tmp_path):
    records, old_rows, old, signal, center, scale = _paired_fixture()
    bundle = dict(indices=np.array([0], dtype=np.int64), forecast_only=np.zeros((2, 1, 25, 23), np.float32))
    ref.seal_forecasts(tmp_path, bundle)
    changed = dict(values=signal["values"], available=signal["available"].copy())
    changed["available"][145, 7] = False
    with pytest.raises(ValueError, match="reference|mask"):
        ref.score_sealed(tmp_path, records, old_rows, old, {1: 170}, center, scale, lambda _: changed)
    path = tmp_path / "forecasts.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="seal"):
        ref.score_sealed(tmp_path, records, old_rows, old, {1: 170}, center, scale,
                         lambda _: pytest.fail("future read before valid seal"))


def test_shot_macro_uses_paired_losses_and_handles_empty_groups():
    def row(shot, loss, baseline):
        cell = dict(mean_loss=loss, delta_vs_native=loss-baseline,
                    delta_vs_risk_taught=loss-(baseline+1), delta_vs_hold=loss-(baseline+2))
        return dict(shot=shot, stratum="safe", eligible=True, views={v: cell for v in fidelity.VIEWS})
    rows = [row(1, 1.0, 0.5), row(1, 3.0, 2.5), row(2, 10.0, 9.5)]
    s = ref.summarize(rows)
    assert s["native"]["safe"]["mean_loss"] == 6.0
    assert s["native"]["safe"]["delta_vs_native"] == 0.5
    assert s["native"]["safe"]["delta_vs_risk_taught"] == -0.5
    assert s["native"]["safe"]["delta_vs_hold"] == -1.5
    assert s["native"][fidelity.STRATA[-1]]["mean_loss"] is None


def test_prior_claims_require_source_only_and_forecast_arm():
    done = dict(status="COMPLETED", source_unchanged=True, east_reads=0, test_signals_read=0)
    protocol = dict(scope="SOURCE_ONLY_DIRECT_DIAGNOSTIC_DECODER_NOT_JTEXT_ONLINE", source_sha256=PARENT_SHA256,
                    source_parameters_updated=0, east_reads=0, test_signals_read=0,
                    arms=["forecast_only", "mean_meta", "worst_meta"],
                    absolute_maintenance_weight=0.2, initial_parameter_anchor_weight=1e-4)
    ref.validate_prior_claims(done, protocol)
    for field in ("east_reads", "test_signals_read", "source_parameters_updated"):
        bad = dict(protocol, **{field: 1})
        with pytest.raises(ValueError):
            ref.validate_prior_claims(done, bad)
    with pytest.raises(ValueError):
        ref.validate_prior_claims(dict(done, source_unchanged=False), protocol)


def test_prior_archive_hashes_are_checked_before_checkpoint_use(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    prior.mkdir()
    trial = b"synthetic-archived-producer"
    trial_sha = hashlib.sha256(trial).hexdigest()
    checkpoint = b"synthetic-not-deserialized"
    checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
    monkeypatch.setattr(ref, "TRIAL_SHA256", trial_sha)
    monkeypatch.setattr(ref, "PRIOR_SHA256", checkpoint_sha)
    (prior / "decoders.pt").write_bytes(checkpoint)
    with zipfile.ZipFile(prior / "code_snapshot.zip", "w") as archive:
        archive.writestr(ref.TRIAL_NAME, trial)
    protocol = dict(scope="SOURCE_ONLY_DIRECT_DIAGNOSTIC_DECODER_NOT_JTEXT_ONLINE",
                    source_sha256=PARENT_SHA256, source_parameters_updated=0,
                    east_reads=0, test_signals_read=0, arms=["forecast_only"],
                    absolute_maintenance_weight=0.2, initial_parameter_anchor_weight=1e-4,
                    bound_code_sha256={str(fidelity.io23.ROOT / ref.TRIAL_NAME): trial_sha})
    (prior / "protocol.json").write_text(json.dumps(protocol))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    done = dict(status="COMPLETED", source_unchanged=True, east_reads=0, test_signals_read=0,
                artifact_sha256={p.name: digest(p) for p in prior.iterdir()})
    (prior / "completed.json").write_text(json.dumps(done))
    ref.verify_prior(prior)
    with zipfile.ZipFile(prior / "code_snapshot.zip", "w") as archive:
        archive.writestr(ref.TRIAL_NAME, b"tampered-producer")
    done["artifact_sha256"]["code_snapshot.zip"] = digest(prior / "code_snapshot.zip")
    (prior / "completed.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="producer hash"):
        ref.verify_prior(prior)


def test_reference_claims_require_same_design_source_and_locked_coverage():
    state = {name: "a" * 64 for name in fidelity.HEADS}
    done = dict(status="COMPLETED_SOURCE_VAL_FIDELITY", source_and_heads_unchanged=True,
                source_updates=0, head_updates=0, test_signal_reads=0, east_signal_reads=0,
                records=946, complete_horizon=764, incomplete_horizon=182,
                state_sha256_before=state, state_sha256_after=state)
    protocol = dict(scope="SOURCE_VAL_POSTQUERY_FIDELITY_NOT_ADAPTIVE_FORECAST_OR_RISK_GATE",
                    source_sha256=PARENT_SHA256, source_role="val", horizon=25, records=946,
                    heads=list(fidelity.HEADS), views=list(fidelity.VIEWS),
                    design_completion_sha256="b" * 64, fit_teaching_objective="risk_utility",
                    raw_sha256={str(i): "c" * 64 for i in range(145)},
                    adaptive_forecast_evaluated=False, risk_gate_changed=False,
                    operational_gain_measured=False)
    ref.validate_reference_claims(done, protocol, "b" * 64)
    with pytest.raises(ValueError, match="reference"):
        ref.validate_reference_claims(done, protocol, "d" * 64)
    with pytest.raises(ValueError, match="reference"):
        ref.validate_reference_claims(dict(done, complete_horizon=763), protocol, "b" * 64)
    with pytest.raises(ValueError, match="reference"):
        ref.validate_reference_claims(done, dict(protocol, source_role="test"), "b" * 64)


def test_reference_rows_must_match_coordinates_and_prediction_indices():
    records, old_rows, old, *_ = _paired_fixture()
    ref.validate_reference_rows(records, old_rows, old["indices"])
    bad = deepcopy(old_rows)
    bad[0]["query"] += 1
    with pytest.raises(ValueError, match="reference"):
        ref.validate_reference_rows(records, bad, old["indices"])
    bad = deepcopy(old_rows)
    bad[1]["prediction_index"] = 0
    with pytest.raises(ValueError, match="reference"):
        ref.validate_reference_rows(records, bad, old["indices"])


def test_missing_comparison_plan_fails_before_preflight_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "PLAN", tmp_path / "missing-plan.md")
    monkeypatch.setattr(ref, "verify_prior", lambda *_a: pytest.fail("preflight must not run"))
    with pytest.raises(FileNotFoundError, match="comparison plan"):
        ref.main(tmp_path / "reference", tmp_path / "design", tmp_path / "prior", tmp_path / "output")


@pytest.mark.parametrize("change_comparison_plan", [False, True])
def test_synthetic_main_writes_one_head_after_seal_without_gpu(source, tmp_path, monkeypatch, change_comparison_plan):
    reference, design, prior, output = (tmp_path / name for name in ("reference", "design", "prior", "output"))
    comparison_plan = tmp_path / "FORECAST_ONLY_REFERENCE_PLAN_20261104.md"
    comparison_plan.write_text("Synthetic comparison plan; no real data.")
    monkeypatch.setattr(ref, "PLAN", comparison_plan)
    for root in (reference, design, prior):
        root.mkdir()
        (root / "completed.json").write_text("{}")
    (reference / "forecasts.npz").write_bytes(b"synthetic-reference")
    (reference / "prediction_seal.json").write_text(json.dumps({"forecast_sha256": fidelity.sha(reference / "forecasts.npz")}))
    ids = list(range(1_000_000, 1_000_145))
    records = [dict(shot=ids[i % 145], origin=127, query=140 if i < 764 else 169,
                    anticipation_stratum=fidelity.STRATA[i % 4]) for i in range(946)]
    old_rows = [dict(eligible=i < 764, missing_future_points=0 if i < 764 else 25,
                     views={v: {"scorable": True} for v in fidelity.VIEWS} if i < 764 else None)
                for i in range(946)]
    old = dict(indices=np.arange(764, dtype=np.int64), forecasts=np.zeros((2, 4, 764, 25, 23), np.float32))
    hashes = {str(sid): "a" * 64 for sid in ids}
    parent = deepcopy(source)
    state = fidelity.state_digest(parent)
    context = dict(records=records, rows=old_rows, old=old,
                   protocol={"plan_sha256": fidelity.sha(fidelity.PLAN), "raw_sha256": hashes},
                   completed={"state_sha256_before": {"native_reference": state}},
                   provenance={"input_hashes": hashes})
    evidence = []
    monkeypatch.setattr(ref, "verify_prior", lambda *_a: {"scope": "synthetic-source-only"})
    monkeypatch.setattr(ref, "verify_reference", lambda *_a: context)
    monkeypatch.setattr(fidelity, "configure", lambda: None)
    monkeypatch.setattr(fidelity, "load_parent", lambda _p: (parent, {}))
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU synthetic test")
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"source_sha256": PARENT_SHA256,
                        "heads": {"forecast_only": deepcopy(source.forecast_head.state_dict())}})
    monkeypatch.setattr(fidelity.io23, "read_signals", lambda _d, _s: {"times": np.arange(170)})
    def windows(rows, _schedule, _device, *, role):
        assert role == "val"
        for sid in ids:
            fidelity.io23.read_signals("J-TEXT", sid)
        return {"xq": torch.zeros(1), "aq": torch.zeros(1)}, {}, hashes
    monkeypatch.setattr(fidelity, "load_windows", windows)
    def forecast(_base, _head, _pack, _location, indices, _old):
        assert not (output / "prediction_seal.json").exists()
        return dict(indices=indices.copy(), forecast_only=np.zeros((2, 764, 25, 23), np.float32)), 0.0
    monkeypatch.setattr(ref, "forecast_pack", forecast)
    def future(sid, _allowed, _hashes, _lengths):
        assert (output / "prediction_seal.json").is_file()
        fidelity.io23.read_signals("J-TEXT", sid)
        evidence.append(sid)
        return {}
    monkeypatch.setattr(fidelity, "read_future_shot", future)
    def score(out, recs, previous, _old, _lengths, _center, _scale, read_signal):
        assert (out / "prediction_seal.json").is_file()
        read_signal(ids[0])
        if change_comparison_plan:
            comparison_plan.write_text("Changed while run was in progress.")
        rows = []
        for i, r in enumerate(recs):
            cell = dict(scorable=True, mean_loss=1.0, delta_vs_native=0.1,
                        delta_vs_risk_taught=-0.1, delta_vs_hold=0.2)
            rows.append(dict(shot=r["shot"], stratum=r["anticipation_stratum"], eligible=i < 764,
                             views={v: cell for v in fidelity.VIEWS} if i < 764 else None))
        return rows, {"single_vs_existing_score_window": 0.0, "reference_recalculation": 0.0}
    monkeypatch.setattr(ref, "score_sealed", score)
    if change_comparison_plan:
        with pytest.raises(AssertionError, match="bound"):
            ref.main(reference, design, prior, output)
        return
    ref.main(reference, design, prior, output)
    done = json.loads((output / "completed.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert done["status"] == "COMPLETED_SOURCE_VAL_FORECAST_ONLY_FIDELITY"
    assert done["records"] == 946 and done["complete_horizon"] == 764
    assert done["state_sha256_before"] == done["state_sha256_after"] and evidence == [ids[0]]
    with np.load(output / "forecasts.npz", allow_pickle=False) as z:
        assert set(z.files) == {"indices", "forecast_only"}
    assert protocol["comparison_plan_sha256"] == fidelity.sha(comparison_plan)
    assert protocol["input_sha256"][str(comparison_plan)] == fidelity.sha(comparison_plan)
    assert protocol["input_sha256"][str(fidelity.PLAN)] == fidelity.sha(fidelity.PLAN)
    with zipfile.ZipFile(output / "code.zip") as archive:
        assert archive.read(comparison_plan.name) == comparison_plan.read_bytes()
        assert archive.read(fidelity.PLAN.name) == fidelity.PLAN.read_bytes()
    fidelity.verify_manifest(output, done["artifact_sha256"])
