"""CPU-only sealed forecast-fidelity tests; no real shot arrays or GPU."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.dual_history_ttt_20261101.stream import targets
from experiments.risk_aligned_online.native_token_probe_20261104.core import forward
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
    sha256,
)
from experiments.risk_aligned_online.source_anticipation_20261104 import fidelity


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
def frozen_source_unchanged(source):
    before = {k: v.detach().clone() for k, v in source.state_dict().items()}
    hooks = {
        name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
        for name, module in source.named_modules()
    }
    yield
    assert all(not m.training for m in source.modules())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in before.items())
    assert hooks == {
        name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
        for name, module in source.named_modules()
    }


def test_future_targets_require_complete_time_horizon_and_real_mask():
    query = 140
    values = np.arange(170 * 23, dtype=np.float32).reshape(170, 23)
    available = np.ones_like(values, dtype=bool)
    available[query, 4] = False
    available[query + 2, 5] = False
    signal = dict(values=values, available=available)
    actual = fidelity.future_target(signal, query, "native")
    old, mask = targets(signal, query, query + 25)
    assert actual["eligible"] and actual["missing_future_points"] == 0
    np.testing.assert_array_equal(actual["truth"], old["real"][0])
    np.testing.assert_array_equal(actual["mask"], mask[0])
    assert not actual["mask"][:, 10].any()
    assert not actual["mask"][:, 4].any()
    assert not actual["mask"][1, 5] and not actual["mask"][-2, 5]
    view = fidelity.future_target(signal, query, "no_MNM_CIII")
    assert not view["mask"][:, 1:3].any()
    np.testing.assert_array_equal(view["truth"], actual["truth"])
    short = fidelity.future_target(signal, 151, "native")
    assert short["eligible"] is False and short["missing_future_points"] == 7
    assert short["truth"] is None and short["mask"] is None


def test_forecasts_use_only_query_prefix_and_skip_incomplete(source):
    generator = torch.Generator(device="cpu").manual_seed(8)
    xq = source.center + source.scale * 0.25 * torch.randn(2, 128, 23, generator=generator)
    aq = torch.rand(2, 128, 23, generator=generator) > 0.1
    pack = {"xq": xq, "aq": aq, "truth": torch.full((2, 25, 23), float("nan"))}
    records = [dict(shot=1, query=140), dict(shot=2, query=151)]
    lengths = {1: 170, 2: 170}
    heads = {"native_reference": None}
    for name in fidelity.ARMS:
        heads[name] = deepcopy(source.forecast_head).requires_grad_(False).eval()
    with torch.no_grad():
        heads[fidelity.ARMS[0]].bias.add_(0.1)
    before = {k: {n: v.clone() for n, v in h.state_dict().items()} for k, h in heads.items() if h is not None}
    first = fidelity.forecast_pack(source, heads, pack, {0: 0, 1: 1}, records, lengths)
    changed_future = dict(pack, truth=torch.randn(2, 25, 23))
    second = fidelity.forecast_pack(source, heads, changed_future, {0: 0, 1: 1}, records, lengths)
    np.testing.assert_array_equal(first["indices"], [0])
    assert first["forecasts"].shape == (2, 4, 1, 25, 23)
    assert first["hold"].shape == (2, 1, 25, 23)
    np.testing.assert_array_equal(first["forecasts"], second["forecasts"])
    np.testing.assert_array_equal(first["hold"], second["hold"])
    with torch.no_grad():
        expected = forward(source, xq[:1], aq[:1], torch.zeros(1, 128))["forecast"]
    np.testing.assert_array_equal(first["forecasts"][0, 0, 0], expected[0].numpy())
    assert not np.array_equal(first["forecasts"][0, 0, 0], first["forecasts"][0, 1, 0])
    for name, saved in before.items():
        assert all(torch.equal(v, heads[name].state_dict()[n]) for n, v in saved.items())
        assert all(p.grad is None for p in heads[name].parameters())


def test_source_coordinate_smooth_l1_and_masked_hold():
    center = np.zeros(23)
    scale = np.ones(23)
    truth = np.zeros((25, 23))
    truth[0, 0] = np.sinh(2.0)
    truth[1, 1] = np.sinh(0.5)
    truth[3, 3] = np.nan
    mask = np.zeros((25, 23), dtype=bool)
    mask[0, 0] = mask[1, 1] = True
    forecasts = np.zeros((4, 25, 23))
    forecasts[1, 0, 0] = 1.0
    hold = np.zeros((25, 23))
    result = fidelity.score_window(forecasts, hold, truth, mask, center, scale)
    assert result["observed_values"] == 2
    assert result["loss"]["native_reference"] == pytest.approx((1.5 + 0.125) / 2)
    assert result["loss"][fidelity.ARMS[0]] == pytest.approx((0.5 + 0.125) / 2)
    assert result["hold_loss"] == pytest.approx(result["loss"]["native_reference"])
    empty = fidelity.score_window(forecasts, hold, truth, np.zeros_like(mask), center, scale)
    assert empty["observed_values"] == 0 and empty["loss"] is None and empty["hold_loss"] is None
    forecasts[0, 1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fidelity.score_window(forecasts, hold, truth, mask, center, scale)


def test_shot_macro_then_group_is_not_window_weighted():
    def row(shot, value, stratum="safe"):
        score = dict(
            observed_values=25,
            loss={h: value + i for i, h in enumerate(fidelity.HEADS)},
            hold_loss=value + 10,
        )
        return dict(shot=shot, stratum=stratum, eligible=True, views={v: score for v in fidelity.VIEWS})
    rows = [row(1, 1.0), row(1, 3.0), row(2, 10.0), row(3, 8.0, fidelity.STRATA[1])]
    summary = fidelity.summarize(rows)
    result = summary["native"]["native_reference"]["safe"]
    assert result["shots"] == 2 and result["windows"] == 3
    assert result["mean_loss"] == 6.0 and result["delta_vs_native"] == 0
    assert result["delta_vs_hold"] == -10.0
    assert summary["native"][fidelity.ARMS[0]]["safe"]["delta_vs_native"] == 1.0
    assert summary["native"]["native_reference"][fidelity.STRATA[-1]]["mean_loss"] is None


def _bundle_and_signal():
    values = np.arange(170 * 23, dtype=np.float32).reshape(170, 23) / 1000
    available = np.ones_like(values, dtype=bool)
    signal = dict(values=values, available=available)
    last = np.arcsinh(values[140].astype(np.float64))
    hold = np.broadcast_to(last, (2, 1, 25, 23)).copy()
    forecasts = np.zeros((2, 4, 1, 25, 23), dtype=np.float32)
    return dict(indices=np.array([0], dtype=np.int64), forecasts=forecasts, hold=hold), signal


def test_scoring_reads_only_after_seal_and_excludes_short_horizon(tmp_path):
    bundle, signal = _bundle_and_signal()
    records = [dict(shot=1, query=140, anticipation_stratum="safe"),
               dict(shot=1, query=151, anticipation_stratum="safe")]
    fidelity.seal_forecasts(tmp_path, bundle)
    calls = []
    def read_signal(sid):
        assert (tmp_path / "prediction_seal.json").is_file()
        calls.append(sid)
        return signal
    rows = fidelity.score_sealed(tmp_path, records, {1: 170}, np.zeros(23), np.ones(23), read_signal)
    assert calls == [1]
    assert rows[0]["eligible"] and rows[0]["views"]["native"]["observed_values"] == 550
    assert not rows[1]["eligible"] and rows[1]["missing_future_points"] == 7
    assert rows[1]["prediction_index"] is None and rows[1]["views"] is None


def test_tampered_forecast_is_rejected_before_reading_any_future(tmp_path):
    bundle, _ = _bundle_and_signal()
    fidelity.seal_forecasts(tmp_path, bundle)
    path = tmp_path / "forecasts.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    def forbidden(_sid):
        pytest.fail("unsealed forecast must not read future")
    with pytest.raises(ValueError, match="seal"):
        fidelity.score_sealed(tmp_path, [dict(shot=1, query=140, anticipation_stratum="safe")],
                              {1: 170}, np.zeros(23), np.ones(23), forbidden)


def test_nonfinite_forecasts_are_rejected_before_seal(tmp_path):
    bundle, _ = _bundle_and_signal()
    bundle["forecasts"][0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fidelity.seal_forecasts(tmp_path, bundle)
    assert not (tmp_path / "forecasts.npz").exists()
    assert not (tmp_path / "prediction_seal.json").exists()


def test_complete_horizon_with_no_observed_targets_is_unscorable(tmp_path):
    bundle, signal = _bundle_and_signal()
    signal["available"][141:166] = False
    fidelity.seal_forecasts(tmp_path, bundle)
    records = [dict(shot=1, query=140, anticipation_stratum="safe")]
    rows = fidelity.score_sealed(tmp_path, records, {1: 170}, np.zeros(23), np.ones(23), lambda _s: signal)
    assert rows[0]["eligible"] and rows[0]["prediction_index"] == 0
    for view in fidelity.VIEWS:
        assert rows[0]["views"][view]["scorable"] is False
        assert rows[0]["views"][view]["loss"] is None
        assert rows[0]["views"][view]["hold_loss"] is None
    assert fidelity.summarize(rows)["native"]["native_reference"]["overall"]["mean_loss"] is None


def test_validate_val_coordinates_rejects_role_leaks_and_duplicates():
    ids = list(range(1_000_000, 1_000_145))
    archived = []
    for i in range(946):
        origin = 127 + 25 * (i // 145)
        archived.append(dict(shot=ids[i % 145], origin=origin, maturity=origin + 25,
                             query=origin + 26, role="val", device="J-TEXT",
                             feedback_observed=True, disruptive=False, label=0.0,
                             group="safe", lead_ms=None))
    records = [fidelity.annotate_val(r, disruptive=False, down=None, query_time=2.0, horizon_ms=25.0)
               for r in archived]
    fidelity.validate_val_records(records, archived, ids)
    changed = deepcopy(records)
    changed[0]["role"] = "test"
    with pytest.raises(ValueError):
        fidelity.validate_val_records(changed, archived, ids)
    changed = deepcopy(records)
    changed[0] = changed[1]
    with pytest.raises(ValueError):
        fidelity.validate_val_records(changed, archived, ids)


def test_manifest_rejects_traversal_and_changed_bytes(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    digest = hashlib.sha256(b"a").hexdigest()
    assert fidelity.verify_manifest(root, {"a.bin": digest}) == {"a.bin": digest}
    with pytest.raises(ValueError):
        fidelity.verify_manifest(root, {"../a.bin": digest})
    (root / "a.bin").write_bytes(b"b")
    with pytest.raises(ValueError, match="hash"):
        fidelity.verify_manifest(root, {"a.bin": digest})


def test_future_reader_checks_role_and_hash_before_access(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.bin"
    path.write_bytes(b"synthetic-only")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(fidelity.io23, "path_for", lambda device, sid: path)
    def reader(device, sid):
        calls.append((device, sid))
        return dict(values=np.ones((170, 23), np.float32),
                    available=np.ones((170, 23), dtype=bool))
    monkeypatch.setattr(fidelity.simple_data, "read_shot", reader)
    with pytest.raises(ValueError, match="VAL"):
        fidelity.read_future_shot(2, {1}, {"1": digest}, {1: 170})
    assert calls == []
    with pytest.raises(ValueError, match="hash"):
        fidelity.read_future_shot(1, {1}, {"1": "0" * 64}, {1: 170})
    assert calls == []
    result = fidelity.read_future_shot(1, {1}, {"1": digest}, {1: 170})
    assert calls == [("J-TEXT", 1)] and result["values"].shape == (170, 23)


def test_fit_design_provenance_rejects_swapped_packet_and_tampering(tmp_path):
    fit, design, other = (tmp_path / name for name in ("fit", "design", "other"))
    for root in (fit, design, other):
        root.mkdir()
    def dump(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    dump(design / "protocol.json", {"source_sha256": PARENT_SHA256})
    dump(design / "val_records.json", [])
    (design / "code.zip").write_bytes(b"design-snapshot")
    design_files = {p.name: digest(p) for p in design.iterdir()}
    dump(design / "completed.json", {"status": "COMPLETED", "artifact_sha256": design_files})
    (fit / "heads.pt").write_bytes(b"not-deserialized-in-this-test")
    code = Path(fidelity.__file__).resolve()
    with zipfile.ZipFile(fit / "code.zip", "w") as z:
        z.write(code, str(code.relative_to(fidelity.io23.ROOT)))
    protocol = dict(source_sha256=PARENT_SHA256, design=str(design.resolve()),
                    arms=list(fidelity.ARMS), source_parameter_updates=0, teaching_objective="risk_utility",
                    input_sha256={str(design / "completed.json"): digest(design / "completed.json"),
                                  str(DEFAULT_PARENT): PARENT_SHA256},
                    code_sha256={str(code): digest(code)})
    dump(fit / "protocol.json", protocol)
    fit_files = {p.name: digest(p) for p in fit.iterdir()}
    fit_done = {"status": "TRAINED_NOT_VALIDATED", "artifact_sha256": fit_files,
                "source_and_native_head_unchanged": True,
                "val_signal_reads": 0, "test_signal_reads": 0, "east_signal_reads": 0}
    dump(fit / "completed.json", fit_done)
    fidelity.verify_fit_design(fit, design)
    declared_cosine = dict(protocol, teaching_objective="cosine")
    dump(fit / "protocol.json", declared_cosine)
    cosine_done = dict(fit_done, artifact_sha256=dict(fit_files, **{"protocol.json": digest(fit / "protocol.json")}))
    dump(fit / "completed.json", cosine_done)
    assert fidelity.verify_fit_design(fit, design)["teaching_objective"] == "cosine"
    for unsupported in (None, "unknown"):
        altered = dict(protocol)
        if unsupported is None:
            del altered["teaching_objective"]
        else:
            altered["teaching_objective"] = unsupported
        dump(fit / "protocol.json", altered)
        changed_done = dict(fit_done, artifact_sha256=dict(fit_files, **{"protocol.json": digest(fit / "protocol.json")}))
        dump(fit / "completed.json", changed_done)
        with pytest.raises(ValueError, match="teaching_objective"):
            fidelity.verify_fit_design(fit, design)
    dump(fit / "protocol.json", protocol)
    dump(fit / "completed.json", fit_done)
    with pytest.raises((FileNotFoundError, ValueError)):
        fidelity.verify_fit_design(fit, other)
    (design / "val_records.json").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        fidelity.verify_fit_design(fit, design)


def test_synthetic_main_seals_before_postquery_read_and_writes_complete_artifacts(
    source, tmp_path, monkeypatch
):
    fit, design, output = (tmp_path / name for name in ("fit", "design", "output"))
    fit.mkdir()
    design.mkdir()
    (fit / "completed.json").write_text("{}")
    (design / "completed.json").write_text("{}")
    ids = list(range(1_000_000, 1_000_145))
    records = [dict(shot=ids[i % 145], query=140 + i // 145,
                    anticipation_stratum=fidelity.STRATA[i % 4]) for i in range(946)]
    (design / "val_records.json").write_text(json.dumps(records))
    lengths = {sid: 147 for sid in ids}
    lengths[ids[0]] = 166
    fake_hashes = {str(sid): "a" * 64 for sid in ids}
    parent = deepcopy(source)
    head_states = {name: deepcopy(source.forecast_head.state_dict()) for name in fidelity.ARMS}
    evidence = []
    monkeypatch.setattr(fidelity, "verify_fit_design", lambda *_a: {"teaching_objective": "risk_utility"})
    monkeypatch.setattr(fidelity, "validate_val_records", lambda *_a: None)
    monkeypatch.setattr(fidelity, "load_records", lambda role: (records, ids, {"input_hashes": fake_hashes}))
    monkeypatch.setattr(fidelity, "configure", lambda: None)
    monkeypatch.setattr(fidelity, "load_parent", lambda _p: (parent, {}))
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU synthetic test")
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"source_sha256": PARENT_SHA256, "heads": head_states})
    monkeypatch.setattr(fidelity.io23, "read_signals", lambda _d, sid: {"times": np.arange(lengths[sid])})
    def windows(rows, schedule, device, *, role):
        assert role == "val" and len(rows) == 946
        for sid in ids:
            fidelity.io23.read_signals("J-TEXT", sid)
        values = source.center.view(1, 1, 23).expand(len(rows), 128, 23)
        return {"xq": values, "aq": torch.ones_like(values, dtype=torch.bool)}, dict(enumerate(range(len(rows)))), fake_hashes
    monkeypatch.setattr(fidelity, "load_windows", windows)
    def future(sid, allowed, hashes, seen_lengths):
        assert (output / "prediction_seal.json").is_file()
        assert sid in allowed and hashes[str(sid)] == fake_hashes[str(sid)]
        fidelity.io23.read_signals("J-TEXT", sid)
        evidence.append(sid)
        values = np.broadcast_to(source.center.detach().numpy(), (seen_lengths[sid], 23)).copy()
        return dict(values=values, available=np.ones_like(values, dtype=bool))
    monkeypatch.setattr(fidelity, "read_future_shot", future)
    fidelity.main(fit, design, output)
    result = json.loads((output / "completed.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    rows = json.loads((output / "rows.json").read_text())
    assert result["status"] == "COMPLETED_SOURCE_VAL_FIDELITY"
    assert result["records"] == 946 and result["complete_horizon"] == 1
    assert result["incomplete_horizon"] == 945 and evidence == [ids[0]]
    assert set(result["state_sha256_before"]) == set(fidelity.HEADS)
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert result["source_val_prepare_signal_shots"] == 145 and result["source_val_postseal_signal_shots"] == 1
    assert fidelity.PLAN.name == "SOURCE_FORECAST_FIDELITY_PLAN_20261104.md"
    assert protocol["plan_sha256"] == hashlib.sha256(fidelity.PLAN.read_bytes()).hexdigest()
    assert protocol["fit_teaching_objective"] == "risk_utility"
    assert rows[0]["eligible"] and all(not r["eligible"] and r["views"] is None for r in rows[1:])
    fidelity.verify_manifest(output, result["artifact_sha256"])
