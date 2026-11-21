import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104 import core, evaluation
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import state_digest
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def models():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cfg = Config(fast_site="tokens", fast_rule="normalized", input_view="source")
    base = Simple23(cfg, torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    repair = core.TokenRepair()
    with torch.no_grad():
        repair.net[-1].weight.normal_(0, 0.03)
        repair.net[-1].bias.fill_(0.03)
    repair.eval().requires_grad_(False)
    constants = {"no_MNM": torch.full((64,), 2.0),
                 "no_MNM_CIII": torch.full((64,), -3.0)}
    return base, repair, constants


def test_all_arms_use_core_eligibility_and_rms_bound(models):
    base, repair, constants = models
    x = torch.randn(1, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    available[:, :8] = False                 # padding, never repair
    available[:, 8:16, 2] = False           # native observation missing MNM
    available[:, 16:24, 1] = False          # native observation missing CIII
    for arm in ("native", "constant", "conditional", "wrong_context"):
        out = evaluation.evaluate_batch(base, repair, constants, x, available,
                                        view="native", arm=arm)
        delta = out["embedding_delta"]
        assert torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0]))
        assert torch.equal(delta[:, 3:], torch.zeros_like(delta[:, 3:]))
        assert (delta.double().square().mean(-1).sqrt() <= 0.25).all()
        if arm == "native":
            assert torch.count_nonzero(delta) == 0
        else:
            assert torch.count_nonzero(delta[:, 1:3]) > 0
    constant = evaluation.evaluate_batch(base, repair, constants, x, available,
                                          view="native", arm="constant")
    assert constant["embedding_delta"][0, 1, 0] > 0
    assert constant["embedding_delta"][0, 2, 0] < 0
    # The underlying core still rejects model/dtype/availability violations.
    with pytest.raises(ValueError):
        evaluation.evaluate_batch(base, repair, constants, x.double(), available,
                                  view="native", arm="wrong_context")


def test_wrong_context_reverses_full_current_window_not_batch(models):
    base, repair, constants = models
    torch.manual_seed(51)
    x = torch.randn(2, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    available[:, :, 2] = False
    available[0, :24] = False
    available[1, :8] = False
    native = evaluation.evaluate_batch(base, repair, constants, x, available,
                                       view="no_MNM", arm="native")
    wrong = evaluation.evaluate_batch(base, repair, constants, x, available,
                                      view="no_MNM", arm="wrong_context")
    groups = available.reshape(2, 16, 8, 23)
    valid = groups.any(-1)
    fractions = groups[..., (1, 2)].sum(-2).float() / valid.sum(-1).clamp_min(1)[..., None]
    context = torch.cat((native["embedding_before"], fractions), dim=-1).flip(1)
    eligible = wrong["repair_tokens"]
    expected = repair(context[eligible])
    assert expected.double().square().mean(-1).sqrt().max() < 0.25
    torch.testing.assert_close(wrong["embedding_delta"][eligible], expected)
    for i in range(2):
        single = evaluation.evaluate_batch(base, repair, constants, x[i:i+1], available[i:i+1],
                                           view="no_MNM", arm="wrong_context")
        torch.testing.assert_close(single["embedding_delta"], wrong["embedding_delta"][i:i+1])


def test_prefix_curves_do_not_depend_on_future_or_batch_boundaries(models):
    base, repair, constants = models
    rng = np.random.default_rng(7)
    values = rng.normal(size=(12, 23)).astype(np.float32)
    available = np.ones_like(values, dtype=bool)
    available[1, 1] = False
    shot = dict(shot=1, values=values, available=available)
    original = evaluation.predict_curves(base, repair, constants, shot, "cpu", batch_size=4)
    changed = dict(shot=1, values=values.copy(), available=available.copy())
    changed["values"][7:] += 1000
    changed["available"][7:, 1:] = False
    altered = evaluation.predict_curves(base, repair, constants, changed, "cpu", batch_size=3)
    assert len(original) == 12
    for name in original:
        np.testing.assert_allclose(original[name][:7], altered[name][:7], rtol=1e-6, atol=1e-7)


def test_strict_ten_ms_scoring_and_fixed_threshold():
    times = np.array([0.0, 0.005, 0.01])
    scores = np.array([0.8, 0.2, 0.2])
    at_boundary = evaluation.score_source(times, scores, True, 0.010, 0.5)
    assert at_boundary["alarm"] and at_boundary["lead_ms"] == 10.0
    assert not at_boundary["timely"]
    assert evaluation.score_source(times, scores, True, 0.011, 0.5)["timely"]
    assert not evaluation.score_source(times, scores, True, 0.011, 0.9)["alarm"]
    assert evaluation.score_source(times, scores, False, None, 0.5)["alarm"]


def _sha(path):
    return evaluation.sha(Path(path))


@pytest.fixture
def source_fixture(tmp_path, monkeypatch, models):
    base, repair, constants = models
    parent = tmp_path / "source.pt"
    parent.write_bytes(b"fixture-parent")
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    source_sha = "a" * 64
    fit = tmp_path / "fit"
    fit.mkdir()
    torch.save(dict(repair=repair.state_dict(), constant_delta=constants,
                    source_sha256=source_sha, trained_not_validated=True), fit / "repair.pt")
    protocol = dict(source_sha256=source_sha, source_state_sha256=state_digest(base),
                    threshold=0.5, steps=512, fixed_last_iterate=True, source_parameter_updates=0,
                    input_sha256={}, code_sha256={}, hard_outcome_reads=0,
                    val_signal_reads=0, test_signal_reads=0, east_signal_reads=0)
    (fit / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    completed = dict(status="TRAINED_EMBEDDING_REPAIR_NOT_VALIDATED", steps=512,
                     artifact_sha256={p.name: _sha(p) for p in fit.iterdir()})
    (fit / "completed.json").write_text(json.dumps(completed), encoding="utf-8")
    paths = {}
    for sid, disruptive in ((101, True), (102, False), (103, False)):
        path = tmp_path / f"{sid}.hdf5"
        paths[sid] = path
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            for channel in io23.RAW23:
                v = np.linspace(0.1, 0.2, 6, dtype=np.float32)
                if channel == "CIII":
                    v[0] = -10.0  # valid source observation, not EAST sentinel
                dataset = data.create_dataset(channel, data=v)
                dataset.attrs["SampleRate"] = 1000.0
                dataset.attrs["StartTime"] = 0.0
            meta = f.create_group("meta")
            meta.create_dataset("IsDisrupt", data=int(disruptive))
            meta.create_dataset("DownTime", data=0.010)
    monkeypatch.setattr(io23, "path_for", lambda device, sid: paths[int(sid)])
    monkeypatch.setattr(io23, "source_roles", lambda: dict(train=[103], val=[102], test=[101]))
    monkeypatch.setattr(io23, "JTEXT_INDEX", index)
    monkeypatch.setattr(evaluation, "DEFAULT_PARENT", parent)
    monkeypatch.setattr(evaluation, "PARENT_SHA256", source_sha)
    monkeypatch.setattr(evaluation, "FIT_COMPLETED_SHA256", _sha(fit / "completed.json"))
    monkeypatch.setattr(evaluation, "load_parent", lambda path: (base, {"threshold": 0.5}))
    return dict(base=base, repair=repair, constants=constants, fit=fit, paths=paths, root=tmp_path)


def test_fit_hash_and_parent_binding_are_checked_before_loading(source_fixture):
    fit = source_fixture["fit"]
    evaluation.load_fit(fit, expected_source_sha=evaluation.PARENT_SHA256, device="cpu")
    with pytest.raises(ValueError, match="source"):
        evaluation.load_fit(fit, expected_source_sha="b" * 64, device="cpu")
    with (fit / "repair.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash|SHA|sha"):
        evaluation.load_fit(fit, expected_source_sha=evaluation.PARENT_SHA256, device="cpu")


def test_only_locked_val_or_test_role_accepted(source_fixture):
    root = source_fixture["root"]
    for bad in ("train", "east", "TEST"):
        with pytest.raises(ValueError, match="role"):
            evaluation.run_source(root / ("out_" + bad), role=bad,
                                  fit_dir=source_fixture["fit"], compute_device="cpu")
        assert not (root / ("out_" + bad)).exists()


def test_end_to_end_cpu_seals_all_arms_before_hdf_reveal(source_fixture, monkeypatch):
    fixture = source_fixture
    output = fixture["root"] / "out"
    original_outcome = io23.read_outcome
    observed = []
    def checked_outcome(device, sid):
        seal_path = output / "shots" / f"0001_{sid}_seal.json"
        seal = json.loads(seal_path.read_text())
        archive = output / "shots" / f"0001_{sid}.npz"
        assert seal["curve_sha256"] == _sha(archive)
        assert seal["current_outcome_used_in_prediction"] is False
        with np.load(archive, allow_pickle=False) as values:
            assert len(values.files) == 13  # 12 curves plus time
        observed.append((device, sid))
        return original_outcome(device, sid)
    monkeypatch.setattr(io23, "read_outcome", checked_outcome)
    raw = simple_data.read_shot("J-TEXT", 101)
    assert raw["values"][0, 1] == -10 and raw["available"][0, 1]
    before_base, before_fit = state_digest(fixture["base"]), _sha(fixture["fit"] / "repair.pt")
    result = evaluation.run_source(output, role="test", fit_dir=fixture["fit"],
                                   compute_device="cpu", batch_size=4)
    assert observed == [("J-TEXT", 101)]
    assert result["neural_updates"] == 0
    assert state_digest(fixture["base"]) == before_base
    assert _sha(fixture["fit"] / "repair.pt") == before_fit
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["role"] == "test" and protocol["threshold"] == 0.5
    assert protocol["independent_future_validation"] is False
    assert json.loads((output / "raw_hashes.json").read_text())["101"] == _sha(fixture["paths"][101])
    per = json.loads((output / "per_shot.json").read_text())
    assert len(per) == 12 and all(rows[0]["disruptive"] for rows in per.values())
    assert all(rows[0]["down_s"] == 0.01 for rows in per.values())
    assert len(json.loads((output / "paired.json").read_text())) > 0
    assert (output / "summary.json").is_file()


def test_validation_uses_only_val_and_records_safe_hdf_outcome(source_fixture, monkeypatch):
    fixture = source_fixture
    output = fixture["root"] / "val_out"
    original_read = io23.read_signals
    calls = []
    def checked(device, shot, *args, **kwargs):
        calls.append((device, shot))
        return original_read(device, shot, *args, **kwargs)
    monkeypatch.setattr(io23, "read_signals", checked)
    result = evaluation.run_source(output, role="val", fit_dir=fixture["fit"],
                                   compute_device="cpu", batch_size=6)
    assert calls == [("J-TEXT", 102)]
    assert result["source_signal_reads"] == 1
    per = json.loads((output / "per_shot.json").read_text())
    assert all(not rows[0]["disruptive"] and rows[0]["down_s"] is None for rows in per.values())


def test_failed_prediction_never_reveals_outcome(source_fixture, monkeypatch):
    fixture = source_fixture
    outcomes = []
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic prediction failure")
    monkeypatch.setattr(evaluation, "predict_curves", fail)
    monkeypatch.setattr(io23, "read_outcome", lambda *args: outcomes.append(args))
    with pytest.raises(RuntimeError, match="synthetic prediction"):
        evaluation.run_source(fixture["root"] / "failed", role="test", fit_dir=fixture["fit"],
                              compute_device="cpu")
    assert outcomes == []


def test_cli_configures_cpu_and_uses_existing_cuda_configure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluation, "configure", lambda: calls.append("existing_cuda_configure"))
    monkeypatch.setattr(evaluation, "run_source", lambda *args, **kwargs:
                        dict(status="fixture", role=kwargs["role"], seconds=0))
    threads = torch.get_num_threads()
    state = torch.random.get_rng_state()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        evaluation.main(["--output", str(tmp_path / "cpu"), "--compute-device", "cpu"])
        assert torch.initial_seed() == 131 and torch.get_num_threads() == 2
        assert not torch.backends.mha.get_fastpath_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert calls == []
        evaluation.main(["--output", str(tmp_path / "cuda"), "--compute-device", "cuda"])
        assert calls == ["existing_cuda_configure"]
    finally:
        torch.set_num_threads(threads)
        torch.random.set_rng_state(state)
        torch.backends.mha.set_fastpath_enabled(fastpath)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
