import numpy as np
import pytest
import torch
import h5py
import json
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.cold20_compare_20261104 import run as runner
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)

from experiments.risk_aligned_online.cold20_compare_20261104.run import (
    predict_curves,
    MODES,
)
from experiments.risk_aligned_online.cold20_compare_20261104.training import (
    make_readout,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_initial_prefix_all_curves_equal_and_future_changes_do_not_rewrite_past():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(1)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    x = np.random.default_rng(1).normal(size=(10, 23)).astype(np.float32)
    shot = dict(values=x, available=np.ones_like(x, dtype=bool))
    before = predict_curves(base, None, shot, device="cpu", batch_size=4)
    assert set(before) == set(MODES)
    assert all(np.array_equal(v, before["native"]) for v in before.values())
    shot["values"][6:] += 10
    after = predict_curves(base, None, shot, device="cpu", batch_size=3)
    for name in before:
        np.testing.assert_allclose(
            before[name][:6], after[name][:6], atol=1e-7, rtol=1e-6
        )


def test_trained_models_are_explicit_and_only_their_own_curves_change():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    models = {
        name: make_readout(base, "bias").requires_grad_(False)
        for name in MODES
        if name != "native"
    }
    with torch.no_grad():
        models["bias"].risk_head.bias.add_(1)
    x = np.ones((3, 23), np.float32)
    shot = dict(values=x, available=np.ones_like(x, dtype=bool))
    result = predict_curves(base, models, shot, device="cpu")
    assert (result["bias"] > result["native"]).all()
    for name in ("head", "norm", "qv_lora"):
        assert np.array_equal(result[name], result["native"])
    with pytest.raises(ValueError):
        predict_curves(base, {"head": models["head"]}, shot, device="cpu")


def test_complete_cpu_replay_trains_only_after_twenty_sealed_truths(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(4)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    original = state_digest(base)
    ids = list(range(1, 23))
    paths = {}
    reference = tmp_path / "reference"
    reference.mkdir()
    mapping = dict(gain=[1.0] * 23, offset=[0.0] * 23)
    mapping["gain"][12] = 0.5
    seals = []
    hashes = {}
    for sid in ids:
        path = tmp_path / f"{sid}.hdf5"
        paths[sid] = path
        with h5py.File(path, "w") as f:
            for j, name in enumerate(io23.RAW23):
                v = np.linspace(0.1, 0.5, 300, dtype=np.float32) + 0.01 * j
                ds = f.create_dataset("data/" + name, data=v)
                ds.attrs["SampleRate"] = 1000.0
                ds.attrs["StartTime"] = 0.0
            f["meta/IsDisrupt"] = int(sid % 2 == 1)
            f["meta/DownTime"] = 0.250
    monkeypatch.setattr(io23, "path_for", lambda domain, sid: paths[int(sid)])
    for index, sid in enumerate(ids, 1):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        signal = prepare_actual(raw, index, mapping)
        scores = predict_curves(base, None, signal, device="cpu")["native"]
        path = reference / f"{index:04d}_{sid}.npz"
        np.savez_compressed(path, times=raw["times"], map_after20=scores)
        seal = dict(
            shot=sid,
            curve_sha256=runner.sha(path),
            current_outcome_used_in_prediction=False,
        )
        (reference / f"{index:04d}_{sid}_seal.json").write_text(json.dumps(seal))
        seals.append(seal)
        hashes[str(sid)] = runner.sha(paths[sid])
    monkeypatch.setattr(
        runner,
        "load_inputs",
        lambda: dict(
            ids=ids,
            order=[dict(shot=s) for s in ids],
            mapping=mapping,
            hashes=hashes,
            seals=seals,
            baseline_shots=reference,
            bound={},
            baseline_protocol=dict(threshold=0.5, source_state_sha256=original),
        ),
    )
    monkeypatch.setattr(runner, "load_parent", lambda _: (base, dict(threshold=0.5)))
    monkeypatch.setattr(runner, "STEPS", 2)
    monkeypatch.setattr(runner, "BATCH_SIZE", 4)
    output = tmp_path / "output"
    reveals = []
    fits = []
    truth_reader = io23.read_outcome
    fit = runner.fit_prefix

    def reveal(domain, sid):
        index = ids.index(sid) + 1
        prefix = output / "shots" / f"{index:04d}_{sid}"
        seal = json.loads(prefix.with_name(prefix.name + "_seal.json").read_text())
        assert runner.sha(prefix.with_suffix(".npz")) == seal["curve_sha256"]
        with np.load(prefix.with_suffix(".npz")) as f:
            if index <= 20:
                assert all(np.array_equal(f[m], f["native"]) for m in MODES)
        reveals.append(sid)
        return truth_reader(domain, sid)

    def prefix_fit(source, past, truth, allowed, mapping, out, **kwargs):
        assert reveals == ids[:20] and list(allowed) == ids[:20]
        assert set(past) == set(truth) == set(ids[:20])
        fits.append(tuple(allowed))
        return fit(source, past, truth, allowed, mapping, out, **kwargs)

    monkeypatch.setattr(io23, "read_outcome", reveal)
    monkeypatch.setattr(runner, "fit_prefix", prefix_fit)
    result = runner.run(output, device="cpu")
    assert (
        result["suffix_training_shots"] == 0 and result["fit_label_shots"] == ids[:20]
    )
    assert len(fits) == 1 and reveals == ids and state_digest(base) == original
    details = json.loads((output / "fit/fitted.json").read_text())
    assert {m: d["trained_parameters"] for m, d in details.items()} == runner.COUNTS
    for name, d in details.items():
        assert d["initial_state_sha256"] != d["fitted_state_sha256"]
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[m]["startup20"] == summary["native"]["startup20"] for m in MODES)
