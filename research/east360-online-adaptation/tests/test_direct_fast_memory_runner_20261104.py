import numpy as np
import torch
import pytest
import h5py
import json

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.direct_fast_memory_20261104 import run as runner
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import Injection
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)


def test_prefix_has_no_adaptation_and_all_controls_equal():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    s = dict(values=np.ones((145, 23), np.float32), available=np.ones((145, 23), bool))
    curves, rows = runner.predict_curves(base, None, None, None, s, device="cpu")
    assert not rows
    assert all(np.array_equal(v, curves["native"]) for v in curves.values())


def test_zero_connection_still_logs_nonzero_signal_learning_but_no_risk_gain():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    v = np.random.default_rng(1).normal(size=(160, 23)).astype("float32")
    s = dict(values=v, available=np.ones_like(v, bool))
    projection = Injection().requires_grad_(False)
    curves, rows = runner.predict_curves(
        base, torch.zeros(168, 70), projection, torch.zeros(168), s, device="cpu"
    )
    assert rows["real"][0]["weight_displacement"] > 0
    assert all(np.array_equal(v, curves["native"]) for v in curves.values())


@pytest.mark.parametrize("rule", ("lms", "ridge"))
def test_complete_cpu_flow_seals_current_prediction_before_hdf_and_fits_only_ended20(
    tmp_path, monkeypatch, rule
):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(5)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    before = state_digest(base)
    ids, paths, seals, hashes = list(range(1, 23)), {}, [], {}
    reference = tmp_path / "reference"
    reference.mkdir()
    mapping = dict(gain=[1.0] * 23, offset=[0.0] * 23)
    for index, sid in enumerate(ids, 1):
        paths[sid] = tmp_path / f"{sid}.hdf5"
        with h5py.File(paths[sid], "w") as f:
            for j, name in enumerate(io23.RAW23):
                ds = f.create_dataset(
                    "data/" + name,
                    data=(np.arange(300, dtype=np.float32) * 0.01 + j * 0.1),
                )
                ds.attrs["SampleRate"], ds.attrs["StartTime"] = 1000.0, 0.0
            f["meta/IsDisrupt"], f["meta/DownTime"] = sid % 2, 0.3
    monkeypatch.setattr(io23, "path_for", lambda domain, sid: paths[sid])
    for index, sid in enumerate(ids, 1):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        scores, _ = runner.predict_curves(
            base, None, None, None, prepare_actual(raw, index, mapping), device="cpu"
        )
        path = reference / f"{index:04d}_{sid}.npz"
        np.savez_compressed(path, times=raw["times"], map_after20=scores["native"])
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
            baseline_protocol=dict(threshold=0.5, source_state_sha256=before),
        ),
    )
    monkeypatch.setattr(runner, "load_parent", lambda p: (base, dict(threshold=0.5)))
    monkeypatch.setattr(runner, "STEPS", 2)
    output = tmp_path / "output"
    reveals = []
    reader = io23.read_outcome

    def reveal(domain, sid):
        index = ids.index(sid) + 1
        path = output / "shots" / f"{index:04d}_{sid}.npz"
        seal = json.loads(path.with_name(path.stem + "_seal.json").read_text())
        assert runner.sha(path) == seal["curve_sha256"]
        reveals.append(sid)
        return reader(domain, sid)

    monkeypatch.setattr(io23, "read_outcome", reveal)
    fitted = runner.fit_prefix

    def fit(base, past, truth, allowed, **kwargs):
        assert reveals == ids[:20] and set(past) == set(truth) == set(allowed) == set(
            ids[:20]
        )
        return fitted(base, past, truth, allowed, **kwargs)

    monkeypatch.setattr(runner, "fit_prefix", fit)
    result = runner.run(output, device="cpu", rule=rule)
    assert reveals == ids and state_digest(base) == before
    assert (
        result["suffix_training_shots"] == 0 and result["fit_label_shots"] == ids[:20]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert all(
        summary[m]["startup20"] == summary["native"]["startup20"] for m in runner.ARMS
    )
