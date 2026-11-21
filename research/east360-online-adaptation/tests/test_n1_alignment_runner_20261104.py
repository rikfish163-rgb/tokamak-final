import numpy as np
import pytest
import torch
import h5py
import json

from experiments.risk_aligned_online.n1_alignment_20261104.run import (
    make_views,
    predict_curves,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.n1_alignment_20261104 import run as runner
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)


def fixture():
    x = np.arange(300 * 23, dtype=np.float32).reshape(300, 23) + 1
    a = np.ones_like(x, bool)
    a[:, 2] = False
    return dict(device="EAST", shot=20, values=x, available=a), dict(
        gain=[1.0] * 12 + [0.1] + [1.0] * 10, offset=[0.0] * 23
    )


def test_alignment_changes_only_declared_values_and_never_masks():
    s, m = fixture()
    before = s["values"].copy()
    v = make_views(s, 21, m, 0.02)
    assert np.array_equal(s["values"], before)
    for a in v.values():
        assert np.array_equal(a["available"], v["unmapped"]["available"])
    assert np.array_equal(
        v["n1"]["values"][:, 11], v["unmapped"]["values"][:, 11] * np.float32(0.02)
    )
    assert np.array_equal(v["both"]["values"][:, 12], v["density"]["values"][:, 12])
    assert np.array_equal(v["both"]["values"][:, 11], v["n1"]["values"][:, 11])
    for j in range(23):
        if j not in (11, 12):
            assert np.array_equal(
                v["both"]["values"][:, j], v["unmapped"]["values"][:, j]
            )


def test_first20_have_no_future_alignment_and_risk_parent_is_unchanged():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    s, m = fixture()
    views = make_views(s, 20, m, None)
    b = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    before = {k: v.clone() for k, v in b.state_dict().items()}
    p = predict_curves(b, views, device="cpu")
    assert all(np.array_equal(v, p["unmapped"]) for v in p.values())
    assert all(torch.equal(v, before[k]) for k, v in b.state_dict().items())
    with pytest.raises(ValueError):
        make_views(s, 21, m, None)


def test_complete_cpu_replay_reveals_before_calibration_and_preserves_prefix(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    before = state_digest(base)
    ids = list(range(1, 23))
    paths = {}
    hashes = {}
    seals = []
    reference = tmp_path / "reference"
    reference.mkdir()
    mapping = dict(gain=[1.0] * 12 + [0.1] + [1.0] * 10, offset=[0.0] * 23)
    for sid in ids:
        paths[sid] = tmp_path / f"{sid}.hdf5"
        with h5py.File(paths[sid], "w") as f:
            for j, name in enumerate(io23.RAW23):
                ds = f.create_dataset(
                    "data/" + name,
                    data=np.arange(300, dtype=np.float32) * 0.01 + j + 0.1,
                )
                ds.attrs["SampleRate"], ds.attrs["StartTime"] = 1000.0, 0.0
            f["meta/IsDisrupt"], f["meta/DownTime"] = sid % 2, 0.3
    monkeypatch.setattr(io23, "path_for", lambda domain, sid: paths[sid])
    for idx, sid in enumerate(ids, 1):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        curves = runner.predict_curves(
            base, runner.make_views(raw, idx, mapping, 1.0), device="cpu"
        )
        p = reference / f"{idx:04d}_{sid}.npz"
        np.savez_compressed(
            p,
            times=raw["times"],
            unmapped=curves["unmapped"],
            map_after20=curves["density"],
        )
        seal = dict(
            shot=sid,
            curve_sha256=runner.sha(p),
            current_outcome_used_in_prediction=False,
        )
        (reference / f"{idx:04d}_{sid}_seal.json").write_text(json.dumps(seal))
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
            baseline_protocol=dict(source_state_sha256=before, threshold=0.5),
        ),
    )
    monkeypatch.setattr(runner, "load_parent", lambda p: (base, dict(threshold=0.5)))
    monkeypatch.setattr(
        runner, "source_reference", lambda ids: dict(value=1.0, source_safe_ids=[])
    )
    output = tmp_path / "output"
    reveals = []
    reader = io23.read_outcome

    def reveal(domain, sid):
        p = output / "shots" / f"{ids.index(sid) + 1:04d}_{sid}.npz"
        seal = json.loads(p.with_name(p.stem + "_seal.json").read_text())
        assert runner.sha(p) == seal["curve_sha256"]
        reveals.append(sid)
        return reader(domain, sid)

    monkeypatch.setattr(io23, "read_outcome", reveal)
    original_fit = runner.fit_gain

    def fit(reference, past, truth, allowed):
        assert (
            reveals == ids[:20]
            and allowed == ids[:20]
            and set(past) == set(truth) == set(allowed)
        )
        return original_fit(reference, past, truth, allowed)

    monkeypatch.setattr(runner, "fit_gain", fit)
    done = runner.run(output, device="cpu")
    assert (
        done["neural_updates"] == 0 and reveals == ids and state_digest(base) == before
    )
    summary = json.loads((output / "summary.json").read_text())
    assert all(
        summary[a]["startup20"] == summary["unmapped"]["startup20"] for a in runner.ARMS
    )
