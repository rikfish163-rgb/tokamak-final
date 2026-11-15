import json

import h5py
import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.shadow_commit_20261104 import run as runner
from experiments.risk_aligned_online.shadow_commit_20261104.training import (
    records_and_pools,
)
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)


def test_complete_flow_only_fits_after_twenty_sealed_truths_and_saves_issued_forecasts(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    before = state_digest(base)
    ids = list(range(1, 23))
    paths = {}
    seals = []
    hashes = {}
    reference = tmp_path / "reference"
    reference.mkdir()
    mapping = dict(gain=[1.0] * 23, offset=[0.0] * 23)
    for sid in ids:
        paths[sid] = tmp_path / f"{sid}.hdf5"
        with h5py.File(paths[sid], "w") as f:
            for j, name in enumerate(io23.RAW23):
                ds = f.create_dataset(
                    "data/" + name,
                    data=np.arange(300, dtype=np.float32) * 0.01 + j * 0.1,
                )
                ds.attrs["SampleRate"], ds.attrs["StartTime"] = 1000.0, 0.0
            f["meta/IsDisrupt"], f["meta/DownTime"] = sid % 2, 0.3
    monkeypatch.setattr(io23, "path_for", lambda domain, sid: paths[sid])
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        curves, _, _ = runner.predict_curves(
            base, None, None, prepare_actual(raw, i, mapping), device="cpu"
        )
        assert all(np.array_equal(x, curves["native"]) for x in curves.values())
        path = reference / f"{i:04d}_{sid}.npz"
        np.savez_compressed(path, times=raw["times"], map_after20=curves["native"])
        seal = dict(
            shot=sid,
            curve_sha256=runner.sha(path),
            current_outcome_used_in_prediction=False,
        )
        (reference / f"{i:04d}_{sid}_seal.json").write_text(json.dumps(seal))
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
    monkeypatch.setattr(runner, "load_parent", lambda _: (base, dict(threshold=0.5)))
    monkeypatch.setattr(runner, "STEPS", 2)
    output = tmp_path / "output"
    reveals = []
    reader = io23.read_outcome

    def reveal(domain, sid):
        path = output / "shots" / f"{ids.index(sid) + 1:04d}_{sid}.npz"
        seal = json.loads(path.with_name(path.stem + "_seal.json").read_text())
        assert runner.sha(path) == seal["curve_sha256"]
        reveals.append(sid)
        return reader(domain, sid)

    monkeypatch.setattr(io23, "read_outcome", reveal)

    def initial(allowed, past, truth, device):
        assert reveals == ids[:20] and allowed == ids[:20]
        return (
            torch.zeros(168, 70, device=device),
            records_and_pools(past, truth, allowed)[0],
            {},
        )

    monkeypatch.setattr(runner, "load_initial", initial)
    done = runner.run(output, device="cpu")
    assert reveals == ids and done["suffix_training_shots"] == 0
    assert state_digest(base) == before
    s = json.loads((output / "summary.json").read_text())
    assert all(s[m]["startup20"] == s["native"]["startup20"] for m in runner.ARMS)
    with np.load(output / "shots" / "0021_21_forecasts.npz") as f:
        assert "guard_real__candidate_prediction" in f
        assert f["guard_real__issued_at"].tolist() == [127, 159, 191, 223, 255, 287]
