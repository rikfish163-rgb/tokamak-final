import json

import h5py
import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.joint_lora_20261104 import run as runner
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)


def test_complete_fixture_preserves_prediction_then_reveal_and_shared_slow(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(
            Config(fast_site="tokens", input_view="source", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    original = state_digest(base)
    ids = list(range(1, 23))
    paths = {}
    hashes = {}
    seals = []
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
    for index, sid in enumerate(ids, 1):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        signal = prepare_actual(raw, index, mapping)
        scores = runner.native_curve(base, signal, device="cpu")
        path = reference / f"{index:04d}_{sid}.npz"
        np.savez_compressed(path, times=raw["times"], map_after20=scores)
        seal = dict(
            shot=sid,
            curve_sha256=runner.sha(path),
            current_outcome_used_in_prediction=False,
        )
        path.with_name(path.stem + "_seal.json").write_text(json.dumps(seal))
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
    monkeypatch.setattr(runner, "WARM_STEPS", 2)
    monkeypatch.setattr(runner, "EOS_STEPS", 1)
    monkeypatch.setattr(
        runner, "load_initial", lambda ids, device: torch.zeros(168, 70, device=device)
    )
    output = tmp_path / "output"
    reveals = []
    reader = io23.read_outcome

    def revealed(domain, sid):
        seal = json.loads(
            (output / "shots" / f"{ids.index(sid) + 1:04d}_{sid}_seal.json").read_text()
        )
        path = output / "shots" / f"{ids.index(sid) + 1:04d}_{sid}.npz"
        assert runner.sha(path) == seal["curve_sha256"]
        reveals.append(sid)
        return reader(domain, sid)

    monkeypatch.setattr(io23, "read_outcome", revealed)
    done = runner.run(output, device="cpu")
    assert reveals == ids and done["postseal_outcomes"] == 22
    assert done["inter_shots"] == [21, 22]
    assert state_digest(base) == original
    p = json.loads((output / "per_shot.json").read_text())
    for arm in runner.ARMS:
        assert p[arm][:20] == p["native"][:20]
    train = json.loads((output / "inter" / "0021_21.json").read_text())
    assert train["available_ended_shots"] == ids[:21]
    assert set(train["training_shots"]) <= set(ids[:21])
    assert 21 in train["training_shots"]
    assert train["shared_slow_for"] == ["inter", "both"]
