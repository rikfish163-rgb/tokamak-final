"""Independent complete-curve recount plus fixed four-shot neural replay.

Decision arithmetic is independent of the experiment's scoring helper. The
neural replay deliberately reuses its forward/gradient implementation, and
is described as reproducibility evidence, not an independent model proof.
"""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, LEARNERS, load_model, predict


def independent_counts(rows):
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    tp = sum(r["timely"] for r in rows)
    fp = sum(not r["disruptive"] and r["alarm"] for r in rows)
    return dict(
        n=len(rows),
        n_disrupt=nd,
        n_safe=ns,
        tp=tp,
        fp=fp,
        tpr=tp / nd,
        fpr=fp / ns,
        J=tp / nd - fp / ns,
        early_over3000=sum(r["timely"] and r["lead_ms"] > 3000 for r in rows),
    )


def verify(root):
    output = root / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED" and p["arms"] == list(ARMS)
    assert p["outcome_authority"] == "HDF5" and p["persistent_neural_updates"] == 0
    assert p["threshold"] == 0.7258946895599365
    assert p["first_update"] == 159 and p["update_interval"] == 128
    reference = Path(p["reference"])
    assert sha(reference / "completed.json") == p["reference_completed_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    assert sha(Path(p["cold_state"])) == p["cold_state_sha256"]
    assert sha(reference / "input_map.json") == p["input_map_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        assert set(archive.namelist()) == set(p["code_sha256"])
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
            assert sha(io23.ROOT / path) == digest
    saved = read(root / "per_shot.json")
    order = p["order"]
    assert len(order) == len({r["shot"] for r in order}) == 360
    hash_inputs = read(reference / "input_hashes.json")
    rows = {a: [] for a in ARMS}
    aggregated = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in LEARNERS
    }
    for i, record in enumerate(order):
        sid, prefix = record["shot"], f"{i + 1:04d}_{record['shot']}"
        path, logs = (
            root / "shots" / f"{prefix}.npz",
            root / "shots" / f"{prefix}_updates.json",
        )
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and sha(logs) == seal["updates_sha256"]
        assert seal["outcome_available"] is False and seal["shot"] == sid
        assert sha(io23.path_for("EAST", sid)) == hash_inputs[str(sid)]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as ref,
        ):
            assert set(data.files) == set(ARMS) | {"times"}
            assert np.array_equal(data["times"], ref["times"])
            assert np.array_equal(data["cold"], ref["cold"])
            for arm in ARMS:
                curve, times = data[arm], data["times"]
                assert curve.shape == times.shape and np.isfinite(curve).all()
                assert ((curve >= 0) & (curve <= 1)).all()
                if i < 20:
                    assert np.array_equal(curve, ref["source"])
                if arm in LEARNERS:
                    assert np.array_equal(curve[:160], data["cold"][:160])
                # Same preterminal tolerance as primary classification, independently expressed.
                eligible = (
                    np.ones(len(times), bool)
                    if not flag
                    else (down - times) * 1000 >= -1e-6
                )
                hit = np.flatnonzero(eligible & (curve >= p["threshold"]))
                lead = (
                    float(round((down - times[hit[0]]) * 1000, 9))
                    if flag and len(hit)
                    else None
                )
                row = dict(
                    shot=sid,
                    disruptive=flag,
                    alarm=bool(len(hit)),
                    lead_ms=lead,
                    timely=lead is not None and lead > 30,
                )
                assert row == saved[arm][i]
                rows[arm].append(row)
            length = len(data["times"])
        log = read(logs)
        assert log["reset_each_shot"]
        expected = list(range(159, length, 128)) if i >= 20 else []
        for arm in LEARNERS:
            history = log["updates"][arm]
            assert [u["target_last"] for u in history] == expected
            previous = [0.0] * 32
            for u in history:
                assert u["origin"] + 25 == u["target_last"] == u["risk_sealed_through"]
                assert u["target_first"] == u["origin"] + 1
                assert u["first_affected"] == u["target_last"] + 1
                assert u["beta32_before"] == previous
                b = np.asarray(u["beta32_after"]).reshape(16, 2)
                assert (
                    np.isfinite(b).all()
                    and (b >= 0).all()
                    and (b.sum(-1) <= 1 + 1e-7).all()
                )
                assert len(u["gradient32"]) == 32 and np.isfinite(u["gradient32"]).all()
                assert np.isclose(
                    np.linalg.norm(u["gradient32"]),
                    u["gradient_norm"],
                    atol=0,
                    rtol=1e-12,
                )
                assert np.isclose(
                    np.linalg.norm(
                        (
                            np.asarray(u["beta32_after"], dtype=np.float32)
                            - np.asarray(previous, dtype=np.float32)
                        ).astype(np.float64)
                    ),
                    u["step_norm"],
                    atol=0,
                    rtol=1e-12,
                )
                if u["observed"]:
                    assert u["loss_retained"] <= u["loss_before"]
                    assert u["accepted"] == (u["loss_proposal"] < u["loss_before"])
                else:
                    assert not u["accepted"] and u["loss_before"] is None
                if not u["accepted"]:
                    assert u["beta32_after"] == previous
                previous = u["beta32_after"]
                aggregated[arm]["events"] += 1
                aggregated[arm]["accepted"] += u["accepted"]
                aggregated[arm]["nonzero_gradients"] += u["gradient_norm"] > 0
                aggregated[arm]["observed_scalars"] += u["observed"]
        for event in zip(*(log["updates"][a] for a in LEARNERS), strict=True):
            assert len({u["mask_sha256"] for u in event}) == 1
            assert len({u["observed"] for u in event}) == 1
    assert aggregated == done["update_statistics"]
    for arm in ARMS:
        for scope, subset in (
            ("full360", rows[arm]),
            ("warmup20", rows[arm][:20]),
            ("suffix340", rows[arm][20:]),
        ):
            assert independent_counts(subset) == done["metrics"][arm][scope]
    m, cold = done["metrics"]["real"]["full360"], done["metrics"]["cold"]["full360"]
    assert done["operational_gate"] == (
        m["fpr"] <= 0.2 and m["tp"] >= cold["tp"] and m["J"] > cold["J"]
    )
    assert done["control_separation_gate"] == all(
        m["J"] > done["metrics"][a]["full360"]["J"]
        for a in ARMS
        if a not in ("cold", "real")
    )
    configure()
    model, _, _ = load_model()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mapping = read(reference / "input_map.json")
    replay = []
    for i in (20, 118, 153, 359):
        sid, prefix = order[i]["shot"], f"{i + 1:04d}_{order[i]['shot']}"
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: s[k] for k in ("values", "available")}, mapping)
        curves, logs = predict(model, signal)
        with np.load(
            root / "shots" / f"{prefix}.npz", allow_pickle=False
        ) as saved_curve:
            assert all(np.array_equal(curves[a], saved_curve[a]) for a in ARMS)
        assert logs == read(root / "shots" / f"{prefix}_updates.json")
        replay.append(sid)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items())
    result = dict(
        status="PASS",
        complete_curves_recounted=360 * len(ARMS),
        exact_parent_curves=360,
        neural_replay_curves=len(replay) * len(ARMS),
        replay_shots=replay,
        replay_all_update_receipts_exact=True,
        update_statistics=aggregated,
        source_tensors_unchanged=True,
        forward_gradient_implementation_reused=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        independent_future_confirmation=False,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
