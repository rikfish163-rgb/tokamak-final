"""Independent3600 decisions, exact copied controls, fixed4-shot fresh replay."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.verify import independent_counts
from ..east_context_operator_20261031.run import configure, read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, COPY, FRESH, predict


def verify(root):
    output = root / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    p, done, rows = (
        read(root / "protocol.json"),
        read(root / "completed.json"),
        read(root / "per_shot.json"),
    )
    assert done["status"] == "COMPLETED" and p["arms"] == list(ARMS)
    assert p["outcome_authority"] == "HDF5" and p["persistent_neural_updates"] == 0
    assert (
        p["threshold"] == 0.7258946895599365
        and p["anchor"] == 0.01
        and p["step_size"] == 0.1
    )
    old, reference = Path(p["old"]), Path(p["reference"])
    assert sha(old / "completed.json") == p["old_completed_sha256"]
    assert sha(old / "protocol.json") == p["old_protocol_sha256"]
    assert sha(old / "verification.json") == p["old_verification_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    assert sha(Path(p["cold_state"])) == p["cold_state_sha256"]
    assert sha(reference / "input_map.json") == p["input_map_sha256"]
    old_protocol = read(old / "protocol.json")
    assert old_protocol["source_sha256"] == p["source_sha256"]
    assert old_protocol["cold_state_sha256"] == p["cold_state_sha256"]
    shared = (
        "experiments/risk_aligned_online/dual_history_ttt_20261101/model.py",
        "experiments/risk_aligned_online/dual_history_ttt_20261101/stream.py",
        "experiments/risk_aligned_online/simple23.py",
        "experiments/risk_aligned_online/causal_span_20261101/core.py",
        "experiments/risk_aligned_online/window_history_diagnostic_20261101/core.py",
        "experiments/risk_aligned_online/lifecycle_stream_20261025/model.py",
        "experiments/risk_aligned_online/delayed_recurrent_memory_20261025/model.py",
    )
    assert all(
        p["code_sha256"][name] == old_protocol["code_sha256"][name] for name in shared
    )
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        assert set(archive.namelist()) == set(p["code_sha256"])
        for name, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert sha(io23.ROOT / name) == digest
    order = p["order"]
    assert len(order) == len({r["shot"] for r in order}) == 360
    assert order == read(old / "protocol.json")["order"]
    input_hashes = read(reference / "input_hashes.json")
    aggregate = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in FRESH[1:]
    }
    temporal_support = dict(
        events=0,
        no_shape_degrees=0,
        nonzero_shape_degrees=0,
        accepted_without_shape_degrees={a: 0 for a in FRESH[1:]},
    )
    paired_alarms = []
    for i, record in enumerate(order):
        sid, prefix = record["shot"], f"{i + 1:04d}_{record['shot']}"
        path, logs = (
            root / "shots" / f"{prefix}.npz",
            root / "shots" / f"{prefix}_updates.json",
        )
        old_path = old / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and sha(logs) == seal["updates_sha256"]
        assert sha(old_path) == seal["copied_from_sha256"]
        assert (
            seal["outcome_available"] is False
            and seal["shot"] == sid
            and seal["index"] == i + 1
        )
        assert sha(io23.path_for("EAST", sid)) == input_hashes[str(sid)]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(old_path, allow_pickle=False) as prior,
        ):
            assert set(data.files) == set(ARMS) | {"times"}
            assert np.array_equal(data["times"], prior["times"])
            assert np.array_equal(data["cold"], prior["cold"])
            assert all(
                np.array_equal(data[a], prior[old_name]) for a, old_name in COPY.items()
            )
            times = data["times"]
            first_indices = {}
            for arm in ARMS:
                curve = data[arm]
                assert curve.shape == times.shape and np.isfinite(curve).all()
                assert ((curve >= 0) & (curve <= 1)).all()
                if i < 20:
                    assert np.array_equal(curve, prior["cold"])
                if arm in FRESH[1:]:
                    assert np.array_equal(curve[:160], data["cold"][:160])
                valid = (
                    np.ones(len(times), bool)
                    if not flag
                    else (down - times) * 1000 >= -1e-6
                )
                hit = np.flatnonzero(valid & (curve >= p["threshold"]))
                first_indices[arm] = int(hit[0]) if len(hit) else None
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
                assert row == rows[arm][i]
            length = len(times)
            paired_alarms.append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    absolute=first_indices["absolute_real"],
                    centered=first_indices["centered_real"],
                )
            )
        log = read(logs)
        assert log["reset_each_shot"]
        expected = list(range(159, length, 128)) if i >= 20 else []
        for arm in FRESH[1:]:
            previous = [0.0] * 32
            assert [u["target_last"] for u in log["updates"][arm]] == expected
            for u in log["updates"][arm]:
                assert u["origin"] + 25 == u["target_last"] == u["risk_sealed_through"]
                assert (
                    u["first_affected"] == u["target_last"] + 1
                    and u["target_first"] == u["origin"] + 1
                )
                assert u["beta32_before"] == previous
                before = np.asarray(previous, np.float32)
                after = np.asarray(u["beta32_after"], np.float32)
                assert len(after) == len(u["gradient32"]) == 32
                assert np.isfinite(after).all() and np.isfinite(u["gradient32"]).all()
                assert (after >= 0).all() and (after.reshape(16, 2).sum(-1) <= 1).all()
                assert np.isclose(
                    np.linalg.norm(u["gradient32"]),
                    u["gradient_norm"],
                    rtol=1e-12,
                    atol=0,
                )
                assert np.isclose(
                    np.linalg.norm((after - before).astype(np.float64)),
                    u["step_norm"],
                    rtol=1e-12,
                    atol=0,
                )
                if u["observed"]:
                    assert u["accepted"] == (u["loss_proposal"] < u["loss_before"])
                    assert u["loss_retained"] == (
                        u["loss_proposal"] if u["accepted"] else u["loss_before"]
                    )
                else:
                    assert not u["accepted"] and u["loss_retained"] is None
                if not u["accepted"]:
                    assert u["beta32_after"] == previous
                previous = u["beta32_after"]
                aggregate[arm]["events"] += 1
                aggregate[arm]["accepted"] += u["accepted"]
                aggregate[arm]["nonzero_gradients"] += u["gradient_norm"] > 0
                aggregate[arm]["observed_scalars"] += u["observed"]
        for same_time in zip(*(log["updates"][a] for a in FRESH[1:]), strict=True):
            assert len({u["mask_sha256"] for u in same_time}) == 1
            assert len({u["observed"] for u in same_time}) == 1
        if i >= 20:
            signals = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            for event in zip(*(log["updates"][a] for a in FRESH[1:]), strict=True):
                origin, maturity = event[0]["origin"], event[0]["target_last"]
                mask = signals["available"][origin + 1 : maturity + 1].copy()
                mask &= signals["available"][origin]
                mask[:, 10] = False
                mask &= mask[::-1].copy()
                expected_hash = hashlib.sha256(mask[None].tobytes()).hexdigest()
                assert all(
                    u["mask_sha256"] == expected_hash
                    and u["observed"] == int(mask.sum())
                    for u in event
                )
                has_shape = bool((mask.sum(0) >= 2).any())
                temporal_support["events"] += 1
                temporal_support["nonzero_shape_degrees"] += has_shape
                temporal_support["no_shape_degrees"] += not has_shape
                for arm, u in zip(FRESH[1:], event, strict=True):
                    temporal_support["accepted_without_shape_degrees"][arm] += (
                        not has_shape and u["accepted"]
                    )
    assert aggregate == done["update_statistics"]
    assert (
        done["fresh_neural_curve_count"] == 1360
        and done["reused_prefix_curve_count"] == 80
    )
    assert done["copied_curve_count"] == 2160 and done["fresh_arm_curve_count"] == 1440
    for arm in ARMS:
        for scope, subset in (
            ("full360", rows[arm]),
            ("warmup20", rows[arm][:20]),
            ("suffix340", rows[arm][20:]),
        ):
            assert independent_counts(subset) == done["metrics"][arm][scope]
    m, c = (
        done["metrics"]["centered_real"]["full360"],
        done["metrics"]["cold"]["full360"],
    )
    assert done["operational_gate"] == (
        m["fpr"] <= 0.2 and m["tp"] >= c["tp"] and m["J"] > c["J"]
    )
    assert done["control_separation_gate"] == all(
        m["J"] > done["metrics"][a]["full360"]["J"]
        for a in (
            "centered_reverse",
            "centered_hold",
            "long_data",
            "snapshot_data",
            "equal_data",
        )
    )
    paired = dict(
        tp_delta=m["tp"] - done["metrics"]["absolute_real"]["full360"]["tp"],
        fp_delta=m["fp"] - done["metrics"]["absolute_real"]["full360"]["fp"],
        J_delta=m["J"] - done["metrics"]["absolute_real"]["full360"]["J"],
        rescued_D=[],
        lost_D=[],
        removed_FP=[],
        added_FP=[],
        first_alarm_index_changed=[
            r for r in paired_alarms if r["absolute"] != r["centered"]
        ],
    )
    for before, after in zip(rows["absolute_real"], rows["centered_real"], strict=True):
        if before["disruptive"]:
            if after["timely"] and not before["timely"]:
                paired["rescued_D"].append(before["shot"])
            if before["timely"] and not after["timely"]:
                paired["lost_D"].append(before["shot"])
        else:
            if before["alarm"] and not after["alarm"]:
                paired["removed_FP"].append(before["shot"])
            if after["alarm"] and not before["alarm"]:
                paired["added_FP"].append(before["shot"])
    configure()
    model, _, _ = load_model()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mapping = read(reference / "input_map.json")
    replay = []
    for i in (20, 118, 153, 359):
        sid, prefix = order[i]["shot"], f"{i + 1:04d}_{order[i]['shot']}"
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        curves, logs = predict(model, signal)
        with np.load(root / "shots" / f"{prefix}.npz", allow_pickle=False) as stored:
            assert all(np.array_equal(curves[a], stored[a]) for a in FRESH)
        assert logs == read(root / "shots" / f"{prefix}_updates.json")
        replay.append(sid)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        complete_curves_recounted=3600,
        exact_parent_curves=360,
        exact_copied_control_curves=2160,
        exact_warmup_curves=200,
        fresh_neural_curves=1360,
        replay_fresh_curves=16,
        replay_shots=replay,
        all_replay_receipts_exact=True,
        source_tensors_unchanged=True,
        forward_gradient_implementation_reused=True,
        independent_future_confirmation=False,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        shared_forward_target_files_matched=len(shared),
        temporal_observation_support=temporal_support,
        centered_vs_absolute=paired,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
