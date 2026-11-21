"""Independent endpoint indices and recount, plus fixed-shot CUDA replay."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform


def main(root):
    output = root / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED" and done["no_neural_adaptation"]
    reference = Path(p["reference"])
    assert sha(reference / "completed.json") == p["reference_completed_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    assert sha(Path(p["cold_state"])) == p["cold_state_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    saved = read(root / "per_shot.json")
    ids = [r["shot"] for r in p["order"]]
    assert len(ids) == len(set(ids)) == 360
    for i, sid in enumerate(ids):
        prefix = f"{i + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and seal["outcome_available"] is False
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as ref,
        ):
            assert np.array_equal(data["times"], ref["times"])
            assert np.array_equal(data["d1"], ref["cold"])
            assert np.array_equal(data["d1"][:254], data["d2"][:254])
            if i < 20:
                assert all(np.array_equal(data[a], ref["source"]) for a in p["arms"])
            for arm in p["arms"]:
                probability = data[arm]
                assert np.isfinite(probability).all()
                valid = (
                    np.ones(len(probability), bool)
                    if not flag
                    else data["times"] <= down + 1e-9
                )
                hit = np.flatnonzero(valid & (probability >= p["threshold"]))
                lead = (
                    float(round((down - data["times"][hit[0]]) * 1000, 9))
                    if flag and len(hit)
                    else None
                )
                decision = dict(
                    shot=sid,
                    disruptive=flag,
                    alarm=bool(len(hit)),
                    lead_ms=lead,
                    timely=lead is not None and lead > 30,
                )
                assert decision == saved[arm][i]
    for arm, rows in saved.items():
        for scope, subset in (
            ("full360", rows),
            ("warmup20", rows[:20]),
            ("suffix340", rows[20:]),
        ):
            nd = sum(r["disruptive"] for r in subset)
            ns = len(subset) - nd
            tp = sum(r["timely"] for r in subset)
            fp = sum(r["alarm"] and not r["disruptive"] for r in subset)
            m = done["metrics"][arm][scope]
            assert (m["n_disrupt"], m["n_safe"], m["tp"], m["fp"]) == (nd, ns, tp, fp)
            assert (
                m["tpr"] == tp / nd
                and m["fpr"] == fp / ns
                and m["J"] == tp / nd - fp / ns
            )
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    model.restore(torch.load(p["cold_state"], map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    mapping = read(reference / "input_map.json")
    assert sha(reference / "input_map.json") == p["input_map_sha256"]
    selected = [20, 118, 153, 359]
    replayed = 0
    for index in selected:
        sid = ids[index]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: s[k] for k in ("values", "available")}, mapping)
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as saved_curve:
            for arm, step in (("d1", 1), ("d2", 2)):
                parts = []
                for lo in range(0, len(signal["values"]), 32):
                    xs, masks = [], []
                    for end in range(lo, min(lo + 32, len(signal["values"]))):
                        if step == 2 and end >= 254:
                            take = list(range(end - 254, end + 1, 2))
                            x, m = signal["values"][take], signal["available"][take]
                        else:
                            x, m = (
                                np.zeros((128, 23), np.float32),
                                np.zeros((128, 23), bool),
                            )
                            start = max(0, end - 127)
                            x[-(end - start + 1) :] = signal["values"][start : end + 1]
                            m[-(end - start + 1) :] = signal["available"][
                                start : end + 1
                            ]
                        xs.append(x)
                        masks.append(m)
                    with torch.no_grad():
                        h, valid = model.tokens(
                            torch.as_tensor(np.stack(xs), device="cuda"),
                            torch.as_tensor(np.stack(masks), device="cuda"),
                        )
                        z = model.risk(h, valid, h.new_zeros(len(h), 92))
                        parts.append(z.sigmoid().cpu().numpy())
                assert np.array_equal(np.concatenate(parts), saved_curve[arm]), (
                    sid,
                    arm,
                    "replay mismatch",
                )
                replayed += 1
    result = dict(
        status="PASS",
        complete_curves_recounted=720,
        reference_curves_exact=360,
        prefix_d1_before254_preserved=360,
        independent_index_replay_curves=replayed,
        replay_shots=[ids[i] for i in selected],
        neural_forward_reused=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        independent_future_confirmation=False,
    )
    _durable_json(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
