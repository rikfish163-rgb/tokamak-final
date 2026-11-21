"""Recount all saved decisions and replay NumPy-built counterfactual windows."""

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


def decision(times, probability, flag, down, threshold):
    eligible = np.ones(len(times), bool) if not flag else times <= down + 1e-9
    hit = np.flatnonzero(eligible & (probability >= threshold))
    first = int(hit[0]) if len(hit) else None
    lead = (
        float(round((down - times[first]) * 1000, 9))
        if flag and first is not None
        else None
    )
    return dict(
        alarm=first is not None, lead_ms=lead, timely=lead is not None and lead > 30
    ), first


def numpy_windows(signal, ends, view):
    windows, masks = [], []
    for end in ends:
        start = max(0, end - 127)
        x = np.zeros((128, 23), np.float32)
        m = np.zeros((128, 23), bool)
        x[-(end - start + 1) :] = signal["values"][start : end + 1]
        m[-(end - start + 1) :] = signal["available"][start : end + 1]
        if view == "reverse_past":
            actual = max(0, 127 - end)
            x[actual:127] = x[actual:127][::-1].copy()
            m[actual:127] = m[actual:127][::-1].copy()
        elif view in ("last_observed", "mean_observed"):
            for c in range(23):
                observed = x[m[:, c], c].copy()
                if len(observed):
                    value = (
                        observed[-1]
                        if view == "last_observed"
                        else np.float32(observed.astype(np.float64).mean())
                    )
                    x[m[:, c], c] = value
        elif view != "original":
            raise ValueError(view)
        windows.append(x)
        masks.append(m)
    return np.stack(windows), np.stack(masks)


def main(root):
    if (root / "verification.json").exists():
        raise FileExistsError(root / "verification.json")
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED" and not done["method_promoted"]
    reference = Path(p["reference"])
    assert sha(reference / "completed.json") == p["reference_completed_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    assert sha(Path(p["cold_state"])) == p["cold_state_sha256"]
    assert sha(reference / "input_map.json") == p["input_map_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    saved = read(root / "per_shot.json")
    old_rows = read(reference / "per_shot.json")
    timing = read(root / "timing_per_shot.json")
    names = p["views"]
    nd = ns = 0
    for index, order in enumerate(p["order"]):
        sid = order["shot"]
        prefix = f"{index + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == seal["sha256"] and seal["outcome_available"] is False
        flag, down = io23.read_outcome("EAST", sid)
        nd += int(flag)
        ns += int(not flag)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as ref,
        ):
            assert np.array_equal(data["times"], ref["times"])
            assert np.array_equal(data["original"], ref["cold"])
            for v in names:
                probability = data[v]
                assert (
                    probability.shape == data["times"].shape
                    and np.isfinite(probability).all()
                )
                d, _ = decision(data["times"], probability, flag, down, p["threshold"])
                assert dict(shot=sid, disruptive=flag, **d) == saved[v][index]
                if index < 20:
                    assert np.array_equal(probability, ref["source"])
            for a in old_rows:
                d, first = decision(ref["times"], ref[a], flag, down, p["threshold"])
                assert dict(shot=sid, is_disrupt=flag, **d) == old_rows[a][index]
                record = timing[a][index]
                assert record["first_alarm_index"] == first
                assert all(record[k] == value for k, value in d.items())
    assert nd == ns == 180
    for v in names:
        for scope, subset in (
            ("full360", saved[v]),
            ("warmup20", saved[v][:20]),
            ("suffix340", saved[v][20:]),
        ):
            tp = sum(r["timely"] for r in subset)
            fp = sum(r["alarm"] and not r["disruptive"] for r in subset)
            nd = sum(r["disruptive"] for r in subset)
            ns = len(subset) - nd
            actual = done["metrics"][v][scope]
            assert actual["tp"] == tp and actual["fp"] == fp
            assert actual["tpr"] == tp / nd and actual["fpr"] == fp / ns
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    model.restore(torch.load(p["cold_state"], map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    selected = [20, 118, 153, 359]
    replayed = 0
    for index in selected:
        sid = p["order"][index]["shot"]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform(
            {k: s[k] for k in ("values", "available")},
            read(reference / "input_map.json"),
        )
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as data:
            for v in names:
                parts = []
                for lo in range(0, len(signal["values"]), 32):
                    ends = range(lo, min(lo + 32, len(signal["values"])))
                    x, m = numpy_windows(signal, ends, v)
                    with torch.no_grad():
                        h, valid = model.tokens(
                            torch.as_tensor(x, device="cuda"),
                            torch.as_tensor(m, device="cuda"),
                        )
                        z = model.risk(h, valid, h.new_zeros(len(h), 92))
                    parts.append(z.sigmoid().cpu().numpy())
                assert np.array_equal(np.concatenate(parts), data[v]), (
                    sid,
                    v,
                    "independent window replay mismatch",
                )
                replayed += 1
    result = dict(
        status="PASS",
        complete_curves_recounted=1440,
        reference_cold_curves_exact=360,
        reference_timing_decisions_recounted=2880,
        numpy_window_replay_exact_curves=replayed,
        replay_shots=[p["order"][i]["shot"] for i in selected],
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        neural_forward_implementation_reused=True,
        independent_future_confirmation=False,
    )
    _durable_json(root / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
