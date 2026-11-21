"""Full-stream independent decision recount and selected original-path replay."""

import argparse
import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ROUTES, bag_backward, prepare_cache
from .run import predict


def classify_independent(times, probability, flag, down, threshold):
    eligible = np.ones(len(times), bool) if not flag else times <= down + 1e-9
    indices = np.flatnonzero(eligible & (probability >= threshold))
    lead = (
        float(round((down - times[indices[0]]) * 1000, 9))
        if flag and len(indices)
        else None
    )
    return dict(
        alarm=bool(len(indices)), lead_ms=lead, timely=lead is not None and lead > 30
    )


def main(root):
    output = root / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED" and p["steps"] == 256
    reference = Path(p["reference"])
    assert sha(reference / "completed.json") == p["reference_completed_sha256"]
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    assert sha(Path(p["cold_state"])) == p["cold_state_sha256"]
    assert sha(reference / "input_map.json") == p["input_map_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    saved = read(root / "per_shot.json")
    access = read(root / "preparation_access.json")
    ids = [r["shot"] for r in p["order"]]
    assert len(ids) == len(set(ids)) == 360 and access["ended_ids"] == ids[:20]
    assert access["first_affected_rank"] == 21
    hashes = read(reference / "input_hashes.json")
    for index, sid in enumerate(ids):
        prefix = f"{index + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        receipt = read(root / "shots" / f"{prefix}_seal.json")
        assert sha(path) == receipt["sha256"] and receipt["outcome_available"] is False
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        if index < 20:
            assert access["seals"][str(sid)] == receipt["sha256"]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as ref,
        ):
            assert np.array_equal(data["times"], ref["times"])
            assert np.array_equal(data["cold"], ref["cold"])
            assert np.array_equal(data["legacy"], ref["intra"])
            for arm in p["arms"]:
                assert np.isfinite(data[arm]).all()
                d = classify_independent(
                    data["times"], data[arm], flag, down, p["threshold"]
                )
                assert dict(shot=sid, disruptive=flag, **d) == saved[arm][index]
                if index < 20:
                    assert np.array_equal(data[arm], ref["source"])
    for arm, rows in saved.items():
        for scope, subset in (
            ("full_stream", rows),
            ("warmup20", rows[:20]),
            ("suffix340", rows[20:]),
        ):
            nd = sum(r["disruptive"] for r in subset)
            ns = len(subset) - nd
            tp = sum(r["timely"] for r in subset)
            fp = sum(r["alarm"] and not r["disruptive"] for r in subset)
            m = done["metrics"][arm][scope]
            assert (
                m["tp"] == tp
                and m["fp"] == fp
                and m["n_disrupt"] == nd
                and m["n_safe"] == ns
            )
            assert (
                m["tpr"] == tp / nd
                and m["fpr"] == fp / ns
                and m["J"] == tp / nd - fp / ns
            )
            assert m["early_over3000"] == sum(
                r["timely"] and r["lead_ms"] > 3000 for r in subset
            )
    _durable_json(
        root / "count_audit.json",
        dict(status="PASS", complete_curves_recounted=3600, reference_curves_exact=720),
    )
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    assert (
        load_context_memory(model, Path(p["context_source"]["path"]))
        == p["context_source"]
    )
    model.restore(torch.load(p["cold_state"], map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    mapping = read(reference / "input_map.json")
    targets = []
    for sid in ids[:20]:
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        s = simple_data.attach_outcome(s, "EAST", {})
        targets.append(
            dict(s, **transform({k: s[k] for k in ("values", "available")}, mapping))
        )
    cache, constant = prepare_cache(model, targets)
    assert torch.equal(
        constant.cpu(),
        torch.load(root / "preparation/constant_direction.pt", weights_only=True),
    )
    manifest = read(root / "preparation/cache_manifest.json")
    for sid, r in cache.items():
        assert {
            k: v for k, v in r.items() if k not in ("h", "valid", "delta")
        } == manifest[str(sid)]
    groups = {
        d: [sid for sid, r in cache.items() if r["disruptive"] == d]
        for d in (False, True)
    }
    rng = np.random.default_rng(2131)
    pairs = [
        [groups[d][int(rng.integers(len(groups[d])))] for d in (False, True)]
        for _ in range(256)
    ]
    assert pairs == read(root / "preparation/pairs.json")
    trace = read(root / "preparation/training.json")
    assert len(trace) == 256 and all(
        r["step"] == i and r["pair"] == pairs[i] for i, r in enumerate(trace)
    )
    initial = model.memory.output.weight.detach()
    final = torch.load(root / "preparation/readouts.pt", weights_only=True)
    boundary = math.log(p["threshold"] / (1 - p["threshold"]))
    replay = []
    for step in (0, 127, 255):
        state = torch.load(
            root / "preparation/states" / f"{step:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for route in ROUTES:
            weight = state[route]["weight"].cuda().requires_grad_(True)
            optimizer = torch.optim.Adam([weight], lr=0.001)
            optimizer.load_state_dict(state[route]["optimizer"])
            optimizer.zero_grad(set_to_none=True)
            bags = [
                dict(
                    shot=sid, **bag_backward(model, cache[sid], weight, route, boundary)
                )
                for sid in pairs[step]
            ]
            penalty = 1e-4 * (weight - initial).square().sum()
            penalty.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_([weight], 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            recorded = trace[step][route]
            assert (
                bags == recorded["bags"]
                and float(penalty.detach()) == recorded["penalty"]
            )
            assert norm == recorded["gradient_norm"]
            assert (
                float((weight - initial).norm().detach())
                == recorded["retained_change_norm"]
            )
            if step == 255:
                assert torch.equal(weight.detach().cpu(), final[route])
            replay.append(
                dict(
                    step=step,
                    route=route,
                    loss_gradient_change_norm_exact=True,
                    final_parameters_exact=step == 255,
                )
            )
    del cache, targets
    weights = {k: v.cuda() for k, v in final.items()}
    selected = [20, 118, 153, 359]
    replayed = 0
    for index in selected:
        sid = ids[index]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        curves, _ = predict(
            model,
            transform({k: s[k] for k in ("values", "available")}, mapping),
            weights,
            constant,
        )
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as data:
            for arm, probability in curves.items():
                assert np.array_equal(probability, data[arm]), (
                    sid,
                    arm,
                    "curve replay mismatch",
                )
                replayed += 1
    result = dict(
        status="PASS",
        complete_curves_recounted=3600,
        reference_curves_exact=720,
        optimizer_replays=replay,
        replay_shots=[ids[i] for i in selected],
        exact_inference_replay_curves=replayed,
        original_learning_and_forward_reused=True,
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
