"""Independent full-stream recount, parameter constraints and selected optimizer replay."""

import argparse
import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .core import gain_basis, backward_bag
from .prepare import ARMS


def main(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise FileExistsError(output)
    p = read(root / "protocol.json")
    done = read(root / "completed.json")
    reference = Path(p["reference"])
    assert (
        done["status"] == "COMPLETED"
        and sha(reference / "completed.json") == p["reference_completed_sha256"]
    )
    assert sha(Path(p["source_checkpoint"])) == p["source_sha256"]
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for path, digest in p["code_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    ids = [r["shot"] for r in p["order"]]
    assert len(ids) == len(set(ids)) == 360
    assert p["order"] == read(reference / "order.json")
    access = read(root / "preparation_access.json")
    assert access["ended_ids"] == ids[:20] and access["first_affected_rank"] == 21
    density = read(root / "density.json")
    assert density == read(reference / "input_map.json")
    trace = read(root / "preparation/training.json")
    pairs = read(root / "preparation/pairs.json")
    assert len(trace) == len(pairs) == 256
    target = {}
    candidates = {}
    recorded_candidates = read(root / "preparation/candidate_endpoints.json")
    for sid in ids[:20]:
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        s = simple_data.attach_outcome(raw, "EAST", {})
        target[sid] = s
        if s["is_disrupt"]:
            lead = np.round((s["down"] - s["times"]) * 1000, 9)
            eligible = s["valid"] & (s["y"] == 1) & (lead > 30) & (lead <= 125)
        else:
            eligible = s["valid"]
            assert eligible.all() and not s["y"].any()
        candidates[sid] = np.flatnonzero(eligible)
        assert (
            len(candidates[sid])
            and candidates[sid].tolist() == recorded_candidates[str(sid)]
        )
    groups = {
        d: [sid for sid in ids[:20] if target[sid]["is_disrupt"] == d]
        for d in (False, True)
    }
    rng = np.random.default_rng(2131)
    assert pairs == [
        [groups[d][int(rng.integers(len(groups[d])))] for d in (False, True)]
        for _ in range(256)
    ]
    previous = {a: [0.0] * (1 if a.startswith("density") else 9) for a in ARMS}
    for step, row in enumerate(trace):
        assert row["step"] == step and row["pair"] == pairs[step]
        for name in ARMS:
            record = row[name]
            assert record["theta_before"] == previous[name]
            assert (
                len(record["theta_after"]) == len(previous[name])
                and np.isfinite(record["theta_after"]).all()
            )
            assert math.isclose(
                record["penalty"],
                0.01 * np.square(record["theta_before"]).sum(),
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
            for bag, sid in zip(record["bags"], pairs[step], strict=True):
                assert (
                    bag["shot"] == sid
                    and bag["disruptive"] == target[sid]["is_disrupt"]
                )
                assert (
                    bag["candidates"] == len(candidates[sid])
                    and bag["peak_end"] in candidates[sid]
                )
                assert bag["aggregation"] == name.split("_")[1]
            previous[name] = record["theta_after"]
    maps = {a: read(root / "preparation" / f"{a}.json") for a in ARMS}
    for name, mapping in maps.items():
        assert (
            mapping["calibration_ids"] == ids[:20]
            and mapping["theta"] == previous[name]
        )
        b = gain_basis(name.split("_")[0]).numpy()
        assert mapping["basis"] == b.tolist()
        gain = np.asarray(density["gain"]) * np.exp(np.asarray(mapping["theta"]) @ b)
        assert (
            np.allclose(gain, mapping["gain"], rtol=1e-14, atol=0) and (gain > 0).all()
        )
        assert np.array_equal(
            np.asarray(mapping["gain"])[~b.any(0)],
            np.asarray(density["gain"])[~b.any(0)],
        )
        assert not any(mapping["offset"])
        if name.startswith("structured"):
            assert math.isclose(
                mapping["gain"][17], mapping["gain"][15] ** 2, rel_tol=1e-14
            )
    saved = read(root / "per_shot.json")
    rows = {a: [] for a in p["arms"]}
    hashes = read(reference / "input_hashes.json")
    for i, sid in enumerate(ids):
        prefix = f"{i + 1:04d}_{sid}"
        path = root / "shots" / f"{prefix}.npz"
        seal = read(root / "shots" / f"{prefix}_seal.json")
        assert (
            seal["shot"] == sid
            and seal["index"] == i + 1
            and not seal["outcome_available"]
            and sha(path) == seal["sha256"]
        )
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        if i < 20:
            assert access["prediction_seals"][str(sid)] == seal["sha256"]
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(reference / "shots" / f"{prefix}.npz", allow_pickle=False) as ref,
        ):
            assert set(data.files) == set(p["arms"]) | {"times"} and np.array_equal(
                data["times"], ref["times"]
            )
            assert np.array_equal(data["source"], ref["source"]) and np.array_equal(
                data["density"], ref["aligned"]
            )
            for name in rows:
                assert np.isfinite(data[name]).all()
                r = classify(data["times"], data[name], flag, down, p["threshold"])
                assert dict(shot=sid, is_disrupt=flag, **r) == saved[name][i]
                rows[name].append(dict(shot=sid, disruptive=flag, **r))
                if i < 20:
                    assert np.array_equal(data[name], data["source"])
    metrics = {a: counts(rs) for a, rs in rows.items()}
    for a, rs in rows.items():
        for scope, subset in (
            ("full360", rs),
            ("warmup20", rs[:20]),
            ("suffix340", rs[20:]),
        ):
            result = counts(subset)
            for key in (
                "tp",
                "fp",
                "n_disrupt",
                "n_safe",
                "tpr",
                "fpr",
                "J",
                "early_over3000",
            ):
                assert result[key] == done["metrics"][a][scope][key]
    comparisons = {
        a + "__minus__density": flips(rows["density"], rows[a]) for a in ARMS
    }
    for scope in ("density", "structured"):
        comparisons[scope + "_peak__minus__mean"] = flips(
            rows[scope + "_mean"], rows[scope + "_peak"]
        )
    report = dict(
        status="PASS",
        metrics=metrics,
        flips=comparisons,
        complete_curves=2160,
        reference_curves_exact=720,
        source_hash_verified=True,
        exact_preparation_prefix=ids[:20],
        sampling_reconstructed=True,
        full_candidate_sets_verified=True,
        all1024_parameter_transitions_checked=True,
        gains_positive_and_constraints_verified=True,
        completed_sha256=sha(root / "completed.json"),
        independent_confirmation=False,
    )
    _durable_json(root / "count_audit.json", report)
    configure()
    base, _ = load_parent(Path(p["source_checkpoint"]))
    base.cuda().eval().requires_grad_(False)
    replay = []
    prior = torch.as_tensor(density["gain"], dtype=torch.float64, device="cuda")
    boundary = math.log(p["threshold"] / (1 - p["threshold"]))
    for step in (0, 127, 255):
        state = torch.load(
            root / "preparation/optimizer_states" / f"{step:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name in ARMS:
            theta = state[name]["theta"].cuda().detach().requires_grad_(True)
            opt = torch.optim.Adam([theta], lr=0.03)
            opt.load_state_dict(state[name]["optimizer"])
            basis = gain_basis(name.split("_")[0]).cuda()
            opt.zero_grad(set_to_none=True)
            bags = []
            for sid in pairs[step]:
                s = target[sid]
                b = backward_bag(
                    base,
                    {k: s[k] for k in ("values", "available")},
                    candidates[sid],
                    s["is_disrupt"],
                    boundary,
                    name.split("_")[1],
                    theta,
                    prior,
                    basis,
                    weight=0.5,
                    device="cuda",
                )
                bags.append(dict(shot=sid, disruptive=s["is_disrupt"], **b))
            penalty = 0.01 * theta.square().sum()
            penalty.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_([theta], 1.0, error_if_nonfinite=True)
            )
            opt.step()
            expected = trace[step][name]
            assert bags == expected["bags"] and norm == expected["gradient_norm"]
            assert (
                float(penalty.detach()) == expected["penalty"]
                and theta.detach().cpu().tolist() == expected["theta_after"]
            )
            replay.append(
                dict(step=step, arm=name, loss_gradient_parameters_exact=True)
            )
    _durable_json(
        root / "optimizer_replay.json",
        dict(status="PASS", cases=replay, original_update_implementation_reused=True),
    )
    selected = [20, 118, 153, 359]
    exact = 0
    for index in selected:
        sid = ids[index]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        with np.load(
            root / "shots" / f"{index + 1:04d}_{sid}.npz", allow_pickle=False
        ) as data:
            for name in rows:
                gain = (
                    np.ones(23)
                    if name == "source"
                    else np.asarray(
                        (density if name == "density" else maps[name])["gain"]
                    )
                )
                values = raw["values"].copy()
                mask = raw["available"]
                product = (values.astype(np.float64) * gain).astype(np.float32)
                values[mask] = product[mask]
                scores = original_predictions(base, dict(values=values, available=mask))
                assert np.array_equal(scores, data[name]), (sid, name, "curve replay")
                exact += 1
    report.update(
        optimizer_replay_cases=len(replay),
        optimizer_replay_exact=True,
        inference_replay_shots=[ids[i] for i in selected],
        inference_replay_exact_curves=exact,
        verifier_sha256=sha(Path(__file__)),
    )
    _durable_json(output, report)
    print({k: v for k, v in report.items() if k != "flips"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
