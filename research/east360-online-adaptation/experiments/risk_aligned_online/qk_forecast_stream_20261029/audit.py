"""Recount all sealed curves and replay head fitting plus changed-shot inference.

Outcome access here is retrospective evaluation, never a deployment update.
"""

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..fixed_test_protocol import validate_test_ids
from ..gradient_split_20261029.audit import count
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS, UPDATES, STEP, fit_head, predict


def decision(sid, times, scores, flag, down, threshold):
    times, scores = np.asarray(times), np.asarray(scores)
    if (
        times.ndim != 1
        or scores.shape != times.shape
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(scores).all()
        or (np.diff(times) <= 0).any()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("finite ordered times and probability trace required")
    valid = (
        np.round((down - times) * 1000, 9) >= -1e-6
        if flag
        else np.ones(len(times), bool)
    )
    indices = np.flatnonzero((scores >= threshold) & valid)
    alarm = bool(len(indices))
    lead = (
        float(round((down - times[indices[0]]) * 1000, 9)) if alarm and flag else None
    )
    return dict(
        shot=sid,
        is_disrupt=flag,
        alarm=alarm,
        lead_ms=lead,
        timely=bool(lead is not None and lead > 30),
    )


def paired(before, after):
    rescued, harmed = [], []
    for old, new in zip(before, after):
        a = old["timely"] if old["is_disrupt"] else not old["alarm"]
        b = new["timely"] if new["is_disrupt"] else not new["alarm"]
        if b and not a:
            rescued.append(old["shot"])
        if a and not b:
            harmed.append(old["shot"])
    return dict(rescued=rescued, harmed=harmed)


def signal(sid, mapping):
    raw = apply_input_contract(
        simple_data.read_shot("EAST", sid, rate_contract="nominal")
    )
    return transform({k: raw[k] for k in ("values", "available")}, mapping)


def replay(root, p, comparisons):
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA replay required")
    nets = {}
    for name in ("original", "source", "cold"):
        base, parent = load_parent(DEFAULT_PARENT)
        nets[name] = MatrixAttention(base.cuda())
        if name != "original":
            base.forecast_head.load_state_dict(
                torch.load(
                    Path(p["source_root"]) / "forecast_head.pt",
                    map_location="cuda",
                    weights_only=True,
                )
            )
    if parent["threshold"] != p["threshold"]:
        raise ValueError("threshold changed")
    frozen = snapshot(nets["original"])
    ref = Path(p["reference"])
    mapping = read(ref, "input_map.json")
    slow = torch.load(ref / "models/cold.pt", map_location="cuda", weights_only=True)[
        "slow"
    ]
    records = [
        dict(shot=sid, device="EAST", **signal(sid, mapping)) for sid in p["order"][:20]
    ]

    def deny(*args, **kwargs):
        raise AssertionError("replay fitting must use only loaded ended signals")

    with (
        patch.object(io23, "read_outcome", side_effect=deny),
        patch.object(io23, "read_signals", side_effect=deny),
    ):
        fitted = fit_head(nets["cold"], records, slow)
    saved = read(root, "cold_head_training.json")
    if fitted != saved:
        raise ValueError(
            "cold auxiliary training receipts/losses not reproduced exactly"
        )
    weights = torch.load(root / "cold_head.pt", map_location="cuda", weights_only=True)
    if any(
        not torch.equal(v, weights[k])
        for k, v in nets["cold"].base.forecast_head.state_dict().items()
    ):
        raise ValueError("cold auxiliary weights not reproduced bitwise")
    for net in nets.values():
        if any(not torch.equal(v, frozen[k]) for k, v in snapshot(net).items()):
            raise ValueError("head fit changed classifier")
    # All decision-changing shots plus three fixed-index unchanged controls.
    chosen = {p["order"][i] for i in (20, 180, 359)}
    chosen.update(
        sid for v in comparisons.values() for ids in v.values() for sid in ids
    )
    errors = []
    for sid in sorted(chosen, key=p["order"].index):
        sig = signal(sid, mapping)
        with patch.object(io23, "read_outcome", side_effect=deny):
            scores, info = predict(nets, sig, slow)
        i = p["order"].index(sid)
        name = f"{i + 1:04d}_{sid}"
        seal = read(root / "shots", name + "_seal.json")
        if info != seal["updates"]:
            raise ValueError("replayed feedback receipts differ")
        with np.load(root / "shots" / f"{name}.npz") as z:
            error = max(float(np.abs(scores[k] - z[k]).max()) for k in ARMS)
            if error > 2e-6 or any(
                not np.array_equal(scores[k] >= p["threshold"], z[k] >= p["threshold"])
                for k in ARMS
            ):
                raise ValueError("replayed risk/crossings differ")
        errors.append(dict(shot=sid, max_probability_error=error))
        print(dict(replayed_shot=sid, error=error), flush=True)
    return dict(
        cold_head_bitwise_reproduced=True,
        training_receipts_exact=True,
        head_steps=fitted["steps"],
        classifier_unchanged=True,
        shots=errors,
        scope="all changed shots and indices21,181,360; not all360 network replay",
    )


def run(root):
    if (root / "audit.json").exists():
        raise FileExistsError(root / "audit.json")
    p, done = read(root, "protocol.json"), read(root, "complete.json")
    validate_test_ids("EAST", p["order"])
    if not done["complete"] or tuple(p["arms"]) != ARMS or p["step"] != STEP:
        raise ValueError("completed declared arms required")
    if p["source_sha256"] != PARENT_SHA256 or sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError("source changed")
    for name, field in [
        ("protocol.json", "protocol_sha256"),
        ("per_shot.json", "per_shot_sha256"),
        ("receipts.json", "receipts_sha256"),
        ("cold_head.pt", "cold_head_sha256"),
        ("cold_head_training.json", "cold_head_training_sha256"),
    ]:
        if sha(root / name) != done[field]:
            raise ValueError("artifact digest differs: " + name)
    entry = "experiments/risk_aligned_online/qk_forecast_stream_20261029/run.py"
    plan = "QK_FORECAST_STREAM_PLAN.md"
    if entry not in p["code_sha256"] or sha(io23.ROOT / plan) != p["plan_sha256"]:
        raise ValueError("entry/plan provenance incomplete")
    with zipfile.ZipFile(root / "code_snapshot.zip") as z:
        for name, h in dict(p["code_sha256"], **{plan: p["plan_sha256"]}).items():
            if (
                hashlib.sha256(z.read(name)).hexdigest() != h
                or sha(io23.ROOT / name) != h
            ):
                raise ValueError("snapshot or current implementation differs")
    source, previous, ref = (
        Path(p[k]) for k in ("source_root", "previous", "reference")
    )
    if (
        sha(source / "complete.json") != p["source_complete_sha256"]
        or sha(source / "forecast_head.pt") != p["source_head_sha256"]
        or sha(previous / "complete.json") != p["previous_complete_sha256"]
    ):
        raise ValueError("previous artifacts differ")
    for name, h in p["reference_hashes"].items():
        if sha(ref / name) != h:
            raise ValueError("cold reference differs")
    declared, receipts = read(root, "per_shot.json"), read(root, "receipts.json")
    mapping, hashes = read(ref, "input_map.json"), read(ref, "input_hashes.json")
    actual = {k: [] for k in ARMS}
    totals = {
        k: dict(
            updates=0,
            opportunities=0,
            observed_targets=0,
            decreased=0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
        )
        for k in UPDATES
    }
    differences, early_error, max_norm = [], 0.0, 0.0
    for i, sid in enumerate(p["order"]):
        if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw data changed")
        name = f"{i + 1:04d}_{sid}"
        seal = read(root / "shots", name + "_seal.json")
        if (
            seal != receipts[i]
            or seal["sha256"] != sha(root / "shots" / f"{name}.npz")
            or seal["outcome_used"]
            or seal["shot"] != sid
            or seal["index"] != i + 1
            or seal["adaptation_active"] != (i >= 20)
        ):
            raise ValueError("invalid seal or activation")
        oldseal = read(previous / "shots", name + "_seal.json")
        if oldseal["sha256"] != sha(previous / "shots" / f"{name}.npz"):
            raise ValueError("old seal invalid")
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(root / "shots" / f"{name}.npz") as z,
            np.load(previous / "shots" / f"{name}.npz") as old,
        ):
            if (
                not np.array_equal(z["times"], old["times"])
                or np.max(np.abs(z["cold"] - old["cold"])) > 2e-6
                or not np.array_equal(
                    z["cold"] >= p["threshold"], old["cold"] >= p["threshold"]
                )
            ):
                raise ValueError("cold reference prediction changed")
            n = len(z["times"])
            valid = (
                np.round((down - z["times"]) * 1000, 9) >= -1e-6
                if flag
                else np.ones(n, bool)
            )
            evaluation = np.round((down - z["times"]) * 1000, 9) > 30 if flag else valid
            item = dict(shot=sid, flag=flag, arms={}, controls={})
            for k in ARMS:
                row = decision(sid, z["times"], z[k], flag, down, p["threshold"])
                if row != declared[k][i]:
                    raise ValueError("per-shot decision differs")
                actual[k].append(row)
                if i < 20 and not np.array_equal(z[k], z["cold"]):
                    raise ValueError("first20 predictions changed")
                err = float(np.abs(z[k][:256] - z["cold"][:256]).max())
                early_error = max(early_error, err)
                if err > 2e-6 or not np.array_equal(
                    z[k][:256] >= p["threshold"], z["cold"][:256] >= p["threshold"]
                ):
                    raise ValueError("pre-feedback decisions changed")
                item["arms"][k] = dict(
                    max_abs_probability_delta=float(
                        np.abs(z[k][valid] - z["cold"][valid]).max()
                    ),
                    peak_delta=float(
                        z[k][evaluation].max() - z["cold"][evaluation].max()
                    )
                    if evaluation.any()
                    else None,
                )
            for candidate in ("source", "cold"):
                a = z[candidate + "_forecast"][valid].astype(float) - z["cold"][valid]
                for control in ("reverse", "hold"):
                    b = (
                        z[candidate + "_" + control][valid].astype(float)
                        - z["cold"][valid]
                    )
                    norm = np.linalg.norm(a) * np.linalg.norm(b)
                    item["controls"][candidate + "_" + control] = dict(
                        cosine=float(a @ b / norm) if norm else None,
                        max_abs_probability_difference=float(np.abs(a - b).max()),
                    )
            differences.append(item)
        if i < 20:
            if seal["updates"]:
                raise ValueError("cold-prefix feedback unexpectedly active")
            continue
        sig = signal(sid, mapping)
        clocks = list(range(256, n, 128))
        observed, eligible = 0, 0
        for c in clocks:
            here = 0
            for origin in (c - 26, c - 51, c - 76, c - 101):
                mask = (
                    sig["available"][origin + 1 : origin + 26]
                    & np.isfinite(sig["values"][origin + 1 : origin + 26])
                    & sig["available"][origin : origin + 1]
                    & np.isfinite(sig["values"][origin : origin + 1])
                )
                mask[:, 10] = False
                here += int(mask.sum())
            observed += here
            eligible += int(here > 0)
        for k in UPDATES:
            r = seal["updates"][k]
            if (
                r["last_observations"] != [c - 1 for c in clocks]
                or r["opportunities"] != len(clocks)
                or r["observed_targets"] != observed
                or not 0 <= r["updates"] <= eligible
                or not 0 <= r["decreased"] <= len(clocks)
                or r["max_matrix_norm"] > STEP + 1e-6
            ):
                raise ValueError("feedback clock/mask/count/radius mismatch")
            for key in totals[k]:
                totals[k][key] += r[key]
            max_norm = max(max_norm, r["max_matrix_norm"])
    metrics = {
        k: {
            name: count(rows)
            for name, rows in [
                ("full360", r),
                ("cold20", r[:20]),
                ("suffix340", r[20:]),
            ]
        }
        for k, r in actual.items()
    }
    for k in ARMS:
        for section, values in metrics[k].items():
            for field, value in values.items():
                reported = done["results"][k][section][field]
                if reported != value and (
                    reported is None or value is None or abs(reported - value) > 1e-12
                ):
                    raise ValueError("aggregate metric mismatch")
    comparisons = {k: paired(actual["cold"], actual[k]) for k in UPDATES}
    if (
        comparisons != done["comparisons"]
        or {k: v["updates"] for k, v in totals.items()} != done["update_counts"]
    ):
        raise ValueError("paired or update counts differ")
    replayed = replay(root, p, comparisons)
    summary = dict(
        complete=True,
        complete_sha256=sha(root / "complete.json"),
        code_sha256=sha(Path(__file__)),
        snapshot_sha256=sha(root / "code_snapshot.zip"),
        metrics=metrics,
        comparisons=comparisons,
        feedback_totals=totals,
        differences=differences,
        first256_max_abs_error=early_error,
        max_matrix_norm=max_norm,
        replay=replayed,
        scope="all360 sealed decisions; exact cold-head replay; all changed shots + three fixed controls replayed",
        independent_confirmation=False,
    )
    _durable_json(root / "audit.json", summary)
    print(
        dict(
            metrics={k: v["full360"] for k, v in metrics.items()},
            comparisons=comparisons,
            totals=totals,
            replay=replayed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
