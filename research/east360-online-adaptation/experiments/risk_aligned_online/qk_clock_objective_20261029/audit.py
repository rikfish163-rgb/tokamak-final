"""Replay factorial validation and independently recount shot-level changes."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..qk_matched_head_20261029.learning import GROUPS
from ..qk_matched_head_20261029.run import load_split
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import ARMS, PRIOR, evaluate


def describe(rows):
    out = {}
    for group in GROUPS:
        ids = sorted({r["shot"] for r in rows if r["group"] == group})
        changes, probabilities = [], []
        for sid in ids:
            selected = [r for r in rows if r["group"] == group and r["shot"] == sid]
            changes.append(np.mean([r["after"] - r["before"] for r in selected]))
            probabilities.append(np.mean([r["probability_change"] for r in selected]))
        x, p = np.array(changes), np.array(probabilities)
        out[group] = dict(
            shots=len(ids),
            mean=float(x.mean()),
            sd_between_shots=float(x.std(ddof=1)),
            median=float(np.median(x)),
            improved=int((x < -1e-6).sum()),
            worsened=int((x > 1e-6).sum()),
            probability_mean_change=float(p.mean()),
            probability_up=int((p > 1e-6).sum()),
            probability_down=int((p < -1e-6).sum()),
        )
    out["macro"] = float(np.mean([out[g]["mean"] for g in GROUPS]))
    return out


def run(root):
    if (root / "audit.json").exists():
        raise FileExistsError(root / "audit.json")
    protocol, done, result = [
        read(root, f"{k}.json") for k in ("protocol", "complete", "validation")
    ]
    for k in ("protocol", "training", "validation"):
        if sha(root / f"{k}.json") != done[f"{k}_sha256"]:
            raise ValueError("artifact changed")
    entry = "experiments/risk_aligned_online/qk_clock_objective_20261029/run.py"
    if entry not in protocol["code_sha256"]:
        raise ValueError("CLI entry missing from contemporaneous snapshot")
    with zipfile.ZipFile(root / "code_snapshot.zip") as z:
        for name, h in protocol["code_sha256"].items():
            if (
                hashlib.sha256(z.read(name)).hexdigest() != h
                or sha(io23.ROOT / name) != h
            ):
                raise ValueError("recorded or live code changed")
        if (
            hashlib.sha256(z.read("QK_CLOCK_OBJECTIVE_PLAN.md")).hexdigest()
            != protocol["plan_sha256"]
        ):
            raise ValueError("plan snapshot mismatch")
    if sha(io23.ROOT / "QK_CLOCK_OBJECTIVE_PLAN.md") != protocol["plan_sha256"]:
        raise ValueError("live plan changed")
    if (
        sha(DEFAULT_PARENT) != PARENT_SHA256
        or protocol["source_sha256"] != PARENT_SHA256
    ):
        raise ValueError("qualified source changed")
    if sha(PRIOR / "complete.json") != protocol["prior_complete_sha256"]:
        raise ValueError("source teaching parent changed")
    roles = simple_data.source_roles()
    if protocol["train_ids"] != roles["train"] or protocol["val_ids"] != roles["val"]:
        raise ValueError("source roles changed")
    history = read(root, "training.json")["history"]
    if len(history) != 512 or [r["step"] for r in history] != list(range(1, 513)):
        raise ValueError("training budget changed")
    for row in history:
        for name in ARMS:
            r = row[name]
            numbers = (
                [r["objective"], r["head_gradient_norm"]]
                + r["signal"]
                + r["delta"]
                + r["alignment"]
            )
            if not np.isfinite(numbers).all() or r["head_gradient_norm"] <= 0:
                raise ValueError("invalid or absent meta gradient")
    for sid, h in read(PRIOR, "input_hashes.json").items():
        if sha(io23.path_for("J-TEXT", int(sid))) != h:
            raise ValueError("source input changed")
    advanced = simple_data.advanced_times()
    shots = load_split(
        roles["val"], {k: advanced[k] for k in roles["val"] if k in advanced}
    )
    lookup = {s["shot"]: s for s in shots}
    timing = {}
    for clock, receipts in result["receipts"].items():
        occupied = []
        for r in receipts:
            count = (
                r["query"]
                if clock == "fresh"
                else 256 + 128 * ((r["query"] - 256) // 128)
            )
            origins = [count - 26 - 25 * j for j in range(4)]
            if (
                r["count"] != count
                or r["origins"] != origins
                or r["last_target"] != count - 1
            ):
                raise ValueError("noncausal or altered feedback clock")
            s = lookup[r["shot"]]
            if s["is_disrupt"] and r["label"] > 0.5:
                # Post-hoc SOURCE label occupancy only, never an input to adaptation.
                occupied.append(float(np.mean(s["y"][count - 100 : count] > 0.5)))
        timing[clock] = dict(
            danger_endpoints=len(occupied),
            mean_positive_fraction_in_arrived_targets=float(np.mean(occupied)),
            zero_positive_target_endpoints=int(np.sum(np.array(occupied) == 0)),
        )
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nets = {}
    for name in list(ARMS) + ["original", "forecast"]:
        base, _ = load_parent(DEFAULT_PARENT)
        base.cuda()
        nets[name] = MatrixAttention(base)
        if name != "original":
            folder = PRIOR if name == "forecast" else root
            h = read(folder, "complete.json")["heads"][name]
            if sha(folder / f"{name}_head.pt") != h:
                raise ValueError("head changed")
            base.forecast_head.load_state_dict(
                torch.load(
                    folder / f"{name}_head.pt", map_location="cuda", weights_only=True
                )
            )
    replay = evaluate(nets, shots, read(PRIOR, "validation.json"))
    if replay != result:
        raise ValueError("saved finite-step results do not replay exactly")
    description = {
        name: describe(rows) for name, rows in result["per_endpoint"].items()
    }
    screen = {}
    for name, (clock, _) in ARMS.items():
        m = description[f"{name}_true"]
        screen[name] = all(m[g]["mean"] <= 0 for g in ("safe", "danger")) and all(
            m["macro"] < description[k]["macro"] - 1e-6
            for k in (
                f"original_{clock}",
                f"forecast_{clock}",
                f"{name}_reverse",
                f"{name}_hold",
            )
        )
    for name, groups in description.items():
        for g in GROUPS:
            if (
                abs(groups[g]["mean"] - result["summary"][name][g]["mean_risk_change"])
                > 1e-12
            ):
                raise ValueError("independent shot recount differs")
    if screen != done["source_screen"] or screen != result["source_screen"]:
        raise ValueError("acceptance screen differs")
    v1 = io23.ROOT / "probe_outputs/qk_clock_objective_v1"
    # v1 omitted the CLI entry in its snapshot. Keep it, but do not present
    # the second run as an independent seed or patch its old protocol.
    same_seed_reproduction = dict(
        validation_equal=sha(v1 / "validation.json") == sha(root / "validation.json"),
        training_equal=sha(v1 / "training.json") == sha(root / "training.json"),
        heads_equal={
            name: sha(v1 / f"{name}_head.pt") == sha(root / f"{name}_head.pt")
            for name in ARMS
        },
    )
    audited = dict(
        complete=True,
        complete_sha256=sha(root / "complete.json"),
        audit_code_sha256=sha(Path(__file__)),
        arms=len(result["per_endpoint"]),
        paired_endpoints=len(result["receipts"]["fresh"]),
        exact_validation_replay=True,
        description=description,
        timing_occupancy_diagnostic=timing,
        same_seed_reproduction=same_seed_reproduction,
        source_screen=screen,
        no_east_results=True,
        independent_confirmation=False,
    )
    _durable_json(root / "audit.json", audited)
    print(
        dict(
            screen=screen,
            exact_replay=True,
            reproduction=same_seed_reproduction,
            timing=timing,
            true_arms={k: description[f"{k}_true"] for k in ARMS},
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
