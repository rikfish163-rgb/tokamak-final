"""Complete360 protected-update comparison with the same frozen decoder."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.run import metrics
from ..direct_decoder_meta_20261102.core import (
    advance as old_advance,
    clone_head,
    prediction,
)
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lifecycle_stream_20261025.audit import classify, flips
from ..lora_stream_20261102.core import direct, encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from ..warmstart_alignment_20261022.alignment import transform
from .core import LowRankDecoder
from .functional_update import advance, FRACTIONS
from .neural_replay import state_at
from .run import SOURCE
from .stream360 import CV, REFERENCE, digest

PLAN = io23.ROOT / "EAST_FUNCTIONAL_PROTECTION360_PLAN_20261102.md"
OLD = io23.ROOT / "probe_outputs/east_decoder_complete360_v1"
ARMS = (
    "static",
    "old_real",
    "parameter_real",
    "hidden_real",
    "hidden_reverse",
    "hidden_hold",
)


def choices():
    return dict(
        old_real=("old", "real"),
        parameter_real=("parameter", "real"),
        hidden_real=("hidden", "real"),
        hidden_reverse=("hidden", "reverse"),
        hidden_hold=("hidden", "hold"),
    )


def predict(model, signal, basis, head):
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only; no label or future metadata")
    state = {a: model.base.center.new_zeros(4, 64) for a in ARMS[1:]}
    curves, updates = {a: [] for a in ARMS}, {a: [] for a in ARMS[1:]}
    for lo in range(0, len(signal["values"]), 32):
        end = min(lo + 32, len(signal["values"])) - 1
        prefix = {k: v[: end + 1] for k, v in signal.items()}
        pack = encode(model, prefix, np.arange(lo, end + 1), basis)
        with torch.no_grad():
            for a in ARMS:
                w = model.base.center.new_zeros(4, 64) if a == "static" else state[a]
                curves[a].append(
                    forward(model, pack, w)["logit"].sigmoid().cpu().numpy()
                )
        origin = feedback_origin(end)
        if origin is None:
            continue
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        with torch.no_grad():
            issued = {
                a: digest(prediction(model, point, state[a], head).cpu().numpy())
                for a in ARMS[1:]
            }
        truth, mask = targets(prefix, origin, end)
        observed = torch.as_tensor(mask, device="cuda")
        for a, (kind, mode) in choices().items():
            value = torch.as_tensor(truth[mode], device="cuda")
            if kind == "old":
                state[a], r = old_advance(model, point, state[a], value, observed, head)
                r["chosen_fraction"] = 1.0 if r["accepted"] else None
            else:
                state[a], r = advance(
                    model, point, state[a], value, observed, head, kind=kind
                )
            updates[a].append(
                dict(
                    origin=origin,
                    maturity=end,
                    sealed_through=end,
                    first_affected=end + 1,
                    forecast_sha256=issued[a],
                    mask_sha256=digest(mask),
                    target_sha256=digest(truth[mode]),
                    observed=int(observed.sum()),
                    **r,
                )
            )
    result = {a: np.concatenate(v) for a, v in curves.items()}
    if any(
        not np.array_equal(v[:160], result["static"][:160]) for v in result.values()
    ):
        raise AssertionError("changed sealed first160")
    return result, updates


def load():
    configure()
    previous = read(OLD / "protocol.json")
    if read(OLD / "completed.json")["status"] != "COMPLETED":
        raise ValueError("completed old trajectory required")
    for name in ("audit_cpu.json", "verification_neural.json"):
        if read(OLD / name)["status"] != "PASS":
            raise ValueError("verified old trajectory required")
    model, parent, cold = load_model()
    if (
        sha(DEFAULT_PARENT) != SOURCE_SHA
        or sha(cold) != previous["cold_sha256"]
        or float(parent["threshold"]) != previous["threshold"]
    ):
        raise ValueError("immutable parent mismatch")
    ref = read(REFERENCE / "protocol.json")
    calibration = Path(ref["calibration_reference"])
    if sha(calibration / "input_map.json") != previous["input_map_sha256"]:
        raise ValueError("input mapping changed")
    mapping, hashes = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    if (
        sha(SOURCE / "decoders.pt") != previous["prior_sha256"]
        or sha(OLD / "heads.pt") != read(OLD / "calibration.json")["heads_sha256"]
    ):
        raise ValueError("frozen decoder changed")
    prior = clone_head(model)
    prior.load_state_dict(
        torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)[
            "heads"
        ]["forecast_only"],
        strict=True,
    )
    prior.requires_grad_(False)
    basis = torch.load(
        CV / "diagnostic_basis.pt", map_location="cuda", weights_only=True
    )
    expected_basis = read(CV / "completed.json")["artifact_sha256"][
        "diagnostic_basis.pt"
    ]
    if sha(CV / "diagnostic_basis.pt") != expected_basis:
        raise ValueError("frozen diagnostic basis changed")
    head = LowRankDecoder(prior, basis)
    saved = torch.load(OLD / "heads.pt", map_location="cuda", weights_only=True)
    with torch.no_grad():
        head.delta.copy_(saved["worst_meta"])
    head.requires_grad_(False)
    return model, head, source_basis(model), cold, previous, mapping, hashes


def preflight(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    model, head, basis, cold, previous, mapping, _ = load()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sid = previous["head_fit_ids"][0]
    raw = apply_input_contract(
        simple_data.read_shot("EAST", sid, rate_contract="nominal")
    )
    signal = transform({k: raw[k][:416] for k in ("values", "available")}, mapping)
    with patch.object(
        io23,
        "read_outcome",
        side_effect=RuntimeError("outcomes forbidden in preflight"),
    ):
        curves, receipts = predict(model, signal, basis, head)
    checks = []
    for q in (0, 160, 288, 415):
        lo, hi = q // 32 * 32, min(q // 32 * 32 + 32, len(signal["values"]))
        prefix = {k: v[:hi] for k, v in signal.items()}
        for a in ARMS:
            state = torch.as_tensor(
                state_at([] if a == "static" else receipts[a], q), device="cuda"
            )
            with torch.no_grad():
                actual = (
                    direct(model, prefix, np.arange(lo, hi), basis, state)["logit"]
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
            if not np.array_equal(actual, curves[a][lo:hi]):
                raise AssertionError((a, q, "actual first-FFN mismatch"))
            checks.append(dict(arm=a, query=q, exact=True))
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items()):
        raise AssertionError("preflight mutated source")
    root.mkdir()
    write(
        root / "preflight.json",
        dict(
            status="PASS",
            shot=sid,
            points=416,
            actual_ffn_checks=checks,
            first160_unchanged=True,
            suffix_raw_reads=0,
            outcomes_read=0,
            cold_sha256=sha(cold),
            accepted={a: sum(r["accepted"] for r in rs) for a, rs in receipts.items()},
        ),
    )
    print(read(root / "preflight.json"), flush=True)


def main(root, resume=False):
    root = Path(root)
    if root.exists() != resume or (root / "completed.json").exists():
        raise FileExistsError("new output or explicit incomplete --resume required")
    model, head, basis, cold, previous, mapping, hashes = load()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ids = [r["shot"] for r in previous["order"]]
    if (
        len(ids) != 360
        or len(set(ids)) != 360
        or ids[:20] != mapping["calibration_ids"]
    ):
        raise ValueError("fixed cohort or startup roster mismatch")
    started = time.monotonic()
    if resume:
        protocol = read(root / "protocol.json")
        if protocol["old_completed_sha256"] != sha(OLD / "completed.json"):
            raise ValueError("reference changed on resume")
        if any(
            sha(Path(path)) != d for path, d in protocol["bound_code_sha256"].items()
        ):
            raise ValueError("frozen code changed on resume")
    else:
        root.mkdir()
        (root / "shots").mkdir()
        bound = imported_code_hashes()
        bound[str(PLAN)] = sha(PLAN)
        protocol = dict(
            scope="DIAGNOSTIC_FUNCTIONAL_PROTECTION_SAME_FROZEN_HEAD",
            arms=list(ARMS),
            order=previous["order"],
            threshold=previous["threshold"],
            source_sha256=SOURCE_SHA,
            cold_sha256=sha(cold),
            input_map_sha256=previous["input_map_sha256"],
            prior_sha256=sha(SOURCE / "decoders.pt"),
            heads_sha256=sha(OLD / "heads.pt"),
            basis_sha256=sha(CV / "diagnostic_basis.pt"),
            old_completed_sha256=sha(OLD / "completed.json"),
            bound_code_sha256=bound,
            fractions=list(FRACTIONS),
            head_preselected="worst_meta",
            candidate_preselected="hidden_real",
            head_training_steps=0,
            suffix_supervised_updates=0,
            warmup20_cost_included=True,
            method_promoted=False,
            independent_confirmation=False,
        )
        write(root / "protocol.json", protocol)
        with zipfile.ZipFile(
            root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(bound):
                archive.write(path, str(Path(path).relative_to(io23.ROOT)))
    rows = {a: [] for a in ARMS}
    for index, sid in enumerate(ids, 1):
        stem = root / "shots" / f"{index:04d}_{sid}"
        path, records, sealpath = (
            Path(str(stem) + s) for s in (".npz", "_updates.json", "_seal.json")
        )
        if sealpath.exists():
            seal = read(sealpath)
            if (
                sha(path) != seal["curve_sha256"]
                or sha(records) != seal["updates_sha256"]
            ):
                raise ValueError("completed shot checkpoint changed")
            with np.load(path, allow_pickle=False) as saved:
                times, curves = (
                    saved["times"].copy(),
                    {a: saved[a].copy() for a in ARMS},
                )
        else:
            if path.exists() or records.exists():
                raise FileExistsError("unsealed partial shot: preserve and investigate")
            if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
                raise ValueError("raw EAST changed")
            oldpath = OLD / "shots" / f"{index:04d}_{sid}.npz"
            oldseal = read(oldpath.with_name(oldpath.stem + "_seal.json"))
            if sha(oldpath) != oldseal["curve_sha256"]:
                raise ValueError("frozen reference curve changed")
            with np.load(oldpath, allow_pickle=False) as old:
                times, cold_curve, old_curve = (
                    old["times"].copy(),
                    old["static"].copy(),
                    old["worst_real"].copy(),
                )
            if index <= 20:
                curves = {a: cold_curve.copy() for a in ARMS}
                updates = {a: [] for a in ARMS[1:]}
            else:
                raw = apply_input_contract(
                    simple_data.read_shot("EAST", sid, rate_contract="nominal")
                )
                signal = transform(
                    {k: raw[k] for k in ("values", "available")}, mapping
                )
                with patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("outcome before seal"),
                ):
                    curves, updates = predict(model, signal, basis, head)
                if (
                    not np.array_equal(raw["times"], times)
                    or not np.array_equal(curves["static"], cold_curve)
                    or not np.array_equal(curves["old_real"], old_curve)
                ):
                    raise AssertionError(
                        "fresh static or old method differs from reference"
                    )
            with path.open("xb") as handle:
                np.savez_compressed(handle, times=times, **curves)
            write(records, updates)
            write(
                sealpath,
                dict(
                    index=index,
                    shot=sid,
                    curve_sha256=sha(path),
                    updates_sha256=sha(records),
                    current_outcome_used_in_prediction=False,
                    reused_warmup=index <= 20,
                ),
            )
        flag, down = io23.read_outcome("EAST", sid)
        for a in ARMS:
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(times, curves[a], flag, down, previous["threshold"]),
                )
            )
        _durable_json(
            root / "progress.json",
            dict(
                completed=index,
                total=360,
                seconds=time.monotonic() - started,
                metrics_so_far={a: metrics(v) for a, v in rows.items()},
            ),
        )
        if index % 10 == 0:
            print(
                dict(completed=index, total=360, seconds=time.monotonic() - started),
                flush=True,
            )
    if (
        sha(DEFAULT_PARENT) != SOURCE_SHA
        or any(
            not torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items()
        )
        or any(
            not torch.equal(v, head.state_dict()[k].cpu())
            for k, v in head_before.items()
        )
    ):
        raise AssertionError("frozen source/cold/head changed")
    if any(sha(Path(path)) != d for path, d in protocol["bound_code_sha256"].items()):
        raise AssertionError("frozen code changed during stream")
    scores = {
        a: {
            part: metrics(r)
            for part, r in (("full360", v), ("warmup20", v[:20]), ("suffix340", v[20:]))
        }
        for a, v in rows.items()
    }
    base, candidate = scores["static"]["full360"], scores["hidden_real"]["full360"]

    def net(score):
        return score["tp"] - score["fp"]

    gate = dict(
        operational=candidate["fp"] <= 36
        and candidate["tp"] > base["tp"]
        and net(candidate) > net(base),
        controls={
            a: net(candidate) > net(scores[a]["full360"])
            for a in ("old_real", "parameter_real", "hidden_reverse", "hidden_hold")
        },
    )
    gate["passed"] = gate["operational"] and all(gate["controls"].values())
    write(root / "per_shot.json", rows)
    done = dict(
        status="COMPLETED",
        metrics=scores,
        flips={a: flips(rows["static"], v) for a, v in rows.items() if a != "static"},
        gate=gate,
        seconds=time.monotonic() - started,
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
        independent_confirmation=False,
        old_local_gate_remains_failed=True,
        optimizer_steps=0,
        protocol_sha256=sha(root / "protocol.json"),
        per_shot_sha256=sha(root / "per_shot.json"),
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.output)
    else:
        main(args.output, args.resume)
