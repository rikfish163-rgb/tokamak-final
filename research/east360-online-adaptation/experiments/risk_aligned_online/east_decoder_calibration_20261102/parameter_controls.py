"""Matched feedback controls for the already-observed parameter-backtrack arm."""

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
from ..direct_decoder_meta_20261102.core import prediction
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lifecycle_stream_20261025.audit import classify, flips
from ..lora_stream_20261102.core import encode, forward
from ..simple_online import _durable_json
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform
from .functional_stream import load
from .functional_update import advance
from .stream360 import digest

PLAN = io23.ROOT / "EAST_PARAMETER_FEEDBACK_CONTROLS_PLAN_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/east_functional_protection360_v1"
ARMS = ("static", "real", "reverse", "hold")


def evidence_gate(scores):
    base, candidate = scores["static"], scores["real"]

    def net(score):
        return score["tp"] - score["fp"]

    result = dict(
        operational=candidate["fp"] <= 36
        and candidate["tp"] > base["tp"]
        and net(candidate) > net(base),
        controls={a: net(candidate) > net(scores[a]) for a in ("reverse", "hold")},
    )
    result["passed"] = result["operational"] and all(result["controls"].values())
    return result


def predict(model, signal, basis, head):
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only; current outcomes are forbidden")
    states = {a: model.base.center.new_zeros(4, 64) for a in ARMS[1:]}
    curves, records = {a: [] for a in ARMS}, {a: [] for a in ARMS[1:]}
    for lo in range(0, len(signal["values"]), 32):
        end = min(lo + 32, len(signal["values"])) - 1
        prefix = {k: v[: end + 1] for k, v in signal.items()}
        pack = encode(model, prefix, np.arange(lo, end + 1), basis)
        with torch.no_grad():
            for a in ARMS:
                state = (
                    model.base.center.new_zeros(4, 64) if a == "static" else states[a]
                )
                curves[a].append(
                    forward(model, pack, state)["logit"].sigmoid().cpu().numpy()
                )
        origin = feedback_origin(end)
        if origin is None:
            continue
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        with torch.no_grad():
            issued = {
                a: digest(prediction(model, point, states[a], head).cpu().numpy())
                for a in ARMS[1:]
            }
        truth, mask = targets(prefix, origin, end)
        observed = torch.as_tensor(mask, device="cuda")
        for a in ARMS[1:]:
            states[a], r = advance(
                model,
                point,
                states[a],
                torch.as_tensor(truth[a], device="cuda"),
                observed,
                head,
                kind="parameter",
            )
            records[a].append(
                dict(
                    origin=origin,
                    maturity=end,
                    sealed_through=end,
                    first_affected=end + 1,
                    forecast_sha256=issued[a],
                    mask_sha256=digest(mask),
                    target_sha256=digest(truth[a]),
                    observed=int(observed.sum()),
                    **r,
                )
            )
    result = {a: np.concatenate(v) for a, v in curves.items()}
    if any(
        not np.array_equal(v[:160], result["static"][:160]) for v in result.values()
    ):
        raise AssertionError("changed sealed prefix")
    return result, records


def main(root, resume=False):
    root = Path(root)
    if root.exists() != resume or (root / "completed.json").exists():
        raise FileExistsError("new root or explicit incomplete --resume required")
    for name in ("audit_cpu.json", "verification_neural.json"):
        if read(REFERENCE / name)["status"] != "PASS":
            raise ValueError("verified parameter-arm reference required")
    ref = read(REFERENCE / "protocol.json")
    model, head, basis, cold, _, mapping, hashes = load()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ids = [r["shot"] for r in ref["order"]]
    if (
        len(ids) != 360
        or len(set(ids)) != 360
        or mapping["calibration_ids"] != ids[:20]
    ):
        raise ValueError("cohort drift")
    started = time.monotonic()
    if resume:
        protocol = read(root / "protocol.json")
        if protocol["reference_sha256"] != sha(
            REFERENCE / "completed.json"
        ) or protocol["cold_sha256"] != sha(cold):
            raise ValueError("resumed reference changed")
        if any(
            sha(Path(path)) != d for path, d in protocol["bound_code_sha256"].items()
        ):
            raise ValueError("resumed code changed")
    else:
        root.mkdir()
        (root / "shots").mkdir()
        bound = imported_code_hashes()
        bound[str(PLAN)] = sha(PLAN)
        protocol = dict(
            scope="MATCHED_PARAMETER_BACKTRACK_DIAGNOSTIC",
            arms=list(ARMS),
            order=ref["order"],
            threshold=ref["threshold"],
            source_sha256=ref["source_sha256"],
            cold_sha256=sha(cold),
            input_map_sha256=ref["input_map_sha256"],
            heads_sha256=ref["heads_sha256"],
            reference_sha256=sha(REFERENCE / "completed.json"),
            bound_code_sha256=bound,
            warmup20_cost_included=True,
            head_training_steps=0,
            suffix_supervised_updates=0,
            candidate_preselected="real",
            independent_confirmation=False,
            method_promoted=False,
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
        curve, records, sealpath = (
            Path(str(stem) + s) for s in (".npz", "_updates.json", "_seal.json")
        )
        if sealpath.exists():
            seal = read(sealpath)
            if (
                sha(curve) != seal["curve_sha256"]
                or sha(records) != seal["updates_sha256"]
            ):
                raise ValueError("sealed checkpoint changed")
            with np.load(curve, allow_pickle=False) as data:
                times, curves = data["times"].copy(), {a: data[a].copy() for a in ARMS}
        else:
            if curve.exists() or records.exists():
                raise FileExistsError("unsealed partial shot; preserve and investigate")
            if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
                raise ValueError("raw EAST changed")
            oldpath = REFERENCE / "shots" / f"{index:04d}_{sid}.npz"
            oldseal = read(oldpath.with_name(oldpath.stem + "_seal.json"))
            if sha(oldpath) != oldseal["curve_sha256"]:
                raise ValueError("reference changed")
            with np.load(oldpath, allow_pickle=False) as old:
                times, baseline, real_reference = (
                    old["times"].copy(),
                    old["static"].copy(),
                    old["parameter_real"].copy(),
                )
            if index <= 20:
                curves = {a: baseline.copy() for a in ARMS}
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
                    not np.array_equal(times, raw["times"])
                    or not np.array_equal(curves["static"], baseline)
                    or not np.array_equal(curves["real"], real_reference)
                ):
                    raise AssertionError(
                        "fresh static or parameter-real not equal reference"
                    )
            with curve.open("xb") as handle:
                np.savez_compressed(handle, times=times, **curves)
            write(records, updates)
            write(
                sealpath,
                dict(
                    index=index,
                    shot=sid,
                    curve_sha256=sha(curve),
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
                    **classify(times, curves[a], flag, down, ref["threshold"]),
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
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
    ) or any(
        not torch.equal(v, head.state_dict()[k].cpu()) for k, v in head_before.items()
    ):
        raise AssertionError("frozen model changed")
    if any(sha(Path(path)) != d for path, d in protocol["bound_code_sha256"].items()):
        raise AssertionError("frozen code changed")
    scores = {
        a: {
            part: metrics(r)
            for part, r in (("full360", v), ("warmup20", v[:20]), ("suffix340", v[20:]))
        }
        for a, v in rows.items()
    }
    write(root / "per_shot.json", rows)
    done = dict(
        status="COMPLETED",
        metrics=scores,
        gate=evidence_gate({a: scores[a]["full360"] for a in ARMS}),
        flips={a: flips(rows["static"], v) for a, v in rows.items() if a != "static"},
        seconds=time.monotonic() - started,
        source_unchanged=True,
        cold_unchanged=True,
        head_unchanged=True,
        independent_confirmation=False,
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
    args = parser.parse_args()
    main(args.output, args.resume)
