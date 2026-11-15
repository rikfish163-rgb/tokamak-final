"""Checkpointed complete360 intra-only diagnostic, fixed preselected candidate."""

import argparse
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.run import metrics
from ..direct_decoder_meta_20261102.core import advance, clone_head, prediction
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lifecycle_stream_20261025.audit import classify, flips
from ..lora_stream_20261102.core import encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from ..warmstart_alignment_20261022.alignment import transform
from .core import LowRankDecoder
from .run import LOCAL, SOURCE, collect, fit

PLAN = io23.ROOT / "EAST_DECODER_COMPLETE360_PLAN_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/innovation_metric360_v2"
CV = io23.ROOT / "probe_outputs/east_decoder_calibration20_v1"
ARMS = (
    "static",
    "prior_real",
    "forecast_real",
    "worst_real",
    "worst_reverse",
    "worst_hold",
)


def specs(heads):
    return dict(
        prior_real=(heads["prior"], "real"),
        forecast_real=(heads["forecast_only"], "real"),
        worst_real=(heads["worst_meta"], "real"),
        worst_reverse=(heads["worst_meta"], "reverse"),
        worst_hold=(heads["worst_meta"], "hold"),
    )


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def predict(model, signal, basis, heads):
    if set(signal) != {"values", "available"}:
        raise ValueError("prediction accepts signals only, never outcomes")
    choices = specs(heads)
    state = {a: model.base.center.new_zeros(4, 64) for a in choices}
    curves = {a: [] for a in ARMS}
    updates = {a: [] for a in choices}
    for lo in range(0, len(signal["values"]), 32):
        end = min(lo + 32, len(signal["values"])) - 1
        prefix = {k: v[: end + 1] for k, v in signal.items()}
        pack = encode(model, prefix, np.arange(lo, end + 1), basis)
        with torch.no_grad():
            curves["static"].append(
                forward(model, pack, model.base.center.new_zeros(4, 64))["logit"]
                .sigmoid()
                .cpu()
                .numpy()
            )
            for a in choices:
                curves[a].append(
                    forward(model, pack, state[a])["logit"].sigmoid().cpu().numpy()
                )
        origin = feedback_origin(end)
        if origin is None:
            continue
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        with torch.no_grad():
            issued = {
                a: digest(prediction(model, point, state[a], h).cpu().numpy())
                for a, (h, _) in choices.items()
            }
        truth, mask = targets(prefix, origin, end)
        observed = torch.as_tensor(mask, device=model.base.center.device)
        for a, (head, mode) in choices.items():
            state[a], r = advance(
                model,
                point,
                state[a],
                torch.as_tensor(truth[mode], device=model.base.center.device),
                observed,
                head,
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
        raise AssertionError("adaptation changed sealed prefix")
    return result, updates


def main(root, resume=False):
    root = Path(root)
    if root.exists() != resume or (root / "completed.json").exists():
        raise FileExistsError("fresh root or explicit incomplete-run --resume required")
    configure()
    started = time.monotonic()
    ref = read(REFERENCE / "protocol.json")
    if (
        sha(REFERENCE / "completed.json")
        != "f558ee30e33192da29daa7863f58267d25913a9c8dc25a645188ab32cac11014"
    ):
        raise ValueError("frozen complete360 reference changed")
    for name in ("verification_records.json", "verification_neural.json"):
        if read(REFERENCE / name)["status"] != "PASS":
            raise ValueError("baseline is not verified")
    order = ref["order"]
    ids = [r["shot"] for r in order]
    mapping_path = Path(ref["calibration_reference"]) / "input_map.json"
    mapping = read(mapping_path)
    inputs = read(Path(ref["calibration_reference"]) / "input_hashes.json")
    if (
        len(ids) != 360
        or len(set(ids)) != 360
        or mapping["calibration_ids"] != ids[:20]
        or sha(mapping_path) != ref["input_map_sha256"]
    ):
        raise ValueError("cohort or mapping drift")
    model, parent, cold = load_model()
    if (
        sha(cold) != ref["cold_state_sha256"]
        or float(parent["threshold"]) != ref["threshold"]
    ):
        raise ValueError("parent or threshold drift")
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prior = clone_head(model)
    saved = torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)
    prior.load_state_dict(saved["heads"]["forecast_only"], strict=True)
    prior.eval().requires_grad_(False)
    basis = torch.load(
        CV / "diagnostic_basis.pt", map_location="cuda", weights_only=True
    )
    ffn_basis = source_basis(model)
    if resume:
        protocol = read(root / "protocol.json")
        if (
            protocol["order"] != order
            or protocol["prior_sha256"] != sha(SOURCE / "decoders.pt")
            or protocol["cold_sha256"] != sha(cold)
        ):
            raise ValueError("resumed prior/parent/cohort changed")
        for path, d in protocol["bound_code_sha256"].items():
            if sha(Path(path)) != d:
                raise ValueError("resumed stream code changed")
        calibration = read(root / "calibration.json")
        if sha(root / "heads.pt") != calibration["heads_sha256"]:
            raise ValueError("calibrated head checkpoint changed")
        fitted = torch.load(root / "heads.pt", map_location="cuda", weights_only=True)
        heads = {a: LowRankDecoder(prior, basis) for a in fitted}
        for a, h in heads.items():
            with torch.no_grad():
                h.delta.copy_(fitted[a])
            h.requires_grad_(False)
    else:
        root.mkdir(parents=True)
        (root / "shots").mkdir()
        p = read(LOCAL / "protocol.json")
        if p["ids"] != ids[:20]:
            raise ValueError("calibration roster differs")
        examples, _ = collect(model, ffn_basis, p)
        bound = imported_code_hashes()
        bound[str(PLAN)] = sha(PLAN)
        protocol = dict(
            scope="DIAGNOSTIC_COMPLETE360_AFTER_LOCAL_SURROGATE_GATE_FAILED",
            arms=list(ARMS),
            order=order,
            threshold=ref["threshold"],
            source_sha256=SOURCE_SHA,
            cold_sha256=sha(cold),
            input_map_sha256=sha(mapping_path),
            reference_completed_sha256=sha(REFERENCE / "completed.json"),
            prior_sha256=sha(SOURCE / "decoders.pt"),
            local_cv_completed_sha256=sha(CV / "completed.json"),
            local_cv_gate=read(CV / "completed.json")["gate"],
            bound_code_sha256=bound,
            head_fit_ids=ids[:20],
            steps=128,
            head_parameters=2300,
            source_parameters_updated=0,
            warmup20_cost_included=True,
            suffix_supervised_updates=0,
            suffix_head_updates=0,
            method_promoted=False,
            independent_confirmation=False,
            candidate_preselected="worst_real",
        )
        write(root / "protocol.json", protocol)
        with zipfile.ZipFile(
            root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(bound):
                archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        heads, history = fit(model, examples, ids[:20], prior, basis, ref["threshold"])
        write(root / "head_training.json", history)
        torch.save(
            {a: h.delta.detach().cpu() for a, h in heads.items()}, root / "heads.pt"
        )
        write(
            root / "calibration.json",
            dict(
                ids=ids[:20],
                steps=128,
                heads_sha256=sha(root / "heads.pt"),
                history_sha256=sha(root / "head_training.json"),
                only_ended20_used=True,
            ),
        )
    heads["prior"] = LowRankDecoder(prior, basis).requires_grad_(False)
    head_initial = {a: h.delta.detach().clone() for a, h in heads.items()}
    rows = {a: [] for a in ARMS}
    for index, sid in enumerate(ids, 1):
        stem = root / "shots" / f"{index:04d}_{sid}"
        curve = Path(str(stem) + ".npz")
        updates_path = Path(str(stem) + "_updates.json")
        seal = Path(str(stem) + "_seal.json")
        if seal.exists():
            s = read(seal)
            if (
                sha(curve) != s["curve_sha256"]
                or sha(updates_path) != s["updates_sha256"]
            ):
                raise ValueError("checkpointed shot changed")
            with np.load(curve, allow_pickle=False) as data:
                times = data["times"].copy()
                curves = {a: data[a].copy() for a in ARMS}
        else:
            if curve.exists() or updates_path.exists():
                raise FileExistsError(
                    "unsealed partial shot; preserve and investigate before resuming"
                )
            if sha(io23.path_for("EAST", sid)) != inputs[str(sid)]:
                raise ValueError("EAST raw drift")
            old_path = REFERENCE / "shots" / f"{index:04d}_{sid}.npz"
            old_seal = read(old_path.with_name(old_path.stem + "_seal.json"))
            if sha(old_path) != old_seal["sha256"]:
                raise ValueError("reference curve changed")
            with np.load(old_path, allow_pickle=False) as data:
                times = data["times"].copy()
                baseline = data["cold"].copy()
            if index <= 20:
                curves = {a: baseline.copy() for a in ARMS}
                updates = {a: [] for a in ARMS if a != "static"}
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
                    curves, updates = predict(model, signal, ffn_basis, heads)
                if not np.array_equal(raw["times"], times) or not np.array_equal(
                    curves["static"], baseline
                ):
                    raise AssertionError(
                        "fresh static baseline differs from fixed parent"
                    )
            with curve.open("xb") as handle:
                np.savez_compressed(handle, times=times, **curves)
            write(updates_path, updates)
            write(
                seal,
                dict(
                    index=index,
                    shot=sid,
                    curve_sha256=sha(curve),
                    updates_sha256=sha(updates_path),
                    current_outcome_used_in_prediction=False,
                    reused_warmup=index <= 20,
                ),
            )
        # HDF metadata may be read ONLY after a complete curve seal exists.
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
                dict(
                    stage="stream",
                    completed=index,
                    total=360,
                    seconds=time.monotonic() - started,
                ),
                flush=True,
            )
    if (
        sha(DEFAULT_PARENT) != SOURCE_SHA
        or any(
            not torch.equal(v, model.state_dict()[k].detach().cpu())
            for k, v in original.items()
        )
        or any(not torch.equal(v, heads[a].delta) for a, v in head_initial.items())
    ):
        raise AssertionError("source/cold/head changed during suffix")
    if any(sha(Path(path)) != d for path, d in protocol["bound_code_sha256"].items()):
        raise AssertionError("frozen stream code changed")
    score = {
        a: {
            part: metrics(part_rows)
            for part, part_rows in (
                ("full360", v),
                ("warmup20", v[:20]),
                ("suffix340", v[20:]),
            )
        }
        for a, v in rows.items()
    }
    base, candidate = score["static"]["full360"], score["worst_real"]["full360"]
    def net(m):
        return m["tp"] - m["fp"]
    gate = dict(
        operational=candidate["fp"] <= 36
        and candidate["tp"] > base["tp"]
        and net(candidate) > net(base),
        controls={
            a: net(candidate) > net(score[a]["full360"])
            for a in ("prior_real", "forecast_real", "worst_reverse", "worst_hold")
        },
    )
    gate["passed"] = gate["operational"] and all(gate["controls"].values())
    write(root / "per_shot.json", rows)
    done = dict(
        status="COMPLETED",
        metrics=score,
        flips={a: flips(rows["static"], v) for a, v in rows.items() if a != "static"},
        gate=gate,
        seconds=time.monotonic() - started,
        source_unchanged=True,
        cold_unchanged=True,
        heads_unchanged=True,
        no_suffix_supervision=True,
        independent_confirmation=False,
        local_gate_remains_failed=True,
        per_shot_sha256=sha(root / "per_shot.json"),
        protocol_sha256=sha(root / "protocol.json"),
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.output, args.resume)
