"""Seal whole-shot internal component and matched interaction-loss ablations."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from moving_peak_component_audit_20261105 import component_actions


PLAN = r.ROOT / "LAYER_INTERACTION_REPLAY_PLAN_20261105.md"
ARCHIVE = r.ROOT / "probe_outputs/moving_peak_replay360_20261105_v1"
ARCHIVE_SHA = "ebc5ec31ba18305e40a906f6cad9f5b4cd05f9d1c59c1c0e7c61cb2267a4205b"
MASK_ARMS = (
    "static",
    "moving_peak",
    "layer0",
    "layer1",
    "gain",
    "bias",
    "layer0_norm",
    "layer1_norm",
)
FIT_ARMS = (
    "static",
    "interaction_control",
    "interaction_penalty",
    "interaction_control_stale",
    "interaction_penalty_stale",
)
ATOL = 4e-6


def component_targets(action):
    if not torch.isfinite(action).all():
        raise ValueError("finite actions required")
    a = component_actions(action)
    full_norm = action.norm(dim=1)
    out = dict(
        moving_peak=a["full"],
        layer0=a["layer0"],
        layer1=a["layer1"],
        gain=a["gain"],
        bias=a["bias"],
    )
    for name in ("layer0", "layer1"):
        v = a[name]
        if ((full_norm > 0) & (v.norm(dim=1) == 0)).any():
            raise ValueError(f"missing direction for equal-norm control: {name}")
        matched = r.policy.match_norm(v, action)
        if ((matched.norm(dim=1) - full_norm).abs() > 1e-6).any():
            raise ValueError(f"unequal norm for component {name}")
        out[name + "_norm"] = matched
    return out


def require_curve_parity(actual, expected, threshold):
    if actual.ndim != 1 or actual.shape != expected.shape or not actual.size:
        raise ValueError("clock alignment required for curve comparison")
    difference = float(np.max(np.abs(actual - expected)))
    if (
        not np.isfinite(difference)
        or difference > ATOL
        or not np.array_equal(actual >= threshold, expected >= threshold)
    ):
        raise ValueError(f"forecast parity failure: max difference {difference:g}")
    return difference


def run(mode, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    if mode not in ("mask", "fit"):
        raise ValueError("expected mask or fit")
    r.configure()
    order, source_cfg, _ = r.reference()
    if r.sha(ARCHIVE / "completed.json") != ARCHIVE_SHA:
        raise ValueError("original moving-peak replay was changed")
    old = r.read(ARCHIVE / "completed.json")
    if old["status"] != "COMPLETE_MOVING_PEAK_REPLAY" or any(
        r.sha(ARCHIVE / k) != h for k, h in old["artifact_sha256"].items()
    ):
        raise ValueError("intact moving-peak terminal archive required")
    old_cfg = r.read(ARCHIVE / "protocol.json")
    if (
        order != old_cfg["order"]
        or old_cfg["threshold"] != r.THRESHOLD
        or old_cfg["raw_sha256"] != source_cfg["raw_sha256"]
        or old_cfg["label_budget"] != r.BUDGET
        or len(old["sealed"]) != len(order)
    ):
        raise ValueError("moving-peak protocol disagrees with frozen parent")
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    digest = r.state_digest(base)
    if (
        digest != old_cfg["source_state_sha256"]
        or parent["threshold"] != r.THRESHOLD
        or any(p.requires_grad for p in base.parameters())
    ):
        raise ValueError("exact frozen J-TEXT parent required")
    base.cuda()
    mapping_path = r.COLD / "input_map.json"
    mapping = r.read(mapping_path)
    if mapping["calibration_ids"] != order[: r.BUDGET] or (
        r.sha(mapping_path) != source_cfg["input_sha256"][str(mapping_path)]
    ):
        raise ValueError("frozen first20 numerical map required")
    deps = [
        Path(__file__),
        PLAN,
        mapping_path,
        r.DEFAULT_PARENT,
        r.io23.EAST_COHORT,
        r.SOURCE / "protocol.json",
        r.SOURCE / "completed.json",
        ARCHIVE / "protocol.json",
        ARCHIVE / "completed.json",
        ARCHIVE / "fit_moving_peak.npz",
        ARCHIVE / "fit_receipt.json",
        Path(r.__file__),
        Path(r.policy.__file__),
        Path(r.latent.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.latent.make_views.__code__.co_filename),
        Path(r.prepare_actual.__code__.co_filename),
        Path(component_actions.__code__.co_filename),
        Path(r.timeline.__code__.co_filename),
    ]
    initial_fit = r.PARENT / "fit20.npz"
    fit_receipt = ARCHIVE / "fit_receipt.json"
    if mode == "fit":
        from moving_peak_interaction_fit_20261105 import fit_interaction

        if r.sha(initial_fit) != r.INITIAL_FIT:
            raise ValueError("unchanged history20 fit required")
        deps.extend((Path(fit_interaction.__code__.co_filename), initial_fit))
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    arms = MASK_ARMS if mode == "mask" else FIT_ARMS
    compare = "moving_peak" if mode == "mask" else "interaction_control"
    maxima = {"static": 0.0, compare: 0.0}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            mode=mode,
            order=order,
            threshold=r.THRESHOLD,
            arms=arms,
            label_budget=r.BUDGET,
            first_anchor=r.policy.FIRST,
            stride=r.policy.STRIDE,
            radius=r.latent.RADIUS,
            source_state_sha256=digest,
            input_sha256=hashes,
            raw_sha256=old_cfg["raw_sha256"],
            development_only=True,
            source_frozen=True,
            temporary_state_reset=True,
            archive_sha256=ARCHIVE_SHA,
            comparison_requires_same_threshold_crossings=True,
            device=torch.cuda.get_device_name(),
            torch_version=str(torch.__version__),
        ),
    )
    rows = {k: [] for k in arms}
    seals, pending = [], []
    rules, fit_hashes = {}, {}
    print("started layer interaction", mode, flush=True)
    for rank, sid in enumerate(order, 1):
        path = r.io23.path_for("EAST", sid)
        before = r.signature(path)
        expected_hash = old_cfg["raw_sha256"][str(sid)]
        with patch.object(
            r.io23,
            "read_outcome",
            side_effect=RuntimeError("current outcome unavailable before seal"),
        ):
            raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
            times = raw["times"]
            ctx = r.latent.native_context(base, r.prepare_actual(raw, rank, mapping))
            native = ctx[2]
            targets = {}
            if rank > r.BUDGET:
                if mode == "mask":
                    targets = component_targets(r.actions(rules["moving_peak"], ctx))
                else:
                    for k, rule in rules.items():
                        real, stale = r.actions(rule, ctx, stale=True)
                        targets[k], targets[k + "_stale"] = real, stale
            traces, updates = r.forecast(base, ctx, targets)
            curves = {k: traces[k] if k in traces else native for k in arms}
        stem = f"{rank:04d}_{sid}"
        prior = ARCHIVE / "shots" / f"{stem}.npz"
        if r.sha(prior) != old["sealed"][rank - 1]["curve_sha256"]:
            raise ValueError(f"archived shot curve changed: {sid}")
        with np.load(prior, allow_pickle=False) as data:
            if not np.array_equal(data["times"], times):
                raise ValueError(f"archived clock mismatch: {sid}")
            for arm, ref in (("static", "static"), (compare, "moving_peak")):
                d = require_curve_parity(curves[arm], data[ref], r.THRESHOLD)
                maxima[arm] = max(maxima[arm], d)
        if any(
            not np.isfinite(v).all() or (v < 0).any() or (v > 1).any()
            for v in curves.values()
        ):
            raise ValueError(f"invalid probability: {sid}")
        curve = output / "shots" / (stem + ".npz")
        r.save_arrays(curve, times=times, **curves)
        curve_hash = r.sha(curve)
        seal = curve.with_name(stem + "_seal.json")
        r.write_new(
            seal,
            dict(
                shot=sid,
                rank=rank,
                raw_sha256=expected_hash,
                curve_sha256=curve_hash,
                current_outcome_used_in_prediction=False,
                updates=updates,
                active_fit_sha256=fit_hashes.copy(),
            ),
        )
        seal_hash = r.sha(seal)
        if r.signature(path) != before or r.sha(path) != expected_hash:
            raise ValueError(f"raw shot changed: {sid}")
        flag, down = r.io23.read_outcome("EAST", sid)
        if r.signature(path) != before:
            raise ValueError(f"raw shot changed during reveal: {sid}")
        scored = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **r.timeline(times, v, flag, down, r.THRESHOLD),
            )
            for k, v in curves.items()
        }
        for k, v in scored.items():
            rows[k].append(v)
        reveal = curve.with_name(stem + "_revealed.json")
        r.write_new(reveal, dict(shot=sid, after_seal_sha256=seal_hash, rows=scored))
        seals.append(
            dict(
                shot=sid,
                curve_sha256=curve_hash,
                seal_sha256=seal_hash,
                revealed_sha256=r.sha(reveal),
            )
        )
        if mode == "fit" and rank <= r.BUDGET:
            pending.append((sid, raw, flag, down))
        if rank == r.BUDGET:
            if mode == "mask":
                fit = ARCHIVE / "fit_moving_peak.npz"
                rules["moving_peak"] = r.load_rule(fit, base.center.device)
                fit_hashes = {"moving_peak": r.sha(fit)}
            else:
                history = [
                    (
                        s,
                        r.latent.native_context(base, r.prepare_actual(v, 21, mapping)),
                        v["times"],
                        f,
                        d,
                    )
                    for s, v, f, d in pending
                ]
                rules, fit_hashes = fit_interaction(
                    base, history, output, initial_fit, fit_receipt
                )
                if set(rules) != set(FIT_ARMS[1:3]) or set(fit_hashes) != set(rules):
                    raise ValueError("matched interaction pair required")
                pending.clear()
                del history
        if rank % 25 == 0 or rank == len(order):
            print("sealed and revealed", mode, rank, len(order), flush=True)
    if (
        r.state_digest(base) != digest
        or any(p.grad is not None for p in base.parameters())
        or any(r.sha(p) != h for p, h in hashes.items())
    ):
        raise AssertionError("frozen source or input changed")
    summary = {k: v for k, v in r.sections(rows).items() if k in ("full360", "after20")}
    r.write_new(output / "per_shot.json", rows)
    r.write_new(output / "summary.json", summary)
    files = ["protocol.json", "per_shot.json", "summary.json"]
    if mode == "fit":
        files.extend(("fit_receipt.json", *(f"fit_{k}.npz" for k in rules)))
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER_INTERACTION_REPLAY",
            mode=mode,
            n_shots=len(order),
            n_curves=len(order) * len(arms),
            sealed=seals,
            parity_max_error=maxima,
            fit_sha256=fit_hashes,
            artifact_sha256={k: r.sha(output / k) for k in files},
            seconds=time.monotonic() - started,
        ),
    )
    print("complete", mode, summary["full360"]["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mask", "fit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.output)
