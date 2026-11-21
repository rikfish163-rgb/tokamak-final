"""Same-site, equal-magnitude timing and donor controls on the fixed EAST replay."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from verify_confirmation_raw_replay_20261105 import pair


PLAN = r.ROOT / "LAYER_TIMING_CONTROL_PLAN_20261105.md"
MASK_RUN = r.ROOT / "probe_outputs/layer_interaction_masks360_20261105_v1"
MASK_CHECK = (
    r.ROOT / "probe_outputs/layer_interaction_masks360_verification_20261105_v1.json"
)
SITES = ("layer0", "layer1")
CONTROLS = ("stale", "initial", "donor")
ARMS = (
    "static",
    *(k for site in SITES for k in (site, *(f"{site}_{c}" for c in CONTROLS))),
)


def timing_targets(real, stale, donor):
    if (
        real.ndim != 2
        or real.shape != stale.shape
        or real.shape[1] != 256
        or donor.ndim != 2
        or donor.shape[1] != 256
        or not len(real)
        or not len(donor)
    ):
        raise ValueError("nonempty aligned dual-site direction sequences required")
    if any(not torch.isfinite(v).all() for v in (real, stale, donor)):
        raise ValueError("nonfinite direction sequences")
    ix = torch.arange(len(real), device=real.device).clamp_max(len(donor) - 1)
    out = {}
    for i, site in enumerate(SITES):
        section = slice(i * 128, (i + 1) * 128)
        orig = real[:, section]
        norm = orig.norm(dim=1)
        choices = {
            site: orig,
            site + "_stale": r.policy.match_norm(stale[:, section], orig),
            site + "_initial": r.policy.match_norm(orig[:1].expand_as(orig), orig),
            site + "_donor": r.policy.match_norm(donor[ix, section], orig),
        }
        for k, v in choices.items():
            if ((v.norm(dim=1) - norm).abs() > 1e-6).any():
                raise ValueError(f"direction control must keep same norm: {k}")
            out[k] = torch.zeros_like(real)
            out[k][:, section] = v
    return out, max(0, len(real) - len(donor))


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, source, _ = r.reference()
    check = r.read(MASK_CHECK)
    parent = r.read(MASK_RUN / "completed.json")
    cfg = r.read(MASK_RUN / "protocol.json")
    if (
        check["status"] != "PASS_LAYER_INTERACTION_HDF5_RECOUNT"
        or check["mode"] != "mask"
        or check["n_curves"] != 2880
        or check["producer_sha256"] != r.sha(MASK_RUN / "completed.json")
        or any(r.sha(MASK_RUN / p) != h for p, h in parent["artifact_sha256"].items())
        or cfg["order"] != order
        or cfg["threshold"] != r.THRESHOLD
        or cfg["raw_sha256"] != source["raw_sha256"]
    ):
        raise ValueError("verified whole-curve mask parent required")
    base, par = r.load_parent(r.DEFAULT_PARENT)
    digest = r.state_digest(base)
    if digest != cfg["source_state_sha256"] or par["threshold"] != r.THRESHOLD:
        raise ValueError("qualified unchanged J-TEXT checkpoint required")
    base.cuda()
    map_path = r.COLD / "input_map.json"
    mapping = r.read(map_path)
    if r.sha(map_path) != source["input_sha256"][str(map_path)] or (
        mapping["calibration_ids"] != order[:20]
    ):
        raise ValueError("locked numerical map changed")
    fit = layer.ARCHIVE / "fit_moving_peak.npz"
    fit_hash = r.sha(fit)
    if parent["fit_sha256"] != {"moving_peak": fit_hash}:
        raise ValueError("component generator changed")
    deps = [
        Path(__file__),
        PLAN,
        MASK_CHECK,
        MASK_RUN / "completed.json",
        MASK_RUN / "protocol.json",
        r.DEFAULT_PARENT,
        fit,
        map_path,
        Path(layer.__file__),
        Path(r.__file__),
        Path(r.policy.__file__),
        Path(r.latent.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(r.latent.make_views.__code__.co_filename),
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            order=order,
            arms=ARMS,
            threshold=r.THRESHOLD,
            first=r.policy.FIRST,
            stride=r.policy.STRIDE,
            input_sha256=hashes,
            label_budget=20,
            raw_sha256=cfg["raw_sha256"],
            source_state_sha256=digest,
            fit_sha256=fit_hash,
            development_only=True,
            training_steps=0,
            parent_sha256=check["producer_sha256"],
            device=torch.cuda.get_device_name(),
        ),
    )
    rows, seals = {k: [] for k in ARMS}, []
    errors = dict.fromkeys(("static", *SITES), 0.0)
    rule, previous = None, None
    tail_uses = 0
    print("started same-site timing replay", flush=True)
    for rank, sid in enumerate(order, 1):
        path = r.io23.path_for("EAST", sid)
        before = r.signature(path)
        raw_hash = cfg["raw_sha256"][str(sid)]
        with patch.object(
            r.io23,
            "read_outcome",
            side_effect=RuntimeError("current outcome unavailable before seal"),
        ):
            raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
            times = raw["times"]
            ctx = r.latent.native_context(base, r.prepare_actual(raw, rank, mapping))
            targets, tail, donor_id = {}, 0, None
            if rank > 20:
                donor_id, donor = previous
                real, stale = r.actions(rule, ctx, stale=True)
                targets, tail = timing_targets(real, stale, donor)
            traces, updates = r.forecast(base, ctx, targets)
            curves = {k: traces[k] if k in traces else ctx[2] for k in ARMS}
        stem = f"{rank:04d}_{sid}"
        orig = MASK_RUN / "shots" / (stem + ".npz")
        if r.sha(orig) != parent["sealed"][rank - 1]["curve_sha256"]:
            raise ValueError(f"saved mask curve changed: {sid}")
        with np.load(orig, allow_pickle=False) as f:
            if not np.array_equal(f["times"], times):
                raise ValueError(f"clock mismatch: {sid}")
            for k in errors:
                d = layer.require_curve_parity(curves[k], f[k], r.THRESHOLD)
                errors[k] = max(errors[k], d)
        if any(
            not np.isfinite(v).all() or (v < 0).any() or (v > 1).any()
            for v in curves.values()
        ):
            raise FloatingPointError("invalid score curve")
        curve = output / "shots" / (stem + ".npz")
        r.save_arrays(curve, times=times, **curves)
        curve_hash = r.sha(curve)
        seal = curve.with_name(stem + "_seal.json")
        r.write_new(
            seal,
            dict(
                shot=sid,
                rank=rank,
                raw_sha256=raw_hash,
                curve_sha256=curve_hash,
                current_outcome_used_in_prediction=False,
                active_fit_sha256=fit_hash if rank > 20 else None,
                donor_id=donor_id,
                donor_tail_uses=tail,
                updates=updates,
            ),
        )
        seal_hash = r.sha(seal)
        if r.signature(path) != before or r.sha(path) != raw_hash:
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
        for k in ARMS:
            rows[k].append(scored[k])
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
        if rank == 20:
            rule = r.load_rule(fit, base.center.device)
            mapped = r.latent.native_context(base, r.prepare_actual(raw, 21, mapping))
            previous = sid, r.actions(rule, mapped)
            del mapped
        elif rank > 20:
            previous = sid, real
        tail_uses += tail
        if rank % 25 == 0 or rank == len(order):
            print("sealed and revealed timing", rank, len(order), flush=True)
    if (
        digest != r.state_digest(base)
        or any(v.grad is not None for v in base.parameters())
        or any(r.sha(p) != h for p, h in hashes.items())
    ):
        raise ValueError("frozen source or bound inputs changed")
    summary = {k: v for k, v in r.sections(rows).items() if k in ("full360", "after20")}
    for key, subset in (("full360", slice(None)), ("after20", slice(20, None))):
        summary[key]["real_vs_controls"] = {
            site: {
                control: {
                    str(capped).lower(): pair(
                        rows[site][subset], rows[f"{site}_{control}"][subset], capped
                    )
                    for capped in (False, True)
                }
                for control in CONTROLS
            }
            for site in SITES
        }
    r.write_new(output / "per_shot.json", rows)
    r.write_new(output / "summary.json", summary)
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER_TIMING_CONTROL",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=seals,
            artifact_sha256={
                k: r.sha(output / k)
                for k in ("protocol.json", "per_shot.json", "summary.json")
            },
            parity=errors,
            donor_tail_uses=tail_uses,
            seconds=time.monotonic() - start,
        ),
    )
    print("complete timing", summary["full360"]["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
