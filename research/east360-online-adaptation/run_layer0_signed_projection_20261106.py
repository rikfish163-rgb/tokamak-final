"""Fixed first-layer policy, with finite validated signed parameter retraction."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from layer0_signed_projection_20261106 import project_layer0
from layer0_signed_support_20261106 import PLAN, causal_phase_support, load_bank
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, raw_for


CONDITIONAL = r.ROOT / "probe_outputs/layer0_conditional835_20261106_v1"
CONDITIONAL_CHECK = (
    r.ROOT / "probe_outputs/layer0_conditional835_verification_20261106_v1.json"
)
FIXED = r.ROOT / "probe_outputs/layer_interaction_masks360_20261105_v1"
ARMS = ("static", "intra", "signed", "shrink_match", "lagged_match", "blind_signed")
RECEIPT_KEYS = (
    "anchor_supported",
    "anchor_positive",
    "anchor_zero",
    "anchor_native_logit",
    "anchor_initial_logit",
    "anchor_projected_logit",
    "anchor_norms",
    "anchor_later_up",
)
EPS = 1e-5


def action_targets(original, lagged, supported, projected):
    if (
        original.ndim != 2
        or original.shape[1] != 128
        or lagged.shape != original.shape
        or projected.values.shape != original.shape
        or supported.shape != original.shape[:1]
        or supported.dtype != torch.bool
        or projected.restricted.shape != supported.shape
        or projected.restricted.dtype != torch.bool
    ):
        raise ValueError("one causal decision per first-layer anchor required")
    if not torch.equal(
        projected.values[~projected.restricted], original[~projected.restricted]
    ):
        raise ValueError("projection changed a non-increasing action")
    guarded = torch.where(supported[:, None], original, projected.values)
    arm = dict(
        intra=original,
        signed=guarded,
        shrink_match=r.policy.match_norm(original, guarded),
        lagged_match=r.policy.match_norm(lagged, guarded),
        blind_signed=projected.values,
    )
    norms = {k: v.norm(dim=1) for k, v in arm.items()}
    if any(
        not torch.isfinite(v).all() or (norms[k] > norms["intra"] + 1e-6).any()
        for k, v in arm.items()
    ) or any(
        (abs(norms[k] - norms["signed"]) > 1e-6).any()
        for k in ("shrink_match", "lagged_match")
    ):
        raise ValueError("directional controls must preserve the signed action norm")
    return {k: torch.cat((v, torch.zeros_like(v)), dim=1) for k, v in arm.items()}


def following_upshifts(native, signed, changed):
    if (
        native.ndim != 1
        or signed.shape != native.shape
        or changed.shape != (len(range(r.policy.FIRST, len(native), r.policy.STRIDE)),)
    ):
        raise ValueError("mismatched anchor and point clocks")
    result = np.zeros(len(changed), dtype=bool)
    for j in np.flatnonzero(changed):
        anchor = r.policy.FIRST + int(j) * r.policy.STRIDE
        use = slice(anchor + 1, min(len(native), anchor + r.policy.STRIDE))
        result[j] = bool((signed[use] > native[use] + 1e-6).any())
    return result


def forecast_signed(base, ctx, rule, keys, bank):
    original, lag = (x[:, :128].contiguous() for x in r.actions(rule, ctx, stale=True))
    allowed = causal_phase_support(keys, bank)["allow"]
    projection = project_layer0(base, ctx, original)
    actions = action_targets(original, lag, allowed, projection)
    curves, stats = r.forecast(base, ctx, actions)
    curves["static"] = ctx[2]
    changed = (projection.restricted & ~allowed).cpu().numpy()
    evidence = dict(
        anchor_supported=allowed.cpu().numpy(),
        anchor_positive=projection.restricted.cpu().numpy(),
        anchor_zero=projection.at_zero.cpu().numpy(),
        anchor_native_logit=projection.native.cpu().numpy(),
        anchor_initial_logit=projection.initial.cpu().numpy(),
        anchor_projected_logit=projection.final.cpu().numpy(),
        anchor_norms=torch.stack([actions[k].norm(dim=1) for k in ARMS[1:]], dim=1)
        .cpu()
        .numpy(),
        anchor_later_up=following_upshifts(curves["static"], curves["signed"], changed),
    )
    return curves, evidence, stats


def run(mode, bank_dir, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, ref, base, source, mapping, map_path, fit_path = parents()
    archive, extra = FIXED, []
    intra_reference = "layer0"
    start_rank = 21
    if mode == "conditional":
        archive = CONDITIONAL
        old_cfg, ref, check = [
            r.read(p)
            for p in (
                archive / "protocol.json",
                archive / "completed.json",
                CONDITIONAL_CHECK,
            )
        ]
        if (
            check["status"] != "PASS_LAYER0_CONDITIONAL_835_HDF5_RECOUNT"
            or check["producer_sha256"] != r.sha(archive / "completed.json")
            or ref["protocol_sha256"] != r.sha(archive / "protocol.json")
            or old_cfg["source_state_sha256"] != source
            or old_cfg["threshold"] != r.THRESHOLD
            or len(old_cfg["order"]) != 835
            or set(old_cfg["order"]) & set(order)
        ):
            raise ValueError("verified disjoint conditional replay required")
        order, cfg = old_cfg["order"], old_cfg
        intra_reference, start_rank = "intra", 1
        extra = [CONDITIONAL_CHECK]
    elif mode != "fixed":
        raise ValueError("unknown replay mode")
    bank, info = load_bank(bank_dir, base.center.device)
    if (
        info["source_state_sha256"] != source
        or info["fit_sha256"] != r.sha(fit_path)
        or info["map_sha256"] != r.sha(map_path)
        or (mode == "fixed" and info["original_roles"] != order[:20])
        or (mode == "conditional" and set(info["original_roles"]) & set(order))
    ):
        raise ValueError("frozen calibration, source, and scoring roles required")
    rule = r.load_rule(fit_path, base.center.device)
    raw_digests = cfg["raw_sha256"]
    deps = [
        Path(__file__),
        PLAN,
        Path(load_bank.__code__.co_filename),
        Path(project_layer0.__code__.co_filename),
        Path(as_case.__code__.co_filename),
        Path(r.__file__),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        archive / "completed.json",
        archive / "protocol.json",
        fit_path,
        map_path,
        r.DEFAULT_PARENT,
        bank_dir / "completed.json",
        bank_dir / "bank.npz",
        *extra,
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            mode=mode,
            order=order,
            arms=ARMS,
            raw_sha256=raw_digests,
            threshold=r.THRESHOLD,
            initial_ns=time.monotonic_ns(),
            bank_active_rank=start_rank,
            first=r.policy.FIRST,
            stride=r.policy.STRIDE,
            eps_logit=EPS,
            bank_roles=info["original_roles"],
            source_state_sha256=source,
            input_sha256=hashes,
            source_fit_sha256=r.sha(fit_path),
            bank_receipt_sha256=r.sha(bank_dir / "completed.json"),
            development_only=True,
            new_fit_labels=0,
            inter_updates=0,
            outcome_reads_during_prediction=0,
        ),
    )
    sealed, previous, parity = [], None, dict(static=0.0, intra=0.0)
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("signed replay may not read any current shot outcome"),
    ):
        for rank, sid in enumerate(order, 1):
            raw = raw_for(sid, raw_digests)
            active = rank >= start_rank
            map_rank = rank if mode == "fixed" else 21
            ctx, _, keys = as_case(
                base, raw, map_rank, mapping, rule if active else None
            )
            if active:
                curves, evidence, stats = forecast_signed(base, ctx, rule, keys, bank)
            else:
                curves, evidence, stats = dict.fromkeys(ARMS, ctx[2]), {}, {}
            stem = f"{rank:04d}_{sid}"
            prev = archive / "shots" / (stem + ".npz")
            ref_record = ref["sealed"][rank - 1]
            if r.sha(prev) != ref_record["curve_sha256"]:
                raise ValueError(f"reference curve changed: {sid}")
            with np.load(prev, allow_pickle=False) as a:
                if not np.array_equal(a["times"], raw["times"]):
                    raise ValueError("mismatched reference shot clock")
                for k, old_arm in (("static", "static"), ("intra", intra_reference)):
                    parity[k] = max(
                        parity[k],
                        layer.require_curve_parity(curves[k], a[old_arm], r.THRESHOLD),
                    )
            path = output / "shots" / (stem + ".npz")
            note = output / "shots" / (stem + "_seal.json")
            r.save_arrays(path, times=raw["times"], **curves, **evidence)
            value = dict(
                rank=rank,
                shot=sid,
                raw_sha256=raw_digests[str(sid)],
                curve_sha256=r.sha(path),
                previous_seal_sha256=previous,
                bank_enabled=active,
                seal_ns=time.monotonic_ns(),
                stats=stats,
            )
            r.write_new(note, value)
            previous = r.sha(note)
            sealed.append(dict(value, seal_sha256=previous))
            if rank % 50 == 0 or rank == len(order):
                print(f"signed {mode} {rank}/{len(order)}", flush=True)
    if (
        r.state_digest(base) != source
        or any(p.grad is not None for p in base.parameters())
        or any(r.sha(p) != digest for p, digest in hashes.items())
    ):
        raise ValueError("signed replay mutated a frozen input")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER0_SIGNED_PROJECTION",
            mode=mode,
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=sealed,
            max_reference_error=parity,
            protocol_sha256=r.sha(output / "protocol.json"),
            completed_ns=time.monotonic_ns(),
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fixed", "conditional"), required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.mode, a.bank, a.output)
