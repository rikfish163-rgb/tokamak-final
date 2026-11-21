"""Fixed first-layer within-shot policy on the pre-existing conditional 835 roster."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from confirmation_raw_replay_20261105 import ROSTER, ROSTER_SHA
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for


PLAN = r.ROOT / "LAYER0_CONDITIONAL_ROBUSTNESS_PLAN_20261105.md"
PRIOR = r.ROOT / "probe_outputs/confirmation_raw_conditional835_20261105_v1"
AUDIT = (
    r.ROOT
    / "probe_outputs/confirmation_raw_conditional835_verification_20261105_v2.json"
)
ARMS = ("static", "intra", "lagged", "first_direction")


def same_site_targets(real, stale):
    if real.ndim != 2 or real.shape != stale.shape or real.shape[1] != 256:
        raise ValueError("matched full two-layer direction shapes required")
    orig = real[:, :128]
    fixed = orig[:1].expand_as(orig) if len(orig) else orig
    matched = {
        "intra": orig,
        "lagged": r.policy.match_norm(stale[:, :128], orig),
        "first_direction": r.policy.match_norm(fixed, orig),
    }
    out = {}
    norm = orig.norm(dim=1)
    for k, v in matched.items():
        if not torch.isfinite(v).all() or ((v.norm(dim=1) - norm).abs() > 1e-6).any():
            raise ValueError("same-site direction control cannot change norm")
        out[k] = torch.cat((v, torch.zeros_like(v)), dim=1)
    return out


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    fixed, _, _, base, source, mapping, map_path, fit_path = parents()
    if r.sha(ROSTER) != ROSTER_SHA:
        raise ValueError("conditional roster changed")
    roster, old_cfg, old, audit = [
        r.read(p)
        for p in (ROSTER, PRIOR / "protocol.json", PRIOR / "completed.json", AUDIT)
    ]
    order, raw_digests = roster["ids"], roster["raw_sha256"]
    if (
        roster["status"] != "LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER"
        or roster["independent_validation"] is not False
        or len(order) != len(set(order))
        or len(order) != 835
        or set(order) & set(fixed)
        or old_cfg["order"] != order
        or old_cfg["raw_sha256"] != raw_digests
        or old_cfg["threshold"] != r.THRESHOLD
        or old_cfg["mode"] != "conditional"
        or audit["status"] != "PASS_RAW_CONFIRMATION_RECOUNT"
        or audit["producer_completion_sha256"] != r.sha(PRIOR / "completed.json")
        or old["artifact_sha256"]["protocol.json"] != r.sha(PRIOR / "protocol.json")
    ):
        raise ValueError("wrong conditional baseline or roster")
    rule = r.load_rule(fit_path, base.center.device)
    deps = [
        Path(__file__),
        PLAN,
        ROSTER,
        PRIOR / "protocol.json",
        PRIOR / "completed.json",
        AUDIT,
        fit_path,
        map_path,
        r.DEFAULT_PARENT,
        Path(r.__file__),
        Path(r.policy.__file__),
        Path(r.latent.__file__),
        Path(r.simple_data.__file__),
        Path(r.io23.__file__),
        Path(r.prepare_actual.__code__.co_filename),
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            order=order,
            arms=ARMS,
            raw_sha256=raw_digests,
            threshold=r.THRESHOLD,
            first=r.policy.FIRST,
            stride=r.policy.STRIDE,
            source_state_sha256=source,
            fit_sha256=r.sha(fit_path),
            input_sha256=hashes,
            independent_validation=False,
            mapped_from_first=True,
            added_fit_labels=0,
            inter_updates=0,
            outcome_reads_during_replay=0,
            ordering=roster["ordering"],
            device=torch.cuda.get_device_name(),
            initial_ns=time.monotonic_ns(),
        ),
    )
    sealed, max_err, previous = [], 0.0, None
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError(
            "conditional outcomes are only for the independent scorer"
        ),
    ):
        for i, sid in enumerate(order, 1):
            raw = raw_for(sid, raw_digests)
            ctx = r.latent.native_context(base, r.prepare_actual(raw, 21, mapping))
            real, stale = r.actions(rule, ctx, stale=True)
            curves, stats = r.forecast(base, ctx, same_site_targets(real, stale))
            curves["static"] = ctx[2]
            times = raw["times"]
            stem = f"{i:04d}_{sid}"
            prior = PRIOR / "shots" / (stem + ".npz")
            if r.sha(prior) != old["sealed"][i - 1]["curve_sha256"]:
                raise ValueError(f"archived conditional trace changed: {sid}")
            with np.load(prior, allow_pickle=False) as p:
                if not np.array_equal(times, p["times"]):
                    raise ValueError("conditional signal clock changed")
                max_err = max(
                    max_err,
                    layer.require_curve_parity(
                        curves["static"], p["static"], r.THRESHOLD
                    ),
                )
            path = output / "shots" / (stem + ".npz")
            seal = output / "shots" / (stem + "_seal.json")
            r.save_arrays(path, times=times, **curves)
            note = dict(
                rank=i,
                shot=sid,
                raw_sha256=raw_digests[str(sid)],
                curve_sha256=r.sha(path),
                previous_seal_sha256=previous,
                seal_ns=time.monotonic_ns(),
                stats=stats,
            )
            r.write_new(seal, note)
            previous = r.sha(seal)
            sealed.append(dict(note, seal_sha256=previous))
            if i % 50 == 0 or i == len(order):
                print(
                    f"sealed conditional first-layer replay {i}/{len(order)}",
                    flush=True,
                )
    if (
        r.state_digest(base) != source
        or any(p.grad is not None for p in base.parameters())
        or any(r.sha(p) != d for p, d in hashes.items())
    ):
        raise ValueError("frozen conditional source or input changed")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER0_CONDITIONAL_835",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=sealed,
            static_max_parity_error=max_err,
            protocol_sha256=r.sha(output / "protocol.json"),
            completed_ns=time.monotonic_ns(),
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
