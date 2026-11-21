"""Reconstruct the decision prefixes of every conditional paired alarm flip."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from layer0_signed_support_20261106 import causal_phase_support, load_bank
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, raw_for
from verify_confirmation_raw_replay_20261105 import raw_truth, score


PARENT = r.ROOT / "probe_outputs/layer0_conditional835_20261106_v1"
CHECK = r.ROOT / "probe_outputs/layer0_conditional835_verification_20261106_v1.json"
KINDS = ("tp_gained", "tp_lost", "fp_added", "fp_removed")


def caliper_pairs(a, b):
    if not a or not b:
        return []

    def features(rows):
        return np.asarray([[v["native_at_alarm"], v["elapsed_s"]] for v in rows])

    d = np.abs(features(a)[:, None, :] - features(b)[None, :, :]) / [0.05, 0.5]
    valid = (d <= 1 + 1e-12).all(axis=-1)
    real = np.where(valid, np.square(d).sum(axis=-1), 1e6)
    cost = np.concatenate((real, np.full((len(a), len(a)), 10000.0)), axis=1)
    ai, bi = linear_sum_assignment(cost)
    return [
        (int(i), int(j))
        for i, j in zip(ai, bi, strict=True)
        if j < len(b) and valid[i, j]
    ]


def cosine(a, b):
    n = a.norm() * b.norm()
    return float((a @ b) / n) if n > 0 else None


def summarize(rows):
    fields = (
        "phase_margin",
        "precursor_rms",
        "same_shot_early_rms",
        "cos_lag1",
        "cos_lag4",
        "mask_delta",
        "anchor_logit_delta",
        "elapsed_s",
        "native_at_alarm",
    )
    summary = {}
    for name in fields:
        values = [v[name] for v in rows if v[name] is not None]
        summary[name] = dict(
            n=len(values), median=float(np.median(values)) if values else None
        )
    return dict(
        n=len(rows),
        relative_phase_supported=sum(v["phase_allow"] for v in rows),
        exposed_positive=sum(v["would_restrict_at_anchor"] for v in rows),
        features=summary,
    )


def run(bank_dir, output):
    if output.exists():
        raise FileExistsError(output)
    cfg, done, verified = [
        r.read(p) for p in (PARENT / "protocol.json", PARENT / "completed.json", CHECK)
    ]
    if (
        verified["status"] != "PASS_LAYER0_CONDITIONAL_835_HDF5_RECOUNT"
        or verified["producer_sha256"] != r.sha(PARENT / "completed.json")
        or done["protocol_sha256"] != r.sha(PARENT / "protocol.json")
    ):
        raise ValueError("verified conditional comparison required")
    r.configure()
    _, _, _, base, digest, mapping, _, fit_path = parents()
    if digest != cfg["source_state_sha256"]:
        raise ValueError("different frozen source")
    rule = r.load_rule(fit_path, base.center.device)
    bank, bank_info = load_bank(bank_dir, base.center.device)
    ids = [v["shot"] for v in bank_info["historical_disruptive_rows"] if "slot" in v]
    order, tau = cfg["order"], cfg["threshold"]
    raw_digests = cfg["raw_sha256"]
    observed = {k: [] for k in KINDS}
    sets = verified["paired_intra"]["static"]["strict"]
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("outcome unavailable to prefix feature code"),
    ):
        for kind in KINDS:
            for sid in sorted(sets[kind], key=order.index):
                rank = order.index(sid) + 1
                bound = done["sealed"][rank - 1]
                shot = PARENT / "shots" / f"{rank:04d}_{sid}.npz"
                if r.sha(shot) != bound["curve_sha256"]:
                    raise ValueError(f"sealed curve changed: {sid}")
                raw = raw_for(sid, raw_digests)
                # This audit runs only after the full parent was sealed. Its
                # outcome is used for offline first-alarm scoring, not features.
                truth, times = raw_truth(r.io23.path_for("EAST", sid), sid)
                with np.load(shot, allow_pickle=False) as a:
                    if not np.array_equal(a["times"], times):
                        raise ValueError("shot clock mismatch")
                    ps, pa = a["static"], a["intra"]
                before, after = (score(times, p, truth, tau) for p in (ps, pa))
                transition = {
                    "tp_gained": before["status"] != "tp" and after["status"] == "tp",
                    "tp_lost": before["status"] == "tp" and after["status"] != "tp",
                    "fp_added": before["status"] == "tn" and after["status"] == "fp",
                    "fp_removed": before["status"] == "fp" and after["status"] == "tn",
                }
                if not transition[kind]:
                    raise ValueError("paired flip did not independently reproduce")
                alarm = after if kind in ("tp_gained", "fp_added") else before
                at = alarm["first_index"]
                if at < r.policy.FIRST:
                    raise ValueError("unchangeable pre-update point cannot be a flip")
                for k in ("times", "values", "available"):
                    raw[k] = raw[k][: at + 1].copy()
                ctx, _, keys = as_case(base, raw, 21, mapping, rule)
                phase = causal_phase_support(keys, bank)
                direction = r.actions(rule, ctx)
                x, a = ctx[:2]
                row = len(keys) - 1
                anchor = r.policy.FIRST + r.policy.STRIDE * row
                u = direction[row : row + 1].clone()
                u[:, 128:] = 0
                with torch.no_grad():

                    def z_at(index, delta):
                        v, mask = x[index : index + 1], a[index : index + 1]
                        return r.policy.routed_pair(
                            base, v, mask, r.latent.routes_all(mask), delta
                        )[0]

                    pa_live = float(z_at(at, u).sigmoid())
                    effect = float(z_at(anchor, u) - z_at(anchor, torch.zeros_like(u)))
                layer.require_curve_parity(np.asarray([pa_live]), pa[at : at + 1], tau)
                layer.require_curve_parity(ctx[2], ps[: at + 1], tau)
                old = max(0, anchor - r.policy.STRIDE)
                availability = (
                    a[anchor, -32:].float().mean() - a[old, -32:].float().mean()
                )
                observed[kind].append(
                    dict(
                        shot=sid,
                        rank=rank,
                        at=at,
                        anchor=anchor,
                        actual_lead_ms=alarm["lead_ms"],
                        elapsed_s=float(times[at] - times[0]),
                        native_at_alarm=float(ps[at]),
                        adapted_at_alarm=float(pa[at]),
                        anchor_logit_delta=effect,
                        phase_allow=bool(phase["allow"][-1]),
                        would_restrict_at_anchor=bool(
                            not phase["allow"][-1] and effect > 1e-5
                        ),
                        nearest_cold_shot=ids[int(phase["nearest_bank"][-1])],
                        phase_margin=float(phase["margin"][-1]),
                        precursor_rms=float(phase["precursor_rms"][-1]),
                        same_shot_early_rms=float(phase["early_rms"][-1]),
                        cos_lag1=cosine(u[0, :128], direction[row - 1, :128])
                        if row
                        else None,
                        cos_lag4=cosine(u[0, :128], direction[row - 4, :128])
                        if row >= 4
                        else None,
                        mask_delta=float(availability),
                    )
                )
    a, b = observed["tp_gained"], observed["fp_added"]
    matches = caliper_pairs(a, b)
    r.write_new(
        output,
        dict(
            status="COMPLETE_SIGNED_FLIP_PREFIX_AUDIT",
            cohort={k: summarize(v) for k, v in observed.items()},
            matched_rescues=summarize([a[i] for i, j in matches]),
            matched_false_alarms=summarize([b[j] for i, j in matches]),
            pairs=[[a[i]["shot"], b[j]["shot"]] for i, j in matches],
            rows=observed,
            bank_sha256=r.sha(bank_dir / "completed.json"),
            parent_audit_sha256=r.sha(CHECK),
            scope="Retrospective selected flips, no case-label fit and no standalone validity claim.",
        ),
    )
    print(
        "verified prefixes",
        {k: len(v) for k, v in observed.items()},
        "matched",
        len(matches),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bank", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.bank, args.output)
