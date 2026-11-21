"""Outcome-free causal feedback probe: measured past loss vs future diagnostic loss."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from audit_layer0_arrival_motion_fit_20261106 import errors, METRICS
from audit_layer0_action_utility_20261106 import dump
from fit_layer0_arrival_motion_20261106 import FIRST, STRIDE, load_head
import layer0_arrival_motion_20261106 as motion
from layer0_arrival_update_20261106 import ARMS, actions_at
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for
from verify_confirmation_raw_replay_20261105 import Hashes, need, read


PLAN = r.ROOT / "LAYER0_ARRIVAL_FEEDBACK_DIAGNOSTIC_20261106.md"
FIT = r.ROOT / "probe_outputs/layer0_arrival_motion_fit_20261106_v1"
FIT_AUDIT = r.ROOT / "probe_outputs/layer0_arrival_motion_fit_audit_20261106_v1.json"
FORECASTS = ("hold", "linear", "local", "zero", *ARMS)
RECEIPTS = (
    "support",
    "accepted",
    "accepted_shape",
    "radius",
    "norms",
    "losses",
    "logits",
)


def do_actions(base, head, rule, data, t):
    ctx = r.latent.native_context(base, data)
    raw_action = r.actions(rule, ctx)[:, : motion.WIDTH]
    device = base.center.device
    v, a = [torch.as_tensor(data[k], device=device) for k in ("values", "available")]
    j = torch.as_tensor(t, device=device, dtype=torch.long)
    radius = raw_action[(j - FIRST) // STRIDE].norm(dim=1)
    u, note = actions_at(base, head, ctx, v, a, j, radius)
    return ctx, j, u, note


@torch.no_grad()
def forecast_at_anchor(base, head, ctx, j, u):
    all_p, all_m = [], []
    for off in range(0, len(j), 128):
        sl = slice(off, off + 128)
        x, a = ctx[0][j[sl]].contiguous(), ctx[1][j[sl]].contiguous()
        h, line, loc, obs = motion.origin_terms(x, a, base.center, base.scale)
        z, _ = motion.features(base, x, a)
        p = [h, line, head.local(h, loc), head(h, loc, z)]
        for k in ARMS:
            z, _ = motion.features(base, x, a, u[k][sl])
            p.append(head(h, loc, z))
        all_p.append(torch.stack(p, dim=1).cpu().numpy())
        all_m.append(obs.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_m)


def produce(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, _, base, source, mapping, mpath, rulepath = parents()
    h = Hashes()
    fit, check = read(FIT / "completed.json"), read(FIT_AUDIT)
    if (
        check["status"] != "PASS_ARRIVAL_MOTION_FIDELITY_RECOUNT"
        or check["fit_completed_sha256"] != h(FIT / "completed.json")
        or check["fit_head_sha256"] != h(FIT / "head.npz")
        or h(FIT / "head.npz") != fit["artifact_sha256"]["head.npz"]
    ):
        raise ValueError("verified frozen auxiliary forecast required")
    head = load_head(FIT / "head.npz", base.center.device)
    rule = r.load_rule(rulepath, base.center.device)
    dependencies = [
        Path(__file__),
        PLAN,
        Path(motion.__file__),
        Path(actions_at.__code__.co_filename),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        mpath,
        rulepath,
        FIT / "head.npz",
        FIT_AUDIT,
        r.DEFAULT_PARENT,
        r.ROOT / "layer0_arrival_step_20261106.py",
    ]
    bound = {str(p.resolve()): h(p) for p in dependencies}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            order=order[:20],
            first=FIRST,
            stride=STRIDE,
            horizon=motion.H,
            forecasts=FORECASTS,
            receipts=RECEIPTS,
            source_state_sha256=source,
            map_path=str(mpath),
            input_sha256=bound,
            eligibility_gate=check["enabled"],
            labels_read=0,
            alarm_scoring=False,
        ),
    )
    records, prefix = [], []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no outcome in held diagnostic probe"),
    ):
        for rank, sid in enumerate(order[:20], 1):
            raw = raw_for(sid, cfg["raw_sha256"])
            data = r.prepare_actual(raw, 21, mapping)
            n = len(raw["times"])
            t = np.arange(FIRST, n - motion.H, STRIDE, dtype=np.int64)
            if not len(t):
                raise ValueError(
                    "cold probe requires some complete held windows per shot"
                )
            ctx, j, u, note = do_actions(base, head, rule, data, t)
            predictions, observed = forecast_at_anchor(base, head, ctx, j, u)
            if rank in (1, 10, 20):
                i = len(t) // 2
                end = int(t[i]) + 1
                short = {k: data[k][:end].copy() for k in ("values", "available")}
                short["shot"] = sid
                _, _, q, stat = do_actions(base, head, rule, short, t[: i + 1])
                for k in ("support", "accepted", "accepted_shape"):
                    if not torch.equal(stat[k], note[k][: i + 1]):
                        raise ValueError(
                            f"truncation changed selected write: {sid}/{k}"
                        )
                max_delta = max(float((q[k] - u[k][: i + 1]).abs().max()) for k in ARMS)
                if max_delta > 2e-4:
                    raise ValueError(
                        f"truncation changed action values: {sid}/{max_delta:g}"
                    )
                prefix.append(
                    dict(
                        shot=sid,
                        cut=end,
                        n_anchors=i + 1,
                        max_action_difference=max_delta,
                    )
                )
            path = output / "shots" / f"{rank:02d}_{sid}.npz"
            r.save_arrays(
                path,
                anchors=t,
                predictions=predictions,
                origin_valid=observed,
                **{k: v.cpu().numpy() for k, v in note.items()},
            )
            records.append(
                dict(
                    shot=sid,
                    path=str(path.relative_to(output)),
                    sha256=h(path),
                    raw_sha256=cfg["raw_sha256"][str(sid)],
                    n_points=n,
                    n_forecasts=len(t),
                )
            )
            if rank % 5 == 0:
                print(f"sealed cold feedback {rank}/20", flush=True)
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("parent mutated during probe")
    h.check()
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_OUTCOME_FREE_ARRIVAL_FEEDBACK",
            sealed=records,
            prefix_witness=prefix,
            protocol_sha256=h(output / "protocol.json"),
            seconds=time.monotonic() - start,
        ),
    )


def audit(run, output):
    h = Hashes()
    cfg, done = [read(run / p) for p in ("protocol.json", "completed.json")]
    need(
        done["status"] == "COMPLETE_OUTCOME_FREE_ARRIVAL_FEEDBACK"
        and cfg["labels_read"] == 0
        and cfg["alarm_scoring"] is False
        and tuple(cfg["forecasts"]) == FORECASTS
        and tuple(cfg["receipts"]) == RECEIPTS
        and len(done["sealed"]) == len(cfg["order"]) == 20
        and len(done["prefix_witness"]) == 3
        and done["protocol_sha256"] == h(run / "protocol.json")
        and all(h(p) == sha for p, sha in cfg["input_sha256"].items()),
        "sealed diagnostic replay required",
    )
    for note in done["sealed"]:
        need(h(run / note["path"]) == note["sha256"], "held forecasts changed")
    mapping = read(cfg["map_path"])
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    center, scale = base.center.numpy(), base.scale.numpy()
    groups = {"all": [], "real_active": [], "shape_active": []}
    support_totals, caps, past = {k: 0 for k in RECEIPTS[:3]}, 0.0, []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no HDF outcome in diagnostic scores"),
    ):
        for sid, note in zip(cfg["order"], done["sealed"], strict=True):
            need(
                note["shot"] == sid
                and note["raw_sha256"] == h(r.io23.path_for("EAST", sid)),
                "raw diagnostic changed",
            )
            raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
            sig = r.prepare_actual(raw, 21, mapping)
            with np.load(run / note["path"], allow_pickle=False) as a:
                t, obs = a["anchors"], a["origin_valid"]
                n = len(raw["times"])
                need(
                    n == note["n_points"]
                    and len(t) == note["n_forecasts"]
                    and np.array_equal(t, np.arange(FIRST, n - motion.H, STRIDE))
                    and np.array_equal(obs, sig["available"][t]),
                    "held clock or origin mask changed",
                )
                norm, radius, loss, z = (
                    a[k] for k in ("norms", "radius", "losses", "logits")
                )
                ok, shape = a["accepted"], a["accepted_shape"]
                need(
                    np.isfinite(loss).all()
                    and np.isfinite(z).all()
                    and np.all(norm <= radius[:, None] + 1e-6)
                    and np.allclose(norm[:, 2:], norm[:, :1], atol=1e-6, rtol=0)
                    and np.all((norm[:, 0] > 0) == ok)
                    and np.all((norm[:, 1] > 0) == shape)
                    and np.all(loss[ok, 1] < loss[ok, 0])
                    and np.all(loss[shape, 2] < loss[shape, 0])
                    and np.all(np.abs(z[:, 2] - z[:, 0]) <= 1.001e-4),
                    "matched write, loss descent or shape constraint failed",
                )
                caps = max(caps, float(np.abs(z[:, 2] - z[:, 0]).max()))
                for k in support_totals:
                    support_totals[k] += int(a[k].sum())
                past.append(
                    dict(shot=sid, all_anchors=len(t), mean=loss.mean(0).tolist())
                )
                j = t[:, None] + np.arange(1, motion.H + 1)
                mask = sig["available"][j] & obs[:, None]
                mask[..., [2, 10]] = False
                target = np.arcsinh(
                    (np.where(mask, sig["values"][j], center) - center) / scale
                )
                result, valid = errors(a["predictions"], target, mask)
                for k, s in (
                    ("all", valid),
                    ("real_active", valid & ok),
                    ("shape_active", valid & shape),
                ):
                    if s.any():
                        groups[k].append(
                            dict(
                                shot=sid,
                                n=int(s.sum()),
                                mean=result[s].mean(0).tolist(),
                            )
                        )
    out = {}
    for k, rows in groups.items():
        need(rows, f"no held diagnostic observations for {k}")
        x = np.array([r["mean"] for r in rows])
        out[k] = dict(
            n_shots=len(rows),
            n_anchors=sum(x["n"] for x in rows),
            per_shot=rows,
            mean={
                name: dict(zip(METRICS, v.tolist(), strict=True))
                for name, v in zip(FORECASTS, x.mean(0), strict=True)
            },
            shape_win_vs_zero={
                a: int(
                    (
                        x[:, FORECASTS.index(a), 1] < x[:, FORECASTS.index("zero"), 1]
                    ).sum()
                )
                for a in ARMS
            },
        )
        print(
            k, {j: round(v["shape"], 8) for j, v in out[k]["mean"].items()}, flush=True
        )
    h.check()
    dump(
        output,
        dict(
            status="PASS_OUTCOME_FREE_ARRIVAL_FEEDBACK_RECOUNT",
            parent_sha256=h(run / "completed.json"),
            groups=out,
            receipt_counts=support_totals,
            past_losses=past,
            max_shape_anchor_logit_shift=caps,
            labels_read=0,
            full_alarm_stream_run=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("produce").add_argument("output", type=Path)
    a = sub.add_parser("audit")
    a.add_argument("run", type=Path)
    a.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode == "produce":
        produce(a.output)
    else:
        audit(a.run, a.output)
