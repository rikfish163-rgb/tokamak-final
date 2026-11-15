"""Repeat the sealed, frozen sensitivity test on the existing 835-shot roster."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import alarm_order_match_20261106 as paired
import alarm_order_views_20261106 as views
import analyze_alarm_order_attribution_20261106 as audit
from confirmation_raw_replay_20261105 import ROSTER, ROSTER_SHA
import moving_peak_replay_20261105 as r
import run_alarm_order_attribution_20261106 as origin
from run_layer0_query_memory_20261105 import raw_for
from trajectory_replay_20261105 import reference
from verify_confirmation_raw_replay_20261105 import Hashes, score


PREVIOUS = r.ROOT / "probe_outputs/layer0_direction_span_conditional835_20261106_v1"
PLAN = r.ROOT / "REAL_ALARM_ORDER_CONDITIONAL_REPLICATION_PLAN_20261106.md"
N = 835
RANGES = (("full", 1, N, 24), ("first_half", 1, 417, 10), ("second_half", 418, N, 10))


def replication_summary(rows):
    data = paired._check_rows(rows)
    splits = {key: paired._summarize_split(data, lo, hi) for key, lo, hi, _ in RANGES}
    summaries = {key: v["paired_summary"]["contrast"] for key, v in splits.items()}
    enough = {key: splits[key]["n_pairs"] >= size for key, _, _, size in RANGES}
    positive = {
        key: v["mean"] is not None and v["mean"] > 0 for key, v in summaries.items()
    }
    ci = summaries["full"]["ci95_mean"]
    return dict(
        splits=splits,
        gate=dict(
            passed=all(enough.values())
            and all(positive.values())
            and ci is not None
            and ci[0] > 0,
            enough=enough,
            positive=positive,
            lower_positive=ci is not None and ci[0] > 0,
        ),
        interpretation="separate conditional development roster; not independent",
    )


def references(h):
    fixed, src, _ = reference()
    if h(ROSTER) != ROSTER_SHA:
        raise ValueError("frozen conditional roster changed")
    manifest, cfg, done = (
        r.read(p)
        for p in (ROSTER, PREVIOUS / "protocol.json", PREVIOUS / "completed.json")
    )
    order = manifest["ids"]
    mpath = r.COLD / "input_map.json"
    if (
        manifest["status"] != "LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER"
        or manifest["independent_validation"] is not False
        or len(order) != N
        or len(set(order)) != N
        or set(order) & set(fixed)
        or order != cfg["order"]
        or manifest["raw_sha256"] != cfg["raw_sha256"]
        or cfg["source_map_sha256"] != h(mpath)
        or cfg["source_state_sha256"] != src["source_state_sha256"]
        or cfg["threshold"] != r.THRESHOLD
        or cfg["outcome_reads_during_prediction"] != 0
        or cfg["mode"] != "conditional"
        or done["status"] != "COMPLETE_LAYER0_DIRECTION_SPAN"
        or done["mode"] != "conditional"
        or done["n_shots"] != N
        or done["protocol_sha256"] != h(PREVIOUS / "protocol.json")
        or [x["shot"] for x in done["sealed"]] != order
    ):
        raise ValueError("exact conditional development baseline and map required")
    return order, cfg, done, mpath, r.read(mpath)


def baseline(h, rank, shot, done, digest):
    row = done["sealed"][rank - 1]
    p = PREVIOUS / "shots" / f"{rank:04d}_{shot}.npz"
    note = p.with_name(p.stem + "_seal.json")
    if (
        (row["rank"], row["shot"], row["raw_sha256"]) != (rank, shot, digest)
        or h(p) != row["curve_sha256"]
        or h(note) != row["seal_sha256"]
    ):
        raise ValueError("changed conditional reference static curve")
    with np.load(p, allow_pickle=False) as data:
        return data["times"], data["static"]


def produce(out):
    start, h = time.monotonic(), Hashes()
    if out.exists():
        raise FileExistsError(out)
    r.configure()
    order, cfg, prior, mpath, mapping = references(h)
    base, meta = r.load_parent(r.DEFAULT_PARENT)
    digest = r.state_digest(base)
    if digest != cfg["source_state_sha256"] or meta["threshold"] != cfg["threshold"]:
        raise ValueError("frozen qualified source required")
    base.cuda()
    bind = {
        str(p.resolve()): h(p)
        for p in (
            Path(__file__),
            PLAN,
            origin.PLAN,
            Path(origin.__file__),
            Path(audit.__file__),
            Path(views.__file__),
            Path(paired.__file__),
            Path(r.io23.__file__),
            Path(r.simple_data.__file__),
            Path(r.prepare_actual.__code__.co_filename),
            Path(raw_for.__code__.co_filename),
            Path(origin.past_arrays.__code__.co_filename),
            PREVIOUS / "protocol.json",
            PREVIOUS / "completed.json",
            ROSTER,
            mpath,
            r.DEFAULT_PARENT,
        )
    }
    out.mkdir(parents=True)
    (out / "shots").mkdir()
    r.write_new(
        out / "protocol.json",
        dict(
            order=order,
            mapped_from_first=True,
            labels_read=0,
            new_alarm_curves=0,
            source_state_sha256=digest,
            threshold=cfg["threshold"],
            input_sha256=bind,
        ),
    )
    rows, checks = [], []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("conditional outcome unavailable before seal"),
    ):
        for i, sid in enumerate(order, 1):
            raw = raw_for(sid, cfg["raw_sha256"])
            t, p = baseline(h, i, sid, prior, cfg["raw_sha256"][str(sid)])
            if not np.array_equal(t, raw["times"]):
                raise ValueError("conditional raw clock changed")
            ix, first, near = origin.anchors_for(p, cfg["threshold"])
            a = r.prepare_actual(raw, 21, mapping)
            v, m = a["values"], a["available"]
            d = origin.forward_views(base, v, m, ix, first)
            j = d["future_index"] >= 0
            if not all(
                np.allclose(
                    torch.from_numpy(z).sigmoid().numpy(), p[k], atol=2e-6, rtol=0
                )
                for z, k in (
                    (d["logits"][:, 0], ix),
                    (d["future"][j, 0], d["future_index"][j]),
                )
            ):
                raise ValueError(
                    "conditional native trace differs from the frozen reference"
                )
            if i in (2, 417, 835):
                checks.append(
                    dict(
                        rank=i, shot=sid, check=origin.witness(base, v, m, ix, first, d)
                    )
                )
            path = out / "shots" / f"{i:04d}_{sid}.npz"
            r.save_arrays(path, index=ix, times=t[ix], first=first, near=near, **d)
            rows.append(dict(shot=sid, anchors=len(ix), path=path.name, sha256=h(path)))
            if i % 100 == 0 or i == N:
                print(f"sealed conditional order intervention {i}/{N}", flush=True)
    if r.state_digest(base) != digest or base.encoder._forward_pre_hooks:
        raise ValueError("conditional run changed the source")
    h.check()
    r.write_new(
        out / "completed.json",
        dict(
            status="COMPLETE_CONDITIONAL_ALARM_ORDER_BLIND_SEAL",
            shots=rows,
            prefix=checks,
            protocol_sha256=h(out / "protocol.json"),
            seconds=time.monotonic() - start,
        ),
    )


def recount(root, out):
    start, h = time.monotonic(), Hashes()
    if out.exists():
        raise FileExistsError(out)
    order, cfg, prior, _, mapping = references(h)
    prot, done = (r.read(root / name) for name in ("protocol.json", "completed.json"))
    if (
        done["status"] != "COMPLETE_CONDITIONAL_ALARM_ORDER_BLIND_SEAL"
        or h(root / "protocol.json") != done["protocol_sha256"]
        or not all(h(p) == q for p, q in prot["input_sha256"].items())
        or [x["shot"] for x in done["shots"]] != order
        or prot["order"] != order
        or prot["labels_read"]
        or prot["new_alarm_curves"]
        or not prot["mapped_from_first"]
        or prot["threshold"] != cfg["threshold"]
        or prot["source_state_sha256"] != cfg["source_state_sha256"]
    ):
        raise ValueError("invalid conditional experiment seal")
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    center, scale = base.center.numpy(), base.scale.numpy()
    threshold = cfg["threshold"]
    tau = float(np.log(threshold / (1 - threshold)))
    totals = dict(
        anchors=0,
        supported=dict.fromkeys(audit.ARMS, 0),
        future=0,
        placebo=0,
        placebo_max=0.0,
    )
    first, rows, cpu = [], [], []
    for i, (sid, item) in enumerate(zip(order, done["shots"], strict=True), 1):
        raw = raw_for(sid, cfg["raw_sha256"])
        t, p = baseline(h, i, sid, prior, cfg["raw_sha256"][str(sid)])
        a = r.prepare_actual(raw, 21, mapping)
        v, m = a["values"], a["available"]
        path = root / "shots" / item["path"]
        if h(path) != item["sha256"] or not np.array_equal(t, raw["times"]):
            raise ValueError("changed conditional view seal or clock")
        with np.load(path, allow_pickle=False) as d:
            q = audit.check_archive(d, v, m, t, p, threshold, center, scale)
            if i in (2, 417, 835):
                cpu.append(
                    dict(
                        shot=sid,
                        check=audit.cpu_witness(base, d, v, m, q, center, scale),
                    )
                )
            flag, down = r.io23.read_outcome("EAST", sid)
            y = score(t, p, dict(shot=sid, disruptive=flag, down=down), threshold)
            orig = int(d["index"][d["first"]][0]) if d["first"].any() else None
            if orig != y["first_index"]:
                raise ValueError("conditional HDF5 first crossing mismatch")
            z = audit.first_report(d, y, i, sid, tau)
            if z is not None:
                first.append(z)
            y["grid"] = audit.point_report(d, flag, down, tau)
            rows.append(y)
            totals["anchors"] += len(d["index"])
            for k, count in zip(audit.ARMS, d["supported"].sum(0), strict=True):
                totals["supported"][k] += int(count)
            totals["future"] += int((d["future_index"] >= 0).sum())
            p = d["placebo"][np.isfinite(d["placebo"])]
            totals["placebo"] += len(p)
            if len(p):
                totals["placebo_max"] = max(
                    totals["placebo_max"], float(np.abs(p).max())
                )
    groups = audit.summarize_first(first)
    if totals["placebo_max"] > 2e-5 or tuple(
        groups[k]["n"] for k in ("reasonable_tp", "early_tp", "fp")
    ) != (203, 54, 68):
        raise ValueError("conditional baseline counts or placebo cannot reproduce")
    result = dict(
        status="RECOUNT_CONDITIONAL_ALARM_ORDER",
        seal_sha256=h(root / "completed.json"),
        totals=totals,
        first_groups=groups,
        first_anchors=first,
        cpu_witness=cpu,
        matched=replication_summary(first),
        grid=audit.summarize_grid(rows),
        per_shot=rows,
        seconds=time.monotonic() - start,
    )
    h.check()
    r.write_new(out, result)
    print(totals, result["matched"]["gate"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    commands.add_parser("predict").add_argument("out", type=Path)
    a = commands.add_parser("recount")
    a.add_argument("root", type=Path)
    a.add_argument("out", type=Path)
    args = p.parse_args()
    if args.mode == "predict":
        produce(args.out)
    else:
        recount(args.root, args.out)
