"""Check whether new direction candidates yield learnable historical first-alarm credit."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from layer0_action_utility_20261106 import block_credit
from run_layer0_direction_span_20261106 import (
    ARMS,
    FAMILIES,
    load_basis,
    layer,
    parents,
    predict,
    r,
    raw_for,
)
from verify_confirmation_raw_replay_20261105 import Hashes, need, raw_truth, read
from verify_layer0_signed_projection_20261106 import FIRST, STRIDE, TAU, shot_path
from verify_moving_peak_replay_20261105 import parity


PLAN = r.ROOT / "LAYER0_DIRECTION_HISTORY_PLAN_20261106.md"
HISTORY = r.ROOT / "probe_outputs/layer0_action_credit_history_20261106_v2"
HISTORY_AUDIT = (
    r.ROOT / "probe_outputs/layer0_action_credit_history_audit_20261106_v1.json"
)


def support_counts(rows):
    keys = ("benefit", "harm", "different_sign_value", "benefit_and_harm_same_anchor")
    out = {k: dict(shots=[], anchors=0) for k in keys}
    seen = set()
    for sid, pos, neg in rows:
        if (
            sid in seen
            or pos.ndim != 1
            or pos.shape != neg.shape
            or not np.isin(pos, [-1, 0, 1]).all()
            or not np.isin(neg, [-1, 0, 1]).all()
        ):
            raise ValueError(
                "one row of matched finite signed credits per shot required"
            )
        seen.add(sid)
        benefit, harm = (pos > 0) | (neg > 0), (pos < 0) | (neg < 0)
        for k, mask in zip(
            keys, (benefit, harm, pos != neg, benefit & harm), strict=True
        ):
            count = int(mask.sum())
            if count:
                out[k]["shots"].append(sid)
                out[k]["anchors"] += count
    return out


def produce(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, _, base, source, _, _, _ = parents()
    h = Hashes()
    done = read(HISTORY / "completed.json")  # Does not contain any label.
    completion = h(HISTORY / "completed.json")
    if (
        done["status"] != "COMPLETE_ENDED_ACTION_HISTORY"
        or done["n_shots"] != 36
        or len(done["sealed"]) != 36
        or [x["shot"] for x in done["sealed"][:20]] != order[:20]
        or any(x["group"] != "cold_loo" for x in done["sealed"][:20])
    ):
        raise ValueError("same sealed historical cold leave-one-shot folds required")
    bind = {
        str(p.resolve()): h(p)
        for p in (Path(__file__), PLAN, Path(predict.__code__.co_filename))
    }
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    rows, max_error = [], {"static": 0.0, "intra": 0.0}
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no HDF outcome in historical predictions"),
    ):
        for rank, old in enumerate(done["sealed"][:20], 1):
            sid = old["shot"]
            stem = f"{rank:04d}_{sid}"
            fold_dir = HISTORY / "folds"
            mpath, rulepath, fitpath = (
                fold_dir / f"{stem}_{s}" for s in ("map.json", "rule.npz", "fit.json")
            )
            source_curve = HISTORY / old["file"]
            if (
                rank != old["rank"]
                or h(source_curve) != old["curve_sha256"]
                or any(
                    h(x) != old["dependencies"][str(x.relative_to(HISTORY))]
                    for x in (mpath, rulepath, fitpath)
                )
            ):
                raise ValueError("cold fold artifacts differ from original seal")
            mapping, fit = read(mpath), read(fitpath)
            roles = [x for x in order[:20] if x != sid]
            if (
                mapping["calibration_ids"] != roles
                or fit["training_ids"] != roles
                or fit["held"] != sid
            ):
                raise ValueError("cold fold must exclude this shot from all fitting")
            rule = r.load_rule(rulepath, base.center.device)
            basis, _ = load_basis(rulepath, base.center.device)
            raw = raw_for(sid, cfg["raw_sha256"])
            curves, receipt = predict(base, raw, 21, True, mapping, rule, basis)
            with np.load(source_curve, allow_pickle=False) as a:
                if not np.array_equal(raw["times"], a["times"]):
                    raise ValueError("cold fold clocks changed")
                for k, old_name in (("static", "static"), ("intra", "adapted")):
                    max_error[k] = max(
                        max_error[k],
                        layer.require_curve_parity(curves[k], a[old_name], TAU),
                    )
            path = output / "shots" / (stem + ".npz")
            r.save_arrays(path, times=raw["times"], **curves, **receipt)
            rows.append(
                dict(
                    rank=rank,
                    shot=sid,
                    file=str(path.relative_to(output)),
                    curve_sha256=h(path),
                    raw_sha256=cfg["raw_sha256"][str(sid)],
                )
            )
            if rank % 5 == 0:
                print(f"cold direction history {rank}/20", flush=True)
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("history direction replay modified source")
    h.check()
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_COLD_LOO_DIRECTIONS",
            n_shots=len(rows),
            n_curves=len(rows) * len(ARMS),
            sealed=rows,
            history_completion_sha256=completion,
            input_sha256=bind,
            source_state_sha256=source,
            max_history_parity=max_error,
            threshold=TAU,
            seconds=time.monotonic() - start,
        ),
    )


def audit(cold_dir, fixed, output):
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    hist, hist_cfg = [read(HISTORY / k) for k in ("completed.json", "protocol.json")]
    check, cold, span, span_cfg = (
        read(x)
        for x in (
            HISTORY_AUDIT,
            cold_dir / "completed.json",
            fixed / "completed.json",
            fixed / "protocol.json",
        )
    )
    hist_sha = h(HISTORY / "completed.json")
    need(
        hist["status"] == "COMPLETE_ENDED_ACTION_HISTORY"
        and hist["protocol_sha256"] == h(HISTORY / "protocol.json")
        and check["status"] == "PASS_HISTORY_ACTION_CREDIT_HDF5"
        and check["parent_sha256"] == hist_sha
        and cold["status"] == "COMPLETE_COLD_LOO_DIRECTIONS"
        and cold["n_shots"] == len(cold["sealed"]) == 20
        and cold["history_completion_sha256"] == hist_sha
        and all(h(p) == sha for p, sha in cold["input_sha256"].items())
        and span["status"] == "COMPLETE_LAYER0_DIRECTION_SPAN"
        and span["mode"] == "fixed"
        and span["n_shots"] == 360
        and span["protocol_sha256"] == h(fixed / "protocol.json")
        and span_cfg["threshold"] == hist_cfg["threshold"] == cold["threshold"] == TAU
        and span_cfg["source_state_sha256"]
        == hist_cfg["source_state_sha256"]
        == cold["source_state_sha256"],
        "same sealed 20+16 histories and fixed span replay required",
    )
    labels, cases = hist_cfg["rows"], hist["sealed"]
    need(
        len(labels) == len(cases) == 36 and len({x["shot"] for x in cases}) == 36,
        "wrong label budget",
    )
    groups = {
        g: {f: {u: [] for u in ("isolated", "telescoping")} for f in FAMILIES}
        for g in ("cold_loo", "queried")
    }
    for i, (old, label) in enumerate(zip(cases, labels, strict=True)):
        rank, sid, group = (old[k] for k in ("rank", "shot", "group"))
        need(
            rank == label["rank"] and sid == label["shot"] and group == label["group"],
            "changed historical role",
        )
        stem = f"{rank:04d}_{sid}"
        prior = HISTORY / old["file"]
        need(h(prior) == old["curve_sha256"], "historical reference changed")
        if i < 20:
            note = cold["sealed"][i]
            f = cold_dir / note["file"]
            need(group == "cold_loo", "history order changed")
        else:
            note = span["sealed"][rank - 1]
            f = fixed / "shots" / (stem + ".npz")
            need(
                group == "queried" and note["active"],
                "query direction prediction missing",
            )
        path = shot_path(sid)
        need(
            (note["rank"], note["shot"]) == (rank, sid)
            and h(f) == note["curve_sha256"]
            and h(path) == note["raw_sha256"] == hist_cfg["raw_sha256"][str(sid)],
            "history or raw hash changed",
        )
        truth, clock = raw_truth(path, sid)
        need(
            all(truth[k] == label[k] for k in ("disruptive", "down")),
            "HDF truth differs from original release",
        )
        with (
            np.load(f, allow_pickle=False) as a,
            np.load(prior, allow_pickle=False) as b,
        ):
            need(
                np.array_equal(a["times"], clock) and np.array_equal(b["times"], clock),
                "clock mismatch",
            )
            parity(a["static"], b["static"], TAU, f"{sid}/static")
            parity(a["intra"], b["adapted"], TAU, f"{sid}/intra")
            for k in FAMILIES:
                pos, neg = (
                    block_credit(
                        clock,
                        a["static"],
                        a[f"{k}_{sign}"],
                        disruptive=truth["disruptive"],
                        down=truth["down"],
                        threshold=TAU,
                        first=FIRST,
                        stride=STRIDE,
                    )
                    for sign in ("pos", "neg")
                )
                for u in ("isolated", "telescoping"):
                    groups[group][k][u].append(
                        (sid, pos["capped_" + u], neg["capped_" + u])
                    )
    summaries = {
        g: {k: {u: support_counts(v) for u, v in s.items()} for k, s in fam.items()}
        for g, fam in groups.items()
    }
    covered = {
        k: all(
            len(summaries[g][k]["isolated"][pol]["shots"]) >= n
            for g, n in (("cold_loo", 5), ("queried", 3))
            for pol in ("benefit", "harm")
        )
        for k in FAMILIES
    }
    h.check()
    r.write_new(
        output,
        dict(
            status="PASS_DIRECTION_HISTORY_HDF5",
            groups=summaries,
            gates=covered,
            new_fit_labels=0,
            models_fitted=0,
            cold_completion_sha256=h(cold_dir / "completed.json"),
            fixed_completion_sha256=h(fixed / "completed.json"),
        ),
    )
    for group, value in summaries.items():
        print(
            group,
            {
                k: tuple(len(d["isolated"][p]["shots"]) for p in ("benefit", "harm"))
                for k, d in value.items()
            },
            flush=True,
        )
    print("per-family distinct-shot coverage", covered, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    modes.add_parser("produce").add_argument("output", type=Path)
    a = modes.add_parser("audit")
    for k in ("cold", "fixed", "output"):
        a.add_argument(k, type=Path)
    a = p.parse_args()
    if a.mode == "produce":
        produce(a.output)
    else:
        audit(a.cold, a.fixed, a.output)
