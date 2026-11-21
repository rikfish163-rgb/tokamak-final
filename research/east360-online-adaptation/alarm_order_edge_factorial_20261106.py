"""Decompose the frozen first-anchor reversal response into inner/outer swaps."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from fit_layer0_arrival_motion_20261106 import past_arrays
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import raw_for
from trajectory_replay_20261105 import reference
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = Path(__file__).with_name("REAL_ALARM_ORDER_EDGE_FACTORIAL_PLAN_20261106.md")
SITES = (("recent", 11, 14), ("old", 7, 10))


def outer_swap(values, mask, a, b):
    v, m = np.asarray(values), np.asarray(mask)
    if (
        v.shape != (128, 23)
        or m.shape != v.shape
        or m.dtype != bool
        or not 0 <= a < b < 15
    ):
        raise ValueError(
            "only nonterminal patch swaps in aligned causal windows are allowed"
        )
    m = m.reshape(16, 8, 23)
    if not np.array_equal(m[a], m[b]):
        return None
    x = v.copy().reshape(16, 8, 23)
    x[[a, b]] = x[[b, a]]
    return x.reshape(v.shape)


def decomposition(quad):
    if not len(quad):
        return dict(n=0)
    z = np.asarray(quad, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all():
        raise ValueError("four finite paired logit views required")
    full, inner, outer = (z[:, :1] - z[:, 1:]).T
    interaction = full - inner - outer
    q = dict(full=full, inner=inner, outer=outer, interaction=interaction)
    denom = np.abs(full).sum()
    return dict(
        n=len(z),
        mean={k: float(v.mean()) for k, v in q.items()},
        median_abs={k: float(np.median(np.abs(v))) for k, v in q.items()},
        abs_residual_outer_over_full=float(np.abs(full - outer).sum() / denom)
        if denom
        else None,
        abs_residual_inner_over_full=float(np.abs(full - inner).sum() / denom)
        if denom
        else None,
    )


def seal(root, output):
    start, h = time.monotonic(), Hashes()
    if output.exists():
        raise FileExistsError(output)
    done, p = (r.read(root / name) for name in ("completed.json", "protocol.json"))
    order, cfg, _ = reference()
    if (
        done["status"] != "COMPLETE_ALARM_ORDER_BLIND_SEAL"
        or h(root / "protocol.json") != done["protocol_sha256"]
        or p["order"] != order
        or not all(h(f) == v for f, v in p["input_sha256"].items())
        or [s["shot"] for s in done["shots"]] != order
        or p["labels_read"]
    ):
        raise ValueError("sealed outcome-blind 360 anchor study required")
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    if r.state_digest(base) != p["source_state_sha256"]:
        raise ValueError("qualified source mismatch")
    mapping = r.read(r.COLD / "input_map.json")
    rows, max_error = [], 0.0
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("outcome unavailable before factorial seal"),
    ):
        for rank, s in enumerate(done["shots"], start=1):
            sid = s["shot"]
            path = root / "shots" / s["name"]
            if h(path) != s["sha256"] or rank != s["rank"]:
                raise ValueError("first anchor seal changed")
            with np.load(path, allow_pickle=False) as a:
                first = np.flatnonzero(a["first"])
                if not len(first):
                    continue
                j = first[0]
                idx = int(a["index"][j])
                use = a["supported"][j, :2] & a["supported"][j, 2:]
                z = a["logits"][j].astype(np.float64)
            row = dict(
                shot=sid,
                rank=rank,
                index=idx,
                bands=dict.fromkeys(k for k, *_ in SITES),
            )
            rows.append(row)
            if not use.any():
                continue
            raw = raw_for(sid, cfg["raw_sha256"])
            data = r.prepare_actual(raw, rank, mapping)
            x, m = past_arrays(data["values"], data["available"], np.array([idx]))
            block = [x[0]]
            keys = []
            for k, (name, i, h_) in enumerate(SITES):
                if use[k]:
                    a = outer_swap(x[0], m[0], i, h_)
                    if a is None:
                        raise ValueError(
                            "outer swap must be supported whenever full and inner are"
                        )
                    block.append(a)
                    keys.append((name, k))
            with torch.no_grad():
                pred = base(
                    torch.from_numpy(np.stack(block)),
                    torch.from_numpy(np.repeat(m, len(block), axis=0)),
                )["logit"].numpy()
            max_error = max(max_error, abs(float(pred[0]) - z[0]))
            if max_error > 2e-5:
                raise ValueError("CPU native first point differs")
            for y, (name, k) in zip(pred[1:], keys, strict=True):
                row["bands"][name] = [
                    float(z[0]),
                    float(z[1 + k]),
                    float(z[3 + k]),
                    float(y),
                ]
    h.check()
    r.write_new(
        output,
        dict(
            status="SEALED_FIRST_ANCHOR_EDGE_FACTORIAL",
            original_sha256=h(root / "completed.json"),
            plan_sha256=h(PLAN),
            code_sha256=h(Path(__file__)),
            outcome_reads=0,
            seconds=time.monotonic() - start,
            native_max_error=max_error,
            rows=rows,
        ),
    )


def analyze(sealed, reference_score, out):
    a, b = r.read(sealed), r.read(reference_score)
    if (
        a["status"] != "SEALED_FIRST_ANCHOR_EDGE_FACTORIAL"
        or a["outcome_reads"]
        or b["status"] != "RECOUNT_REAL_ALARM_ORDER_ATTRIBUTION"
        or a["original_sha256"] != b["seal_sha256"]
        or a["code_sha256"] != r.sha(Path(__file__))
        or a["plan_sha256"] != r.sha(PLAN)
    ):
        raise ValueError(
            "compatible outcome-blind factorial and post-seal HDF5 receipt required"
        )
    tagged = {
        (q["shot"], q["rank"], q["index"]): q["class_tag"] for q in b["first_anchors"]
    }
    observed = {(q["shot"], q["rank"], q["index"]) for q in a["rows"]}
    if observed != set(tagged) or len(observed) != len(a["rows"]):
        raise ValueError("first anchors do not match")
    groups = {}
    for subset in ("all", "fp", "reasonable_tp", "early_tp", "tardy"):
        q = [
            x
            for x in a["rows"]
            if subset == "all" or tagged[(x["shot"], x["rank"], x["index"])] == subset
        ]
        groups[subset] = {
            k: decomposition([x["bands"][k] for x in q if x["bands"][k] is not None])
            for k, *_ in SITES
        }
    r.write_new(
        out,
        dict(
            status="RECOUNT_FIRST_ANCHOR_EDGE_FACTORIAL",
            seal_sha256=r.sha(sealed),
            hdf_grouping_sha256=r.sha(reference_score),
            groups=groups,
            new_alarm_curves=0,
        ),
    )
    print(groups["all"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    a = s.add_parser("seal")
    a.add_argument("root", type=Path)
    a.add_argument("out", type=Path)
    a = s.add_parser("recount")
    a.add_argument("seal", type=Path)
    a.add_argument("reference", type=Path)
    a.add_argument("out", type=Path)
    q = p.parse_args()
    if q.mode == "seal":
        seal(q.root, q.out)
    else:
        analyze(q.seal, q.reference, q.out)
