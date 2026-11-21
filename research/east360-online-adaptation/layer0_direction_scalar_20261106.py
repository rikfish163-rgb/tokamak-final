"""Compare internal direction writes to a causal, same-anchor scalar control."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit

from audit_layer0_action_utility_20261106 import best_first, first_is_timely
from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    counts,
    need,
    raw_truth,
    read,
    score,
)
from verify_layer0_signed_projection_20261106 import FIRST, ROOT, STRIDE, TAU, shot_path


PLAN = ROOT / "LAYER0_DIRECTION_SCALAR_PLAN_20261106.md"
FAMILIES = ("full", "in_span", "out_span", "random")


def scalar_shadow(native, written, *, first=FIRST, stride=STRIDE):
    if native.ndim != 1 or written.shape != native.shape or first < 0 or stride < 1:
        raise ValueError(
            "equal-length probability traces and positive block stride required"
        )
    anchors = np.arange(first, len(native), stride)
    y = native.astype(np.float64)
    if not len(anchors):
        return y
    a, b = y[first:], written[anchors].astype(np.float64)
    if not all(((0 < p) & (p < 1)).all() for p in (a, b)):
        raise ValueError(
            "cannot identify finite logits from saturated or invalid probabilities"
        )
    offsets = logit(b) - logit(y[anchors])
    y[first:] = expit(logit(a) + offsets[np.arange(len(a)) // stride])
    y[anchors] = b  # Exact anchor equality independent of numeric round-trip.
    return y


def audit(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    cohorts = {}
    for name, n in (("fixed360", 360), ("conditional835", 835)):
        prefix = ROOT / "probe_outputs" / f"layer0_direction_span_{name}"
        path = prefix.with_name(prefix.name + "_20261106_v1")
        report = prefix.with_name(prefix.name + "_verification_20261106_v1.json")
        cfg, done, verify = [
            read(p) for p in (path / "protocol.json", path / "completed.json", report)
        ]
        need(
            verify["status"] == "PASS_LAYER0_DIRECTION_SPAN_HDF5_RECOUNT"
            and verify["hashes"]["producer_sha256"] == h(path / "completed.json")
            and done["protocol_sha256"] == h(path / "protocol.json")
            and len(cfg["order"]) == len(verify["per_shot"]) == n
            and cfg["threshold"] == TAU,
            "sealed, independently scored direction replay required",
        )
        res = {
            f: dict(
                actual={s: [] for s in ("pos", "neg")},
                menus={
                    "strict": {"max_tp": 0, "min_fp": 0},
                    "capped": {"max_tp": 0, "min_fp": 0},
                },
                real_only_capped=[],
                scalar_only_capped=[],
                real_only_no_cross_prior=[],
                scalar_only_no_cross_prior=[],
                scalar_capped_no_cross_prior=0,
            )
            for f in FAMILIES
        }
        for rank, (sid, previous) in enumerate(
            zip(cfg["order"], verify["per_shot"], strict=True), 1
        ):
            file = path / "shots" / f"{rank:04d}_{sid}.npz"
            raw = shot_path(sid)
            note = done["sealed"][rank - 1]
            truth, t = raw_truth(raw, sid)
            need(
                (note["rank"], note["shot"], previous["rank"], previous["shot"])
                == (rank, sid, rank, sid)
                and (truth["disruptive"], truth["down"])
                == (previous["disruptive"], previous["down"])
                and h(raw) == note["raw_sha256"]
                and h(file) == note["curve_sha256"],
                "scalar diagnostic input changed",
            )
            deep = (
                previous["prior_extended_capped_reason"]
                == "no_eligible_candidate_crossing"
            )
            with np.load(file, allow_pickle=False) as a:
                need(np.array_equal(t, a["times"]), "scalar diagnostic clock mismatch")
                native, intra = a["static"], a["intra"]
                for family, out in res.items():
                    pair = []
                    for sign in ("pos", "neg"):
                        p = (
                            scalar_shadow(native, a[f"{family}_{sign}"])
                            if note["active"]
                            else native
                        )
                        out["actual"][sign].append(score(t, p, truth, TAU))
                        pair.append(p)
                    probs = np.stack([native, intra, *pair])
                    for kind, cap in (("strict", False), ("capped", True)):
                        j = best_first(t, probs, truth, capped=cap)
                        if truth["disruptive"]:
                            useful = first_is_timely(t, truth["down"], j, capped=cap)
                            out["menus"][kind]["max_tp"] += useful
                            if cap:
                                real = previous["menus"][family]["capped"]
                                real_useful = (
                                    real["status"] == "tp" and real["lead_ms"] <= 3000
                                )
                                if real_useful and not useful:
                                    out["real_only_capped"].append(sid)
                                    if deep:
                                        out["real_only_no_cross_prior"].append(sid)
                                if useful and not real_useful:
                                    out["scalar_only_capped"].append(sid)
                                    if deep:
                                        out["scalar_only_no_cross_prior"].append(sid)
                                out["scalar_capped_no_cross_prior"] += bool(
                                    deep and useful
                                )
                        else:
                            out["menus"][kind]["min_fp"] += j < len(t)
        for family, out in res.items():
            out["actual"] = {s: counts(v) for s, v in out["actual"].items()}
            out["real_menus"] = verify["menus"][family]
            out["real_capped_no_cross_prior"] = verify[
                "prior_extended_no_eligible_capped"
            ]["menus"][family]["newly_reachable_capped_first_alarms"]
            need(
                out["real_menus"]["capped"]["max_tp"] - out["menus"]["capped"]["max_tp"]
                == len(out["real_only_capped"]) - len(out["scalar_only_capped"]),
                "capped pair count mismatch",
            )
        cohorts[name] = dict(n_shots=n, report_sha256=h(report), scalar=res)
        print(
            name,
            {
                k: (d["real_menus"]["capped"]["max_tp"], d["menus"]["capped"]["max_tp"])
                for k, d in res.items()
            },
            flush=True,
        )
    h.check()
    from audit_layer0_action_utility_20261106 import dump

    dump(
        output,
        dict(
            status="PASS_SAME_ANCHOR_SCALAR_CONTROL",
            seconds=time.monotonic() - start,
            plan_sha256=h(PLAN),
            code_sha256=h(Path(__file__)),
            cohorts=cohorts,
            scope="Offline comparison of causal arm curves; an outcome-informed menu is not a policy.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    audit(p.parse_args().output)
