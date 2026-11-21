"""Compare feedback policies on the exact same revealed labels and shots."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    RAW_ROOT,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)


def compare(local, local_audit, other, other_audit, output):
    need(not output.exists(), "output exists")
    h = Hashes()
    variants = {
        "uniform_global_layer0": "uniform",
        "uniform_global_matched_kernel": "uniform_matched",
        "disruptive_feedback_only": "disruptive_only",
        "phase_all_early": "phase_all",
        "phase_matched_early": "phase_matched",
    }
    baseline = read(local / "protocol.json").get("memory_variant", "local_all_feedback")
    variant = read(other / "protocol.json")["memory_variant"]
    allowed = {
        "local_all_feedback": {
            "uniform_global_layer0",
            "uniform_global_matched_kernel",
            "disruptive_feedback_only",
        },
        "disruptive_feedback_only": {"phase_all_early", "phase_matched_early"},
    }
    need(
        baseline in allowed and variant in allowed[baseline],
        "unmatched base and alternate policy",
    )
    base_tag = "local" if baseline == "local_all_feedback" else "disruptive_only"
    tag = variants[variant]
    roots = {base_tag: local, tag: other}
    audits = {k: read(p) for k, p in ((base_tag, local_audit), (tag, other_audit))}
    cfg = {k: read(p / "protocol.json") for k, p in roots.items()}
    done = {k: read(p / "completed.json") for k, p in roots.items()}
    for k, folder in roots.items():
        need(
            audits[k]["status"] == "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT"
            and audits[k]["producer_sha256"] == h(folder / "completed.json")
            and audits[k]["n_curves"] == 1440
            and done[k]["protocol_sha256"] == h(folder / "protocol.json"),
            "independently verified producers required",
        )
    for name in (
        "order",
        "arms",
        "threshold",
        "raw_sha256",
        "source_state_sha256",
        "block_size",
        "cold_budget",
        "online_budget",
        "fast_fit_sha256",
    ):
        need(cfg[base_tag][name] == cfg[tag][name], f"different protocol: {name}")
    need(
        cfg[base_tag].get("memory_variant", "local_all_feedback") == baseline
        and cfg[tag]["memory_variant"] == variant,
        "wrong matched ablation variants",
    )
    order, tau = cfg[base_tag]["order"], cfg[base_tag]["threshold"]
    queried = audits[base_tag]["online_queries"]
    need(
        queried == audits[tag]["online_queries"] and len(queried) == 16,
        "different online label manifests",
    )
    blocks, steps = [], {k: 0 for k in roots}
    for i in range(1, 19):
        commits, updates = [], {}
        for k, folder in roots.items():
            paths = [
                folder / "blocks" / f"{i:02d}_{kind}.json"
                for kind in ("commit", "update")
            ]
            bound = done[k]["blocks"][i - 1]
            need(
                h(paths[0]) == bound["commit_sha256"]
                and h(paths[1]) == bound["update_sha256"],
                "variant block evidence changed",
            )
            commits.append(read(paths[0]))
            u = read(paths[1])
            if "fit" in u:
                steps[k] += u["fit"]["steps"]
            updates[k] = u["decision"]
        need(
            all(
                commits[0][k] == commits[1][k]
                for k in ("selected", "rank_end", "ranking")
            ),
            "different label release rank or selection priority",
        )
        blocks.append(
            dict(
                block=i,
                rank_end=commits[0]["rank_end"],
                selected=commits[0]["selected"],
                decisions=updates,
            )
        )
    rows = {k: {a: [] for a in ("static", "intra", "inter", "both")} for k in roots}
    for j, sid in enumerate(order):
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(
            h(raw) == cfg[base_tag]["raw_sha256"][str(sid)],
            "raw changed before comparison",
        )
        truth, times = raw_truth(raw, sid)
        values = {}
        for k, folder in roots.items():
            p = folder / "shots" / f"{j + 1:04d}_{sid}.npz"
            need(h(p) == done[k]["sealed"][j]["curve_sha256"], "changed variant curve")
            with np.load(p, allow_pickle=False) as f:
                need(np.array_equal(f["times"], times), "changed signal clock")
                values[k] = {arm: f[arm] for arm in rows[k]}
            for a, v in values[k].items():
                rows[k][a].append(score(times, v, truth, tau))
        need(
            all(
                np.array_equal(values[base_tag][a], values[tag][a])
                for a in ("static", "intra")
            ),
            "static/intra parity disagreement",
        )
    sections = {}
    for name, subset in (
        ("full360", range(360)),
        ("unqueried_after20", [i for i in range(20, 360) if order[i] not in queried]),
        ("last180", range(180, 360)),
    ):
        r = {
            k: {a: [v[i] for i in subset] for a, v in all_rows.items()}
            for k, all_rows in rows.items()
        }
        value = {k: {a: counts(v) for a, v in m.items()} for k, m in r.items()}
        need(
            all(value[k] == audits[k]["sections"][name]["metrics"] for k in roots),
            "variant counts do not match independent recount",
        )
        sections[name] = {
            "metrics": value,
            f"{base_tag}_vs_{tag}": {
                a: dict(
                    strict=pair(r[base_tag][a], r[tag][a]),
                    capped=pair(r[base_tag][a], r[tag][a], True),
                )
                for a in ("inter", "both")
            },
        }
    h.check()
    report = dict(
        status="PASS_MATCHED_QUERY_AND_RELEASE_MEMORY_ABLATION",
        labels=queried,
        stages=blocks,
        optimizer_steps=steps,
        variants={base_tag: baseline, tag: variant},
        sections=sections,
        verifier_sha256=h(__file__),
        input_audits={
            k: h(p) for k, p in ((base_tag, local_audit), (tag, other_audit))
        },
        scope="Exact label budget and reveal ranks; candidate error/guard decisions and realized optimizer steps can differ.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(report["status"], steps, sections["full360"]["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("local", "local_audit", "other", "other_audit", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    compare(a.local, a.local_audit, a.other, a.other_audit, a.output)
