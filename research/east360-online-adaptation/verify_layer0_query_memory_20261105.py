"""Check a terminal layer-0 replay from raw HDF5 and sealed artifacts (CPU)."""

import argparse
import json
from pathlib import Path

import numpy as np

from layer0_query_protocol_20261105 import (
    ARMS,
    BLOCK,
    FIRST,
    PERSISTENT_ARMS,
    choose_query,
    guard_violations,
    query_progress,
    teaching_indices,
)
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
from verify_moving_peak_replay_20261105 import check_curve, parity


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "probe_outputs/layer_interaction_masks360_20261105_v1"
REFERENCE_SHA = "c887f88cf6a7183074d781d1655d86a15f8fdcf29d14273fcb4137b41e62cec8"
PHASE_MODES = {"phase_all_early": "all", "phase_matched_early": "matched"}
POSITIVE_ONLY = {"disruptive_feedback_only", *PHASE_MODES}


def expected_early_anchors(clock, down, n_pos, mode):
    """Reconstruct the eligible clock and exact deterministic subsample independently."""
    need(mode in ("all", "matched") and n_pos >= 1, "invalid phase verification mode")
    grid = np.arange(FIRST, len(clock), 32, dtype=np.int64)
    early = grid[np.round((down - clock[grid]) * 1000.0, 9) >= 500]
    if mode == "all":
        return early
    if len(early) < n_pos:
        return early[:0]
    if n_pos == 1:
        return early[:1]
    return early[np.arange(n_pos) * (len(early) - 1) // (n_pos - 1)]


def load_state(path, h, digest, n, *, phase=False):
    need(h(path) == digest, "wrong memory state hash")
    kinds = ("keys", "delta", "early_keys") if phase else ("keys", "delta")
    with np.load(path, allow_pickle=False) as f:
        need(
            f["size"].item() == n
            and set(f.files)
            == {
                "size",
                *(f"{kind}_{i}" for i in range(n) for kind in kinds),
            },
            "wrong memory state schema",
        )
        saved = [tuple(f[f"{kind}_{i}"].copy() for kind in kinds) for i in range(n)]
    for key, d, *bank in saved:
        need(
            key.ndim == 2
            and 0 < key.shape[0] <= 10
            and key.shape[1] == 192
            and d.shape == (128,)
            and np.isfinite(key).all()
            and np.isfinite(d).all()
            and np.linalg.norm(d.astype(np.float64)) <= 0.600001,
            "invalid saved memory",
        )
        if phase:
            [early] = bank
            need(
                early.ndim == 2
                and early.shape[0] > 0
                and early.shape[1] == 192
                and early.dtype == key.dtype
                and np.isfinite(early).all(),
                "invalid saved phase reference",
            )
    return saved


def check_proposal(
    run, block, sid, old, info, truth, times, guards, h, threshold, *, phase_mode=None
):
    base = run / "blocks"
    flag, down = truth[sid]["disruptive"], truth[sid]["down"]
    old_curves = {k: old[k] for k in PERSISTENT_ARMS}
    old_scores = {
        k: score(times[sid], v, truth[sid], threshold) for k, v in old_curves.items()
    }
    need(
        info["query_status"]
        == {
            k: {n: v[n] for n in ("status", "first_index", "lead_ms")}
            for k, v in old_scores.items()
        },
        "wrong active query error type",
    )
    wrong = [v for v in old_scores.values() if v["status"] in ("fp", "miss", "tardy")]
    q = teaching_indices(times[sid], old_curves, flag, down, threshold)
    if not wrong:
        need(
            info["decision"] == "already_correct", "fitted a correctly classified query"
        )
        return None
    if not flag and all(v["first_index"] < FIRST for v in wrong):
        need(
            info["decision"] == "error_before_first_write",
            "fitted an unchangeable first alarm",
        )
        return None
    if not len(q):
        need(info["decision"] == "no_eligible_window", "fitted an ineligible window")
        return None
    n_keys = len(np.unique((q - FIRST) // 32))
    if phase_mode is not None:
        need(flag, "phase proposal requires a released disruptive outcome")
        early = expected_early_anchors(times[sid], down, n_keys, phase_mode)
        need(
            info["phase_mode"] == phase_mode
            and info["early_anchor_indices"] == early.tolist(),
            "reference bank does not match the ended shot's early anchors",
        )
        if not len(early):
            need(
                info["decision"] == "no_eligible_early_reference"
                and not any(
                    k in info for k in ("fit", "proposal_sha256", "teaching_indices")
                ),
                "fitted despite missing early phase support",
            )
            return None
    need(
        q.tolist() == info["teaching_indices"] and info["guard_ids"] == guards,
        "changed teaching or guard IDs",
    )
    proposal = load_state(
        base / f"{block:02d}_proposal.npz",
        h,
        info["proposal_sha256"],
        1,
        phase=phase_mode is not None,
    )[0]
    if phase_mode is not None:
        need(len(proposal[2]) == len(early), "saved early bank has the wrong size")
    fit = info["fit"]
    need(
        fit["steps"] == 20
        and fit["n_keys"] == n_keys == len(proposal[0])
        and abs(fit["norm"] - np.linalg.norm(proposal[1])) <= 1e-6
        and np.isfinite([fit["initial_loss"], fit["final_loss"]]).all(),
        "wrong optimizer/prototype receipt",
    )
    file = base / f"{block:02d}_candidate_curves.npz"
    need(h(file) == info["witness_sha256"], "candidate curve witness hash")
    violations = []
    with np.load(file, allow_pickle=False) as f:
        keys = set()
        for g in [sid, *guards]:
            tn = f"{g}_times"
            keys.add(tn)
            need(np.array_equal(f[tn], times[g]), "candidate guard clock mismatch")
            for k in PERSISTENT_ARMS:
                before, after = (f"{g}_{k}_{kind}" for kind in ("before", "after"))
                keys.update((before, after))
                a, b = f[before], f[after]
                check_curve(a, times[g], before)
                check_curve(b, times[g], after)
                need(np.array_equal(a[:FIRST], b[:FIRST]), "memory changed prefix")
                if g == sid:
                    need(
                        np.array_equal(a, old[k]),
                        "query baseline not its committed trace",
                    )
                for v in guard_violations(
                    times[g], a, b, truth[g]["disruptive"], truth[g]["down"], threshold
                ):
                    violations.append(dict(shot=g, arm=k, reason=v))
        need(keys == set(f.files), "candidate guard reads undisclosed shots")
        helpful, diagnostic = query_progress(
            times[sid],
            old_curves,
            {k: f[f"{sid}_{k}_after"] for k in PERSISTENT_ARMS},
            flag,
            down,
            threshold,
        )
    expected = (
        "accepted"
        if helpful and not violations
        else ("rejected_guard" if violations else "no_query_progress")
    )
    need(
        info["decision"] == expected
        and info["diagnostics"] == diagnostic
        and info["violations"] == violations,
        "incorrect guarded write decision",
    )
    return proposal if expected == "accepted" else None


def metrics(rows, indices):
    selected = {a: [rows[a][j] for j in indices] for a in ARMS}
    values = {a: counts(r) for a, r in selected.items()}
    references = (
        ("intra", "static"),
        ("inter", "static"),
        ("both", "static"),
        ("both", "intra"),
        ("both", "inter"),
    )
    return dict(
        metrics=values,
        strict_pairs={
            f"{a}_vs_{b}": pair(selected[a], selected[b]) for a, b in references
        },
        capped_pairs={
            f"{a}_vs_{b}": pair(selected[a], selected[b], True) for a, b in references
        },
    )


def verify(run, output):
    need(not output.exists(), "output already exists")
    h = Hashes()
    cfg, done = (read(run / k) for k in ("protocol.json", "completed.json"))
    old_cfg, old = (read(REFERENCE / k) for k in ("protocol.json", "completed.json"))
    order, tau = cfg["order"], cfg["threshold"]
    variant = cfg.get("memory_variant", "local_all_feedback")
    need(
        variant
        in (
            "local_all_feedback",
            "uniform_global_layer0",
            "uniform_global_matched_kernel",
            "disruptive_feedback_only",
            *PHASE_MODES,
        ),
        "unknown memory variant",
    )
    phase_mode = PHASE_MODES.get(variant)
    if phase_mode is not None:
        need(
            cfg["phase_mode"] == phase_mode
            and cfg["early_min_lead_ms"] == 500
            and cfg["optimizer_kernel"]
            == cfg["inference_kernel"]
            == "phase_competition",
            "wrong phase kernel contract",
        )
    if variant == "uniform_global_matched_kernel":
        need(
            cfg["optimizer_kernel"] == cfg["inference_kernel"] == "uniform",
            "uniform control must bind training and inference",
        )
    need(
        h(REFERENCE / "completed.json") == cfg["reference_sha256"] == REFERENCE_SHA
        and h(REFERENCE / "protocol.json") == old["artifact_sha256"]["protocol.json"]
        and len(order) == len(set(order)) == 360
        and order == old_cfg["order"]
        and tau == old_cfg["threshold"] == 0.7258946895599365
        and cfg["raw_sha256"] == old_cfg["raw_sha256"]
        and cfg["source_state_sha256"] == old_cfg["source_state_sha256"]
        and cfg["fast_fit_sha256"] == old["fit_sha256"]["moving_peak"]
        and cfg["arms"] == list(ARMS)
        and (
            cfg["first"],
            cfg["stride"],
            cfg["block_size"],
            cfg["cold_budget"],
            cfg["online_budget"],
        )
        == (135, 32, 20, 20, 16)
        and cfg["development_only"] is True
        and cfg["shared_inter_memory"] is True
        and cfg["unqueried_hdf5_outcomes_read_in_producer"] is False,
        "changed study or causal contract",
    )
    need(
        done["status"] == "COMPLETE_LAYER0_QUERY_MEMORY"
        and done["n_shots"] == len(done["sealed"]) == 360
        and done["n_curves"] == 1440
        and done["n_reveals"] == 36
        and len(done["blocks"]) == 18
        and done["protocol_sha256"] == h(run / "protocol.json")
        and all(h(p) == d for p, d in cfg["input_sha256"].items()),
        "incomplete or unbound producer",
    )
    truth, times, rows = {}, {}, {k: [] for k in ARMS}
    block_rows, events, queried, guards = [], [], [], []
    prev_seal = prev_block = state = None
    state_content, first_write = [], None
    last_ns = cfg["initial_ns"]
    errors = {"static": 0.0, "intra": 0.0}
    exposure = {"inter": 0, "both": 0}
    active_anchor_count = 0
    fit_count = 0
    for rank, sid in enumerate(order, 1):
        block = (rank - 1) // BLOCK + 1
        stem = run / "shots" / f"{rank:04d}_{sid}"
        arr_path, seal = (
            stem.with_suffix(".npz"),
            stem.with_name(stem.name + "_seal.json"),
        )
        note = read(seal)
        digest = h(seal)
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(
            done["sealed"][rank - 1] == dict(note, seal_sha256=digest)
            and (note["rank"], note["shot"], note["block"]) == (rank, sid, block)
            and h(raw) == note["raw_sha256"] == cfg["raw_sha256"][str(sid)]
            and h(arr_path) == note["curve_sha256"]
            and note["previous_seal_sha256"] == prev_seal
            and note["previous_block_sha256"] == prev_block
            and note["state_sha256"] == state
            and note["state_size"] == len(state_content)
            and last_ns <= note["predict_start_ns"] <= note["seal_ns"],
            f"unbound prediction or memory used before committed reveal: {sid}",
        )
        prev_seal, last_ns = digest, note["seal_ns"]
        truth[sid], times[sid] = raw_truth(raw, sid)
        if "inter" in note["stats"]:
            active = note["stats"]["inter"]["nonzero"]
            need(
                type(active) is int
                and 0 <= active <= len(range(FIRST, len(times[sid]), 32)),
                "wrong number of written memory anchors",
            )
            active_anchor_count += active
        ref = REFERENCE / "shots" / (stem.name + ".npz")
        need(
            h(ref) == old["sealed"][rank - 1]["curve_sha256"], "archive curve mismatch"
        )
        with (
            np.load(arr_path, allow_pickle=False) as a,
            np.load(ref, allow_pickle=False) as b,
        ):
            need(
                set(a.files) == {"times", *ARMS}
                and np.array_equal(a["times"], times[sid])
                and np.array_equal(b["times"], times[sid]),
                "raw/forecast clock mismatch",
            )
            x = {k: a[k].copy() for k in ARMS}
            for k, old_arm in (("static", "static"), ("intra", "layer0")):
                errors[k] = max(errors[k], parity(x[k], b[old_arm], tau, f"{sid}/{k}"))
        for k, trace in x.items():
            check_curve(trace, times[sid], f"{sid}/{k}")
            need(
                np.array_equal(trace[:FIRST], x["static"][:FIRST]),
                "changed first-write prefix",
            )
            rows[k].append(score(times[sid], trace, truth[sid], tau))
        for k, parent in (("inter", "static"), ("both", "intra")):
            delta = float(np.max(np.abs(x[k] - x[parent])))
            if not state_content:
                need(delta == 0, "nonzero intervention without committed memory")
            exposure[k] += int(delta > 1e-6)
        block_rows.append((rank, sid, x, digest))
        if rank % BLOCK:
            continue

        paths = [
            run / "blocks" / f"{block:02d}_{k}.json"
            for k in ("commit", "reveals", "update")
        ]
        commit, reveal, info = [read(p) for p in paths]
        bound = dict(
            block=block,
            commit_sha256=h(paths[0]),
            reveals_sha256=h(paths[1]),
            update_sha256=h(paths[2]),
        )
        if block == 1:
            ids, ranking = order[:20], []
        elif rank == 360:
            ids, ranking = [], []
        else:
            chosen, ranking = choose_query(
                [(i, s, x["static"], x["intra"]) for i, s, x, _ in block_rows], tau
            )
            ids = [chosen]
            queried.extend(ids)
        need(
            done["blocks"][block - 1] == bound
            and commit["block"] == info["block"] == block
            and commit["rank_end"] == rank
            and commit["selected"] == ids
            and commit["ranking"] == ranking
            and commit["seals"]
            == [dict(shot=s, seal_sha256=d) for _, s, _, d in block_rows]
            and commit["state_before_sha256"] == info["state_before_sha256"] == state
            and commit["previous_block_sha256"] == prev_block
            and info["first_next_rank"] == (rank + 1 if rank < 360 else None)
            and reveal["commit_sha256"]
            == info["commit_sha256"]
            == bound["commit_sha256"]
            and info["reveals_sha256"] == bound["reveals_sha256"]
            and [v["shot"] for v in reveal["outcomes"]] == ids
            and last_ns <= commit["committed_ns"],
            f"query/reveal not bound to its sealed block {block}",
        )
        last_ns = commit["committed_ns"]
        for v in reveal["outcomes"]:
            need(
                {k: v[k] for k in ("shot", "disruptive", "down")} == truth[v["shot"]]
                and last_ns <= v["read_start_ns"] <= v["read_end_ns"],
                "wrong label or label accessed before commitment",
            )
            last_ns = v["read_end_ns"]
        need(last_ns <= info["updated_ns"], "write recorded before allowed label read")
        proposal = None
        if block == 1:
            picked = []
            for flag in (True, False):
                picked += [s for s in ids if truth[s]["disruptive"] is flag][:4]
            guards = [s for s in ids if s in picked]
            need(
                info["decision"] == "cold_release"
                and info["guard_ids"] == guards
                and info["bandwidth_anchors_per_shot"] == 8
                and np.isfinite(info["bandwidth"])
                and info["bandwidth"] > 0,
                "cold guard/fit protocol mismatch",
            )
        elif ids:
            s = ids[0]
            pred = next(x for _, candidate, x, _ in block_rows if candidate == s)
            if variant in POSITIVE_ONLY and not truth[s]["disruptive"]:
                need(
                    info["decision"] == "guard_only_safe_outcome"
                    and not any(
                        k in info
                        for k in ("fit", "proposal_sha256", "teaching_indices")
                    ),
                    "safe query must enter guard history without a proposal",
                )
            else:
                proposal = check_proposal(
                    run,
                    block,
                    s,
                    pred,
                    info,
                    truth,
                    times,
                    [*guards, *queried[:-1][-2:]],
                    h,
                    tau,
                    phase_mode=phase_mode,
                )
            if proposal is not None and first_write is None:
                first_write = rank + 1
        else:
            need(info["decision"] == "terminal_no_query", "terminal block cannot learn")
        if block == 1 or proposal is not None:
            name = f"{block:02d}_state.npz"
            need(info["state_file"] == name, "missing committed state")
            expected = state_content if proposal is None else [*state_content, proposal]
            state = info["state_after_sha256"]
            state_content = load_state(
                run / "blocks" / name,
                h,
                state,
                len(expected),
                phase=phase_mode is not None,
            )
            need(
                all(
                    np.array_equal(a, b)
                    for left, right in zip(state_content, expected, strict=True)
                    for a, b in zip(left, right, strict=True)
                ),
                "memory changed beyond the approved addition",
            )
        else:
            need(
                info["state_file"] is None and info["state_after_sha256"] == state,
                "a rejected or absent update changed memory",
            )
        need(info["state_size"] == len(state_content), "wrong recorded state length")
        fit_count += int("fit" in info)
        events.append(
            dict(
                block=block,
                label=None if not ids or block == 1 else truth[ids[0]],
                decision=info["decision"],
                size=len(state_content),
            )
        )
        block_rows = []
        prev_block, last_ns = bound["update_sha256"], info["updated_ns"]
    need(
        done["completed_ns"] >= last_ns
        and len(queried) == len(set(queried)) == 16
        and done["n_accepted"] == len(state_content)
        and done["max_ref_error"] == errors
        and sum(v["disruptive"] for v in truth.values()) == 180,
        "wrong terminal query budget/coverage/source parity",
    )
    if variant == "uniform_global_matched_kernel":
        need(
            done["optimizer_kernel_calls"] == 22 * fit_count,
            "uniform private kernel was not called for every training objective",
        )
    sections = dict(
        full360=metrics(rows, range(360)),
        after20=metrics(rows, range(20, 360)),
        unqueried_after20=metrics(
            rows, [j for j in range(20, 360) if order[j] not in queried]
        ),
        last180=metrics(rows, range(180, 360)),
        blocks=[metrics(rows, range(j, j + BLOCK)) for j in range(0, 360, BLOCK)],
    )
    if first_write is not None:
        sections["after_first_write"] = metrics(rows, range(first_write - 1, 360))
    h.check()
    report = dict(
        status="PASS_LAYER0_FACTORIAL_HDF5_RECOUNT",
        n_shots=360,
        n_curves=1440,
        n_visible_outcomes=36,
        memory_variant=variant,
        online_queries=queried,
        n_accepted=len(state_content),
        first_effective_rank=first_write,
        events=events,
        memory_exposure_shots=exposure,
        memory_active_anchors=active_anchor_count,
        n_proposals=fit_count,
        reference_parity_max=errors,
        producer_sha256=h(run / "completed.json"),
        verifier_sha256=h(__file__),
        sections=sections,
        scope="Fixed development replay; independent raw HDF5 outcome/clock recount and committed selection/guard replay, not a retrain or independent future test.",
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        report["status"], report["n_curves"], sections["full360"]["metrics"], flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    verify(args.run, args.output)
