"""Prequential EAST layer-0 factorial with one shared query-trained local memory."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
import layer0_query_protocol_20261105 as protocol
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r


PLAN = r.ROOT / "LAYER0_QUERY_MEMORY_PLAN_20261105.md"
LOCK = r.ROOT / "LAYER0_INTRA_DEVELOPMENT_LOCK_20261105.md"
MASK = r.ROOT / "probe_outputs/layer_interaction_masks360_20261105_v1"
MASK_SHA = "c887f88cf6a7183074d781d1655d86a15f8fdcf29d14273fcb4137b41e62cec8"
RULE_SHA = "eaf4c97f7aa4a829db8058ea879ff0677c092435e3cd30d38a1ab22127639eac"
BLOCK, ARMS = protocol.BLOCK, protocol.ARMS
N, COLD_BUDGET, ONLINE_BUDGET = 360, 20, 16


def parents():
    order, source, _ = r.reference()
    if r.sha(MASK / "completed.json") != MASK_SHA:
        raise ValueError("frozen layer-0 reference changed")
    done = r.read(MASK / "completed.json")
    if any(r.sha(MASK / name) != h for name, h in done["artifact_sha256"].items()):
        raise ValueError("invalid frozen-layer reference")
    cfg = r.read(MASK / "protocol.json")
    mapping_path = r.COLD / "input_map.json"
    fit_path = layer.ARCHIVE / "fit_moving_peak.npz"
    mapping = r.read(mapping_path)
    if (
        cfg["order"] != order
        or len(order) != N
        or cfg["threshold"] != r.THRESHOLD
        or cfg["raw_sha256"] != source["raw_sha256"]
        or r.sha(mapping_path) != source["input_sha256"][str(mapping_path)]
        or mapping["calibration_ids"] != order[:COLD_BUDGET]
        or r.sha(fit_path) != RULE_SHA
        or done["fit_sha256"] != {"moving_peak": RULE_SHA}
        or (r.policy.FIRST, r.policy.STRIDE) != (protocol.FIRST, protocol.STRIDE)
    ):
        raise ValueError("frozen cohort, map, write clock and fit required")
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    digest = r.state_digest(base)
    if digest != cfg["source_state_sha256"] or parent["threshold"] != r.THRESHOLD:
        raise ValueError("unchanged qualified J-TEXT checkpoint required")
    base.cuda()
    return order, cfg, done, base, digest, mapping, mapping_path, fit_path


def as_case(base, raw, rank, mapping, rule):
    ctx = r.latent.native_context(base, r.prepare_actual(raw, rank, mapping))
    if rule is None:
        if rank > COLD_BUDGET:
            raise ValueError("missing frozen rule after cold start")
        return ctx, None, None
    action = r.actions(rule, ctx)[:, :128].contiguous()
    return ctx, action, memory.keys_for(rule, ctx)


def predict(base, case, episodes, bandwidth, *, all_arms):
    ctx, fast, keys = case
    native = ctx[2]
    if fast is None:
        if episodes or not all_arms:
            raise ValueError("cold prefix cannot use a memory")
        return dict.fromkeys(ARMS, native), {}
    zero = torch.zeros_like(fast)
    targets = {}
    if all_arms:
        targets["intra"] = memory.total_action(fast, zero)
    if episodes:
        slow = memory.memory_actions(keys, episodes, bandwidth)
        targets["inter"] = memory.total_action(zero, slow)
        targets["both"] = memory.total_action(fast, slow)
    elif not all_arms:
        targets["both"] = memory.total_action(fast, zero)
    traces, stats = r.forecast(base, ctx, targets)
    if not episodes:
        traces["inter"] = native
        if all_arms:
            traces["both"] = traces["intra"]
    if all_arms:
        traces["static"] = native
    if any(
        not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
        for p in traces.values()
    ):
        raise FloatingPointError("nonfinite or out-of-range forecast")
    return traces, stats


def save_memory(path, episodes):
    arrays = {"size": np.array(len(episodes), dtype=np.int64)}
    for k, episode in enumerate(episodes):
        arrays[f"keys_{k}"] = episode.keys.detach().cpu().numpy()
        arrays[f"delta_{k}"] = episode.delta.detach().cpu().numpy()
    r.save_arrays(path, **arrays)
    return r.sha(path)


def raw_for(sid, raw_digests):
    path = r.io23.path_for("EAST", sid)
    before = r.signature(path)
    if r.sha(path) != raw_digests[str(sid)]:
        raise ValueError(f"raw shot content changed: {sid}")
    raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
    if before != r.signature(path):
        raise ValueError(f"raw shot changed during signal access: {sid}")
    return raw


def guard_case(result, cases, sid, truth, before, after):
    times, flag, down = truth
    result[f"{sid}_times"] = times
    for arm in protocol.PERSISTENT_ARMS:
        result[f"{sid}_{arm}_before"] = before[arm]
        result[f"{sid}_{arm}_after"] = after[arm]
        issues = protocol.guard_violations(
            times, before[arm], after[arm], flag, down, r.THRESHOLD
        )
        cases.extend(dict(shot=sid, arm=arm, reason=issue) for issue in issues)


def learn(base, current, guard, episodes, bandwidth, sealed, output, block):
    sid, case, truth = current
    times, flag, down = truth
    old = {k: sealed[k] for k in protocol.PERSISTENT_ARMS}
    earlier = {k: r.timeline(times, v, flag, down, r.THRESHOLD) for k, v in old.items()}
    incorrect = [
        k for k, v in earlier.items() if v["status"] in ("fp", "miss", "tardy")
    ]
    if not incorrect:
        return None, dict(decision="already_correct", query_status=earlier)
    if not flag and all(earlier[k]["first_index"] < protocol.FIRST for k in incorrect):
        return None, dict(decision="error_before_first_write", query_status=earlier)
    q = protocol.teaching_indices(times, old, flag, down, r.THRESHOLD)
    if not len(q):
        return None, dict(decision="no_eligible_window", query_status=earlier)
    ctx, fast, keys = case
    candidate, fit = memory.fit_episode(
        base, ctx, fast, keys, episodes, bandwidth, q, flag
    )
    after, _ = predict(base, case, (*episodes, candidate), bandwidth, all_arms=False)
    helpful, diagnostic = protocol.query_progress(
        times, old, after, flag, down, r.THRESHOLD
    )
    arrays, violations = {}, []
    guard_case(arrays, violations, sid, truth, old, after)
    for past_sid, past_case, past_truth in guard:
        if past_sid == sid:
            raise ValueError("query cannot be a prior guard")
        a, _ = predict(base, past_case, episodes, bandwidth, all_arms=False)
        b, _ = predict(
            base, past_case, (*episodes, candidate), bandwidth, all_arms=False
        )
        guard_case(arrays, violations, past_sid, past_truth, a, b)
    witness_path = output / "blocks" / f"{block:02d}_candidate_curves.npz"
    proposal_path = output / "blocks" / f"{block:02d}_proposal.npz"
    r.save_arrays(witness_path, **arrays)
    proposal_sha = save_memory(proposal_path, (candidate,))
    decision = (
        "accepted"
        if helpful and not violations
        else ("rejected_guard" if violations else "no_query_progress")
    )
    receipt = dict(
        decision=decision,
        query_status=earlier,
        teaching_indices=q.tolist(),
        fit=fit,
        diagnostics=diagnostic,
        guard_ids=[v[0] for v in guard],
        violations=violations,
        witness_sha256=r.sha(witness_path),
        proposal_sha256=proposal_sha,
    )
    return (candidate if decision == "accepted" else None), receipt


def shot_file(output, note, suffix):
    return output / "shots" / f"{note['rank']:04d}_{note['shot']}{suffix}"


def sealed_arrays(output, note):
    path = shot_file(output, note, ".npz")
    if (
        r.sha(path) != note["curve_sha256"]
        or r.sha(shot_file(output, note, "_seal.json")) != note["seal_sha256"]
    ):
        raise ValueError("sealed curve changed before query")
    with np.load(path, allow_pickle=False) as arr:
        return {k: arr[k].copy() for k in arr.files}


def cold_setup(base, outcomes, raw_digests, mapping, fit_path):
    rule = r.load_rule(fit_path, base.center.device)
    groups, guard = [], []
    class_count = {True: 0, False: 0}
    for v in outcomes:
        sid, flag, down = v["shot"], v["disruptive"], v["down"]
        raw = raw_for(sid, raw_digests)
        case = as_case(base, raw, 21, mapping, rule)
        keys = case[2]
        if len(keys) < 8:
            raise ValueError("at least eight cold bandwidth anchors required")
        groups.append(keys[np.linspace(0, len(keys) - 1, 8, dtype=int)].clone())
        if class_count[flag] < 4:
            guard.append((sid, case, (raw["times"], flag, down)))
        class_count[flag] += 1
    if class_count != {True: 8, False: 12} or len(guard) != 8:
        raise ValueError("expected 8D/12S authoritative cold labels")
    return rule, memory.fit_bandwidth(groups), guard


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, reference, base, source_digest, mapping, map_path, fit_path = parents()
    raw_digests = cfg["raw_sha256"]
    deps = [
        Path(__file__),
        PLAN,
        LOCK,
        MASK / "completed.json",
        MASK / "protocol.json",
        fit_path,
        map_path,
        r.DEFAULT_PARENT,
        r.io23.EAST_COHORT,
        Path(memory.__file__),
        Path(protocol.__file__),
        Path(r.__file__),
        Path(r.policy.__file__),
        Path(r.latent.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(r.latent.make_views.__code__.co_filename),
        Path(r.timeline.__code__.co_filename),
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    (output / "blocks").mkdir()
    initial_ns = time.monotonic_ns()
    r.write_new(
        output / "protocol.json",
        dict(
            order=order,
            arms=ARMS,
            first=protocol.FIRST,
            stride=protocol.STRIDE,
            block_size=BLOCK,
            cold_budget=COLD_BUDGET,
            online_budget=ONLINE_BUDGET,
            initial_ns=initial_ns,
            threshold=r.THRESHOLD,
            source_state_sha256=source_digest,
            input_sha256=hashes,
            raw_sha256=raw_digests,
            reference_sha256=MASK_SHA,
            fast_fit_sha256=RULE_SHA,
            device=torch.cuda.get_device_name(),
            torch_version=str(torch.__version__),
            development_only=True,
            unqueried_hdf5_outcomes_read_in_producer=False,
            shared_inter_memory=True,
        ),
    )
    oracle = r.io23.read_outcome

    def read_committed(device, sid):
        path = r.io23.path_for(device, sid)
        stamp = r.signature(path)
        if r.sha(path) != raw_digests[str(sid)]:
            raise ValueError(f"raw changed before committed reveal: {sid}")
        label = oracle(device, sid)
        if stamp != r.signature(path):
            raise ValueError(f"raw changed during committed reveal: {sid}")
        return label

    last_seal = last_block = None
    rule = bandwidth = state_sha = None
    episodes, recent = (), []
    cold_guard, batch, sealed, blocks = [], [], [], []
    max_ref_error = {"static": 0.0, "intra": 0.0}
    n_reveals = n_accepted = 0
    print("started locked EAST360 local memory factorial", flush=True)
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("outcome unavailable to signal prediction or fitting"),
    ):
        for rank, sid in enumerate(order, 1):
            begin_ns = time.monotonic_ns()
            raw = raw_for(sid, raw_digests)
            case = as_case(base, raw, rank, mapping, rule)
            curves, stats = predict(base, case, episodes, bandwidth, all_arms=True)
            times = raw["times"]
            stem = f"{rank:04d}_{sid}"
            prior = MASK / "shots" / (stem + ".npz")
            if r.sha(prior) != reference["sealed"][rank - 1]["curve_sha256"]:
                raise ValueError(f"frozen curve changed: {sid}")
            with np.load(prior, allow_pickle=False) as p:
                if not np.array_equal(p["times"], times):
                    raise ValueError(f"signal clock changed: {sid}")
                for arm, old in (("static", "static"), ("intra", "layer0")):
                    err = layer.require_curve_parity(curves[arm], p[old], r.THRESHOLD)
                    max_ref_error[arm] = max(max_ref_error[arm], err)
            array_path = output / "shots" / (stem + ".npz")
            r.save_arrays(array_path, times=times, **curves)
            note = dict(
                rank=rank,
                shot=sid,
                block=(rank - 1) // BLOCK + 1,
                raw_sha256=raw_digests[str(sid)],
                curve_sha256=r.sha(array_path),
                previous_seal_sha256=last_seal,
                previous_block_sha256=last_block,
                state_sha256=state_sha,
                state_size=len(episodes),
                stats=stats,
                predict_start_ns=begin_ns,
                seal_ns=time.monotonic_ns(),
            )
            seal_path = output / "shots" / (stem + "_seal.json")
            r.write_new(seal_path, note)
            last_seal = r.sha(seal_path)
            note["seal_sha256"] = last_seal
            batch.append(note)
            sealed.append(note)
            del case, curves
            if rank % BLOCK:
                continue

            block = rank // BLOCK
            snapshots = [(n, sealed_arrays(output, n)) for n in batch]
            if block == 1:
                selected, scores = order[:COLD_BUDGET], []
            elif rank == N:
                selected, scores = [], []
            else:
                chosen, scores = protocol.choose_query(
                    [
                        (n["rank"], n["shot"], p["static"], p["intra"])
                        for n, p in snapshots
                    ],
                    r.THRESHOLD,
                )
                selected = [chosen]
            path = output / "blocks" / f"{block:02d}_commit.json"
            r.write_new(
                path,
                dict(
                    block=block,
                    rank_end=rank,
                    selected=selected,
                    ranking=scores,
                    seals=[
                        dict(shot=n["shot"], seal_sha256=n["seal_sha256"])
                        for n in batch
                    ],
                    state_before_sha256=state_sha,
                    previous_block_sha256=last_block,
                    committed_ns=time.monotonic_ns(),
                ),
            )
            revealed = protocol.reveal_block(path, selected, read_committed)
            reveal_path = output / "blocks" / f"{block:02d}_reveals.json"
            r.write_new(reveal_path, revealed)
            n_reveals += len(selected)
            state_before = state_sha
            state_file = None
            if block == 1:
                rule, bandwidth, cold_guard = cold_setup(
                    base, revealed["outcomes"], raw_digests, mapping, fit_path
                )
                info = dict(
                    decision="cold_release",
                    guard_ids=[v[0] for v in cold_guard],
                    bandwidth=bandwidth,
                    bandwidth_anchors_per_shot=8,
                )
                state_file = f"{block:02d}_state.npz"
                state_sha = save_memory(output / "blocks" / state_file, episodes)
            elif not selected:
                info = dict(decision="terminal_no_query")
            else:
                label = revealed["outcomes"][0]
                qid, flag, down = selected[0], label["disruptive"], label["down"]
                shot_note, qpred = next(
                    (n, p) for n, p in snapshots if n["shot"] == qid
                )
                raw = raw_for(qid, raw_digests)
                if not np.array_equal(qpred["times"], raw["times"]):
                    raise ValueError("committed query signal clock changed")
                qcase = as_case(base, raw, shot_note["rank"], mapping, rule)
                current = qid, qcase, (raw["times"], flag, down)
                candidate, info = learn(
                    base,
                    current,
                    [*cold_guard, *recent],
                    episodes,
                    bandwidth,
                    qpred,
                    output,
                    block,
                )
                if candidate is not None:
                    episodes = (*episodes, candidate)
                    n_accepted += 1
                    state_file = f"{block:02d}_state.npz"
                    state_sha = save_memory(output / "blocks" / state_file, episodes)
                recent = [*recent, current][-2:]
            update_path = output / "blocks" / f"{block:02d}_update.json"
            r.write_new(
                update_path,
                dict(
                    block=block,
                    commit_sha256=r.sha(path),
                    reveals_sha256=r.sha(reveal_path),
                    state_before_sha256=state_before,
                    state_after_sha256=state_sha,
                    state_file=state_file,
                    state_size=len(episodes),
                    first_next_rank=rank + 1 if rank < N else None,
                    updated_ns=time.monotonic_ns(),
                    **info,
                ),
            )
            last_block = r.sha(update_path)
            blocks.append(
                dict(
                    block=block,
                    commit_sha256=r.sha(path),
                    reveals_sha256=r.sha(reveal_path),
                    update_sha256=last_block,
                )
            )
            batch = []
            print(
                f"sealed through {rank}/{N}, {n_reveals} labels, {n_accepted} writes: "
                f"{info['decision']}",
                flush=True,
            )
    if (
        n_reveals != COLD_BUDGET + ONLINE_BUDGET
        or n_accepted != len(episodes)
        or r.state_digest(base) != source_digest
        or any(p.grad is not None for p in base.parameters())
        or any(r.sha(p) != h for p, h in hashes.items())
    ):
        raise AssertionError("query budget, frozen parent or bound input changed")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER0_QUERY_MEMORY",
            n_shots=N,
            n_curves=N * len(ARMS),
            n_reveals=n_reveals,
            n_accepted=n_accepted,
            sealed=sealed,
            blocks=blocks,
            max_ref_error=max_ref_error,
            protocol_sha256=r.sha(output / "protocol.json"),
            completed_ns=time.monotonic_ns(),
            seconds=time.monotonic() - started,
        ),
    )
    print("complete local memory", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
