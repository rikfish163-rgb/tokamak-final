"""Rebuild all phase banks from authorized raw history and re-run true prefixes."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
from layer0_phase_kernel_20261105 import PhaseEpisode, phase_actions
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, predict, raw_for


RANKS = (21, 41, 81, 181, 321, 359)
PRIOR_CASES = (70861, 73541, 85123, 89549)


def witness(run, verification, output):
    if output.exists():
        raise FileExistsError(output)
    cfg, done, proof = [
        r.read(p) for p in (run / "protocol.json", run / "completed.json", verification)
    ]
    if (
        cfg["memory_variant"] not in ("phase_all_early", "phase_matched_early")
        or proof["status"] != "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT"
        or proof["producer_sha256"] != r.sha(run / "completed.json")
        or r.sha(run / "protocol.json") != done["protocol_sha256"]
    ):
        raise ValueError("verified phase experiment required")
    r.configure()
    order, source_cfg, _, base, source_digest, mapping, _, fit_path = parents()
    raw_digests = cfg["raw_sha256"]
    if order != cfg["order"] or source_cfg["raw_sha256"] != raw_digests:
        raise ValueError("raw witness and producer disagree")
    rule = r.load_rule(fit_path, base.center.device)
    states = {0: ()}
    bank_evidence = []
    n_prefix_active = 0
    with (
        patch.object(
            r.io23, "read_outcome", side_effect=RuntimeError("no new labels in witness")
        ),
        patch.object(memory, "memory_actions", new=phase_actions),
    ):
        for b in range(1, 19):
            updates = run / "blocks" / f"{b:02d}_update.json"
            reveals = run / "blocks" / f"{b:02d}_reveals.json"
            bound = done["blocks"][b - 1]
            if (
                r.sha(updates) != bound["update_sha256"]
                or r.sha(reveals) != bound["reveals_sha256"]
            ):
                raise ValueError("released history changed")
            update = r.read(updates)
            states[b] = states[b - 1]
            if b == 1:
                bandwidth = update["bandwidth"]
            if "fit" not in update:
                continue
            [label] = r.read(reveals)["outcomes"]
            sid = label["shot"]
            rank = order.index(sid) + 1
            if not label["disruptive"] or not 20 * (b - 1) < rank <= 20 * b:
                raise ValueError("prototype from outside released block")
            raw = raw_for(sid, raw_digests)
            keys = as_case(base, raw, rank, mapping, rule)[2]
            q = np.unique((np.asarray(update["teaching_indices"]) - 135) // 32)
            early = (np.asarray(update["early_anchor_indices"]) - 135) // 32
            proposal = run / "blocks" / f"{b:02d}_proposal.npz"
            if r.sha(proposal) != update["proposal_sha256"]:
                raise ValueError("proposal changed")
            with np.load(proposal, allow_pickle=False) as f:
                if not np.array_equal(
                    keys[q].cpu().numpy(), f["keys_0"]
                ) or not np.array_equal(keys[early].cpu().numpy(), f["early_keys_0"]):
                    raise ValueError(f"historical bank not reproduced from raw: {sid}")
                episode = PhaseEpisode(
                    torch.as_tensor(f["keys_0"], device=keys.device),
                    torch.as_tensor(f["delta_0"], device=keys.device),
                    torch.as_tensor(f["early_keys_0"], device=keys.device),
                )
            if update["decision"] == "accepted":
                states[b] = (*states[b], episode)
            bank_evidence.append(
                dict(
                    block=b,
                    shot=sid,
                    early=len(early),
                    precursor=len(q),
                    decision=update["decision"],
                )
            )
        prefix_evidence = []
        ranks = sorted(set(RANKS) | {order.index(sid) + 1 for sid in PRIOR_CASES})
        for rank in ranks:
            sid = order[rank - 1]
            data = raw_for(sid, raw_digests)
            path = run / "shots" / f"{rank:04d}_{sid}.npz"
            bound = done["sealed"][rank - 1]
            state = states[(rank - 1) // 20]
            if (
                r.sha(path) != bound["curve_sha256"]
                or len(state) != bound["state_size"]
            ):
                raise ValueError("wrong sealed current state")
            with np.load(path, allow_pickle=False) as old:
                move = int(np.argmax(np.abs(old["both"] - old["intra"])))
                n = min(
                    len(data["times"]) - 1, max(len(data["times"]) // 2, 200, move + 1)
                )
                for key in ("times", "values", "available"):
                    data[key] = data[key][:n].copy()
                if not np.array_equal(old["times"][:n], data["times"]):
                    raise ValueError("truncated clock changed")
                p, stats = predict(
                    base,
                    as_case(base, data, rank, mapping, rule),
                    state,
                    bandwidth,
                    all_arms=True,
                )
                diffs = {
                    k: layer.require_curve_parity(p[k], old[k][:n], r.THRESHOLD)
                    for k in cfg["arms"]
                }
            active = stats["inter"]["nonzero"] if state else 0
            n_prefix_active += active
            prefix_evidence.append(
                dict(
                    rank=rank,
                    shot=sid,
                    prefix_length=n,
                    active_anchors=active,
                    max_error=diffs,
                )
            )
    if not n_prefix_active or source_digest != r.state_digest(base):
        raise ValueError(
            "prefix witness did not exercise the phase gate or changed the source"
        )
    r.write_new(
        output,
        dict(
            status="PASS_LAYER0_PHASE_RAW_BANKS_AND_PREFIX",
            mode=cfg["phase_mode"],
            banks=bank_evidence,
            prefixes=prefix_evidence,
            prefix_active_anchors=n_prefix_active,
            producer_sha256=r.sha(run / "completed.json"),
            scope="No test-shot outcomes; authentic past releases only; same threshold crossings after truncation.",
        ),
    )
    print(
        "PASS_LAYER0_PHASE_RAW_BANKS_AND_PREFIX",
        len(bank_evidence),
        len(prefix_evidence),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "verification", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    witness(a.run, a.verification, a.output)
