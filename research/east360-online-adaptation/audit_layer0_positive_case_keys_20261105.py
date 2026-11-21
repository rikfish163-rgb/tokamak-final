"""Diagnose which matured keys matched at every flipped or overearly alarm."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, raw_for
from verify_confirmation_raw_replay_20261105 import raw_truth, score


def key_match(key, targets, h):
    if (
        key.shape != (192,)
        or targets.ndim != 2
        or targets.shape[1] != 192
        or not len(targets)
    ):
        raise ValueError(
            "one causal key and nonempty 192-dimensional reference bank required"
        )
    if (
        not np.isfinite(key).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(h)
        or h <= 0
    ):
        raise ValueError("finite key geometry required")
    squares = (targets.astype(np.float64) - key).reshape(-1, 3, 64) ** 2
    j = int(np.argmin(squares.mean(axis=(1, 2))))
    d = float(np.sqrt(squares[j].mean()))
    return dict(
        index=j,
        rms=d,
        distance_over_bandwidth=d / h,
        block_rms=np.sqrt(squares[j].mean(axis=1)).tolist(),
        weight=max(0.0, 1 - d * d / (4 * h * h)) ** 2,
    )


def audit(run, verification, output):
    if output.exists():
        raise FileExistsError(output)
    receipt, cfg, done = [
        r.read(p) for p in (verification, run / "protocol.json", run / "completed.json")
    ]
    if (
        receipt["status"] != "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT"
        or receipt["producer_sha256"] != r.sha(run / "completed.json")
        or cfg["memory_variant"] != "disruptive_feedback_only"
        or r.sha(run / "protocol.json") != done["protocol_sha256"]
    ):
        raise ValueError("verified end-to-end disruptive-feedback replay required")
    changes = receipt["sections"]["full360"]
    role = {}
    for kind, slots in (
        ("strict", ("tp_gained", "tp_lost", "fp_added", "fp_removed")),
        ("capped", ("tp_gained", "tp_lost")),
    ):
        table = changes[f"{kind}_pairs"]["both_vs_intra"]
        for slot in slots:
            for sid in table[slot]:
                role.setdefault(sid, []).append(f"{kind}_{slot}")
    r.configure()
    order, old, _, base, digest, mapping, _, fit_path = parents()
    if (
        order != cfg["order"]
        or old["raw_sha256"] != cfg["raw_sha256"]
        or digest != cfg["source_state_sha256"]
    ):
        raise ValueError("frozen replay source changed")
    raw_hashes = cfg["raw_sha256"]
    rule = r.load_rule(fit_path, base.center.device)
    banks = []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no new label for key geometry"),
    ):
        for block in range(1, 18):
            note = run / "blocks" / f"{block:02d}_update.json"
            revealed = run / "blocks" / f"{block:02d}_reveals.json"
            bound = done["blocks"][block - 1]
            if (
                r.sha(note) != bound["update_sha256"]
                or r.sha(revealed) != bound["reveals_sha256"]
            ):
                raise ValueError("archive changed before geometry audit")
            update = r.read(note)
            if block == 1:
                bandwidth = update["bandwidth"]
            if update["decision"] != "accepted":
                continue
            [label] = r.read(revealed)["outcomes"]
            sid, down = label["shot"], label["down"]
            if not label["disruptive"] or not (order.index(sid) < 20 * block):
                raise ValueError("invalid earlier revealed prototype")
            raw = raw_for(sid, raw_hashes)
            case = as_case(base, raw, order.index(sid) + 1, mapping, rule)
            keys = case[2].cpu().numpy()
            q = (np.asarray(update["teaching_indices"]) - 135) // 32
            anchor = 135 + 32 * np.arange(len(keys))
            clocks = raw["times"][anchor]
            early = np.flatnonzero(np.round((down - clocks) * 1000, 9) >= 500)
            state = run / "blocks" / update["state_file"]
            if r.sha(state) != update["state_after_sha256"]:
                raise ValueError("prototype state changed")
            with np.load(state, allow_pickle=False) as data:
                expected = data[f"keys_{len(banks)}"]
                if not np.array_equal(expected, keys[np.unique(q)]):
                    raise ValueError("failed to reproduce the saved teaching keys")
            banks.append(
                dict(
                    shot=sid,
                    rank_start=20 * block + 1,
                    positives=expected,
                    early=keys[early],
                    early_lead_ms=((down - clocks[early]) * 1000).tolist(),
                )
            )
        rows = []
        for sid in sorted(role, key=order.index):
            rank = order.index(sid) + 1
            raw = raw_for(sid, raw_hashes)
            case = as_case(base, raw, rank, mapping, rule)
            truth, clock = raw_truth(r.io23.path_for("EAST", sid), sid)
            sealed = run / "shots" / f"{rank:04d}_{sid}.npz"
            if r.sha(sealed) != done["sealed"][rank - 1]["curve_sha256"]:
                raise ValueError("sealed case changed")
            with np.load(sealed, allow_pickle=False) as f:
                if not np.array_equal(f["times"], clock):
                    raise ValueError("clock changed")
                items = {
                    k: score(clock, f[k], truth, r.THRESHOLD) for k in ("intra", "both")
                }
            snapshots = []
            for k, value in items.items():
                i = value["first_index"]
                if i is None or i < 135:
                    continue
                v = case[2][(i - 135) // 32].cpu().numpy()
                nearest = []
                for b in banks:
                    if rank < b["rank_start"]:
                        continue
                    pos = key_match(v, b["positives"], bandwidth)
                    early = None
                    if len(b["early"]):
                        early = key_match(v, b["early"], bandwidth)
                        early["lead_ms"] = b["early_lead_ms"][early["index"]]
                    nearest.append(
                        dict(
                            source_shot=b["shot"], teaching=pos, same_prior_early=early
                        )
                    )
                snapshots.append(
                    dict(
                        arm=k,
                        first_index=i,
                        lead_ms=value["lead_ms"],
                        retrieved=nearest,
                    )
                )
            rows.append(
                dict(
                    shot=sid, rank=rank, roles=role[sid], status=items, alarms=snapshots
                )
            )
    r.write_new(
        output,
        dict(
            status="COMPLETE_LAYER0_CASE_KEY_GEOMETRY",
            rows=rows,
            bandwidth=bandwidth,
            positive_query_ids=[b["shot"] for b in banks],
            early_reference_min_lead_ms=500,
            verification_sha256=r.sha(verification),
            scope="Post-hoc geometry on changed first alarms; early reference is not an outcome label or a validated gate.",
        ),
    )
    print("complete changed-case geometry", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "verification", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    audit(a.run, a.verification, a.output)
