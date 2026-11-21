"""Shot-balanced clock-free evidence from the already revealed 20-shot calibration."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, raw_for


PLAN = r.ROOT / "LAYER0_SIGNED_PROJECTION_PLAN_20261106.md"
HISTORY = r.ROOT / "probe_outputs/layer0_disruptive_feedback360_20261105_v1"
AUDIT = (
    r.ROOT / "probe_outputs/layer0_disruptive_feedback360_verification_20261105_v1.json"
)


def paired_clock_indices(times, down):
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
        or not np.isfinite(down)
    ):
        raise ValueError(
            "finite historical clock and released disruption time required"
        )
    grid = np.arange(r.policy.FIRST, len(times), r.policy.STRIDE, dtype=np.int64)
    lead = np.round((down - times[grid]) * 1000.0, 9)
    p = grid[(lead > 30) & (lead <= 125)]
    early = grid[lead >= 500]
    if not len(p) or len(early) < len(p):
        return grid[:0], grid[:0]
    return p, early[np.linspace(0, len(early) - 1, len(p), dtype=np.int64)]


def causal_phase_support(keys, bank):
    """Pick the nearest precursor first; its own past early phase is the opponent."""
    memory._matrix(keys, 192, "causal keys")
    if not bank:
        raise ValueError("at least one paired training shot required")
    minima = []
    for p, e in bank:
        memory._matrix(p, 192, "precursor bank", nonempty=True)
        memory._matrix(e, 192, "early bank", nonempty=True)
        memory._same_type(keys, p)
        memory._same_type(keys, e)
        if len(p) != len(e):
            raise ValueError("same-shot phase banks must have the same cardinality")
        q = keys.to(torch.float64)
        minima.append(
            torch.stack(
                tuple(
                    (q[:, None, :] - b[None].to(torch.float64))
                    .square()
                    .mean(dim=2)
                    .amin(dim=1)
                    for b in (p, e)
                ),
                dim=1,
            )
        )
    all_d = torch.stack(minima, dim=1)
    if not torch.isfinite(all_d).all():
        raise ValueError("phase geometry not finite")
    nearest = all_d[:, :, 0].argmin(dim=1)
    dp, de = (
        all_d[torch.arange(len(keys), device=keys.device), nearest].sqrt().unbind(dim=1)
    )
    # Positive means the competing same-shot early phase is farther away.
    den = de + dp
    margin = (de - dp) / torch.where(den > 0, den, torch.ones_like(den))
    return dict(
        allow=dp < de,
        nearest_bank=nearest,
        precursor_rms=dp,
        early_rms=de,
        margin=margin,
    )


def verified_cold_release():
    done, checked = [r.read(p) for p in (HISTORY / "completed.json", AUDIT)]
    cfg = r.read(HISTORY / "protocol.json")
    p = HISTORY / "blocks" / "01_reveals.json"
    update = HISTORY / "blocks" / "01_update.json"
    commit = HISTORY / "blocks" / "01_commit.json"
    b = done["blocks"][0]
    reveal = r.read(p)
    outcomes = reveal["outcomes"]
    if (
        checked["status"] != "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT"
        or checked["producer_sha256"] != r.sha(HISTORY / "completed.json")
        or r.sha(HISTORY / "protocol.json") != done["protocol_sha256"]
        or r.sha(p) != b["reveals_sha256"]
        or r.sha(update) != b["update_sha256"]
        or r.sha(commit) != b["commit_sha256"]
        or reveal["commit_sha256"] != b["commit_sha256"]
        or r.read(update)["decision"] != "cold_release"
        or [v["shot"] for v in outcomes] != cfg["order"][:20]
        or sum(v["disruptive"] for v in outcomes) != 8
    ):
        raise ValueError(
            "verified cold history released after first twenty seals required"
        )
    return outcomes, (
        p,
        update,
        commit,
        HISTORY / "completed.json",
        HISTORY / "protocol.json",
        AUDIT,
    )


def build(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, _, base, source, mapping, map_path, fit_path = parents()
    released, history_files = verified_cold_release()
    if [v["shot"] for v in released] != order[:20]:
        raise ValueError("cold role does not match the pinned order")
    rule = r.load_rule(fit_path, base.center.device)
    deps = [
        Path(__file__),
        PLAN,
        fit_path,
        map_path,
        r.DEFAULT_PARENT,
        Path(as_case.__code__.co_filename),
        Path(memory.__file__),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        Path(r.io23.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        *history_files,
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    arrays, rows = {}, []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("only archived cold releases may train support"),
    ):
        for rank, label in enumerate(released, 1):
            if not label["disruptive"]:
                continue
            sid, down = label["shot"], label["down"]
            raw = raw_for(sid, cfg["raw_sha256"])
            p, e = paired_clock_indices(raw["times"], down)
            row = dict(
                rank=rank,
                shot=sid,
                down=down,
                raw_sha256=cfg["raw_sha256"][str(sid)],
                precursor=p.tolist(),
                early=e.tolist(),
            )
            if len(p):
                case = as_case(base, raw, 21, mapping, rule)
                keys = case[2].cpu().numpy()
                row["slot"] = len(arrays) // 2
                for k, indices in (("p", p), ("e", e)):
                    arrays[f"{k}_{row['slot']}"] = keys[
                        (indices - r.policy.FIRST) // r.policy.STRIDE
                    ]
            rows.append(row)
    if (
        not arrays
        or len(rows) != 8
        or r.state_digest(base) != source
        or any(r.sha(p) != digest for p, digest in hashes.items())
    ):
        raise ValueError("no valid cold phase bank or a frozen source changed")
    output.mkdir(parents=True)
    r.save_arrays(output / "bank.npz", **arrays)
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_COLD20_SIGNED_PHASE_BANK",
            original_roles=order[:20],
            earliest_fixed_rank=21,
            source_state_sha256=source,
            fit_sha256=r.sha(fit_path),
            map_sha256=r.sha(map_path),
            n_pairs=len(arrays) // 2,
            historical_disruptive_rows=rows,
            bank_sha256=r.sha(output / "bank.npz"),
            input_sha256=hashes,
            seconds=time.monotonic() - start,
        ),
    )
    print(f"sealed {len(arrays) // 2} paired cold phase banks", flush=True)


def load_bank(output, device):
    info = r.read(output / "completed.json")
    if (
        info["status"] != "COMPLETE_COLD20_SIGNED_PHASE_BANK"
        or info["earliest_fixed_rank"] != 21
        or not 1 <= info["n_pairs"] <= 8
        or r.sha(output / "bank.npz") != info["bank_sha256"]
        or any(r.sha(p) != digest for p, digest in info["input_sha256"].items())
    ):
        raise ValueError("bad cold phase bank binding")
    keys = {f"{k}_{j}" for k in ("p", "e") for j in range(info["n_pairs"])}
    with np.load(output / "bank.npz", allow_pickle=False) as f:
        if set(f.files) != keys:
            raise ValueError("bad phase bank arrays")
        bank = tuple(
            tuple(torch.as_tensor(f[f"{k}_{j}"], device=device) for k in ("p", "e"))
            for j in range(info["n_pairs"])
        )
    return bank, info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    build(p.parse_args().output)
