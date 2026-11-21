"""Seal past-only teaching curves for actual first-layer writes and zero writes."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import (
    fit_prefix_map,
)
from layer0_signed_support_20261106 import AUDIT, HISTORY
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for


PLAN = r.ROOT / "LAYER0_ACTION_UTILITY_PLAN_20261106.md"
FIRST, STRIDE, BLOCK = 135, 32, 20
SIGNAL_FIELDS = ("shot", "device", "values", "available")


def release_rows(block, seals, commit, reveal, commit_hash, *, block_size=BLOCK):
    """Only a historical, already sealed and committed block can supply labels."""
    n = block * block_size
    if len(seals) != block_size:
        raise ValueError("complete sealed block required")
    if (
        (commit["block"], commit["rank_end"]) != (block, n)
        or reveal["commit_sha256"] != commit_hash
        or commit["seals"]
        != [dict(shot=x["shot"], seal_sha256=x["seal_sha256"]) for x in seals]
        or commit["committed_ns"] < max(x["seal_ns"] for x in seals)
        or commit["selected"] != [x["shot"] for x in reveal["outcomes"]]
        or len(set(commit["selected"])) != len(commit["selected"])
        or not set(commit["selected"]) <= {x["shot"] for x in seals}
    ):
        raise ValueError("label release does not match committed predictions")
    result = []
    last = commit["committed_ns"]
    for row in reveal["outcomes"]:
        flag, down = row["disruptive"], row["down"]
        if (
            not last < row["read_start_ns"] <= row["read_end_ns"]
            or type(flag) is not bool
            or (flag and (type(down) not in (float, int) or not np.isfinite(down)))
            or (not flag and down is not None)
        ):
            raise ValueError("release has an invalid time or hard outcome")
        result.append(
            dict(
                shot=row["shot"],
                disruptive=flag,
                down=down,
                available_after_rank=n,
                original_read_end_ns=row["read_end_ns"],
                original_commit_sha256=commit_hash,
            )
        )
        last = row["read_end_ns"]
    return result


def load_history(order, raw_sha256, source, fit_path):
    dp, cp = (HISTORY / name for name in ("completed.json", "protocol.json"))
    done, cfg, audit = (r.read(p) for p in (dp, cp, AUDIT))
    if (
        done["status"] != "COMPLETE_LAYER0_QUERY_MEMORY"
        or audit["status"] != "PASS_LAYER0_FACTORIAL_HDF5_RECOUNT"
        or audit["producer_sha256"] != r.sha(dp)
        or done["protocol_sha256"] != r.sha(cp)
        or cfg["order"] != order
        or cfg["raw_sha256"] != raw_sha256
        or cfg["source_state_sha256"] != source
        or cfg["threshold"] != r.THRESHOLD
        or (cfg["first"], cfg["stride"], cfg["cold_budget"], cfg["online_budget"])
        != (FIRST, STRIDE, 20, 16)
        or cfg["input_sha256"][str(fit_path.resolve())] != r.sha(fit_path)
        or len(done["blocks"]) != 18
        or len(done["sealed"]) != 360
    ):
        raise ValueError("verified original history required")
    records, deps = [], [dp, cp, AUDIT]
    rank = {sid: i for i, sid in enumerate(order, 1)}
    for block, note in enumerate(done["blocks"], 1):
        if note["block"] != block:
            raise ValueError("original block order changed")
        commit, reveal = (
            HISTORY / "blocks" / f"{block:02d}_{tag}.json"
            for tag in ("commit", "reveals")
        )
        hashes = (r.sha(commit), r.sha(reveal))
        if hashes != (note["commit_sha256"], note["reveals_sha256"]):
            raise ValueError("original label release changed")
        end = block * BLOCK
        rows = release_rows(
            block,
            done["sealed"][end - BLOCK : end],
            r.read(commit),
            r.read(reveal),
            hashes[0],
        )
        records.extend(
            dict(
                row,
                rank=rank[row["shot"]],
                group="cold_loo" if block == 1 else "queried",
            )
            for row in rows
        )
        deps.extend((commit, reveal))
    if (
        len(records) != audit["n_visible_outcomes"]
        or len(records) != 36
        or [x["shot"] for x in records] != order[:20] + audit["online_queries"]
        or len({x["shot"] for x in records}) != 36
        or sum(x["disruptive"] for x in records[:20]) != 8
        or sum(x["disruptive"] for x in records[20:]) != 4
    ):
        raise ValueError("cold 20 plus same 16 original queries required")
    return records, deps, done


def anchor_inputs(
    times, static, z, adapted, history, norm, *, first=FIRST, stride=STRIDE
):
    idx = np.arange(first, len(times), stride)
    if (
        times.ndim != 1
        or not len(times)
        or not (np.diff(times) > 0).all()
        or not (times.shape == static.shape == z.shape == adapted.shape)
        or history.shape != (len(idx), 128)
        or norm.shape != (len(idx),)
    ):
        raise ValueError("causal columns require corresponding observed anchor clocks")
    result = np.column_stack(
        (z[idx], times[idx] - times[0], adapted[idx] - static[idx], norm, history)
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("anchor features must be finite")
    return result.astype(np.float32)


def context(base, raw, mapping):
    return r.latent.native_context(base, r.prepare_actual(raw, 21, mapping))


def calibration_record(raw):
    """The original map fits signal stats after the pointwise CIII sentinel mask."""
    signal = apply_input_contract(raw)
    return {k: signal[k] for k in SIGNAL_FIELDS}


def refit(base, rows, raws, mapping):
    history, episodes, windows = [], [], []
    for row in rows:
        sid, flag, down, rank = (row[k] for k in ("shot", "disruptive", "down", "rank"))
        raw = raws[sid]
        ctx = context(base, raw, mapping)
        t = raw["times"]
        episode, win = r.latent.episode(
            ctx, t, flag, down, sid, rank, keep_windows=True
        )
        episodes.append(episode)
        windows.append(win)
        history.append((sid, ctx, t, flag, down))
    models, info = r.latent.fit_budget(
        base, episodes, windows, r.THRESHOLD, seed=r.SEED
    )
    rule = models["history"]
    opt = torch.optim.Adam(rule.parameters(), lr=r.latent.LR)
    stages = []
    for permutations in r.shared_orders(len(history) * 8):
        peaks = {
            sid: r.forecast(base, ctx, {"score": r.actions(rule, ctx)})[0]["score"]
            for sid, ctx, *_ in history
        }
        stages.append(
            r.train_round(base, rule, opt, history, peaks, permutations, "moving_peak")
        )
    if (
        any(x["steps"] != 12 for x in stages)
        or info["history"]["optimizer_steps"] != 36
    ):
        raise ValueError("fold optimizer budget did not match frozen pipeline")
    return rule, dict(initial=info["history"], rounds=stages)


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, _, base, source, mapping, map_path, fit_path = parents()
    records, files, archive = load_history(order, cfg["raw_sha256"], source, fit_path)
    files += [
        Path(__file__),
        PLAN,
        r.DEFAULT_PARENT,
        map_path,
        fit_path,
        Path(r.__file__),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(fit_prefix_map.__code__.co_filename),
        Path(apply_input_contract.__code__.co_filename),
    ]
    binding = {str(p.resolve()): r.sha(p) for p in files}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    (output / "folds").mkdir()
    protocol = dict(
        scope="ENDED_HISTORY_ACTION_UTILITY_ONLY",
        rows=records,
        threshold=r.THRESHOLD,
        source_state_sha256=source,
        input_sha256=binding,
        source_fit_sha256=r.sha(fit_path),
        source_map_sha256=r.sha(map_path),
        first=FIRST,
        stride=STRIDE,
        raw_sha256={
            str(row["shot"]): cfg["raw_sha256"][str(row["shot"])] for row in records
        },
        fold_fits_use_other_19_only=True,
        fold_predictions_not_original_stream=True,
        scored_flow_predictions_untouched=True,
        new_labels=0,
        gpu=torch.cuda.get_device_name(),
        feature_columns=[
            "static_logit",
            "elapsed_s",
            "anchor_probability_change",
            "action_norm",
            "128 latent_history columns",
        ],
    )
    r.write_new(output / "protocol.json", protocol)
    sealed, max_parity = [], {"static": 0.0, "adapted": 0.0}
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError(
            "only originally released historical outcomes allowed"
        ),
    ):
        cold = records[:20]
        raw = {row["shot"]: raw_for(row["shot"], cfg["raw_sha256"]) for row in cold}
        stats = mapping["source_statistics"]

        def refit_map(rows, ordered):
            signals = [calibration_record(raw[row["shot"]]) for row in rows]
            return fit_prefix_map(
                stats, signals, order=ordered, count=len(rows), policy=mapping["policy"]
            )

        if refit_map(cold, order) != mapping:
            raise ValueError("full cold map must exactly reproduce original fit")
        frozen_rule = r.load_rule(fit_path, base.center.device)
        for row in records:
            sid, rank = row["shot"], row["rank"]
            is_cold = row["group"] == "cold_loo"
            stem = f"{rank:04d}_{sid}"
            dependencies = {}
            if is_cold:
                train = [x for x in cold if x["shot"] != sid]
                train_ids = [x["shot"] for x in train]
                fold_map = refit_map(train, train_ids + [sid])
                rule, fit_log = refit(base, train, raw, fold_map)
                shot = raw[sid]
                r.write_new(output / "folds" / f"{stem}_map.json", fold_map)
                r.write_new(
                    output / "folds" / f"{stem}_fit.json",
                    dict(training_ids=train_ids, held=sid, **fit_log),
                )
                r.save_arrays(
                    output / "folds" / f"{stem}_rule.npz",
                    **{
                        k: v.detach().cpu().numpy()
                        for k, v in rule.state_dict().items()
                    },
                )
                dependencies = {
                    str(p.relative_to(output)): r.sha(p)
                    for p in (output / "folds").glob(stem + "_*")
                }
            else:
                fold_map, rule = mapping, frozen_rule
                shot = raw_for(sid, cfg["raw_sha256"])
            ctx = context(base, shot, fold_map)
            action = r.actions(rule, ctx)[:, :128].contiguous()
            target = torch.cat((action, torch.zeros_like(action)), dim=1)
            curves, _ = r.forecast(base, ctx, {"adapted": target})
            t, static, adapted = shot["times"], ctx[2], curves["adapted"]
            if not is_cold:
                reference = HISTORY / "shots" / (stem + ".npz")
                if r.sha(reference) != archive["sealed"][rank - 1]["curve_sha256"]:
                    raise ValueError("original queried curves no longer match seal")
                dependencies[str(reference)] = r.sha(reference)
                with np.load(reference, allow_pickle=False) as a:
                    if not np.array_equal(a["times"], t):
                        raise ValueError("queried shot clock changed")
                    for k, p, old in (
                        ("static", static, a["static"]),
                        ("adapted", adapted, a["intra"]),
                    ):
                        error = float(np.max(abs(p - old)))
                        if error > 4e-6 or not np.array_equal(
                            p >= r.THRESHOLD, old >= r.THRESHOLD
                        ):
                            raise ValueError(
                                f"original query curve parity: {sid}/{k}/{error}"
                            )
                        max_parity[k] = max(max_parity[k], error)
            features = anchor_inputs(
                t,
                static,
                ctx[3],
                adapted,
                ctx[5].cpu().numpy(),
                action.norm(dim=1).cpu().numpy(),
            )
            p = output / "shots" / (stem + ".npz")
            r.save_arrays(p, times=t, static=static, adapted=adapted, features=features)
            sealed.append(
                dict(
                    shot=sid,
                    rank=rank,
                    group=row["group"],
                    file=str(p.relative_to(output)),
                    curve_sha256=r.sha(p),
                    dependencies=dependencies,
                )
            )
            if is_cold or len(sealed) % 4 == 0:
                print(f"history {len(sealed)}/36 {row['group']} {sid}", flush=True)
    if (
        r.state_digest(base) != source
        or any(p.grad is not None for p in base.parameters())
        or any(r.sha(p) != h for p, h in binding.items())
    ):
        raise ValueError("source or input changed during historical reconstruction")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_ENDED_ACTION_HISTORY",
            seconds=time.monotonic() - start,
            protocol_sha256=r.sha(output / "protocol.json"),
            n_shots=len(sealed),
            n_curves=2 * len(sealed),
            crossfit_folds=20,
            queried_parity=max_parity,
            sealed=sealed,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
