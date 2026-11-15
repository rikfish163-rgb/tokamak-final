"""Fixed360 research replay of bounded, exact-forward-trained latent adaptation."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from confirmation_raw_replay_20261105 import metrics, signature
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.east_context_operator_20261031.run import configure
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_hindsight_capacity_20261104 import (
    SOURCE,
    THRESHOLD,
    read,
    sha,
    timeline,
    write_new,
)
import latent_trajectory_adapt_20261105 as model
from local_affine_reach_20261105 import COLD
import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import baseline_for, reference
from verify_point_attribution_20261105 import pair

PLAN = io23.ROOT / "LATENT_TRAJECTORY_ADAPT_PLAN_20261105.md"
ARMS = (
    "static",
    *(f"{name}{b}" for b in (20, 100) for name in ("state", "history", "stale")),
)
ATOL = 4e-6


def save_arrays(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def sections(rows):
    output = {}
    for name, subset in (
        ("full360", slice(None)),
        ("after20", slice(20, None)),
        ("after100", slice(100, None)),
    ):
        group = {k: v[subset] for k, v in rows.items()}
        output[name] = dict(
            metrics={k: metrics(v) for k, v in group.items()},
            pairs={
                k: pair(v, group["static"]) for k, v in group.items() if k != "static"
            },
            pairs_capped={
                k: pair(v, group["static"], capped=True)
                for k, v in group.items()
                if k != "static"
            },
        )
    return output


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    configure()
    order, cfg, source_done = reference()
    base, parent = load_parent(DEFAULT_PARENT)
    digest = state_digest(base)
    if (
        parent["threshold"] != THRESHOLD
        or cfg["threshold"] != THRESHOLD
        or digest != cfg["source_state_sha256"]
    ):
        raise ValueError("exact frozen parent state required")
    base.cuda()
    map_path = COLD / "input_map.json"
    mapping = read(map_path)
    if (
        mapping["calibration_ids"] != order[:20]
        or sha(map_path) != cfg["input_sha256"][str(map_path)]
    ):
        raise ValueError("exact fixed numerical map required")
    deps = [
        Path(__file__),
        PLAN,
        DEFAULT_PARENT,
        map_path,
        Path(io23.__file__),
        Path(simple_data.__file__),
        Path(model.__file__),
        Path(policy.__file__),
        Path(model.make_views.__code__.co_filename),
        Path(prepare_actual.__code__.co_filename),
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
        io23.EAST_COHORT,
    ]
    inputs = {str(p.resolve()): sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write_new(
        output / "protocol.json",
        dict(
            scope="FIXED360_LATENT_ADAPTATION_DEVELOPMENT",
            order=order,
            threshold=THRESHOLD,
            input_sha256=inputs,
            arms=ARMS,
            first_anchor=policy.FIRST,
            stride=policy.STRIDE,
            radius=model.RADIUS,
            actual_nonlinear_forward_training=True,
            epochs=model.EPOCHS,
            lr=model.LR,
            minibatch=model.MINIBATCH,
            history_lag=model.CONTRAST_LAG,
            label_budgets=[20, 100],
            numeric_mapping_from_rank=21,
            early_teaching_map="only recompute mapped views after all first20 have ended",
            raw_sha256=cfg["raw_sha256"],
            fixed_source_state_sha256=digest,
            source_frozen=True,
            model_training_after_predict_seal_and_reveal=True,
            per_shot_temporary_reset=True,
            independent_future_validation=False,
            reveal="HDF5 meta/IsDisrupt + meta/DownTime only",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
        ),
    )
    rows = {k: [] for k in ARMS}
    models, fits, fit_hashes = {}, {}, {}
    episodes, windows, pending, seals = [], [], [], []
    max_native_difference = 0.0
    print("started fixed360 latent trajectory", flush=True)
    for rank, sid in enumerate(order, 1):
        raw_path = io23.path_for("EAST", sid)
        before = signature(raw_path)
        raw_hash = cfg["raw_sha256"][str(sid)]
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcome unavailable before seal"),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            times = raw["times"]
            ctx = model.native_context(base, prepare_actual(raw, rank, mapping))
            t_ref, p_ref = baseline_for(rank, sid, source_done)
            native = ctx[2]
            difference = float(np.max(abs(native - p_ref)))
            if (
                not np.array_equal(times, t_ref)
                or difference > ATOL
                or not np.array_equal(native >= THRESHOLD, p_ref >= THRESHOLD)
            ):
                raise ValueError(
                    f"new native does not match archived fixed trace: {sid}, max {difference:g}"
                )
            max_native_difference = max(max_native_difference, difference)
            traces, stats = model.predict(base, ctx, models)
            curves = {k: traces[k] if k in traces else native for k in ARMS}
        path = output / "shots" / f"{rank:04d}_{sid}.npz"
        save_arrays(path, times=times, **curves)
        curve_hash = sha(path)
        seal_path = path.with_name(path.stem + "_seal.json")
        write_new(
            seal_path,
            dict(
                shot=sid,
                rank=rank,
                raw_sha256=raw_hash,
                curve_sha256=curve_hash,
                active_fit_sha256=fit_hashes.copy(),
                updates=stats,
                current_outcome_used_in_prediction=False,
                native_difference=difference,
            ),
        )
        seal_hash = sha(seal_path)
        if signature(raw_path) != before or sha(raw_path) != raw_hash:
            raise ValueError(f"changed raw shot: {sid}")
        flag, down = io23.read_outcome("EAST", sid)
        if signature(raw_path) != before:
            raise ValueError(f"raw changed during reveal: {sid}")
        scored = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(times, v, flag, down, THRESHOLD),
            )
            for k, v in curves.items()
        }
        for k, v in scored.items():
            rows[k].append(v)
        reveal = path.with_name(path.stem + "_revealed.json")
        write_new(reveal, dict(shot=sid, after_seal_sha256=seal_hash, rows=scored))
        seals.append(
            dict(
                shot=sid,
                curve_sha256=curve_hash,
                seal_sha256=seal_hash,
                revealed_sha256=sha(reveal),
            )
        )
        if rank <= 20:
            pending.append((raw, flag, down, sid, rank))
            if rank == 20:
                for hist_raw, hist_flag, hist_down, hist_sid, hist_rank in pending:
                    historical = model.native_context(
                        base, prepare_actual(hist_raw, 21, mapping)
                    )
                    ep, samples = model.episode(
                        historical,
                        hist_raw["times"],
                        hist_flag,
                        hist_down,
                        hist_sid,
                        hist_rank,
                        keep_windows=True,
                    )
                    episodes.append(ep)
                    windows.append(samples)
                pending.clear()
        else:
            ep, samples = model.episode(
                ctx, times, flag, down, sid, rank, keep_windows=rank <= 100
            )
            episodes.append(ep)
            if samples is not None:
                windows.append(samples)
        if rank in (20, 100):
            fitted, fit_report = model.fit_budget(base, episodes, windows, THRESHOLD)
            ids = sorted(set(model.pack(episodes)["shot"].tolist()))
            if set(ids) != set(order[:rank]):
                raise ValueError(
                    "the historical fit cannot silently drop any ended shot"
                )
            tensors = {
                f"{arm}.{k}": v.detach().cpu().numpy()
                for arm, m in fitted.items()
                for k, v in m.state_dict().items()
            }
            fit_file = output / f"fit{rank}.npz"
            save_arrays(fit_file, **tensors)
            models[rank] = fitted
            fit_hashes[str(rank)] = sha(fit_file)
            fits[str(rank)] = dict(
                ids=ids,
                n_shots=len(ids),
                n_examples=sum(len(e["shot"]) for e in episodes),
                disruptive_shots=sum(r["disruptive"] for r in rows["static"]),
                after_all_historical_reveals=True,
                params_sha256=fit_hashes[str(rank)],
                trained=fit_report,
            )
            print("fitted", rank, fits[str(rank)]["n_examples"], fit_report, flush=True)
            if rank == 100:
                windows.clear()
        if rank % 25 == 0 or rank == len(order):
            print("sealed/revealed", rank, len(order), flush=True)
    if (
        state_digest(base) != digest
        or any(p.grad is not None for p in base.parameters())
        or any(sha(k) != v for k, v in inputs.items())
    ):
        raise AssertionError("frozen source or bound input changed")
    bank = model.pack(episodes)
    if set(bank["shot"].tolist()) != set(order):
        raise ValueError("complete diagnostic window coverage required")
    save_arrays(output / "latent_bank.npz", **bank)
    write_new(output / "fit_receipts.json", fits)
    write_new(output / "per_shot.json", rows)
    summary = sections(rows)
    write_new(output / "summary.json", summary)
    names = (
        "protocol.json",
        "fit20.npz",
        "fit100.npz",
        "latent_bank.npz",
        "fit_receipts.json",
        "per_shot.json",
        "summary.json",
    )
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LATENT_TRAJECTORY_ADAPT_DEVELOPMENT",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            max_native_difference=max_native_difference,
            artifact_sha256={k: sha(output / k) for k in names},
            sealed=seals,
            seconds=time.monotonic() - started,
        ),
    )
    print("complete", summary["full360"]["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
