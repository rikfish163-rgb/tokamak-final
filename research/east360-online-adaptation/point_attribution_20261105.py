"""Locked point-objective controls; no tuning or new fit on scored shots."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
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
from joint_lora_hindsight_capacity_20261104 import read, sha, timeline, write_new
from local_affine_reach_20261105 import COLD
import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import (
    FIT_AFTER,
    RIDGE,
    counts,
    flips,
    predict,
    reference,
)
from trajectory_teaching_loss_20261105 import teaching_loss

ROOT = io23.ROOT
OLD = ROOT / "probe_outputs/trajectory_teaching360_20261105_v1"
AUDIT = ROOT / "probe_outputs/trajectory_teaching360_verification_20261105_v1.json"
PLAN = ROOT / "POINT_ATTRIBUTION_PLAN_20261105.md"
MAP = dict(
    static="static",
    point_real="point_real",
    point_lagged="trajectory_lagged",
    point_reversed="trajectory_reversed",
    point_state="trajectory_state",
)


def fit_state_point(bank, tau):
    lead, valid, disruptive, weight = (
        torch.as_tensor(bank[k]) for k in ("lead", "valid", "disruptive", "shot_weight")
    )
    x, z, g = (
        torch.as_tensor(bank[k], dtype=torch.float64) for k in ("state", "z", "g")
    )
    count = valid.sum(1)
    if (count <= 0).any() or int(count.sum()) != len(x):
        raise ValueError("exact same nonempty historical teaching sample required")
    row_weight = torch.repeat_interleave(weight / count, count)
    rms = (row_weight @ x.square()).sqrt()
    scale = torch.where(rms > 1e-10, rms, 1.0)
    w = torch.zeros((84, 256), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        (w,),
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def loss():
        risk = z + (g * policy.action(x, scale, w)).sum(1)
        scores = torch.zeros_like(lead, dtype=torch.float64).masked_scatter(valid, risk)
        return (
            weight @ teaching_loss(scores, lead, disruptive, valid, tau, "point")
            + RIDGE / 2 * w.square().sum()
        )

    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        opt.zero_grad(set_to_none=True)
        obj = loss()
        obj.backward()
        return obj

    initial = float(loss().detach())
    opt.step(closure)
    final = float(loss().detach())
    if not np.isfinite(final) or final > initial + 1e-10:
        raise FloatingPointError("state-only control failed the same fit objective")
    return (scale.detach(), w.detach()), dict(
        before=initial, after=final, closure_calls=calls
    )


def frozen_point_models(point, state):
    # The generic forward runner's "trajectory" slots describe how to route
    # controls, not the training objective of the matrix passed to it here.
    return dict(point_real=point, trajectory_real=point, trajectory_state=state)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("fixed point attribution requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda")
    done, cfg, audit = (
        read(OLD / "completed.json"),
        read(OLD / "protocol.json"),
        read(AUDIT),
    )
    if (
        done["status"] != "COMPLETE_DELAYED_TRAJECTORY_TEACHING360"
        or audit["status"] != "PASS_RECOUNT_COMPLETE_TEACHING360"
        or audit["producer_completion_sha256"] != sha(OLD / "completed.json")
        or any(sha(OLD / k) != h for k, h in done["artifact_sha256"].items())
    ):
        raise ValueError("independently verified original six-arm run required")
    order, parent_protocol, _ = reference()
    if (
        cfg["order"] != order
        or cfg["threshold"] != parent_protocol["threshold"]
        or cfg["fit_indices"] != [21, 100]
    ):
        raise ValueError("identical order, threshold and teaching prefix required")
    threshold = cfg["threshold"]
    tau = float(np.log(threshold / (1 - threshold)))
    base, parent = load_parent(DEFAULT_PARENT)
    if (
        parent["threshold"] != threshold
        or state_digest(base) != cfg["fixed_source_state_sha256"]
    ):
        raise ValueError("unchanged source required")
    base = base.to(device)
    map_path = COLD / "input_map.json"
    mapping = read(map_path)
    if cfg["input_sha256"][str(map_path)] != sha(map_path):
        raise ValueError("unchanged prepared map required")
    bound = {
        str(Path(f).resolve()): sha(Path(f))
        for f in (
            Path(__file__),
            PLAN,
            OLD / "completed.json",
            OLD / "protocol.json",
            OLD / "fit.npz",
            AUDIT,
            Path(predict.__code__.co_filename),
            Path(policy.__file__),
            Path(teaching_loss.__code__.co_filename),
            Path(io23.__file__),
            Path(simple_data.__file__),
            DEFAULT_PARENT,
            map_path,
        )
    }
    with np.load(OLD / "fit.npz", allow_pickle=False) as bank:
        if bank["ids"].tolist() != order[20:100]:
            raise ValueError("only already-ended historical labels")
        point = tuple(
            torch.from_numpy(bank["point_real_" + k]) for k in ("scale", "matrix")
        )
        state, state_fit = fit_state_point(bank, tau)
    models = frozen_point_models(point, state)
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    with (output / "point_state_fit.npz").open("xb") as f:
        np.savez_compressed(f, scale=state[0].numpy(), matrix=state[1].numpy())
    write_new(
        output / "protocol.json",
        dict(
            scope="LOCKED_POINT_FEEDBACK_ATTRIBUTION_SUFFIX260_SHARED_FIRST100",
            input_sha256=bound,
            arms=MAP,
            order=order,
            threshold=threshold,
            fit_indices=[21, 100],
            independent_validation=False,
            prefix_uses_verified_original=True,
            target_label_seen_only_after_suffix_seal=True,
            state_fit=state_fit,
            max_radius=policy.RADIUS,
        ),
    )
    print("state-only point fit", state_fit, flush=True)
    rows, bindings = {k: [] for k in MAP}, []
    for rank in range(FIT_AFTER + 1, 361):
        sid = order[rank - 1]
        binding = done["sealed"][rank - 1]
        path = io23.path_for("EAST", sid)
        stat = path.stat()
        raw_hash = sha(path)
        if raw_hash != parent_protocol["raw_sha256"][str(sid)]:
            raise ValueError("fixed HDF source changed")
        reference_file = OLD / "shots" / f"{rank:04d}_{sid}.npz"
        if binding["shot"] != sid or sha(reference_file) != binding["curve_sha256"]:
            raise ValueError("original point curve changed")
        with np.load(reference_file, allow_pickle=False) as archive:
            t, old_p, previous_point = (
                archive[k].copy() for k in ("times", "static", "point_real")
            )
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        if not np.array_equal(raw["times"], t):
            raise ValueError("clock changed")
        signal = prepare_actual(raw, rank, mapping)
        pred, stats = predict(base, signal, old_p, tau, models, device)
        if not (
            np.array_equal(pred["point_real"], previous_point)
            and np.array_equal(pred["trajectory_real"], previous_point)
        ):
            raise ValueError("locked point arm must be bit-identical to original")
        traces = {k: pred[inner] for k, inner in MAP.items()}
        stem = output / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        with curve.open("xb") as f:
            np.savez_compressed(f, times=t, **traces)
        seal = stem.with_name(stem.name + "_seal.json")
        write_new(
            seal,
            dict(
                shot=sid,
                rank=rank,
                curve_sha256=sha(curve),
                raw_sha256=raw_hash,
                reference_curve_sha256=binding["curve_sha256"],
                stats=stats,
                current_outcome_used_in_prediction=False,
            ),
        )
        flag, down = io23.read_outcome("EAST", sid)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("HDF changed")
        judged = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(t, p, flag, down, threshold),
            )
            for k, p in traces.items()
        }
        for k, row in judged.items():
            rows[k].append(row)
        reveal = stem.with_name(stem.name + "_revealed.json")
        write_new(reveal, dict(shot=sid, after_seal_sha256=sha(seal), rows=judged))
        bindings.append(
            dict(
                shot=sid,
                curve_sha256=sha(curve),
                seal_sha256=sha(seal),
                revealed_sha256=sha(reveal),
            )
        )
        if rank % 20 == 0:
            print("point attribution sealed", rank, "of 360", flush=True)
    prefix = read(OLD / "per_shot.json")["static"][:FIT_AFTER]
    all_rows = {k: prefix + v for k, v in rows.items()}
    report = dict(
        full360={k: counts(v) for k, v in all_rows.items()},
        suffix260={k: counts(v) for k, v in rows.items()},
        paired={
            k: flips(all_rows["point_real"], v)
            for k, v in all_rows.items()
            if k != "point_real"
        },
    )
    if state_digest(base) != cfg["fixed_source_state_sha256"] or any(
        sha(Path(f)) != h for f, h in bound.items()
    ):
        raise ValueError("source or fixed inputs changed")
    write_new(output / "summary.json", report)
    write_new(output / "per_shot.json", all_rows)
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LOCKED_POINT_ATTRIBUTION",
            seconds=time.monotonic() - started,
            reexecuted_suffix_shots=len(bindings),
            sealed=bindings,
            reference_prefix_shots=FIT_AFTER,
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    run(p.parse_args().output)
