"""Causal prior-shot and first-arrival controls for the locked point model."""

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
    BATCH,
    FIT_AFTER,
    counts,
    features_at_arrival,
    flips,
    make_views,
    reference,
    routes_for,
)

ROOT = io23.ROOT
POINT = ROOT / "probe_outputs/trajectory_point_attribution_20261105_v1"
TEACH = ROOT / "probe_outputs/trajectory_teaching360_20261105_v1"
AUDIT = (
    ROOT / "probe_outputs/trajectory_point_attribution_verification_20261105_v1.json"
)
PLAN = ROOT / "POINT_DONOR_CONTROLS_PLAN_20261105.md"
MATCHED = ("previous_clock", "previous_state", "first_direction")
ARMS = ("static", "point_real", *MATCHED, "first_full")


def targets(real, previous, current_state, previous_state, scale):
    if (
        real.ndim != 2
        or previous.ndim != 2
        or not len(real)
        or not len(previous)
        or previous.shape[1] != real.shape[1]
        or scale.ndim != 1
        or current_state.shape != (len(real), len(scale))
        or previous_state.shape != (len(previous), len(scale))
    ):
        raise ValueError("nonempty compatible current and ended donor arrays required")
    if (
        not all(
            torch.isfinite(x).all()
            for x in (real, previous, current_state, previous_state, scale)
        )
        or not (scale > 0).all()
    ):
        raise ValueError("finite directions, states and positive scale required")
    clock = torch.arange(len(real), device=real.device).clamp_max(len(previous) - 1)
    nearest = torch.cdist(
        current_state / scale,
        previous_state / scale,
        compute_mode="donot_use_mm_for_euclid_dist",
    ).argmin(dim=1)
    controls = dict(
        previous_clock=policy.match_norm(previous[clock], real),
        previous_state=policy.match_norm(previous[nearest], real),
        first_direction=policy.match_norm(real[:1].expand_as(real), real),
        first_full=real[:1].clone(),
    )
    return controls, clock, nearest


@torch.no_grad()
def action_context(base, signal, original, tau, point, device):
    x, a = make_views(signal, device)
    feats = features_at_arrival(base, x, a, original, tau)
    return x, a, policy.action(feats["real"], *point), feats["state"]


def predict(base, x, a, original, real, controls):
    fast = dict(point_real=policy.write_fast(real))
    fast.update({k: policy.write_fast(controls[k]) for k in MATCHED})
    first = policy.write_fast(controls["first_full"])
    fast["first_full"] = first.expand_as(real)
    n = len(real)
    arrays = {k: original.copy() for k in ARMS}
    for start in range(policy.FIRST, len(x), BATCH):
        end = min(start + BATCH, len(x))
        q = np.arange(start, end, dtype=np.int64)
        v, mask = x[start:end], a[start:end]
        weights = routes_for(mask, q)
        idx = torch.as_tensor((q - policy.FIRST) // policy.STRIDE, device=x.device)
        with torch.no_grad():
            for name, state in fast.items():
                arrays[name][start:end] = (
                    policy.routed_pair(base, v, mask, weights, state[idx])
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
    if any(
        not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in arrays.values()
    ):
        raise FloatingPointError("valid probabilities required")
    norm = {k: v.norm(dim=1).cpu().numpy() for k, v in fast.items()}
    if any(len(v) != n or (v > policy.RADIUS + 1e-6).any() for v in norm.values()):
        raise ValueError("bounded temporary state required")
    return (
        arrays,
        norm,
        dict(
            n_arrivals=n,
            nonzero_optimizer_writes={
                k: int((v > 0).sum()) if k != "first_full" else int(v[0] > 0)
                for k, v in norm.items()
            },
            norm_mismatches={
                k: int((abs(norm[k] - norm["point_real"]) > 1e-6).sum())
                for k in MATCHED
            },
        ),
    )


def sealed_signal(rank, sid, protocol, mapping, done, directory):
    stem = directory / "shots" / f"{rank:04d}_{sid}"
    path = stem.with_suffix(".npz")
    binding = done["sealed"][rank - 1 if directory == TEACH else rank - FIT_AFTER - 1]
    if (
        binding["shot"] != sid
        or sha(path) != binding["curve_sha256"]
        or sha(stem.with_name(stem.name + "_seal.json")) != binding["seal_sha256"]
    ):
        raise ValueError(f"original forecast seal mismatch: {sid}")
    with np.load(path, allow_pickle=False) as f:
        t, p, old = [f[k].copy() for k in ("times", "static", "point_real")]
    raw_path = io23.path_for("EAST", sid)
    before = raw_path.stat()
    digest = sha(raw_path)
    if digest != protocol["raw_sha256"][str(sid)]:
        raise ValueError("raw file changed")
    raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
    if not np.array_equal(t, raw["times"]):
        raise ValueError("original time coordinates changed")
    return prepare_actual(raw, rank, mapping), t, p, old, raw_path, before, digest


def run(output):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("fixed replay requires CUDA")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda")
    done, cfg, teach_done, audit = [
        read(p)
        for p in (
            POINT / "completed.json",
            POINT / "protocol.json",
            TEACH / "completed.json",
            AUDIT,
        )
    ]
    if (
        done["status"] != "COMPLETE_LOCKED_POINT_ATTRIBUTION"
        or audit["status"] != "PASS_RECOUNT_LOCKED_POINT_ATTRIBUTION"
        or audit["producer_completion_sha256"] != sha(POINT / "completed.json")
        or any(sha(POINT / k) != h for k, h in done["artifact_sha256"].items())
        or any(sha(TEACH / k) != h for k, h in teach_done["artifact_sha256"].items())
    ):
        raise ValueError("original verified point replay and model bank required")
    order, parent_protocol, _ = reference()
    threshold = cfg["threshold"]
    tau = float(np.log(threshold / (1 - threshold)))
    if (
        order != cfg["order"]
        or threshold != parent_protocol["threshold"]
        or cfg["fit_indices"] != [21, 100]
    ):
        raise ValueError("fixed point order/threshold/history required")
    base, parent = load_parent(DEFAULT_PARENT)
    if (
        parent["threshold"] != threshold
        or state_digest(base) != parent_protocol["source_state_sha256"]
    ):
        raise ValueError("unchanged source model required")
    base = base.to(device)
    map_path = COLD / "input_map.json"
    mapping = read(map_path)
    bank_path = TEACH / "fit.npz"
    if any(sha(f) != cfg["input_sha256"][str(f)] for f in (map_path, bank_path)):
        raise ValueError("unchanged point matrix and alignment required")
    with np.load(bank_path, allow_pickle=False) as f:
        point = tuple(
            torch.as_tensor(f["point_real_" + k], device=device, dtype=torch.float32)
            for k in ("scale", "matrix")
        )
    with np.load(POINT / "point_state_fit.npz", allow_pickle=False) as f:
        scale = torch.as_tensor(f["scale"], device=device, dtype=torch.float32)
    files = (
        Path(__file__),
        PLAN,
        DEFAULT_PARENT,
        map_path,
        bank_path,
        POINT / "completed.json",
        POINT / "protocol.json",
        POINT / "point_state_fit.npz",
        TEACH / "completed.json",
        AUDIT,
        Path(policy.__file__),
        Path(features_at_arrival.__code__.co_filename),
        Path(io23.__file__),
        Path(simple_data.__file__),
        Path(prepare_actual.__code__.co_filename),
    )
    hashes = {str(f.resolve()): sha(f) for f in files}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write_new(
        output / "protocol.json",
        dict(
            scope="LOCKED_POINT_DONOR_AND_SINGLE_WRITE_DEVELOPMENT",
            arms=ARMS,
            matched=tuple(MATCHED),
            threshold=threshold,
            order=order,
            reference_prefix_shots=100,
            fit_indices=[21, 100],
            hdf5_truth_reveal_after_seal=True,
            independent_validation=False,
            model_training=False,
            previous_after_end="hold_last_clock",
            first_anchor=policy.FIRST,
            stride=policy.STRIDE,
            radius=policy.RADIUS,
            input_sha256=hashes,
        ),
    )
    sid0 = order[FIT_AFTER - 1]
    last, _, prefix_p, _, _, _, _ = sealed_signal(
        FIT_AFTER, sid0, parent_protocol, mapping, teach_done, TEACH
    )
    _, _, previous, previous_state = action_context(
        base, last, prefix_p, tau, point, device
    )
    donor = sid0
    rows = {k: [] for k in ARMS}
    receipts = []
    for rank, sid in enumerate(order[FIT_AFTER:], FIT_AFTER + 1):
        signal, t, p, old, raw_path, stat, digest = sealed_signal(
            rank, sid, parent_protocol, mapping, done, POINT
        )
        x, a, real, state = action_context(base, signal, p, tau, point, device)
        controls, clock, nearest = targets(real, previous, state, previous_state, scale)
        traces, norm, stats = predict(base, x, a, p, real, controls)
        if not np.array_equal(traces["point_real"], old):
            raise ValueError("real point arm is not bit-identical")
        stem = output / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        control = stem.with_name(stem.name + "_controls.npz")
        with curve.open("xb") as f:
            np.savez_compressed(f, times=t, **traces)
        with control.open("xb") as f:
            np.savez_compressed(
                f, clock=clock.cpu().numpy(), nearest=nearest.cpu().numpy(), **norm
            )
        seal = stem.with_name(stem.name + "_seal.json")
        write_new(
            seal,
            dict(
                rank=rank,
                shot=sid,
                donor=donor,
                donor_rank=rank - 1,
                donor_n_arrivals=len(previous),
                raw_sha256=digest,
                curve_sha256=sha(curve),
                controls_sha256=sha(control),
                stats=stats,
                current_outcome_used_in_prediction=False,
            ),
        )
        flag, down = io23.read_outcome("EAST", sid)
        after = raw_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("HDF changed during prediction")
        judged = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(t, v, flag, down, threshold),
            )
            for k, v in traces.items()
        }
        for k, row in judged.items():
            rows[k].append(row)
        reveal = stem.with_name(stem.name + "_revealed.json")
        write_new(reveal, dict(shot=sid, after_seal_sha256=sha(seal), rows=judged))
        receipts.append(
            dict(
                shot=sid,
                curve_sha256=sha(curve),
                controls_sha256=sha(control),
                seal_sha256=sha(seal),
                revealed_sha256=sha(reveal),
            )
        )
        previous, previous_state, donor = real, state, sid
        if rank % 20 == 0:
            print("sealed donor controls", rank, "of 360", flush=True)
    first = read(POINT / "per_shot.json")["static"][:FIT_AFTER]
    all_rows = {k: first + v for k, v in rows.items()}
    result = dict(
        full360={k: counts(v) for k, v in all_rows.items()},
        suffix260={k: counts(v) for k, v in rows.items()},
        paired={
            k: flips(all_rows["point_real"], v)
            for k, v in all_rows.items()
            if k != "point_real"
        },
    )
    if state_digest(base) != parent_protocol["source_state_sha256"] or any(
        sha(Path(f)) != h for f, h in hashes.items()
    ):
        raise ValueError("source or locked inputs changed")
    write_new(output / "summary.json", result)
    write_new(output / "per_shot.json", all_rows)
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_POINT_DONOR_CONTROLS",
            seconds=time.monotonic() - started,
            reference_prefix_shots=FIT_AFTER,
            reexecuted_suffix_shots=len(receipts),
            sealed=receipts,
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    run(ap.parse_args().output)
