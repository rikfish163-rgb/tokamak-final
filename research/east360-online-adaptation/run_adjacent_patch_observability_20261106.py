"""See whether arrived forecast coordinates span a useful guarded risk direction."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import run_adjacent_patch_capacity_20261106 as cap
import run_adjacent_patch_intra_20261106 as a
from run_h25_late_capacity_20261106 import load_receipts
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = a.r.ROOT / "ADJACENT_PATCH_OBSERVABILITY_PROTOCOL_20261106.md"


def sample(cases):
    result = [q for q in cases if q["late"]]
    for role in a.old.ROLES:
        unique = {}
        for q in sorted(cases, key=lambda v: (v["shot"], v["maturity"])):
            if not q["late"] and q["role"] == role:
                unique.setdefault(q["shot"], q)
        xs = list(unique.values())
        if len(xs) < 12:
            raise ValueError("predeclared 12 distinct cases per role required")
        result.extend(xs[i] for i in np.linspace(0, len(xs) - 1, 12, dtype=np.int64))
    return result


def remove_history(history):
    q = a.old.geo.basis(history)
    return np.eye(q.shape[0]) - q @ q.T


def energy(q, state):
    n = np.linalg.norm(state)
    return float(np.linalg.norm(q.T @ state) / n) if n > 1e-10 else 0.0


def arrived_pairs(raw, m):
    y, observed = a.old.h25.arrived_targets(
        raw["values"], raw["available"], np.array([m - 32]), np.array([m])
    )
    valid = observed[0] & np.isfinite(y[0])
    keep = np.zeros(23, dtype=bool)
    for _, channels in a.feedback.GROUPS:
        keep[list(channels)] = True
    valid &= keep & (valid.sum(0) >= 13)
    return valid[8:] & valid[:-8]


def forecast_jacobian(base, packed, mode, observed):
    mask = torch.as_tensor(observed, device=packed.tokens.device)
    if not bool(mask.any()):
        raise ValueError("the sampled probe did not have observed increment targets")

    def forward(u):
        z = a.op.forward(base, packed, u[None], mode, forecast=True)["forecast"][0]
        return (z[8:] - z[:-8])[mask]

    u = packed.tokens.new_zeros(a.op.WIDTH)
    return (
        torch.autograd.functional.jacobian(forward, u, vectorize=True)
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def risk_gradient(base, packet, mode):
    u = packet.tokens.new_zeros((len(packet.tokens), a.op.WIDTH), requires_grad=True)
    z = a.op.forward(base, packet, u, mode)["logit"]
    (g,) = torch.autograd.grad(z.sum(), u)
    return g.detach().cpu().numpy().astype(np.float64)


def decompose(
    base,
    packed,
    native,
    raw,
    down,
    item,
    mode,
    candidate,
    real,
    reversed_space,
    visible,
):
    u = np.asarray(candidate["state"], dtype=np.float64)
    m = item["maturity"]
    clock = np.arange(m + 1, min(m + 33, len(native)))
    past = np.array([m - 16, m - 8, m], dtype=np.int64)
    eligible = np.asarray(item["eligible"])
    proj = real @ (real.T @ u)
    vectors = dict(
        full=u,
        in_aggregate=proj,
        outside_aggregate=u - proj,
        visible=visible @ (visible.T @ u),
    )
    result = {}
    for name, state in vectors.items():
        best, step = None, None
        for scale in a.pair.SCALES:
            hit = cap.verify_state(
                base,
                packed,
                mode,
                scale * state,
                native,
                raw["times"],
                down,
                past,
                clock,
                eligible,
            )
            if hit is not None:
                step, best = scale, hit
                break
        result[name] = dict(
            relative_norm=float(np.linalg.norm(state) / np.linalg.norm(u)),
            scale=step,
            witness=best,
        )
    return dict(
        motion=mode == 0,
        aggregate_fraction=energy(real, u),
        reversed_fraction=energy(reversed_space, u),
        visible_fraction=energy(visible, u),
        parts=result,
    )


def case(base, p, raw, native, item, group, arrived, history, verify, down):
    m = item["maturity"]
    origin = p.take([m - 32])
    j = p.take([item["target"]])
    output = []
    for mode in (0, 1):
        projector = remove_history(history[mode])
        teacher = projector @ risk_gradient(base, j, mode)[0]
        exact = forecast_jacobian(base, origin, mode, arrived) @ projector
        _, sv, vh = np.linalg.svd(exact, full_matrices=False)
        if not sv[0] > 0:
            raise ValueError("sampled origin is blind to all temporary updates")
        q, rev = [
            a.old.geo.basis(g @ projector) for g in group[2 * mode : 2 * mode + 2]
        ]
        rank = min(q.shape[1], rev.shape[1], int((sv >= sv[0] * 1e-6).sum()))
        q, rev = q[:, :rank], rev[:, :rank]
        coords = {k: vh[sv >= sv[0] * k].T for k in (0.01, 0.001)}
        info = dict(
            mode=a.op.MODES[mode],
            aggregate_rank=rank,
            n_pairs=int(arrived.sum()),
            teacher_norm=float(np.linalg.norm(teacher)),
            max_singular=float(sv[0]),
            rank_01=coords[0.01].shape[1],
            rank_001=coords[0.001].shape[1],
            aggregate=energy(q, teacher),
            reverse=energy(rev, teacher),
            pointwise_same_rank=energy(vh[:rank].T, teacher),
            stable_01=energy(coords[0.01], teacher),
            stable_001=energy(coords[0.001], teacher),
        )
        w = verify.get((item["shot"], m, mode))
        if w is not None:
            info["decomposition"] = decompose(
                base, p, native, raw, down, item, mode, w, q, rev, coords[0.01]
            )
        output.append(info)
    return dict(
        shot=item["shot"],
        maturity=m,
        role=item["role"],
        late=item["late"],
        modes=output,
    )


def candidate_witnesses(capacity):
    chosen = {}
    for case in capacity["rows"]:
        for i, arm in enumerate(("motion_full", "level_full")):
            candidates = [
                p
                for t in case["arms"][arm]
                for p in (t["initial"], t["raw"]["witness"], t["shape"]["witness"])
                if p is not None
                and p["peak"] >= a.TAU
                and p["full_first"]["timely"]
                and not p["equal_scalar_first"]["timely"]
            ]
            if candidates:
                chosen[(case["shot"], case["maturity"], i)] = max(
                    candidates, key=lambda p: p["peak"]
                )
    return chosen


def summarize(rows):
    ans = {}
    for role in ("positive", "protect", "safe", "late"):
        q = [
            v
            for v in rows
            if (v["late"] if role == "late" else v["role"] == role and not v["late"])
        ]
        ans[role] = {}
        ids = [x["shot"] for x in q]
        for j, mode in enumerate(a.op.MODES):
            records = [x["modes"][j] for x in q]
            keys = (
                "aggregate_rank",
                "rank_01",
                "rank_001",
                "aggregate",
                "reverse",
                "pointwise_same_rank",
                "stable_01",
                "stable_001",
            )
            delta = np.asarray(
                [x["pointwise_same_rank"] - x["aggregate"] for x in records]
            )
            ans[role][mode] = dict(
                **{k: a.old.shot_average([x[k] for x in records], ids) for k in keys},
                pointwise_minus_aggregate=a.paired_shot_effect(delta, ids),
            )
    return ans


def run(capacity_path, output):
    if output.exists():
        raise FileExistsError(output)
    h, started = Hashes(), time.monotonic()
    for path in (PLAN, Path(__file__), capacity_path):
        h(path)
    cfg, receipt, _ = load_receipts(a.PREVIOUS, h)
    cap_data = a.r.read(capacity_path)
    if not all(h(p) == sha for p, sha in cap_data["input_sha256"].items()):
        raise ValueError("capacity witnesses changed")
    verify = candidate_witnesses(cap_data)
    previous, inventory = a.bank.src.source_inputs(h)
    raw_sha = {q["shot"]: q["sha256"] for q in inventory}
    by_shot = defaultdict(list)
    for q in sample(receipt["cases"]):
        by_shot[q["shot"]].append(q)
    if not set(by_shot) <= set(previous["roles"]["train"]):
        raise ValueError("TRAIN-only observability diagnosis")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = a.r.load_parent(a.r.DEFAULT_PARENT)
    if a.r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("source encoder changed")
    projection = a.op.basis(base)
    rows = []
    for j, (sid, cases) in enumerate(by_shot.items(), 1):
        if h(a.r.io23.path_for("J-TEXT", sid)) != raw_sha[sid]:
            raise ValueError("original raw source changed")
        raw = a.r.simple_data.read_shot("J-TEXT", sid)
        x, available = a.bank.prior.views(base, raw)
        packed = a.op.encode(base, x, available, projection)
        native = a.bank.prior.scan(base, x, available).detach().cpu().numpy()
        clock, ix = np.unique([q["maturity"] for q in cases], return_inverse=True)
        g, history, support = a.gradients(base, packed, raw, clock)
        if not support.all():
            raise ValueError("matched 497-window study reported full common support")
        flag, down = a.r.io23.read_outcome("J-TEXT", sid)
        for q, i in zip(cases, ix, strict=True):
            if flag != (q["role"] != "safe"):
                raise ValueError("frozen control role disagrees with HDF5")
            pairs = arrived_pairs(raw, q["maturity"])
            rows.append(
                case(
                    base, packed, raw, native, q, g[i], pairs, history[i], verify, down
                )
            )
        if j % 10 == 0 or j == len(by_shot):
            print(f"Forecast observability {j}/{len(by_shot)} shots", flush=True)
    if a.r.state_digest(base) != cfg["source_state_sha256"] or any(
        x.grad is not None for x in base.parameters()
    ):
        raise ValueError("forecast Jacobian changed the model")
    h.check()
    ans = summarize(rows)
    a.r.write_new(
        output,
        dict(
            status="MEASURED_TRAIN_OBSERVABILITY_NOT_POLICY",
            rows=rows,
            summary=ans,
            input_sha256={str(k): v[1] for k, v in h.cache.items()},
            elapsed_seconds=time.monotonic() - started,
            new_val_test_east_reads=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capacity", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.capacity, args.out)
