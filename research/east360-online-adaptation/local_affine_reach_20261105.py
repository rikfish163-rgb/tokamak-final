"""Offline, depth-matched logit reach for an absolute-clock LayerNorm update."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)
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
import joint_lora_hindsight_capacity_20261104 as sealed

ROOT = io23.ROOT
OUT = ROOT / "probe_outputs/local_affine_reach_20261105_v1"
FRESH = ROOT / "probe_outputs/hdf5_outcome_authority_20261105_v1"
COLD = ROOT / "probe_outputs/stratified_cold20_context360_v1"
PLAN = ROOT / "LOCAL_AFFINE_REACH_PLAN_20261105.md"
ROUTES = (
    ("layer0_recent", 0, "recent"),
    ("layer1_recent", 1, "recent"),
    ("layer0_older", 0, "older"),
    ("layer1_cls", 1, "cls"),
)
RADII = (0.05, 0.50)


def arrival_for(queries):
    q = np.asarray(queries)
    if q.ndim != 1 or q.dtype.kind not in "iu" or (q < 135).any():
        raise ValueError("a scheduled observation must already have arrived")
    return 135 + ((q.astype(np.int64) - 135) // 32) * 32


def local_weights(available, queries, maturities, route):
    if (
        available.ndim != 3
        or available.shape[1:] != (128, 23)
        or available.dtype != torch.bool
    ):
        raise ValueError("128 by 23 availability required")
    if route not in ("recent", "older", "cls"):
        raise ValueError("fixed route required")
    q = torch.as_tensor(queries, device=available.device)
    m = torch.as_tensor(maturities, device=available.device)
    if (
        q.shape != (len(available),)
        or m.shape != q.shape
        or q.is_floating_point()
        or m.is_floating_point()
        or (m < 135).any()
        or ((m - 135) % 32 != 0).any()
        or (q < m).any()
        or (q >= m + 32).any()
    ):
        raise ValueError("use only the latest arrived feedback clock")
    ends = q[:, None] - 8 * torch.arange(15, -1, -1, device=q.device)[None]
    starts = ends - 7
    valid = available.reshape(len(available), 16, 8, 23).any(dim=(2, 3))
    recent = (starts >= m[:, None] - 31) & (ends <= m[:, None])
    older = (starts >= m[:, None] - 95) & (ends <= m[:, None] - 64)
    complete = (~(recent | older) | valid).all(1)
    if not ((recent.sum(1) >= 3) & (older.sum(1) >= 3)).all():
        raise ValueError("causal patch geometry must retain three or four tokens")
    if route == "cls":
        mask = torch.cat((torch.zeros_like(valid), complete[:, None]), dim=1)
    else:
        mask = torch.cat(
            (
                (recent if route == "recent" else older) & complete[:, None],
                torch.zeros_like(complete[:, None]),
            ),
            dim=1,
        )
    return mask.float() * (4 / mask.sum(1, keepdim=True).clamp_min(1)).sqrt()


def routed_logit(base, x, a, weights, delta, layer):
    if (
        layer not in (0, 1)
        or base.cfg.dim != 64
        or base.cfg.layers != 2
        or weights.shape != (len(x), 17)
        or delta.shape != (len(x), 128)
    ):
        raise ValueError("one fixed layer and a 128D affine state per example required")
    module = base.encoder.layers[layer].norm1
    dg, db = delta.chunk(2, dim=-1)
    shifts = []

    def intervene(norm, args, original):
        normalized = F.layer_norm(args[0], norm.normalized_shape, eps=norm.eps)
        update = weights[..., None] * (normalized * dg[:, None] + db[:, None])
        shifts.append(update)
        return original + update

    handle = module.register_forward_hook(intervene)
    try:
        z = base(x, a)["logit"]
    finally:
        handle.remove()
    if len(shifts) != 1:
        raise RuntimeError("expected exactly one norm1 intervention")
    return z, shifts[0].flatten(1).norm(dim=1)


def value_and_gradient(base, x, a, weights, layer):
    delta = x.new_zeros((len(x), 128), requires_grad=True)
    z, _ = routed_logit(base, x, a, weights, delta, layer)
    (g,) = torch.autograd.grad(z.sum(), (delta,))
    if not torch.isfinite(z).all() or not torch.isfinite(g).all():
        raise FloatingPointError("nonfinite logit or local Jacobian")
    return z.detach(), g.detach()


def trial(base, signal, queries, sign):
    rows = {name: [] for name, _, _ in ROUTES}
    bare = []
    for batch in np.array_split(queries, (len(queries) + 63) // 64):
        x, a = window_batch(signal, batch, device="cpu")
        clocks = arrival_for(batch)
        with torch.no_grad():
            original = base(x, a)["logit"]
        bare.append(original.numpy())
        for name, layer, route in ROUTES:
            weights = local_weights(a, batch, clocks, route)
            z, g = value_and_gradient(base, x, a, weights, layer)
            if not torch.equal(z, original):
                raise ValueError("zero affine state changed the parent logits")
            length = g.norm(dim=1)
            supported = weights.any(dim=1)
            if torch.any(~supported & (length != 0)):
                raise ValueError("an unsupported intervention has a gradient")
            unit = sign * g / torch.where(length > 0, length, 1.0)[:, None]
            fields = [supported.float(), length]
            with torch.no_grad():
                for radius in RADII:
                    after, energy = routed_logit(
                        base, x, a, weights, unit * radius, layer
                    )
                    fields.extend([after, energy])
            rows[name].append(torch.stack(fields, dim=1).numpy())
    return np.concatenate(bare), {k: np.concatenate(v) for k, v in rows.items()}


def shot_summary(query, baseline, table, kind, lead, threshold):
    result = {}
    for name, values in table.items():
        entry = dict(supported=int(values[:, 0].sum()))
        for i, radius in enumerate(RADII):
            z = values[:, 2 + 2 * i].astype(np.float64)
            energy = values[:, 3 + 2 * i]
            prob = 1 / (1 + np.exp(-z))
            cross = (
                (prob >= threshold) if kind != "fp_first_point" else (prob < threshold)
            )
            if not np.isfinite(values).all() or not np.isfinite(prob).all():
                raise FloatingPointError("finite oracle result required")
            entry[str(radius)] = dict(
                crosses=bool(cross.any()),
                crossings=int(cross.sum()),
                first_cross_query=int(query[cross][0]) if cross.any() else None,
                first_cross_lead_ms=float(lead[cross][0])
                if lead is not None and cross.any()
                else None,
                max_signed_logit_change=float(
                    ((z - baseline) * (1 if lead is not None else -1)).max()
                ),
                max_first_order_signed_change=float(radius * values[:, 1].max()),
                max_activation_l2=float(energy.max()),
            )
        result[name] = entry
    return result


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    protocol, done, ids, meta = sealed.verify_inputs()
    threshold = protocol["threshold"]
    if sealed.sha(FRESH / "hdf5_metadata.json") != sealed.V5_METADATA_SHA:
        raise ValueError(
            "fresh HDF5 reconciliation and sealed-source inventory disagree"
        )
    files = (
        Path(__file__),
        PLAN,
        Path(sealed.__file__),
        DEFAULT_PARENT,
        Path(window_batch.__code__.co_filename),
        Path(prepare_actual.__code__.co_filename),
        Path(io23.__file__),
        Path(simple_data.__file__),
        COLD / "input_map.json",
        COLD / "input_hashes.json",
        sealed.SOURCE / "completed.json",
        sealed.SOURCE / "protocol.json",
        FRESH / "hdf5_metadata.json",
    )
    bindings = {str(f.resolve()): sealed.sha(f) for f in files}
    mapping = sealed.read(COLD / "input_map.json")
    hashes = sealed.read(COLD / "input_hashes.json")
    if mapping["calibration_ids"] != ids[:20]:
        raise ValueError("unchanged causal prefix mapping required")
    base, parent = load_parent(DEFAULT_PARENT)
    state = state_digest(base)
    if parent["threshold"] != threshold:
        raise ValueError("unchanged threshold required")
    OUT.mkdir()
    sealed.write_new(
        OUT / "protocol.json",
        dict(
            scope="POSTHOC_PER_ENDPOINT_LOCAL_NORM1_DIRECTION_REACH_NOT_ONLINE",
            input_sha256=bindings,
            order=ids,
            radii=RADII,
            routes=ROUTES,
            threshold=threshold,
            first_possible_arrival=135,
            stride=32,
            first20_frozen=True,
            no_training_or_label_based_deployable_policy=True,
            point_fields=[
                "support",
                "grad_norm",
                "z_after_.05",
                "energy_.05",
                "z_after_.5",
                "energy_.5",
            ],
        ),
    )
    points, records = [], []
    for index, sid in enumerate(ids[20:], 21):
        times, probs, seal = sealed.sealed_curve(index, sid, done, meta)
        truth = meta[sid]
        disruptive, down = truth["disruptive"], truth["down"]
        record = sealed.timeline(times, probs, disruptive, down, threshold)
        kind = record["status"]
        if kind in ("miss", "tardy"):
            lead_all = np.round((down - times) * 1000, 9)
            query = np.flatnonzero(
                (lead_all > 30) & (lead_all <= 125) & (np.arange(len(times)) >= 135)
            )
            if not len(query):
                raise ValueError("sealed failure has no scheduled positive window")
            lead = lead_all[query]
        elif kind == "fp" and record["first_index"] >= 135:
            query = np.array([record["first_index"]], dtype=np.int64)
            lead, kind = None, "fp_first_point"
        else:
            continue
        path = io23.path_for("EAST", sid)
        stat = path.stat()
        raw_sha = sealed.sha(path)
        if raw_sha != hashes[str(sid)] or (stat.st_size, stat.st_mtime_ns) != (
            truth["size"],
            truth["mtime_ns"],
        ):
            raise ValueError(f"changed original HDF: {sid}")
        signal = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        if io23.read_outcome("EAST", sid) != (disruptive, down) or not np.array_equal(
            signal["times"], times
        ):
            raise ValueError(f"HDF truth or original sample clock changed: {sid}")
        signal = prepare_actual(signal, index, mapping)
        z0, table = trial(base, signal, query, 1 if disruptive else -1)
        stat_after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise ValueError(f"raw HDF changed during read: {sid}")
        error = np.abs(1 / (1 + np.exp(-z0.astype(np.float64))) - probs[query])
        if float(error.max()) > 3e-6:
            raise ValueError(f"fixed-source logit parity failed: {sid}")
        points.append((sid, query, z0, probs[query], table))
        records.append(
            dict(
                shot=sid,
                stream_index=index,
                kind=kind,
                candidates=len(query),
                raw_sha256=raw_sha,
                curve_sha256=seal["curve_sha256"],
                max_source_parity_error=float(error.max()),
                routes=shot_summary(query, z0, table, kind, lead, threshold),
            )
        )
        if len(records) % 10 == 0:
            print(
                "finished selected shots",
                len(records),
                "last stream rank",
                index,
                flush=True,
            )
    tally = {
        g: sum(r["kind"] == g for r in records)
        for g in ("miss", "tardy", "fp_first_point")
    }
    if tally != {"miss": 26, "tardy": 37, "fp_first_point": 19}:
        raise ValueError(f"fixed failure strata changed: {tally}")
    summary = {}
    for g, n in tally.items():
        rr = [r for r in records if r["kind"] == g]
        summary[g] = dict(
            shots=n,
            points=sum(r["candidates"] for r in rr),
            routes={
                name: {
                    str(radius): dict(
                        crossing_shots=sum(
                            r["routes"][name][str(radius)]["crosses"] for r in rr
                        ),
                        crossing_points=sum(
                            r["routes"][name][str(radius)]["crossings"] for r in rr
                        ),
                        median_shot_max_signed_logit=float(
                            np.median(
                                [
                                    r["routes"][name][str(radius)][
                                        "max_signed_logit_change"
                                    ]
                                    for r in rr
                                ]
                            )
                        ),
                        median_shot_max_activation_l2=float(
                            np.median(
                                [
                                    r["routes"][name][str(radius)]["max_activation_l2"]
                                    for r in rr
                                ]
                            )
                        ),
                    )
                    for radius in RADII
                }
                for name, _, _ in ROUTES
            },
        )
    if state != state_digest(base) or any(
        sealed.sha(Path(f)) != h for f, h in bindings.items()
    ):
        raise ValueError("source parameters or bound inputs changed")
    np.savez_compressed(
        OUT / "points.npz",
        shot=np.concatenate([np.full(len(q), sid) for sid, q, _, _, _ in points]),
        query=np.concatenate([p[1] for p in points]),
        baseline_logit=np.concatenate([p[2] for p in points]),
        baseline_probability=np.concatenate([p[3] for p in points]),
        **{name: np.concatenate([p[4][name] for p in points]) for name, _, _ in ROUTES},
    )
    sealed.write_new(OUT / "per_shot.json", records)
    sealed.write_new(OUT / "summary.json", summary)
    sealed.write_new(
        OUT / "completed.json",
        dict(
            status="COMPLETE_POSTHOC_LOCAL_AFFINE_REACH",
            shots=len(records),
            points=sum(r["candidates"] for r in records),
            seconds=time.monotonic() - started,
            artifact_sha256={f.name: sealed.sha(f) for f in OUT.iterdir()},
        ),
    )
    print(
        {
            g: {
                n: {r: d["crossing_shots"] for r, d in v.items()}
                for n, v in s["routes"].items()
            }
            for g, s in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
