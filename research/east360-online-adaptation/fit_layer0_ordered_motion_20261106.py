"""Train only on J-TEXT signals, then seal causal order-odd held predictions."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from fit_layer0_arrival_motion_20261106 import (
    BATCH,
    COUNT,
    EPOCHS,
    FIRST,
    SEED,
    STRIDE,
    read_training_signal,
)
import layer0_ordered_motion_20261106 as ordered
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = r.ROOT / "LAYER0_ORDERED_MOTION_PLAN_20261106.md"
H = ordered.motion.H
ARMS = ("zero", "persistence", "local", "joint", "reverse", "stale")


def origins_for(n, role):
    if role == "cold":
        o = np.arange(FIRST, n, STRIDE, dtype=np.int64) - H
        return o[o >= ordered.MIN_ORIGIN]
    if role not in ("train", "val"):
        raise ValueError("forecast role must be train, val or cold")
    lo, hi = ordered.MIN_ORIGIN, n - H - 1
    return (
        np.linspace(lo, hi, min(COUNT, hi - lo + 1), dtype=np.int64)
        if lo <= hi
        else np.empty(0, dtype=np.int64)
    )


def make_rows(v, a, sid, origins, role):
    if role not in ("train", "val", "cold"):
        raise ValueError("invalid forecast role")
    x, m = ordered.pair_windows(v, a, origins, source=role != "cold")
    rows = dict(
        x=x, a=m, shot=np.full(len(origins), sid, dtype=np.int64), origin=origins
    )
    if role == "train":
        j = origins[:, None] + np.arange(1, H + 1, dtype=np.int64)
        if (j >= len(v)).any():
            raise ValueError("incomplete future TRAIN target")
        rows.update(y=v[j], m=a[j])
    return rows


def observations(sid, role, cfg, mapping, hashes):
    if role == "cold":
        raw = raw_for(sid, cfg["raw_sha256"])
        data = r.prepare_actual(raw, 21, mapping)
        digest = cfg["raw_sha256"][str(sid)]
    else:
        data, digest = read_training_signal(sid, hashes)
    return data["values"], data["available"], digest


def collect(base, ids, hashes):
    parts, shot_index, inventory = [], [], []
    device = base.center.device
    with torch.no_grad():
        for i, sid in enumerate(ids):
            v, a, digest = observations(sid, "train", None, None, hashes)
            o = origins_for(len(v), "train")
            inventory.append(
                dict(shot=sid, raw_sha256=digest, n_points=len(v), n_forecasts=len(o))
            )
            if len(o):
                q = make_rows(v, a, sid, o, "train")
                x, m, y, f = [
                    torch.as_tensor(q[k], device=device) for k in ("x", "a", "y", "m")
                ]
                raw, _, history, end = ordered.history(
                    x[:, 0], m[:, 0], base.center, base.scale
                )
                actual, _ = ordered.context_pair(base, x, m)
                target, use = ordered.target(
                    y, f, end, history, base.center, base.scale
                )
                supported = use.any((1, 2))
                parts.append(tuple(z[supported] for z in (raw, actual, target, use)))
                shot_index.extend([sid] * int(supported.sum()))
            if (i + 1) % 160 == 0 or i + 1 == len(ids):
                print(f"ordered TRAIN shots: {i + 1}/{len(ids)}", flush=True)
    if not shot_index:
        raise ValueError("no supported ordered TRAIN targets")
    vals = tuple(torch.cat(v) for v in zip(*parts, strict=True))
    _, inv, n = np.unique(shot_index, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(
        1.0 / (len(n) * n[inv]), dtype=vals[0].dtype, device=device
    )
    return vals, weights, inventory, len(n)


def fit(data, weights):
    raw, actual, truth, support = data
    # Do not center: the model output must be odd under a reversed difference.
    rms = (actual.square() * weights[:, None]).sum(0).sqrt().clamp_min(1e-3)
    head = ordered.OrderedHead(rms)
    traces = {}
    for name, lr in (("local", 0.003), ("joint", 0.001)):
        head.local_weight.requires_grad_(name == "local")
        head.context.requires_grad_(name == "joint")
        optimizer = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )
        if name == "joint":
            with torch.no_grad():
                local = head.local(raw)
        rng = torch.Generator().manual_seed(SEED)
        curve = []
        for epoch in range(EPOCHS):
            total = 0.0
            for batch in torch.randperm(len(raw), generator=rng).split(BATCH):
                j = batch.to(raw.device)
                optimizer.zero_grad(set_to_none=True)
                pred = (
                    head.local(raw[j])
                    if name == "local"
                    else local[j] + head.context_motion(actual[j])
                )
                err, _ = ordered.masked_loss(pred, truth[j], support[j])
                loss = (err * weights[j]).sum()
                (loss * len(raw) / len(j)).backward()
                optimizer.step()
                total += float(loss.detach())
            curve.append(total)
            if epoch + 1 in (1, 8, 16, EPOCHS):
                print(f"ordered {name} epoch {epoch + 1}: {total:.8g}", flush=True)
        traces[name] = curve
    return head.eval().requires_grad_(False), traces


def load_head(path, device):
    with np.load(path, allow_pickle=False) as a:
        state = {k: torch.as_tensor(a[k], device=device) for k in a.files}
    head = ordered.OrderedHead(state["scale"])
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


@torch.no_grad()
def infer_rows(base, head, q):
    if set(q) != {"x", "a", "shot", "origin"}:
        raise ValueError("held inference received extra payload")
    x, a = [torch.as_tensor(q[k], device=base.center.device) for k in ("x", "a")]
    raw, prev, pm, end = ordered.history(x[:, 0], a[:, 0], base.center, base.scale)
    actual, earlier = ordered.context_pair(base, x, a)
    local = head.local(raw)
    delta = head.context_motion(actual)
    stale = head.context_motion(head.match_stale(actual, earlier))
    pred = torch.stack(
        (
            torch.zeros_like(prev),
            prev,
            local,
            local + delta,
            local - delta,
            local + stale,
        ),
        dim=1,
    )
    if not torch.isfinite(pred).all():
        raise FloatingPointError("non-finite held ordered motion predictions")
    return dict(
        shot=q["shot"],
        origin=q["origin"],
        predictions=pred.cpu().numpy(),
        origin_valid=end.cpu().numpy(),
        past_valid=pm.cpu().numpy(),
    )


def predict(base, head, ids, role, cfg, mapping, hashes):
    data, inventory, witness = [], [], []
    for i, sid in enumerate(ids):
        v, a, digest = observations(sid, role, cfg, mapping, hashes)
        o = origins_for(len(v), role)
        inventory.append(
            dict(shot=sid, raw_sha256=digest, n_points=len(v), n_forecasts=len(o))
        )
        if len(o):
            f = infer_rows(base, head, make_rows(v, a, sid, o, role))
            data.append(f)
            if role == "cold" and i in (0, 9, 19):
                count = (len(o) + 1) // 2
                cut = o[count - 1] + 1
                raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
                end = len(raw["values"])
                # Repeat the raw preprocessing with a shorter record, not only the encoder.
                cut_raw = {
                    k: x[:cut]
                    if isinstance(x, np.ndarray) and x.ndim and len(x) == end
                    else x
                    for k, x in raw.items()
                }
                d = r.prepare_actual(cut_raw, 21, mapping)
                short = infer_rows(
                    base,
                    head,
                    make_rows(d["values"], d["available"], sid, o[:count], role),
                )
                keys = ("origin", "shot", "past_valid", "origin_valid")
                if not all(np.array_equal(f[k][:count], short[k]) for k in keys):
                    raise ValueError(
                        "causal truncation changed held origins or support"
                    )
                p, q = f["predictions"][:count], short["predictions"]
                if not np.allclose(p, q, rtol=1e-5, atol=5e-6):
                    raise ValueError("unarrived suffix changed held predictions")
                witness.append(
                    dict(
                        shot=sid,
                        cutoff=int(cut),
                        n=count,
                        max_abs=float(np.max(np.abs(p - q))),
                    )
                )
    if not data:
        raise ValueError("empty held signal role")
    return (
        {k: np.concatenate([d[k] for d in data]) for k in data[0]},
        inventory,
        witness,
    )


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    cold, cfg, _, base, source, mapping, mpath, _ = parents()
    cold = cold[:20]
    roles, hashes = r.simple_data.source_roles(), Hashes()
    paths = [
        Path(__file__),
        PLAN,
        Path(ordered.__file__),
        Path(ordered.motion.__file__),
        Path(read_training_signal.__code__.co_filename),
        Path(parents.__code__.co_filename),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        r.io23.JTEXT_INDEX,
        r.DEFAULT_PARENT,
        mpath,
    ]
    binding = {str(p.resolve()): hashes(p) for p in paths}
    output.mkdir(parents=True)
    r.write_new(
        output / "protocol.json",
        dict(
            scope="OUTCOME_FREE_ORDER_ODD_MOTION_FIDELITY",
            training=roles["train"],
            validation=roles["val"],
            cold_signal_only=cold,
            held_test_n=len(roles["test"]),
            source_state_sha256=source,
            inputs=binding,
            mapping=str(mpath),
            arms=ARMS,
            forecast_horizon=H,
            window=ordered.motion.W,
            shape_lag=ordered.motion.LAG,
            pair_offset=ordered.OFFSET,
            start_origin=ordered.MIN_ORIGIN,
            epochs=EPOCHS,
            seed=SEED,
            outcomes_read=0,
            source_test_signals_read=0,
            source_risk_updates=0,
            first=FIRST,
            stride=STRIDE,
        ),
    )
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("ordered signal fit has no outcomes"),
    ):
        vals, w, inventory, count = collect(base, roles["train"], hashes)
        head, traces = fit(vals, w)
        report = dict(
            train=inventory,
            training_supported_shots=count,
            training_supported_rows=len(w),
            loss=traces,
        )
        del vals, w
        r.save_arrays(
            output / "head.npz",
            **{k: v.detach().cpu().numpy() for k, v in head.state_dict().items()},
        )
        r.save_arrays(
            output / "normalization.npz",
            center=base.center.cpu().numpy(),
            scale=base.scale.cpu().numpy(),
        )
        for name, ids in (("val", roles["val"]), ("cold", cold)):
            p, report[name], witness = predict(
                base, head, ids, name, cfg, mapping, hashes
            )
            if name == "cold":
                report["cold_raw_prefix_witness"] = witness
            r.save_arrays(output / f"{name}_predictions.npz", **p)
            print(
                f"sealed ordered {name}: {len(ids)} shots, {len(p['origin'])} origins",
                flush=True,
            )
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("ordered forecast fit altered source state")
    hashes.check()
    r.write_new(output / "inventory.json", report)
    artifacts = (
        "protocol.json",
        "inventory.json",
        "head.npz",
        "normalization.npz",
        "val_predictions.npz",
        "cold_predictions.npz",
    )
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_ORDERED_MOTION_SOURCE_AND_HELD_FORECASTS",
            artifact_sha256={k: hashes(output / k) for k in artifacts},
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
