"""Fit source-only direct motion residuals and seal outcome-free held forecasts."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer0_arrival_motion_20261106 as motion
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = r.ROOT / "LAYER0_ARRIVAL_MOTION_PLAN_20261106.md"
FIRST, STRIDE, SEED = 135, 32, 131
COUNT, BATCH, EPOCHS = 16, 256, 24
ARMS = ("hold", "linear", "native", "local", "joint")


def source_origins(n):
    start, stop = motion.W - 1, n - motion.H - 1
    return (
        np.linspace(start, stop, min(COUNT, stop - start + 1), dtype=np.int64)
        if stop >= start
        else np.empty(0, dtype=np.int64)
    )


def past_arrays(values, mask, origins):
    if (
        values.ndim != 2
        or values.shape[1] != motion.C
        or mask.shape != values.shape
        or mask.dtype != bool
        or origins.ndim != 1
        or origins.dtype.kind not in "iu"
        or (origins < 0).any()
        or (origins >= len(values)).any()
    ):
        raise ValueError("causal windows require valid observations and origin indices")
    ix = origins[:, None] + np.arange(1 - motion.W, 1, dtype=np.int64)
    safe = np.maximum(ix, 0)
    return np.where((ix >= 0)[:, :, None], values[safe], 0.0), mask[safe] & (ix >= 0)[
        :, :, None
    ]


def source_mask(windows_mask, parity):
    m = windows_mask.copy()
    m[..., 2] = False  # MNM is not available in the EAST main signal contract.
    m[np.asarray(parity) % 2 == 1, :, 1] = False  # fixed CIII-view augmentation
    return m


def read_training_signal(sid, h):
    path = r.io23.path_for("J-TEXT", sid)
    before = h(path)
    raw = r.simple_data.read_shot("J-TEXT", sid)
    if raw["shot"] != sid or raw["device"] != "J-TEXT" or before != h(path):
        raise ValueError("source diagnostic record changed")
    return raw, before


def gather(raw, values, mask, origins, *, source, include_target):
    x, a = past_arrays(values, mask, origins)
    if source:
        a = source_mask(a, np.arange(len(origins)))
    rows = dict(
        x=x,
        a=a,
        shot=np.full(len(origins), raw["shot"], dtype=np.int64),
        origin=origins,
    )
    if include_target:
        j = origins[:, None] + np.arange(1, motion.H + 1)
        rows.update(y=values[j], m=mask[j])
    return rows


def batches(ids, role, cfg, mapping, h, *, include_target):
    """Only offline TRAIN returns future targets. Held prediction exposes past arrays only."""
    if (include_target and role != "train") or role not in ("train", "val", "cold"):
        raise ValueError("only TRAIN may provide future signal to the fitter")
    for lo in range(0, len(ids), 16):
        chunks, inventory = [], []
        for sid in ids[lo : lo + 16]:
            if role == "cold":
                raw = raw_for(sid, cfg["raw_sha256"])
                digest = cfg["raw_sha256"][str(sid)]
                signal = r.prepare_actual(raw, 21, mapping)
                v, a = signal["values"], signal["available"]
                origins = np.arange(FIRST, len(v), STRIDE, dtype=np.int64) - motion.H
            else:
                raw, digest = read_training_signal(sid, h)
                v, a = raw["values"], raw["available"]
                origins = source_origins(len(v))
            inventory.append(
                dict(
                    shot=sid,
                    raw_sha256=digest,
                    n_points=len(v),
                    n_forecasts=len(origins),
                )
            )
            if len(origins):
                chunks.append(
                    gather(
                        raw,
                        v,
                        a,
                        origins,
                        source=role != "cold",
                        include_target=include_target,
                    )
                )
        batch = (
            {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
            if chunks
            else {}
        )
        yield batch, inventory


def collect_training(base, ids, cfg, mapping, h):
    all_batches, shots, inventory = [], [], []
    device = base.center.device
    with torch.no_grad():
        for q, rows in batches(ids, "train", cfg, mapping, h, include_target=True):
            inventory.extend(rows)
            if q:
                for i in range(0, len(q["x"]), BATCH):
                    j = slice(i, i + BATCH)
                    x, a, y, m = [
                        torch.as_tensor(q[k][j], device=device)
                        for k in ("x", "a", "y", "m")
                    ]
                    hold, _, local, observed = motion.origin_terms(
                        x, a, base.center, base.scale
                    )
                    ctx, _ = motion.features(base, x, a)
                    target, valid = motion.target_terms(
                        y, m, observed, base.center, base.scale
                    )
                    supported = motion.row_loss(hold, target, valid)["supported"]
                    all_batches.append(
                        tuple(v[supported] for v in (hold, local, ctx, target, valid))
                    )
                    shots.extend(q["shot"][j][supported.cpu().numpy()].tolist())
            if len(inventory) % 160 == 0 or len(inventory) == len(ids):
                print(f"read J-TEXT TRAIN {len(inventory)}/{len(ids)}", flush=True)
    if not shots:
        raise ValueError("no paired TRAIN motion targets")
    values = tuple(torch.cat(v) for v in zip(*all_batches, strict=True))
    _, inv, cnt = np.unique(shots, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(
        1.0 / (len(cnt) * cnt[inv]), dtype=values[0].dtype, device=device
    )
    if not torch.isclose(weights.sum(), torch.ones((), device=device)):
        raise ValueError("equal-weight TRAIN shot accounting failed")
    return values, weights, inventory, len(cnt)


def fit(values, weight):
    hold, local, ctx, y, m = values
    mu = (ctx * weight[:, None]).sum(0)
    sd = ((ctx - mu).square() * weight[:, None]).sum(0).sqrt().clamp_min(1e-3)
    head = motion.MotionHead(mu, sd).to(hold.device)
    report = {}
    for phase, lr in (("local", 0.003), ("joint", 0.001)):
        head.local_weight.requires_grad_(phase == "local")
        head.context.requires_grad_(phase == "joint")
        optimizer = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )
        rng = torch.Generator().manual_seed(SEED)
        curve = []
        for epoch in range(EPOCHS):
            avg = 0.0
            for ids in torch.randperm(len(hold), generator=rng).split(BATCH):
                j = ids.to(hold.device)
                optimizer.zero_grad(set_to_none=True)
                pred = (
                    head.local(hold[j], local[j])
                    if phase == "local"
                    else head(hold[j], local[j], ctx[j])
                )
                err = motion.row_loss(pred, y[j], m[j])["combined"]
                loss = (err * weight[j]).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite direct motion fit")
                (loss * len(hold) / len(j)).backward()
                optimizer.step()
                avg += float(loss.detach())
            curve.append(avg)
            if (epoch + 1) in (1, 8, 16, EPOCHS):
                print(f"motion {phase} epoch {epoch + 1}: {avg:.7g}", flush=True)
        report[phase] = curve
    head.eval().requires_grad_(False)
    for p in head.parameters():
        p.grad = None
    return head, report


def load_head(path, device):
    with np.load(path, allow_pickle=False) as a:
        state = {k: torch.as_tensor(a[k], device=device) for k in a.files}
    head = motion.MotionHead(state["mean"], state["scale"]).to(device)
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


@torch.no_grad()
def predict_split(base, head, ids, role, cfg, mapping, h):
    out, inventory = [], []
    device = base.center.device
    for q, rows in batches(ids, role, cfg, mapping, h, include_target=False):
        inventory.extend(rows)
        if not q:
            continue
        if set(q) != {"x", "a", "shot", "origin"}:
            raise ValueError("held prediction payload includes future information")
        for i in range(0, len(q["x"]), BATCH):
            j = slice(i, i + BATCH)
            x, a = [torch.as_tensor(q[k][j], device=device) for k in ("x", "a")]
            hold, linear, local, observed = motion.origin_terms(
                x, a, base.center, base.scale
            )
            ctx, native = motion.features(base, x, a)
            y = torch.stack(
                (hold, linear, native, head.local(hold, local), head(hold, local, ctx)),
                dim=1,
            )
            if not torch.isfinite(y).all():
                raise FloatingPointError("nonfinite sealed forecast")
            out.append(
                dict(
                    shot=q["shot"][j],
                    origin=q["origin"][j],
                    predictions=y.cpu().numpy(),
                    origin_valid=observed.cpu().numpy(),
                )
            )
    if not out:
        raise ValueError("empty held diagnostic role")
    return {k: np.concatenate([v[k] for v in out]) for k in out[0]}, inventory


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, cfg, _, base, source, mapping, map_path, _ = parents()
    roles = r.simple_data.source_roles()
    h = Hashes()
    deps = [
        Path(__file__),
        PLAN,
        Path(motion.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        r.io23.JTEXT_INDEX,
        r.DEFAULT_PARENT,
        map_path,
    ]
    binding = {str(p.resolve()): h(p) for p in deps}
    output.mkdir(parents=True)
    r.write_new(
        output / "protocol.json",
        dict(
            scope="OUTCOME_FREE_DIRECT_25_MOTION_FIDELITY",
            training=roles["train"],
            validation=roles["val"],
            cold_signal_only=order[:20],
            held_test_n=len(roles["test"]),
            source_state_sha256=source,
            inputs=binding,
            mapping=str(map_path),
            arms=ARMS,
            forecast_horizon=motion.H,
            window=motion.W,
            shape_lag=motion.LAG,
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
        side_effect=RuntimeError("motion fidelity may never read outcomes"),
    ):
        values, weights, train, support = collect_training(
            base, roles["train"], cfg, mapping, h
        )
        head, losses = fit(values, weights)
        training_rows = len(weights)
        del values, weights
        r.save_arrays(
            output / "head.npz",
            **{k: v.cpu().numpy() for k, v in head.state_dict().items()},
        )
        # Normalization is from the unchanged qualified parent, for independent CPU re-scoring.
        r.save_arrays(
            output / "normalization.npz",
            center=base.center.cpu().numpy(),
            scale=base.scale.cpu().numpy(),
        )
        report = dict(
            train=train,
            training_supported_rows=training_rows,
            training_supported_shots=support,
            loss=losses,
        )
        for role, ids in (("val", roles["val"]), ("cold", order[:20])):
            out, inventory = predict_split(base, head, ids, role, cfg, mapping, h)
            report[role] = inventory
            r.save_arrays(output / f"{role}_predictions.npz", **out)
            print(
                f"sealed {role}: {len(ids)} shots, {len(out['origin'])} forecasts",
                flush=True,
            )
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("fit altered the frozen parent")
    h.check()
    r.write_new(output / "inventory.json", report)
    names = (
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
            status="COMPLETE_ARRIVAL_MOTION_SOURCE_AND_HELD_FORECASTS",
            artifact_sha256={k: h(output / k) for k in names},
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
