"""Explanatory diagnostic: quantify the arrival gradient that reverses with order."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from audit_layer0_action_utility_20261106 import dump
from fit_layer0_arrival_motion_20261106 import FIRST, STRIDE, load_head
import layer0_arrival_motion_20261106 as motion
from layer0_arrival_update_20261106 import episode
import moving_peak_replay_20261105 as r
from probe_layer0_arrival_feedback_20261106 import FIT, FIT_AUDIT
from run_layer0_query_memory_20261105 import parents, raw_for


def geometry(a, b):
    a, b = (np.asarray(x, dtype=np.float64) for x in (a, b))
    if (
        a.ndim != 2
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("aligned finite gradient matrices required")
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if (denom <= 0).any():
        raise ValueError("cosine needs nonzero gradient pairs")
    cos = (a * b).sum(1) / denom
    odd = np.square(a - b).sum(1) / (2 * (np.square(a).sum(1) + np.square(b).sum(1)))
    return np.stack((cos, odd), axis=1)


def run(out):
    if out.exists():
        raise FileExistsError(out)
    r.configure()
    order, cfg, _, base, source, mapping, map_path, _ = parents()
    check = r.read(FIT_AUDIT)
    if check["fit_head_sha256"] != r.sha(FIT / "head.npz"):
        raise ValueError("frozen observed-only fit required")
    head = load_head(FIT / "head.npz", base.center.device)
    dev = base.center.device
    rows = []
    names = ("combined", "level", "shape")
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("order diagnostic has no shot labels"),
    ):
        for sid in order[:20]:
            raw = raw_for(sid, cfg["raw_sha256"])
            data = r.prepare_actual(raw, 21, mapping)
            ctx = r.latent.native_context(base, data)
            v, a = [
                torch.as_tensor(data[k], device=dev) for k in ("values", "available")
            ]
            m = torch.arange(FIRST, len(v) - motion.H, STRIDE, device=dev)
            metrics = {k: [] for k in names}
            for t in m.split(128):
                x, mask, hold, local, y, observed = episode(
                    base, ctx[0], ctx[1], v, a, t
                )
                u = x.new_zeros((len(x), motion.WIDTH), requires_grad=True)
                context, _ = motion.features(base, x, mask, u)
                pred = head(hold, local, context)
                support = observed & observed.flip(1)
                direct = motion.row_loss(pred, y, support)
                reverse = motion.row_loss(pred, y.flip(1), support)
                for key in names:
                    g, h = [
                        torch.autograd.grad(z[key].sum(), u, retain_graph=True)[
                            0
                        ].detach()
                        for z in (direct, reverse)
                    ]
                    keep = (
                        (g.norm(dim=1) > 0) & (h.norm(dim=1) > 0) & direct["supported"]
                    )
                    metrics[key].append(
                        geometry(g[keep].cpu().numpy(), h[keep].cpu().numpy())
                    )
            rec = {key: np.concatenate(values) for key, values in metrics.items()}
            if any(not len(v) for v in rec.values()):
                raise ValueError(f"unsupported explanatory geometry for {sid}")
            rows.append(
                dict(
                    shot=sid,
                    windows=len(m),
                    stats={
                        k: dict(
                            n_pairs=len(x),
                            median_cosine=float(np.median(x[:, 0])),
                            median_order_energy=float(np.median(x[:, 1])),
                        )
                        for k, x in rec.items()
                    },
                )
            )
    if r.state_digest(base) != source or any(
        p.grad is not None for p in base.parameters()
    ):
        raise ValueError("order diagnostic modified source")
    stats = {
        k: {
            field: float(np.mean([v["stats"][k][field] for v in rows]))
            for field in ("median_cosine", "median_order_energy")
        }
        for k in names
    }
    dump(
        out,
        dict(
            status="PASS_OUTCOME_FREE_ORDER_GRADIENT_GEOMETRY",
            rows=rows,
            shot_equal_means_of_medians=stats,
            fit_head_sha256=check["fit_head_sha256"],
            map_sha256=r.sha(map_path),
            source_state_sha256=source,
            scope="Explanatory gradient comparison after failed fidelity gate; no risk outcomes or fit.",
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
