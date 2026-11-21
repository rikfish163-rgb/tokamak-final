"""Post-run diagnostic: distance to the 12 historical safe shots at adapted peaks."""

import argparse
from pathlib import Path

import numpy as np
import torch

import moving_peak_replay_20261105 as r


def nearest_rms(point, support, scale):
    point, support, scale = [
        np.asarray(v, dtype=np.float64) for v in (point, support, scale)
    ]
    if (
        point.ndim != 1
        or point.shape != scale.shape
        or support.ndim != 2
        or support.shape[1:] != point.shape
        or not len(support)
    ):
        raise ValueError("point, scale and nonempty compatible support required")
    if (
        any(not np.isfinite(v).all() for v in (point, support, scale))
        or not (scale > 0).all()
    ):
        raise ValueError("finite features and positive scale required")
    dist = np.square((support - point) / scale)
    d = dist.mean(axis=1)
    j = int(d.argmin())
    return float(np.sqrt(d[j])), j


def describe(rows):
    fields = (
        "initial_ip_kA",
        "clock_s",
        "native_at_peak",
        "moving_peak",
        "delta",
        "state_distance",
        "history_distance",
        "action_norm",
    )
    if not rows:
        return dict(n=0, quartiles=None)
    return dict(
        n=len(rows),
        quartiles={
            k: np.quantile([v[k] for v in rows], [0.25, 0.5, 0.75]).tolist()
            for k in fields
        },
        near_radius=sum(v["action_norm"] >= 0.99 * r.latent.RADIUS for v in rows),
    )


def run(run_path, output):
    if output.exists():
        raise FileExistsError(output)
    completed = run_path / "completed.json"
    done = r.read(completed)
    if done["status"] != "COMPLETE_MOVING_PEAK_REPLAY" or any(
        r.sha(run_path / k) != v for k, v in done["artifact_sha256"].items()
    ):
        raise ValueError("unchanged terminal replay required")
    cfg, saved = [r.read(run_path / k) for k in ("protocol.json", "per_shot.json")]
    order, original, _ = r.reference()
    if cfg["order"] != order or cfg["raw_sha256"] != original["raw_sha256"]:
        raise ValueError("fixed ordered cohort required")
    r.configure()
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    mapping = r.read(r.COLD / "input_map.json")
    scale = (
        r.load_rule(r.PARENT / "fit20.npz", base.center.device, namespace="history.")
        .scale.cpu()
        .numpy()
    )
    mover = r.load_rule(run_path / "fit_moving_peak.npz", base.center.device)
    prepared, support_ids, rows = [], [], []
    support = None
    for rank, sid in enumerate(order, 1):
        flag, down = r.io23.read_outcome("EAST", sid)
        before, after = [saved[k][rank - 1] for k in ("static", "moving_peak")]
        if any(
            (v["shot"], v["disruptive"], v["down"]) != (sid, flag, down)
            for v in (before, after)
        ):
            raise ValueError("shot provenance disagrees with live HDF5")
        if flag:
            continue
        path = r.io23.path_for("EAST", sid)
        if r.sha(path) != cfg["raw_sha256"][str(sid)]:
            raise ValueError(f"raw HDF5 input changed: {sid}")
        raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
        ctx = r.latent.native_context(base, r.prepare_actual(raw, 21, mapping))
        features = torch.cat((ctx[4], ctx[5]), dim=1).cpu().numpy()
        if rank <= 20:
            prepared.append(features)
            support_ids.extend([sid] * len(features))
            if len(prepared) == 12:
                support = np.concatenate(prepared, axis=0)
                print("built safe support", len(set(support_ids)), flush=True)
            continue
        trace = run_path / "shots" / f"{rank:04d}_{sid}.npz"
        if r.sha(trace) != done["sealed"][rank - 1]["curve_sha256"]:
            raise ValueError("completed curves changed")
        with np.load(trace, allow_pickle=False) as f:
            if not np.array_equal(raw["times"], f["times"]) or not np.array_equal(
                ctx[2], f["static"]
            ):
                raise ValueError("new context does not reproduce its native trace")
            moved = f["moving_peak"]
        first = r.policy.FIRST
        q = first + int(np.argmax(moved[first:]))
        idx = (q - first) // r.policy.STRIDE
        if support is None or len(prepared) != 12:
            raise ValueError("expected all 12 eligible ended safe histories")
        d_s, _ = nearest_rms(features[idx, :66], support[:, :66], scale[:66])
        d_h, j = nearest_rms(features[idx], support, scale)
        norm = float(r.actions(mover, ctx)[idx].norm())
        rows.append(
            dict(
                shot=sid,
                rank=rank,
                prior_fp=before["status"] == "fp",
                moving_fp=after["status"] == "fp",
                peak_index=q,
                clock_s=float(raw["times"][q] - raw["times"][0]),
                initial_ip_kA=original["order"][rank - 1]["initial_ip_A"] / 1000,
                native_at_peak=float(ctx[2][q]),
                moving_peak=float(moved[q]),
                delta=float(moved[q] - ctx[2][q]),
                state_distance=d_s,
                history_distance=d_h,
                nearest_safe_shot=support_ids[j],
                action_norm=norm,
            )
        )
        if len(rows) % 25 == 0:
            print("held safe diagnosed", len(rows), flush=True)
    groups = dict(
        new_fp=[x for x in rows if not x["prior_fp"] and x["moving_fp"]],
        stay_tn=[x for x in rows if not x["prior_fp"] and not x["moving_fp"]],
        static_fp=[x for x in rows if x["prior_fp"]],
    )
    stats = {k: describe(v) for k, v in groups.items()}
    if len(rows) != 168 or len(set(support_ids)) != 12:
        raise ValueError("complete fixed held-safe and training-safe scopes required")
    r.write_new(
        output,
        dict(
            status="COMPLETE_MOVING_PEAK_SUPPORT_DIAGNOSTIC",
            producer_completion_sha256=r.sha(completed),
            train_safe_ids=sorted(set(support_ids)),
            support_anchors=len(support_ids),
            stats=stats,
            rows=rows,
            scope="Posthoc safe-shot comparison at each adapted maximum from index135; distances standardized by the frozen first20 feature scales, not an operational rule.",
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.run, a.output)
