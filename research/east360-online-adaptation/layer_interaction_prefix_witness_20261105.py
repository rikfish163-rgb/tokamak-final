"""Re-execute fixed raw prefixes without any EAST outcome calls."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import layer_interaction_replay_20261105 as layer
import layer_timing_control_20261105 as timing
import moving_peak_replay_20261105 as r


RANKS = (20, 21, 37, 181, 359)
OUTPUTS = dict(
    mask="layer_interaction_masks360_20261105_v1",
    fit="layer_interaction_fit360_20261105_v1",
    timing="layer_timing_control360_20261105_v1",
)


def witness(output):
    if output.exists():
        raise FileExistsError(output)
    runs = {k: r.ROOT / "probe_outputs" / p for k, p in OUTPUTS.items()}
    ends, configs = [
        {k: r.read(p / name) for k, p in runs.items()}
        for name in ("completed.json", "protocol.json")
    ]
    order, source, _ = r.reference()
    for k in runs:
        if (
            configs[k]["order"] != order
            or configs[k]["threshold"] != r.THRESHOLD
            or configs[k]["raw_sha256"] != source["raw_sha256"]
            or ends[k]["n_shots"] != 360
            or any(
                r.sha(runs[k] / p) != d for p, d in ends[k]["artifact_sha256"].items()
            )
        ):
            raise ValueError(f"invalid witness parent: {k}")
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    if (
        parent["threshold"] != r.THRESHOLD
        or r.state_digest(base) != source["source_state_sha256"]
    ):
        raise ValueError("frozen source required")
    base.cuda()
    device = base.center.device
    moving = layer.ARCHIVE / "fit_moving_peak.npz"
    if r.sha(moving) != ends["mask"]["fit_sha256"]["moving_peak"]:
        raise ValueError("wrong moving fit")
    rule = r.load_rule(moving, device)
    fits = {
        k: r.load_rule(runs["fit"] / f"fit_{k}.npz", device)
        for k in layer.FIT_ARMS[1:3]
    }
    map_path = r.COLD / "input_map.json"
    if r.sha(map_path) != source["input_sha256"][str(map_path)]:
        raise ValueError("wrong frozen input map")
    mapping = r.read(map_path)

    def raw(sid):
        if r.sha(r.io23.path_for("EAST", sid)) != source["raw_sha256"][str(sid)]:
            raise ValueError(f"raw changed: {sid}")
        return r.simple_data.read_shot("EAST", sid, rate_contract="nominal")

    rows = []
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no labels are available to the prefix witness"),
    ):
        for rank in RANKS:
            sid = order[rank - 1]
            data = raw(sid)
            n = min(len(data["times"]) - 1, max(200, len(data["times"]) // 2))
            for key in ("times", "values", "available"):
                data[key] = data[key][:n].copy()
            if n <= r.policy.FIRST:
                raise ValueError("prefix must contain at least one adaptation anchor")
            ctx = r.latent.native_context(base, r.prepare_actual(data, rank, mapping))
            actions = {k: {} for k in runs}
            if rank > 20:
                real, stale = r.actions(rule, ctx, stale=True)
                actions["mask"] = layer.component_targets(real)
                prev = r.latent.native_context(
                    base,
                    r.prepare_actual(raw(order[rank - 2]), max(21, rank - 1), mapping),
                )
                actions["timing"], _ = timing.timing_targets(
                    real, stale, r.actions(rule, prev)
                )
                for k, fit in fits.items():
                    real, stale = r.actions(fit, ctx, stale=True)
                    actions["fit"][k], actions["fit"][k + "_stale"] = real, stale
            differences = {}
            for k, target in actions.items():
                predicted, _ = r.forecast(base, ctx, target)
                curve = runs[k] / "shots" / f"{rank:04d}_{sid}.npz"
                if r.sha(curve) != ends[k]["sealed"][rank - 1]["curve_sha256"]:
                    raise ValueError(f"wrong sealed curve: {k}/{sid}")
                with np.load(curve, allow_pickle=False) as p:
                    if not np.array_equal(data["times"], p["times"][:n]):
                        raise ValueError("different clock")
                    differences[k] = {
                        arm: layer.require_curve_parity(
                            predicted[arm] if arm in predicted else ctx[2],
                            p[arm][:n],
                            r.THRESHOLD,
                        )
                        for arm in configs[k]["arms"]
                    }
            rows.append(dict(rank=rank, shot=sid, n=n, max_abs_error=differences))
    r.write_new(
        output,
        dict(
            status="PASS_TRUNCATED_LAYER_UPDATE_PREFIXES",
            rows=rows,
            n_curves=sum(len(v) for row in rows for v in row["max_abs_error"].values()),
            producer_completion_sha256={
                k: r.sha(p / "completed.json") for k, p in runs.items()
            },
            scope="Raw-signal prefixes and unchanged threshold crossings; not a wall-clock deployment test.",
        ),
    )
    print("PASS_TRUNCATED_LAYER_UPDATE_PREFIXES", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    witness(p.parse_args().output)
