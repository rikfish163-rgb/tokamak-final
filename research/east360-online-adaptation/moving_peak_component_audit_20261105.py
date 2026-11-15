"""Post-reveal, same-endpoint internal intervention; not a new alarm policy."""

import argparse
from pathlib import Path

import numpy as np
import torch

import moving_peak_replay_20261105 as r


def component_actions(d):
    if d.ndim != 2 or d.shape[1] != 256:
        raise ValueError("two 64D gain/bias pairs required")
    layer0, layer1 = d.clone(), d.clone()
    layer0[:, 128:] = 0
    layer1[:, :128] = 0
    gain, bias = d.clone(), d.clone()
    gain[:, 64:128] = gain[:, 192:] = 0
    bias[:, :64] = bias[:, 128:192] = 0
    return dict(
        zero=torch.zeros_like(d),
        full=d,
        layer0=layer0,
        layer1=layer1,
        gain=gain,
        bias=bias,
    )


def endpoint_error(native, adapted, sealed_native, sealed_adapted):
    return max(
        abs(float(native) - float(sealed_native)),
        abs(float(adapted) - float(sealed_adapted)),
    )


def audit(run, output):
    if output.exists():
        raise FileExistsError(output)
    done = r.read(run / "completed.json")
    if done["status"] != "COMPLETE_MOVING_PEAK_REPLAY" or any(
        r.sha(run / k) != v for k, v in done["artifact_sha256"].items()
    ):
        raise ValueError("verified terminal producer inputs required")
    cfg, saved = [r.read(run / k) for k in ("protocol.json", "per_shot.json")]
    r.configure()
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    rule = r.load_rule(run / "fit_moving_peak.npz", base.center.device)
    mapping = r.read(r.COLD / "input_map.json")
    rows = []
    for rank, sid in enumerate(cfg["order"][20:], 21):
        raw_path = r.io23.path_for("EAST", sid)
        if r.sha(raw_path) != cfg["raw_sha256"][str(sid)]:
            raise ValueError(f"raw data changed: {sid}")
        flag, down = r.io23.read_outcome("EAST", sid)
        old, new = [saved[k][rank - 1] for k in ("static", "moving_peak")]
        if any(
            (v["shot"], v["disruptive"], v["down"]) != (sid, flag, down)
            for v in (old, new)
        ):
            raise ValueError("saved outcome disagrees with HDF")
        raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
        ctx = r.latent.native_context(base, r.prepare_actual(raw, rank, mapping))
        path = run / "shots" / f"{rank:04d}_{sid}.npz"
        if r.sha(path) != done["sealed"][rank - 1]["curve_sha256"]:
            raise ValueError("forecast changed")
        with np.load(path, allow_pickle=False) as f:
            times = f["times"]
            if not np.array_equal(raw["times"], times) or not np.array_equal(
                ctx[2], f["static"]
            ):
                raise ValueError("different native forecast or clock")
            eligible = np.arange(r.policy.FIRST, len(times))
            if flag:
                lead = np.round((down - times[eligible]) * 1000, 9)
                eligible = eligible[(lead > 30) & (lead <= 125)]
            if not len(eligible):
                raise ValueError(f"no eligible comparison endpoint: {sid}")
            q = int(eligible[np.argmax(f["moving_peak"][eligible])])
            target_p = float(f["moving_peak"][q])
        index = (q - r.policy.FIRST) // r.policy.STRIDE
        acts = component_actions(r.actions(rule, ctx)[index : index + 1])
        with torch.no_grad():
            x, a = ctx[0][q : q + 1], ctx[1][q : q + 1]
            z = {
                k: float(
                    r.policy.routed_pair(base, x, a, r.latent.routes_all(a), v).item()
                )
                for k, v in acts.items()
            }
        p = {k: float(np.exp(-np.logaddexp(0.0, -v))) for k, v in z.items()}
        parity = endpoint_error(p["zero"], p["full"], ctx[2][q], target_p)
        if parity > 4e-6:
            raise ValueError(
                f"causal endpoint does not match complete replay: {sid}/{q}: {parity}"
            )
        if flag:
            group = (
                "new_danger_point"
                if p["zero"] < r.THRESHOLD <= p["full"]
                else "other_disruptive"
            )
        else:
            group = (
                "static_fp"
                if old["status"] == "fp"
                else ("new_fp" if new["status"] == "fp" else "stay_tn")
            )
        interaction = z["full"] - z["layer0"] - z["layer1"] + z["zero"]
        rows.append(
            dict(
                shot=sid,
                rank=rank,
                group=group,
                index=q,
                parity_error=parity,
                logits=z,
                probabilities=p,
                layer_interaction=interaction,
                gain_bias_interaction=z["full"] - z["gain"] - z["bias"] + z["zero"],
            )
        )
        if rank % 50 == 0:
            print("component audit through rank", rank, flush=True)
    summary = {}
    for name in sorted(set(v["group"] for v in rows)):
        group = [v for v in rows if v["group"] == name]
        summary[name] = dict(
            shots=len(group),
            at_or_above_threshold={
                k: sum(v["probabilities"][k] >= r.THRESHOLD for v in group)
                for k in group[0]["probabilities"]
            },
            median_logit_delta={
                k: float(
                    np.median([v["logits"][k] - v["logits"]["zero"] for v in group])
                )
                for k in group[0]["logits"]
            },
            median_layer_interaction=float(
                np.median([v["layer_interaction"] for v in group])
            ),
            median_gain_bias_interaction=float(
                np.median([v["gain_bias_interaction"] for v in group])
            ),
        )
    if len(rows) != 340:
        raise ValueError("complete post-budget scope required")
    r.write_new(
        output,
        dict(
            status="COMPLETE_MOVING_PEAK_COMPONENT_AUDIT",
            summary=summary,
            rows=rows,
            scope="One retrospective endpoint per shot at the adapted maximum: safe all eligible times, disruptive only 30<lead<=125 ms. Component masks preserve the retained coefficients without renormalizing; this is not a whole-shot score or a deployable filter.",
            producer_completion_sha256=r.sha(run / "completed.json"),
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.run, a.output)
