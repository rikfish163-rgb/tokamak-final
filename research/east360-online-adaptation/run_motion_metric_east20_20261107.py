"""Apply frozen V2 only on already revealed EAST preparation shots."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import motion_metric_model_20261107 as model
from motion_metric_kernel_20261107 import MotionMetric, match_state
from run_motion_metric_20261107 import ARMS, aggregate, batch, state
from experiments.risk_aligned_online.learned_update_loss_20261023.learning import context_features


def prepare_history(base, *, include_past=False, encoder=model.encode_windows):
    """The same already-revealed20 and predeclared windows, never later shots."""
    mapping = r.read(r.COLD / "input_map.json")
    ids = mapping["calibration_ids"]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("exactly original preparation20 required")
    raw_reader = r.simple_data.read_shot
    arrays = {k: [] for k in ("x0", "a0", "xr", "ar", "xq", "aq", "truth", "mask")}
    if include_past:
        arrays.update({k: [] for k in ("xp", "ap", "xpp", "app")})
    rows, skipped = [], []
    class_counts = dict(disruptive=0, safe=0)
    for sid in ids:
        raw = raw_reader("EAST", sid, rate_contract="nominal")
        flag, down = r.io23.read_outcome("EAST", sid)
        class_counts["disruptive" if flag else "safe"] += 1
        signal = r.prepare_actual(raw, 21, mapping)
        lead = np.round((down-raw["times"])*1000, 9) if flag else None
        if flag:
            queries = []
            for request in (250, 150, 100, 60, 35):
                candidates = np.flatnonzero((np.arange(len(lead)) >= 169) & (lead > 30))
                q = int(candidates[np.argmin(np.abs(lead[candidates]-request))]) if len(candidates) else None
                if q is None or abs(lead[q]-request) > 2 or q in queries:
                    skipped.append(dict(shot=sid, requested_lead=request))
                else:
                    queries.append(q)
        else:
            queries = np.unique(np.linspace(169, len(raw["times"])-1, 5, dtype=int)).tolist()
        for q in queries:
            origin, release = q-42, q-16
            group = "safe" if not flag else "timely" if lead[q] <= 125 else "earlier"
            rows.append(dict(shot=sid, query=q, origin=origin, release=release, group=group,
                label=int(group == "timely"), disruptive=bool(flag), down=down if flag else None,
                query_time=float(raw["times"][q]), lead_ms=float(lead[q]) if flag else None))
            for tag, end in (("0", origin), ("r", release), ("q", q)):
                x, a = r.simple_data.window(signal, end, 128)
                arrays[f"x{tag}"].append(x)
                arrays[f"a{tag}"].append(a)
            if include_past:
                for tag, end in (("p", release-8), ("pp", release-16)):
                    x, a = r.simple_data.window(signal, end, 128)
                    arrays[f"x{tag}"].append(x)
                    arrays[f"a{tag}"].append(a)
            arrays["truth"].append(signal["values"][origin+1:origin+26])
            arrays["mask"].append(signal["available"][origin+1:origin+26])
    if class_counts != dict(disruptive=8, safe=12):
        raise ValueError("preparation HDF5 classes changed")
    values = {k: torch.as_tensor(np.stack(v), device="cuda") for k, v in arrays.items()}
    basis = model.basis_for(base)
    times = ("0", "r", "q", "p", "pp") if include_past else ("0", "r", "q")
    packs = {s: encoder(base, values[f"x{s}"], values[f"a{s}"], basis) for s in times}
    motion, mask = model.motion_target(base, values["x0"], values["a0"], values["truth"], values["mask"])
    context = context_features(values["xr"], values["ar"], base.center, base.scale)
    return dict(packs=packs, motion=motion, mask=mask, context=context), rows, dict(
        ids=ids, skipped=skipped, class_counts=class_counts)


def main(fit, out):
    start = time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    cfg = r.read(fit / "protocol.json")
    if cfg["geometry"] != "magnitude" or r.read(fit / "fit.json")["status"] != "FIT_COMPLETE":
        raise ValueError("frozen completed V2 required")
    saved = torch.load(fit / "metrics.pt", map_location="cuda", weights_only=True)
    names = ("conditional", "global", "auxiliary125") if "auxiliary125" in saved else ("conditional", "global")
    models = {k: MotionMetric().cuda().eval() for k in names}
    for k, metric in models.items():
        metric.load_state_dict(saved[k])
        metric.requires_grad_(False)
    data, rows, history = prepare_history(base)
    packs, motion, mask, context = (data[k] for k in ("packs", "motion", "mask", "context"))
    states = {}
    jobs = [
        ("conditional", models["conditional"], "real", False),
        ("fixed", None, "real", False), ("global", models["global"], "real", True),
        ("time", models["conditional"], "time", False),
        ("channel", models["conditional"], "channel", False),
        ("reverse", models["conditional"], "reverse", False)]
    if "auxiliary125" in models:
        jobs.append(("auxiliary125", models["auxiliary125"], "real", False))
    for arm, metric, mode, global_context in jobs:
        updated, _ = state(base, packs["0"], context, motion, mask, metric, mode=mode,
                           global_context=global_context, update_scale=cfg["update_scale"])
        states[arm] = updated.detach()
    for arm in ("time", "channel", "reverse"):
        states[arm] = match_state(states[arm], states["conditional"])
    with torch.no_grad():
        logits = {k: model.forward_state(base, packs["q"], w)["logit"].cpu().numpy() for k, w in states.items()}
        logits["static"] = packs["q"]["native"].cpu().numpy()
        at_release = model.forward_state(base, packs["r"], states["conditional"])["logit"]
        logits["scalar"] = (packs["q"]["native"] + at_release-packs["r"]["native"]).cpu().numpy()
    out.mkdir(parents=True, exist_ok=False)
    r.save_arrays(out / "predictions.npz", **logits,
                  **{f"norm_{k}": v.flatten(1).norm(dim=1).cpu().numpy() for k, v in states.items()})
    r.write_new(out / "sealed.json", dict(scope="ENDED20_OFFLINE_DIAGNOSTIC_NOT_DEPLOYMENT",
        rows=rows, fit=str(fit), label_shots_added=0, **history,
        source_files={str(p): p.read_text() for p in (Path(__file__), r.ROOT / "MOTION_METRIC_PROTOCOL_20261107.md")}))
    tau = math.log(parent["threshold"]/(1-parent["threshold"]))
    output = []
    for i, row in enumerate(rows):
        flag, down = r.io23.read_outcome("EAST", row["shot"])
        lead = round((down-row["query_time"])*1000, 9) if flag else None
        group = "safe" if not flag else "timely" if 30 < lead <= 125 else "earlier"
        if group != row["group"] or (flag and lead <= 30):
            raise ValueError("HDF5 strata changed")
        losses = {k: float(np.logaddexp(0., (1-2*row["label"])*(float(v[i])-tau))) for k, v in logits.items()}
        output.append(dict(row, before_bce=losses["static"], after_bce=losses,
                           state_norm={k: float(v[i].norm()) for k, v in states.items()}))
    arms = (*ARMS, "auxiliary125") if "auxiliary125" in models else ARMS
    table = aggregate(output, arms=arms)
    r.write_new(out / "result.json", dict(status="ENDED20_POINT_DIAGNOSTIC_NOT_STREAM_SCORE", table=table,
                                            rows=output, seconds=time.monotonic()-start))
    print(table, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.fit, args.out)
