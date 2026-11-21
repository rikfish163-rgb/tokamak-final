"""One frozen full360 comparison of first-FFN plasticity directions."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import motion_metric_model_20261107 as native
import plasticity_basis_model_20261107 as model
from motion_metric_kernel_20261107 import match_state
from run_plasticity_basis_20261107 import BASES, load_bases, state
from run_shape_motion_stream_20261107 import FIRST, STRIDE, BATCH, schedule
from run_layer0_query_memory_20261105 import raw_for
from run_rolling_safety_20261106 import cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


FIT = r.ROOT / "probe_outputs/plasticity_basis_source_20261107_v1"
PREVIOUS = r.ROOT / "probe_outputs/shape_motion360_20261107_v1"
ARMS = ("static", *BASES, "time", "channel", "scalar")
NEURAL = ARMS[1:-1]


def take(pack, indices):
    return {k: v[indices] for k, v in pack.items()}


def predict_shot(base, signal, bases, scales, *, warm=False):
    x, available = r.latent.make_views(signal, base.center.device)
    n = len(x)
    initial = bases["svd"]
    pieces = [model.encode_windows(base, x[lo:lo+BATCH], available[lo:lo+BATCH], initial)
              for lo in range(0, n, BATCH)]
    pack = {k: torch.cat([p[k] for p in pieces]) for k in pieces[0]}
    native_probability = pack["native"].sigmoid().cpu().numpy()
    curves = {k: native_probability.copy() for k in ARMS}
    origins, releases = schedule(n)
    if warm or not len(origins):
        return curves, dict(releases=0, adapted_points=0, control_norm_error=0., unmatched_controls=0)
    values = torch.as_tensor(signal["values"], device=base.center.device)
    valid = torch.as_tensor(signal["available"], device=base.center.device)
    states, anchors = {k: [] for k in NEURAL}, []
    norm_error, unmatched = 0., 0
    for lo in range(0, len(origins), BATCH):
        o, release = origins[lo:lo+BATCH], releases[lo:lo+BATCH]
        target = torch.as_tensor(o[:, None]+np.arange(1, 26)[None], device=x.device)
        motion, mask = native.motion_target(base, x[o], available[o], values[target], valid[target])
        data = dict(packs={"0": take(pack, o)}, motion=motion, mask=mask)
        current = {k: state(base, data, p, scales).detach() for k, p in bases.items()}
        for mode in ("time", "channel"):
            wrong = state(base, data, bases["risk"], scales, mode=mode).detach()
            current[mode] = match_state(wrong, current["risk"])
            error = (current[mode].flatten(1).norm(dim=1)-current["risk"].flatten(1).norm(dim=1)).abs()
            norm_error = max(norm_error, float(error.max()))
            unmatched += int((error > 1e-6).sum())
        for name, w in current.items():
            states[name].append(w)
        with torch.no_grad():
            anchor = model.forward_state(base, take(pack, release), current["risk"], bases["risk"])["logit"]
            anchors.append(anchor-pack["native"][release])
    states = {k: torch.cat(v) for k, v in states.items()}
    anchor = torch.cat(anchors)
    with torch.no_grad():
        for lo in range(FIRST, n, BATCH):
            ends = torch.arange(lo, min(lo+BATCH, n), device=x.device)
            active = (ends-FIRST)//STRIDE
            for name, w in states.items():
                p = bases[name] if name in BASES else bases["risk"]
                curves[name][lo:lo+len(ends)] = model.forward_state(
                    base, take(pack, ends), w[active], p)["logit"].sigmoid().cpu().numpy()
            curves["scalar"][lo:lo+len(ends)] = (pack["native"][ends]+anchor[active]).sigmoid().cpu().numpy()
    if any(not np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError("nonfinite plasticity risk curve")
    return curves, dict(releases=len(origins), adapted_points=n-FIRST,
        control_norm_error=norm_error, unmatched_controls=unmatched,
        state_norm={k: dict(median=float(w.flatten(1).norm(dim=1).median()),
                           maximum=float(w.flatten(1).norm(dim=1).max())) for k, w in states.items()})


def predict(out, fit):
    start = time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    bases, trained = load_bases(fit, base)
    reference, cfg, archive, (real, stale, warm) = cohort("fixed360")
    if parent["threshold"] != r.THRESHOLD or trained["source_state"] != cfg["source_state_sha256"] or warm != 20:
        raise ValueError("same original360 parent, threshold and preparation required")
    if r.read(PREVIOUS / "completed.json")["status"] != "SEALED_SHAPE_MOTION360":
        raise ValueError("completed fixed-direction baseline required")
    mapping = r.read(r.COLD / "input_map.json")
    if mapping["calibration_ids"] != cfg["order"][:20]:
        raise ValueError("original preparation20 mapping required")
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    (out / "shots").mkdir()
    r.write_new(out / "protocol.json", dict(status="FROZEN_PLASTICITY_BASIS360", order=cfg["order"],
        arms=ARMS, warm=warm, first=FIRST, stride=STRIDE, threshold=r.THRESHOLD,
        source=str(r.DEFAULT_PARENT), source_state=trained["source_state"], fit=str(fit), scales=trained["scales"],
        baseline=str(reference), fixed_direction_reference=str(PREVIOUS), added_label_fit=0, inter_updates=0,
        source_files={str(p): p.read_text() for p in (Path(__file__), Path(model.__file__), Path(native.__file__),
            r.ROOT / "run_plasticity_basis_20261107.py",
            r.ROOT / "experiments/risk_aligned_online/lora_stream_20261102/core.py",
            r.ROOT / "PLASTICITY_BASIS_PROTOCOL_20261107.md")}))
    records, witnesses = [], []
    with patch.object(r.io23, "read_outcome", side_effect=RuntimeError("outcomes unavailable during prediction")):
        for rank, note in enumerate(archive["sealed"], 1):
            sid = note["shot"]
            t, baseline, _, _ = recorded(reference, note, rank, real, stale)
            raw = raw_for(sid, cfg["raw_sha256"])
            np.testing.assert_array_equal(t, raw["times"])
            signal = r.prepare_actual(raw, rank, mapping)
            curves, info = predict_shot(base, signal, bases, trained["scales"], warm=rank <= warm)
            static_error = require_parity(curves["static"], baseline)
            with np.load(PREVIOUS / "shots" / f"{rank:04d}_{sid}.npz", allow_pickle=False) as f:
                fixed_error = require_parity(curves["svd"], f["fixed_shape"])
            if rank in (21, 181, 359):
                stop = min(len(t), max(FIRST+1, len(t)//2))
                partial, _ = predict_shot(base, {k: signal[k][:stop] for k in ("values", "available")},
                                          bases, trained["scales"])
                witnesses.append(dict(shot=sid, points=stop,
                    maximum_error=max(require_parity(v, curves[k][:stop]) for k, v in partial.items())))
            r.save_arrays(out / "shots" / f"{rank:04d}_{sid}.npz", **{k: v for k, v in curves.items() if k != "static"})
            records.append(dict(shot=sid, rank=rank, static_error=static_error, fixed_error=fixed_error, **info))
            if rank % 30 == 0:
                print("predicted", rank, "of", len(cfg["order"]), "seconds", round(time.monotonic()-start, 1), flush=True)
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()):
        raise ValueError("source changed")
    r.write_new(out / "completed.json", dict(status="SEALED_PLASTICITY_BASIS360", records=records,
                                               witnesses=witnesses, seconds=time.monotonic()-start))
    print("sealed", len(records), flush=True)


def recount(out):
    start = time.monotonic()
    cfg, done = (r.read(out / f"{k}.json") for k in ("protocol", "completed"))
    reference, original, archive, (real, stale, warm) = cohort("fixed360")
    if (done["status"] != "SEALED_PLASTICITY_BASIS360" or cfg["order"] != original["order"]
            or [q["shot"] for q in done["records"]] != cfg["order"] or cfg["threshold"] != r.THRESHOLD):
        raise ValueError("complete same-order sealed360 required")
    rows = {k: [] for k in ARMS}
    for rank, note in enumerate(archive["sealed"], 1):
        sid = note["shot"]
        t, baseline, _, _ = recorded(reference, note, rank, real, stale)
        with np.load(out / "shots" / f"{rank:04d}_{sid}.npz", allow_pickle=False) as f:
            curves = {k: f[k].astype(np.float64) for k in ARMS if k != "static"}
        curves["static"] = baseline.astype(np.float64)
        for curve in curves.values():
            require_parity(curve[:FIRST], baseline[:FIRST])
            if rank <= warm:
                require_parity(curve, baseline)
        truth, clock = raw_truth(r.io23.path_for("EAST", sid), sid)
        np.testing.assert_array_equal(t, clock)
        for name, curve in curves.items():
            rows[name].append(score(t, curve, truth, r.THRESHOLD))
    metrics = {k: counts(v) for k, v in rows.items()}
    if (metrics["static"]["tp"], metrics["static"]["fp"]) != (116, 29):
        raise ValueError("original HDF5 baseline changed")
    for k, value in metrics.items():
        value["j_30_to_3000"] = value["tp_30_to_3000"]/value["n_disruptive"]-value["fpr"]
        value["tp_30_to_125"] = sum(q["status"] == "tp" and q["lead_ms"] <= 125 for q in rows[k])
        value["fpr_cap_met"] = {str(cap): value["fpr"] <= cap for cap in (.10, .15, .20)}
    candidate = metrics["risk"]
    operational = candidate["fpr"] <= .20 and candidate["tp"] > metrics["static"]["tp"]
    attribution = (candidate["tp_30_to_3000"] > metrics["static"]["tp_30_to_3000"]
        and all(candidate["j_30_to_3000"] > metrics[k]["j_30_to_3000"]
                for k in ("static", "svd", "random", "feedback", "wrong_feedback", "time", "channel", "scalar"))
        and sum(q["unmatched_controls"] for q in done["records"]) == 0)
    r.write_new(out / "result.json", dict(status="HDF5_PLASTICITY_BASIS360_RECOUNT", metrics=metrics,
        paired={k: dict(strict=pair(rows[k], rows["static"]), capped=pair(rows[k], rows["static"], capped=True))
                for k in ARMS if k != "static"},
        prefix20={k: counts(v[:20]) for k, v in rows.items()}, suffix340={k: counts(v[20:]) for k, v in rows.items()},
        rows=rows, operational_screen=operational, attribution_screen=attribution,
        eligible_for_inter_experiment=operational and attribution, seconds=time.monotonic()-start))
    print({k: (v["tp"], v["tp_30_to_3000"], v["fp"]) for k, v in metrics.items()}, flush=True)
    print("operational", operational, "attribution", attribution, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--fit", type=Path, default=FIT)
    args = parser.parse_args()
    (predict(args.out, args.fit) if args.phase == "predict" else recount(args.out))
