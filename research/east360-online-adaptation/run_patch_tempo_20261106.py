"""Frozen-source causal patch-spacing experiment; HDF5 scoring is separate."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import patch_tempo_20261106 as model
from layer0_action_utility_20261106 import alarm_envelope
from rolling_safety_policy_20261106 import scalar_control
from run_layer0_query_memory_20261105 import raw_for
from run_rolling_safety_20261106 import cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


ARMS = ("static", "short", "long", "current", "delayed", "scalar")
WITNESS = {"fixed360": (21, 181, 359), "conditional835": (1, 418, 835)}


def source_reference():
    root = r.ROOT / "probe_outputs/temporal_scale_audit_v1"
    done = r.read(root / "completed.json")
    audit = r.read(root / "protocol.json")
    summary = r.read(root / "summary.json")["J-TEXT"]["record_windows"]["ALL"]
    # Short training shots contribute no full128 reference window. Membership
    # is still TRAIN1353; the archived window statistic has 1317 eligible shots.
    if (done["source_shots"] != 1353 or audit["source_ids"] != r.simple_data.source_roles()["train"]
            or not 0 < summary["n_shots"] <= done["source_shots"]):
        raise ValueError("reference must be the archived source TRAIN1353 audit")
    reference = np.array([[summary["channels"][c][f"D{lag}"]["median"]
                           for c in model.CHANNELS] for lag in (4, 8, 16)])
    if reference.shape != (3, 5) or not np.isfinite(reference).all() or (reference < 0).any():
        raise ValueError("finite nonnegative source variograms required")
    return reference


@torch.no_grad()
def inference(base, signal, interval, ends):
    result = np.empty(len(ends), np.float32)
    for lo in range(0, len(ends), 256):
        take = ends[lo:lo+256]
        x, a = model.gather(signal["values"], signal["available"], take, interval)
        x, a = torch.as_tensor(x, device="cuda"), torch.as_tensor(a, device="cuda")
        result[lo:lo+len(take)] = base(x, a)["logit"].sigmoid().cpu().numpy()
    return result


@torch.no_grad()
def predict_shot(base, raw, signal, reference, *, warm=False):
    n = len(signal["values"])
    native = inference(base, signal, 8, np.arange(n))
    menu = np.repeat(native[None], 3, axis=0)
    columns = [r.io23.RAW23.index(c) for c in model.CHANNELS]
    selected, delayed, support = model.clock(
        raw["values"][:, columns], raw["available"][:, columns], reference)
    if warm:
        selected[:], delayed[:] = 0, 0
    else:
        for j, interval in ((1, 4), (2, 16)):
            menu[j, model.FIRST:] = inference(base, signal, interval, np.arange(model.FIRST, n))
    point = np.arange(n)
    curves = dict(static=native, short=menu[1], long=menu[2],
                  current=menu[selected, point], delayed=menu[delayed, point])
    curves["scalar"] = scalar_control(native, curves["current"],
                                      first=model.FIRST, stride=model.STRIDE)
    return curves, selected, delayed, support


@torch.no_grad()
def live_prefix(base, raw, signal, reference, stop):
    """Only the selected historical window is evaluated at each endpoint."""
    columns = [r.io23.RAW23.index(c) for c in model.CHANNELS]
    choice, _, _ = model.clock(raw["values"][:stop, columns],
                               raw["available"][:stop, columns], reference)
    result = np.empty(stop, np.float32)
    for j, interval in enumerate(model.INTERVALS):
        ends = np.flatnonzero(choice == j)
        result[ends] = inference(base, {k: signal[k][:stop] for k in ("values", "available")},
                                 interval, ends)
    return result, choice


def predict(out, name):
    start = time.monotonic()
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    reference = source_reference()
    path, cfg, archive, (old_real, old_stale, warm) = cohort(name)
    if parent["threshold"] != r.THRESHOLD or r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("same qualified parent and threshold required")
    mapping = r.read(r.COLD / "input_map.json")
    frozen = {k: v.detach().clone() for k, v in base.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    (out / "shots").mkdir()
    r.write_new(out / "protocol.json", dict(
        cohort=name, order=cfg["order"], arms=ARMS, reference=reference.tolist(),
        reference_eligible_shots=r.read(r.ROOT / "probe_outputs/temporal_scale_audit_v1/summary.json")
                                  ["J-TEXT"]["record_windows"]["ALL"]["n_shots"],
        reference_channels=model.CHANNELS, intervals=model.INTERVALS,
        first=model.FIRST, stride=model.STRIDE, warm=warm, threshold=r.THRESHOLD,
        source=str(r.DEFAULT_PARENT), source_state=cfg["source_state_sha256"],
        fit_labels=0, inter_updates=0, adaptation="per-shot variogram EMA",
        baseline=str(path), baseline_completion_sha256=r.sha(path / "completed.json"),
        source_files={str(f): f.read_text() for f in
                      (Path(__file__), Path(model.__file__), r.ROOT / "PATCH_TEMPO_PROTOCOL_20261106.md")}))
    records, witnesses = [], []
    with patch.object(r.io23, "read_outcome", side_effect=RuntimeError("outcomes unavailable during prediction")):
        for rank, note in enumerate(archive["sealed"], 1):
            sid = note["shot"]
            times, static, _, _ = recorded(path, note, rank, old_real, old_stale)
            raw = raw_for(sid, cfg["raw_sha256"])
            np.testing.assert_array_equal(raw["times"], times)
            signal = r.prepare_actual(raw, rank if name == "fixed360" else 21, mapping)
            curves, choice, delayed, support = predict_shot(base, raw, signal, reference, warm=rank <= warm)
            error = require_parity(curves["static"], static)
            if rank in WITNESS[name]:
                stop = min(len(times), max(model.FIRST+1, len(times)//2))
                live, partial = live_prefix(base, raw, signal, reference, stop)
                np.testing.assert_array_equal(partial, choice[:stop])
                witnesses.append(dict(shot=sid, points=stop,
                                       probability_error=require_parity(live, curves["current"][:stop])))
            file = out / "shots" / f"{rank:04d}_{sid}.npz"
            r.save_arrays(file, **{k: v for k, v in curves.items() if k not in ("static", "scalar")},
                          choice=choice, delayed_choice=delayed, support=support)
            records.append(dict(shot=sid, rank=rank, static_error=error,
                                choices=np.bincount(choice, minlength=3).tolist(),
                                delayed_different=int((choice != delayed).sum())))
            if rank % 60 == 0:
                print("predicted", rank, "of", len(cfg["order"]), "seconds", round(time.monotonic()-start, 1), flush=True)
    if any(not torch.equal(frozen[k], v) for k, v in base.state_dict().items()):
        raise ValueError("source parameters changed")
    r.write_new(out / "completed.json", dict(status="SEALED_PATCH_TEMPO", records=records,
                                             witnesses=witnesses, seconds=time.monotonic()-start))
    print("sealed", len(records), flush=True)


def recount(out):
    start = time.monotonic()
    cfg, done = (r.read(out / f"{s}.json") for s in ("protocol", "completed"))
    path, original, archive, (real, stale, warm) = cohort(cfg["cohort"])
    if (done["status"] != "SEALED_PATCH_TEMPO" or cfg["order"] != original["order"]
            or [q["shot"] for q in done["records"]] != cfg["order"]
            or r.sha(path / "completed.json") != cfg["baseline_completion_sha256"]):
        raise ValueError("complete sealed same-cohort predictions required")
    rows = {k: [] for k in ARMS}
    envelope = dict(strict=0, capped=0, fp_min=0, total_safe=0)
    for rank, note in enumerate(archive["sealed"], 1):
        sid = note["shot"]
        t, static, _, _ = recorded(path, note, rank, real, stale)
        with np.load(out / "shots" / f"{rank:04d}_{sid}.npz", allow_pickle=False) as f:
            curves = {k: f[k].astype(np.float64) for k in ("short", "long", "current", "delayed")}
        for curve in curves.values():
            require_parity(curve[:model.FIRST], static[:model.FIRST])
            curve[:model.FIRST] = static[:model.FIRST]
        curves["static"] = static.astype(np.float64)
        curves["scalar"] = scalar_control(static, curves["current"], first=model.FIRST, stride=model.STRIDE)
        truth, clock = raw_truth(r.io23.path_for("EAST", sid), sid)
        np.testing.assert_array_equal(t, clock)
        for k in ARMS:
            rows[k].append(score(t, curves[k], truth, r.THRESHOLD))
        feasible = alarm_envelope(t, np.stack([curves[k] for k in ("static", "short", "long")]),
                                  disruptive=truth["disruptive"], down=truth["down"],
                                  threshold=r.THRESHOLD, first=model.FIRST, stride=model.STRIDE)
        if truth["disruptive"]:
            lead = np.round((truth["down"] - t[feasible[feasible < len(t)]])*1000, 9)
            envelope["strict"] += int((lead > 30).any())
            envelope["capped"] += int(((lead > 30) & (lead <= 3000)).any())
        else:
            envelope["total_safe"] += 1
            envelope["fp_min"] += int(len(t) not in feasible)
    metrics = {k: counts(v) for k, v in rows.items()}
    expected = (116, 29) if cfg["cohort"] == "fixed360" else (257, 68)
    if (metrics["static"]["tp"], metrics["static"]["fp"]) != expected:
        raise ValueError("baseline HDF5 counts changed")
    for k, v in metrics.items():
        v["tp_30_to_125"] = sum(q["status"] == "tp" and q["lead_ms"] <= 125 for q in rows[k])
        v["j_capped"] = v["tp_30_to_3000"] / v["n_disruptive"] - v["fpr"]
    current = metrics["current"]
    passes = (current["fpr"] <= .2 and current["tp"] > metrics["static"]["tp"]
              and current["tp_30_to_3000"] > metrics["static"]["tp_30_to_3000"]
              and all(current["j_capped"] > metrics[k]["j_capped"] for k in ARMS if k != "current"))
    result = dict(status="HDF5_PATCH_TEMPO_RECOUNT", metrics=metrics, envelope=envelope,
                  paired={k: dict(strict=pair(rows[k], rows["static"]),
                                  capped=pair(rows[k], rows["static"], capped=True))
                          for k in ARMS if k != "static"}, rows=rows,
                  seconds=time.monotonic()-start, promote=passes)
    r.write_new(out / "result.json", result)
    print({k: (v["tp"], v["tp_30_to_3000"], v["fp"]) for k, v in metrics.items()}, flush=True)
    print("hindsight only", envelope, "promote", passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--cohort", choices=("fixed360", "conditional835"), default="fixed360")
    args = parser.parse_args()
    (predict(args.out, args.cohort) if args.phase == "predict" else recount(args.out))
