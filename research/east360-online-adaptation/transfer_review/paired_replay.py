"""Run from code/ with PYTHONPATH=. . Fixed-source, disjoint temporal transfer probe."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binom

from zero_cold_start import backbone, data
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from risk_aligned_online.warmstart_alignment_20261022 import alignment

OUT = Path(__file__).with_name("paired_replay.json")
REF = 100  # samples, 1 kHz
DENSITY = data.RAW23.index("ne_nG")


def device_statistics(device, ids):
    rows = []
    for sid in ids:
        sig = data.read_signals(device, sid)
        rows.append(dict(device=device, shot=sid, values=sig["values"], available=sig["available"]))
    return alignment.fit_statistics(rows, device=device, allowed_ids=ids)


@torch.inference_mode()
def scores(model, sig, gain):
    if isinstance(gain, dict):
        values = alignment.transform({k:sig[k] for k in ("values", "available")}, gain)["values"]
    else:
        values = sig["values"].copy()
        values[:, DENSITY] *= gain
    x, m = backbone._windows(values, sig["available"], model.cfg["window"], "cuda")
    return torch.cat([model(x[i:i+2048], m[i:i+2048]) for i in range(0, len(x), 2048)]).cpu().numpy()


def threshold(rows, field):
    # Monotone alarm: smallest boundary above the first forbidden normal peak.
    peaks = sorted([np.max(r[field]) for r in rows if not r["dis"]], reverse=True)
    max_fp = (len(peaks) - 1) // 5
    return float(np.nextafter(float(peaks[max_fp]), np.inf))


def evaluate(rows, field, boundary, deadline=30, normal_history=None, rank=None,
             donor_seed=None, seed_at=0, audit=None, freeze_after_seed=False):
    counts = dict(tp=0, late=0, miss=0, fp=0, tn=0, startup_fp=0, adv_covered=0, beat_adv=0)
    capacity = len(normal_history) if normal_history is not None else 0
    for i, r in enumerate(rows):
        if i == seed_at and donor_seed is not None:
            normal_history = donor_seed.copy()
        if normal_history is not None:
            n = len(normal_history)
            order = rank if n == capacity else min(n, int(binom.ppf(.95, n, .8)) + 1)
            boundary = float(np.nextafter(np.sort(normal_history)[order-1], np.inf))
            if field == "relative":
                boundary = max(0., boundary)
        seq = r[field].astype(np.float64)
        times = r["times"]
        valid = times <= r["down"] + 1e-9 if r["dis"] else np.ones(len(times), bool)
        idx = np.flatnonzero((seq >= boundary) & valid)
        lead = round(float(1000*(r["down"] - times[idx[0]])), 6) if len(idx) and r["dis"] else None
        if audit is not None:
            audit.append((r["shot"], float(times[idx[0]]) if len(idx) else None, lead, boundary))
        if r["dis"]:
            counts["miss" if lead is None else "tp" if lead > deadline else "late"] += 1
            if r["advanced"] is not None:
                counts["adv_covered"] += 1
                counts["beat_adv"] += int(lead is not None and lead > r["advanced"])
        else:
            if normal_history is not None and (not freeze_after_seed or i < seed_at):
                if len(normal_history) == capacity:
                    normal_history.pop(0)
                normal_history.append(float(seq.max()))
            counts["fp" if len(idx) else "tn"] += 1
            counts["startup_fp"] += int(len(idx) > 0 and idx[0] < REF)
    counts["timely_tpr"] = counts["tp"] / (counts["tp"] + counts["late"] + counts["miss"])
    counts["fpr"] = counts["fp"] / (counts["fp"] + counts["tn"])
    return counts


def main():
    model, ckpt = backbone.load_source_model()
    splits = data.load_jtext_splits()
    cohorts = data.load_east_cohorts()
    dev = sorted(cohorts["dev360"])
    ext = sorted(cohorts["east_indep2593"])
    ids = sorted(dev + ext)
    donors = ids[:5]
    assert set(donors).isdisjoint(dev)
    source = device_statistics("J-TEXT", splits["train"])
    target = device_statistics("EAST", donors)
    gain = alignment.make_map(source, target, "density_gain")["gain"][DENSITY]
    affine = alignment.make_map(source, target, "moments_all")
    maps = {"identity":1., "density":gain, "moments":affine}
    val_scores = []
    for sid in splits["val"]:
        sig = data.read_signals("J-TEXT", sid)
        val_scores.append(scores(model, sig, 1.))
    source_anchor = float(np.median([np.median(z[:REF]) for z in val_scores]))
    records = {"J-TEXT": [], **{arm:{} for arm in maps}}

    def record(device, sid, times, z, advanced):
        relative = z.astype(float) - source_anchor
        relative[REF:] = z[REF:].astype(float) - np.median(z[:REF])
        dis, down = data.read_outcome(device, sid)
        return dict(shot=sid, times=times, dis=dis, down=down, advanced=advanced.get(sid), absolute=z, relative=relative)

    advanced = data.read_advanced_time("J-TEXT")
    for sid, z in zip(splits["val"], val_scores):
        sig = data.read_signals("J-TEXT", sid)
        records["J-TEXT"].append(record("J-TEXT", sid, sig["times"], z, advanced))
    advanced = data.read_advanced_time("EAST")
    donor_records = {arm:[] for arm in maps}
    for sid in ids:
        sig = data.read_signals("EAST", sid)
        raw = scores(model, sig, 1.)
        for arm, mapping in maps.items():
            z = scores(model, sig, mapping) if arm != "identity" and sid > donors[-1] else raw
            records[arm][sid] = record("EAST", sid, sig["times"], z, advanced)
            if sid in donors:
                after = scores(model, sig, mapping) if arm != "identity" else raw
                donor_records[arm].append(record("EAST", sid, sig["times"], after, advanced))
    result = dict(checkpoint=hashlib.sha256((data.ARTIFACTS/"jtext_source.pt").read_bytes()).hexdigest(),
                  donors=donors, density_gain=gain, moment_gain=affine["gain"], moment_offset=affine["offset"],
                  source_anchor=source_anchor, dev_fit_last=dev[179], results={}, dev_capacity={})
    sets = {"dev360":dev, "dev_last180":dev[180:], "extension_all":ext,
            "extension_after_dev_fit":[s for s in ext if s > dev[179]]}
    for arm in maps:
        p = float(ckpt["val_threshold_fpr5"])
        b = float(np.log(p / (1 - p)))
        result["results"][f"{arm}/absolute/checkpoint"] = {
            "boundary":b, **{name:evaluate([records[arm][s] for s in subset], "absolute", b)
                             for name,subset in sets.items()}}
        for field in ("absolute", "relative"):
            # Retrospective capacity of the score at an identical false-alarm count.
            # Never apply or select this development-fitted boundary on the extension.
            development = [records[arm][s] for s in dev]
            cap = threshold(development, field)
            result["dev_capacity"][f"{arm}/{field}"] = evaluate(development, field, cap)
            for calibration, fit in (("JTEXT_val", records["J-TEXT"]), ("EAST_first180", [records[arm][s] for s in dev[:180]])):
                b = threshold(fit, field)
                if field == "relative":
                    b = max(0., b)
                key = f"{arm}/{field}/{calibration}"
                result["results"][key] = {"boundary":b, **{name:evaluate([records[arm][s] for s in subset], field, b) for name,subset in sets.items()}}
            normal_seed = [float(r[field].max()) for r in records["J-TEXT"] if not r["dis"]]
            ranks = {
                "past_normal_peaks": int(np.ceil(.8 * (len(normal_seed) + 1))),
                "past_normal_tolerance": int(binom.ppf(.95, len(normal_seed), .8)) + 1,
            }
            for name, rank in ranks.items():
                key = f"{arm}/{field}/{name}"
                result["results"][key] = {"normal_memory":len(normal_seed), "rank":rank, **{
                    cohort:evaluate([records[arm][s] for s in subset], field, 0.,
                                    normal_history=list(normal_seed), rank=rank)
                    for cohort, subset in (("dev360",dev), ("extension_all",ext))}}
            peaks = [float(r[field].max()) for r in donor_records[arm]]
            key = f"{arm}/{field}/donor_seeded_tolerance"
            entry = {"normal_memory":len(normal_seed), "donor_peaks":peaks}
            for cohort, subset in (("dev360",dev), ("extension_all",ext)):
                rows = [records[arm][s] for s in subset]
                seed_at = 0 if cohort == "dev360" else len(donors)

                def seeded_counts(freeze):
                    audit = []
                    counts = evaluate(rows, field, 0., normal_history=list(normal_seed),
                                      rank=ranks["past_normal_tolerance"], donor_seed=peaks,
                                      seed_at=seed_at, audit=audit, freeze_after_seed=freeze)
                    counts["timely_ids"] = [s for s, _, lead, _ in audit if lead is not None and lead > 30]
                    counts["fp_ids"] = [s for s, t, lead, _ in audit if t is not None and lead is None]
                    return counts

                entry[cohort] = seeded_counts(False)
                if arm == "density" and field == "absolute":
                    frozen = result["results"].setdefault("density/absolute/donor_seeded_frozen",
                                                          {"donor_peaks":peaks})
                    frozen[cohort] = seeded_counts(True)
            result["results"][key] = entry
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
