#!/usr/bin/env python3
"""Read-only CPU audit of saved maturity-clock factorial outputs.

Reuses only the independent prior audit's arithmetic/IO helpers.
Imports the isolated window utility, never any experiment runner or model.
JSON is written to stdout; --rows includes all four cells' per-shot records.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import zipfile
import hashlib
import numpy as np
import torch
from audit_anchored_feedback_20261030 import (
    ARMS, GROUPS, EXPECTED_KEYS, read_json, digest, load_npz,
    shot_rows, group_rows, gate,
)

torch.set_num_threads(1)
OLD = "origin_teacher_origin_step"
NEW = ("origin_teacher_mature_step", "mature_teacher_origin_step", "mature_teacher_mature_step")
PRIMARY = NEW[2]


def audit(root, include_rows=False):
    out = root/"probe_outputs/maturity_jacobian_cold20_v1"
    p, c = read_json(out/"protocol.json"), read_json(out/"complete.json")
    origin = Path(p["origin"])
    op, oc = read_json(origin/"protocol.json"), read_json(origin/"complete.json")
    previous = Path(op["previous"])
    records, folds = read_json(out/"records.json"), read_json(out/"folds.json")
    ids = list(dict.fromkeys(r["shot"] for r in records))
    sids = np.array([r["shot"] for r in records])
    labels = np.array([r["group"] for r in records])
    issues, count, errors, hashes = [], collections.Counter(), collections.defaultdict(list), {}

    def check(condition, message):
        count["assertions"] += 1
        if not condition:
            issues.append(message)

    def compare(actual, saved, category):
        if isinstance(saved, dict):
            for key, value in saved.items():
                if key not in actual:
                    issues.append(category+" missing "+key)
                else:
                    compare(actual[key], value, category)
        elif isinstance(saved, (float, int)) and not isinstance(saved, bool):
            error = abs(float(actual)-float(saved))
            errors[category].append(error)
            check(error <= 1e-12, category+" numeric mismatch")
        else:
            check(actual == saved, category+" value mismatch")

    for name, base, mapping in (
        ("new_artifacts", out, c["artifacts"]),
        ("origin_artifacts", origin, p["origin_hashes"]),
        ("observation_inputs", previous, op["input_hashes"]),
        ("parent_references", Path(op["parent"]), op["parent_hashes"]),
        ("new_live_code", root, p["code_sha256"]),
        ("origin_live_code", root, op["code_sha256"]),
    ):
        bad = [n for n, v in mapping.items() if digest(base/n) != v]
        hashes[name] = dict(checked=len(mapping), mismatches=bad)
        check(not bad, name+" hashes")
    check(p["origin_hashes"] == oc["artifacts"], "origin manifest")
    hashes["origin_complete"] = dict(sha256=digest(origin/"complete.json"))
    check(hashes["origin_complete"]["sha256"] == p["origin_complete_sha256"], "origin completion hash")
    with zipfile.ZipFile(out/"code_snapshot.zip") as z:
        check(set(z.namelist()) == set(p["code_sha256"]), "snapshot member set")
        bad = [n for n, v in p["code_sha256"].items() if hashlib.sha256(z.read(n)).hexdigest() != v]
        hashes["snapshot"] = dict(checked=len(z.namelist()), mismatches=bad)
        check(not bad, "snapshot member hashes")
    source = root/"experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
    hashes["source"] = dict(path=str(source), sha256=digest(source))
    check(hashes["source"]["sha256"] == p["source_sha256"] == op["source_sha256"], "source hash")
    hashes["arithmetic_helper"] = dict(path=str(root/"audit_anchored_feedback_20261030.py"),
                                      sha256=digest(root/"audit_anchored_feedback_20261030.py"))
    check(records == read_json(origin/"records.json") == read_json(previous/"records.json"), "records unchanged")
    check(folds == read_json(origin/"folds.json") == read_json(previous/"folds.json"), "folds unchanged")
    check(len(records) == c["opportunities"], "opportunities")
    check(p["cells"] == list(NEW) and p["primary"] == c["primary"] == PRIMARY, "declared cells and primary")
    check(collections.Counter(s for f in folds for s in f["held"]) == collections.Counter(ids),
          "held partition")
    check(all(set(f["fit"]).isdisjoint(f["held"]) and set(f["fit"])|set(f["held"]) == set(ids)
              for f in folds), "fold separation")
    check(all(r["origin"] >= 127 and (r["origin"]-127) % 25 == 0 and
              r["maturity"] == r["origin"]+25 and r["query"] == r["origin"]+50 for r in records), "clocks")
    check(len({(r["shot"], r["origin"], r["query"]) for r in records}) == len(records), "unique clocks")
    expected = {f"{cell}_fold{f['fold']}_{sid}.npz" for cell in NEW for f in folds for sid in ids}
    check({x.name for x in out.glob("*.npz")} == expected, "new NPZ set")

    data = torch.load(previous/"observations.pt", map_location="cpu", weights_only=True)
    x, mask, later, later_mask = [data[n] for n in ("x", "mask", "later_x", "later_mask")]
    check(all(v.shape == (len(records),128,23) for v in (x,mask,later,later_mask)), "window shapes")
    check(torch.equal(x[:,50:],later[:,:78]), "all 78-point overlapping values")
    check(torch.equal(mask[:,50:],later_mask[:,:78]), "all 78-point overlapping masks")
    # Independent timeline: unique union covers relative times 0..177, select 25..152.
    union_values = np.concatenate((x.numpy(),later.numpy()[:,78:]),axis=1)
    union_masks = np.concatenate((mask.numpy(),later_mask.numpy()[:,78:]),axis=1)
    expected_x, expected_mask = union_values[:,25:153],union_masks[:,25:153]
    spec = importlib.util.spec_from_file_location(
        "_audit_window_only",root/"experiments/risk_aligned_online/maturity_jacobian_20261030/window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual_x, actual_mask = module.maturity_window(x,mask,later,later_mask)
    check(np.array_equal(actual_x.numpy(),expected_x), "production window vs independent timeline")
    check(np.array_equal(actual_mask.numpy(),expected_mask), "production masks vs independent timeline")
    check(not actual_x.requires_grad and not actual_mask.requires_grad, "detached windows")
    for value in (float("nan"),float("inf"),-float("inf"),-12345.):
        changed_later, changed_mask = later.clone(), later_mask.clone()
        changed_later[:,103:] = value
        changed_mask[:,103:] = ~changed_mask[:,103:]
        xx, mm = module.maturity_window(x,mask,changed_later,changed_mask)
        check(torch.equal(xx,actual_x) and torch.equal(mm,actual_mask), "future suffix mutation")
        count["future_mutation_window_cases"] += len(records)
    changed_x, changed_mask = x.clone(), mask.clone()
    changed_x[:,:25] = float("nan")
    changed_mask[:,:25] = ~changed_mask[:,:25]
    xx, mm = module.maturity_window(changed_x,changed_mask,later,later_mask)
    check(torch.equal(xx,actual_x) and torch.equal(mm,actual_mask), "discarded old prefix ignored")
    count["discarded_prefix_window_cases"] += len(records)
    # Rejection tests are pure slicing/validation, not neural replay.
    for source_index, time_index in ((0,25),(0,127),(2,78),(2,102)):
        inputs = [v[:1].clone() for v in (x,mask,later,later_mask)]
        inputs[source_index][0,time_index,0] = float("nan")
        try:
            module.maturity_window(*inputs)
            check(False, "used nonfinite must reject")
        except ValueError:
            check(True, "used nonfinite rejected")
        count["used_nonfinite_rejections"] += 1
    for mask_only in (False,True):
        inputs = [v[:1].clone() for v in (x,mask,later,later_mask)]
        if mask_only:
            inputs[3][0,0,0] = ~inputs[3][0,0,0]
        else:
            inputs[2][0,0,0] += 1
        try:
            module.maturity_window(*inputs)
            check(False, "overlap mismatch must reject")
        except ValueError:
            check(True, "overlap mismatch rejected")
        count["overlap_mismatch_rejections"] += 1
    window_report = dict(
        windows=len(records), overlap_points_per_window=78,
        overlap_scalar_values=int(x[:,50:].numel()), overlap_mask_values=int(mask[:,50:].numel()),
        mature_scalar_values=int(actual_x.numel()), mature_mask_values=int(actual_mask.numel()),
        relative_origin=[0,127], relative_query=[50,177], relative_mature=[25,152],
        excluded_query_suffix=[153,177], cache_context_unchanged=True)
    context, support = data["context"].numpy(), data["support"].numpy()
    check(context.shape == (len(records),358) and np.isfinite(context).all(),"context shape")
    check(np.array_equal(support,(context[:,128:].reshape(-1,5,46)[:,:,23:]>0).any(axis=(1,2))),
          "support equals cached availability")

    models_report, draw_report = [], []
    for f in folds:
        fold = f["fold"]
        old = torch.load(origin/f"anchored_fold{fold}.pt",map_location="cpu",weights_only=True)
        new = torch.load(out/f"mature_fold{fold}.pt",map_location="cpu",weights_only=True)
        check(old["spec"] == new["spec"] == f, "model fold identity")
        check(torch.equal(old["mean"],new["mean"]), "exact prior mean reuse")
        check(all(torch.equal(v,new["scale"][n]) for n,v in old["scale"].items()), "exact prior scale reuse")
        check(set(old["state_dict"]) == set(new["state_dict"]), "parameter keys")
        for model in (old,new):
            check(sum(v.numel() for v in model["state_dict"].values()) == 14316, "parameters")
            check(all(torch.isfinite(v).all().item() for v in model["state_dict"].values()), "finite model weights")
        stat = {n:v.numpy() for n,v in old["scale"].items()}
        scaled = context.copy()
        scaled[:,:128] = (context[:,:128]-stat["hcenter"])/stat["hscale"]
        b = scaled[:,128:].reshape(-1,5,46)
        b[:,:,:23] = np.where(b[:,:,23:]>0,b[:,:,:23]/stat["change_scale"],0.)
        mean = np.stack([scaled[sids==s].mean(axis=0) for s in f["fit"]]).mean(axis=0)
        error = float(np.max(np.abs(mean-new["mean"].numpy())))
        check(error < 3e-6,"fit-only control mean")
        donor = np.asarray(read_json(origin/f"fold{fold}_donors.json"))
        check(donor.shape == (len(records),) and np.isin(sids[donor],f["fit"]).all() and
              (sids[donor] != sids).all(), "inherited fit-only other-shot donor")
        count["inherited_donor_assignments"] += len(donor)
        models_report.append(dict(fold=fold,mean_max_error=error,
                                  fit_contexts=int(np.isin(sids,f["fit"]).sum())))
        tr = read_json(out/f"anchored_fold{fold}_training.json")
        older = read_json(origin/f"anchored_fold{fold}_training.json")
        oldest = read_json(previous/f"representation_fold{fold}_training.json")
        draws = np.asarray(tr["draws"])
        check(draws.shape == (256,24), "draw count")
        check(tr["draws"] == older["draws"] == oldest["draws"], "all paired historical draws")
        check(np.isin(sids[draws],f["fit"]).all(), "fit-only draws")
        check((labels[draws[:,:12]]=="safe").all() and (labels[draws[:,12:]]=="danger").all(),
              "balanced safe/danger-only draws")
        pools = {g:{s:np.flatnonzero((sids==s)&(labels==g)) for s in sorted(f["fit"])
                    if ((sids==s)&(labels==g)).any()} for g in ("safe","danger")}
        rng = np.random.default_rng(131)
        reproduced = []
        for _ in range(256):
            row = []
            for g in ("safe","danger"):
                for _ in range(12):
                    sid = int(rng.choice(sorted(pools[g])))
                    row.append(int(rng.choice(pools[g][sid])))
            reproduced.append(row)
        check(reproduced == tr["draws"], "independent seed131 draw replay")
        check([h["step"] for h in tr["history"]] == list(range(1,257)), "step sequence")
        check(all(np.isfinite([h["loss"],h["gradient_norm"],h["mean_matrix_norm"]]).all()
                  for h in tr["history"]), "finite training history")
        count["training_steps"] += len(tr["history"])
        count["draw_positions"] += draws.size
        draw_report.append(dict(fold=fold,first32_loss=float(np.mean([h["loss"] for h in tr["history"][:32]])),
                                last32_loss=float(np.mean([h["loss"] for h in tr["history"][-32:]]))))

    predictions, all_rows, fold_groups, control_report = {},[],[],[]
    fit_saved = read_json(out/"fit_groups.json")
    old_fit_saved = read_json(origin/"fit_groups.json")
    for cell in (OLD,)+NEW:
        scope = "old" if cell == OLD else "new"
        for f in folds:
            fold = f["fold"]
            parts = []
            for sid in ids:
                file = (origin/f"anchored_fold{fold}_{sid}.npz" if cell == OLD else
                        out/f"{cell}_fold{fold}_{sid}.npz")
                a = load_npz(file)
                ix = sids == sid
                check(set(a) == EXPECTED_KEYS, "NPZ key set")
                for key,value in a.items():
                    check(value.shape == (int(ix.sum()),) and np.isfinite(value).all(),"NPZ shape and finite")
                check(np.array_equal(a["support"],support[ix]),"cached support unchanged")
                if cell != OLD:
                    base = load_npz(origin/f"anchored_fold{fold}_{sid}.npz")
                    check(np.array_equal(a["cold"],base["cold"]),"cold baseline exact")
                    count["new_cold_array_comparisons"] += 1
                parts.append(a)
                count[scope+"_npz_files"] += 1
                count[scope+"_arrays"] += len(a)
                count[scope+"_prediction_rows"] += int(ix.sum())
            packed = {key:np.concatenate([a[key] for a in parts]) for key in parts[0]}
            predictions[cell,fold] = packed
            valid = (packed["real_raw_norm"]>0)&(packed["reverse_raw_norm"]>0)
            check(np.array_equal(valid,packed["direction_nonzero"]),"saved nonzero mask")
            norm_error = 0.
            for arm in ARMS:
                check((packed[arm+"_raw_norm"]>=0).all() and (packed[arm+"_raw_norm"]<=.100001).all(),
                      "raw postprojection radius")
                check((packed[arm+"_norm"]>=0).all() and (packed[arm+"_norm"]<=.100001).all(),
                      "matched radius")
                check((np.abs(packed[arm+"_cos_real"])<=1.000001).all(),"cosine bounds")
                expected = (packed["real_raw_norm"] if arm=="real" else
                            np.where(packed[arm+"_raw_norm"]>0,packed["real_raw_norm"],0.))
                norm_error = max(norm_error,float(np.max(np.abs(packed[arm+"_norm"]-expected))))
            check(norm_error <= 1e-6,"norm matching")
            for key in ("mean_feedback_norm","mean_feedback_raw_norm","mean_feedback_cos_real"):
                check(np.count_nonzero(packed[key])==0,"zero mean control")
            check(np.array_equal(packed["mean_feedback"],packed["cold"]),"mean logits exactly cold")
            count[scope+"_zero_mean_rows"] += len(records)
            control_report.append(dict(cell=cell,fold=fold,norm_error=norm_error,
                                       supported=int(packed["support"].sum()),nonzero=int(valid.sum()),
                                       real_norm_min=float(packed["real_norm"].min()),
                                       real_norm_median=float(np.median(packed["real_norm"])),
                                       real_norm_max=float(packed["real_norm"].max())))
            for split in ("fit","held"):
                selected = np.isin(sids,f[split])
                rr = [row for row,keep in zip(records,selected) if keep]
                pp = {n:v[selected] for n,v in packed.items()}
                rows = shot_rows(rr,pp)
                gg = group_rows(rows)
                all_rows.extend(dict(cell=cell,fold=fold,split=split,**row) for row in rows)
                count[scope+"_"+split+"_shot_arm_rows"] += len(rows)
                count[scope+"_"+split+"_arm_window_BCEs"] += sum(row["windows"] for row in rows)
                fold_groups.append(dict(cell=cell,fold=fold,split=split,groups=gg))
                if split == "fit":
                    saved = (next(v["groups"] for v in old_fit_saved if v["fold"]==fold and v["kind"]=="anchored")
                             if cell == OLD else next(v["groups"] for v in fit_saved
                                                      if v["fold"]==fold and v["cell"]==cell))
                    compare(gg,saved,scope+"_fit_groups")
                    count[scope+"_fit_group_cells"] += sum(len(x) for x in gg.values())
    held_groups,held_predictions,gates = {},{},{}
    for cell in (OLD,)+NEW:
        parts = []
        for sid in ids:
            fold = next(f["fold"] for f in folds if sid in f["held"])
            parts.append({k:v[sids==sid] for k,v in predictions[cell,fold].items()})
        packed = {k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
        held_predictions[cell] = packed
        check([sid for sid in ids for r in records if r["shot"]==sid] == sids.tolist(),"packed order")
        rows = shot_rows(records,packed)
        groups = group_rows(rows)
        held_groups[cell] = groups
        saved = (read_json(origin/"anchored_held_per_shot.json") if cell==OLD else
                 read_json(out/f"{cell}_held_per_shot.json"))
        lookup = {(r["shot"],r["group"],r["arm"]):r for r in rows}
        check(len(saved)==len(lookup),"held rows count")
        check(len({(r["shot"],r["group"],r["arm"]) for r in saved})==len(saved),"held unique keys")
        for row in saved:
            compare(lookup[row["shot"],row["group"],row["arm"]],row,
                    ("old" if cell==OLD else "new")+"_held_per_shot")
        reference = oc["summaries"]["anchored"] if cell==OLD else c["summaries"][cell]
        compare(groups,reference["groups"],("old" if cell==OLD else "new")+"_held_groups")
        count[("old" if cell==OLD else "new")+"_held_group_cells"] += sum(len(v) for v in groups.values())
        status,details = gate(groups,packed)
        if cell != OLD:
            for group in ("safe","danger"):
                better = groups["real"][group]["mean_delta_bce"] < held_groups[OLD]["real"][group]["mean_delta_bce"]-1e-6
                if not better:
                    status["reasons"].append(group+": not better than origin-teacher/origin-step")
            status["passed"] = not status["reasons"]
            check(reference["primary"] == (cell==PRIMARY),"primary tag")
        compare(status,reference["gate"],"gate")
        gates[cell] = dict(**status,**details)

    contrasts = {}
    for name,new,base in (
        ("fixed_origin_teacher_change_window",NEW[0],OLD),
        ("fixed_mature_teacher_change_window",PRIMARY,NEW[1]),
        ("change_teacher_keep_origin_window",NEW[1],OLD),
        ("primary_vs_old",PRIMARY,OLD),
        ("change_teacher_keep_mature_window",PRIMARY,NEW[0]),
    ):
        contrasts[name] = dict(new=new,base=base,groups={})
        for group in GROUPS:
            a = {r["shot"]:r for r in all_rows if r["cell"]==new and r["split"]=="held"
                 and r["arm"]=="real" and r["group"]==group}
            b = {r["shot"]:r for r in all_rows if r["cell"]==base and r["split"]=="held"
                 and r["arm"]=="real" and r["group"]==group}
            difference = {s:a[s]["delta_bce"]-b[s]["delta_bce"] for s in sorted(a)}
            vv = np.array(list(difference.values()))
            contrasts[name]["groups"][group] = dict(
                mean=float(vv.mean()),median=float(np.median(vv)),
                improved=int((vv< -1e-6).sum()),worsened=int((vv>1e-6).sum()),
                unchanged=int((np.abs(vv)<=1e-6).sum()),
                per_shot=[dict(shot=s,old_delta=b[s]["delta_bce"],new_delta=a[s]["delta_bce"],
                               difference=difference[s]) for s in sorted(a)])
        norm_delta = held_predictions[new]["real_norm"].astype(float)-held_predictions[base]["real_norm"]
        contrasts[name]["real_norm_difference"] = dict(mean=float(norm_delta.mean()),
            median=float(np.median(norm_delta)),max_abs=float(np.abs(norm_delta).max()))
    result = dict(
        issues=issues,coverage=dict(count),hashes=hashes,
        max_absolute_errors={k:max(v,default=0.) for k,v in errors.items()},
        windows=window_report,models=models_report,draws=draw_report,
        record_groups=dict(collections.Counter(labels)),
        held_groups=held_groups,gates=gates,fold_groups=fold_groups,
        controls=control_report,contrasts=contrasts,
        primary_exceptions=[r for r in all_rows if r["cell"]==PRIMARY and r["split"]=="held"
                            and r["arm"]=="real" and r["group"] in ("safe","danger")
                            and r["delta_bce"]>1e-6])
    if include_rows:
        result["per_shot_rows"] = all_rows
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--rows",action="store_true")
    args = parser.parse_args()
    result = audit(args.root,args.rows)
    print(json.dumps(result,allow_nan=False,indent=2))
    raise SystemExit(bool(result["issues"]))

