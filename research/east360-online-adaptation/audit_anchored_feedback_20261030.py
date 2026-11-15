#!/usr/bin/env python3
"""Read-only CPU audit; JSON to stdout, --rows includes all per-shot values.

No experiment imports, model evaluation, optimization, HDF5 access or file writes.
Run with the project's rl_env Python and -B to disable bytecode writes.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

torch.set_num_threads(1)
ARMS = ("real", "reverse", "mean_feedback", "donor")
KINDS = ("ordinary", "anchored")
GROUPS = ("safe", "danger", "early", "late")
EXPECTED_KEYS = {"cold", "support", "direction_nonzero"} | {
    arm + suffix for arm in ARMS
    for suffix in ("", "_norm", "_raw_norm", "_cos_real")
}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def bce(logit, label):
    return np.logaddexp(0., (1. - 2. * label) * logit.astype(np.float64))


def shot_rows(records, predictions):
    rows = []
    for sid in sorted({r["shot"] for r in records}):
        for group in GROUPS:
            selected = np.array([r["shot"] == sid and r["group"] == group for r in records])
            if not selected.any():
                continue
            label = int(group in ("danger", "late"))
            before = bce(predictions["cold"][selected], label).mean()
            for arm in ARMS:
                after = bce(predictions[arm][selected], label).mean()
                rows.append(dict(
                    shot=sid, group=group, arm=arm, windows=int(selected.sum()),
                    supported=int(predictions["support"][selected].sum()),
                    before_bce=float(before), after_bce=float(after),
                    delta_bce=float(after - before),
                    mean_logit_change=float(
                        (predictions[arm][selected].astype(np.float64) -
                         predictions["cold"][selected]).mean())))
    return rows


def group_rows(rows):
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for group in GROUPS:
            selected = [r for r in rows if r["arm"] == arm and r["group"] == group]
            delta = np.array([r["delta_bce"] for r in selected])
            output[arm][group] = dict(
                shots=len(selected),
                supported_shots=sum(r["supported"] > 0 for r in selected),
                mean_delta_bce=float(delta.mean()) if len(delta) else None,
                min_delta_bce=float(delta.min()) if len(delta) else None,
                max_delta_bce=float(delta.max()) if len(delta) else None,
                improved=int((delta < -1e-6).sum()),
                worsened=int((delta > 1e-6).sum()))
    return output


def gate(groups, predictions):
    nonzero = ((predictions["real_raw_norm"] > 0) &
               (predictions["reverse_raw_norm"] > 0))
    cosine = (float(np.median(predictions["reverse_cos_real"][nonzero]))
              if nonzero.any() else None)
    different = int((np.abs(predictions["real"][nonzero] -
                            predictions["reverse"][nonzero]) > 1e-4).sum())
    fraction = different / int(nonzero.sum()) if nonzero.any() else 0.
    failures, conditions = [], {}
    for group, minimum in (("safe", 2), ("danger", 9)):
        real = groups["real"][group]
        name = group + ": no mean improvement"
        conditions[name] = (real["mean_delta_bce"] is not None and
                            real["mean_delta_bce"] < -1e-6)
        name = group + ": too few improved shots"
        conditions[name] = real["improved"] >= minimum
        for arm in ("reverse", "donor"):
            rival = groups[arm][group]["mean_delta_bce"]
            name = group + ": no advantage over " + arm
            conditions[name] = (real["mean_delta_bce"] is not None and
                                rival is not None and real["mean_delta_bce"] < rival - 1e-6)
    conditions["feedback specificity failed"] = (
        cosine is not None and cosine < .99 and fraction > .5)
    failures = [name for name, passed in conditions.items() if not passed]
    return dict(passed=not failures, reasons=failures,
                real_reverse_median_cosine=cosine, different_later_fraction=fraction,
                mean_feedback_not_used_as_utility_gate=True), dict(
                    criterion_pass_by_failure_label=conditions, nonzero_count=int(nonzero.sum()),
                    different_count=different)


def audit(root, include_rows=False):
    output = root / "probe_outputs/anchored_feedback_cold20_v1"
    protocol = read_json(output / "protocol.json")
    complete = read_json(output / "complete.json")
    previous = Path(protocol["previous"])
    old_complete = read_json(previous / "complete.json")
    old_protocol = read_json(previous / "protocol.json")
    records = read_json(output / "records.json")
    folds = read_json(output / "folds.json")
    fit_saved = read_json(output / "fit_groups.json")
    old_train = read_json(previous / "train_groups.json")
    ids = old_protocol["warmup_ids"]
    shot_ids = np.array([r["shot"] for r in records])
    group_ids = np.array([r["group"] for r in records])
    issues, count, hashes = [], collections.Counter(), {}
    errors = collections.defaultdict(list)

    def check(condition, message):
        count["assertions"] += 1
        if not condition:
            issues.append(message)

    def compare(actual, expected, category):
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in actual:
                    issues.append(category + " missing " + key)
                else:
                    compare(actual[key], value, category)
        elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
            error = abs(float(actual) - float(expected))
            errors[category].append(error)
            check(error <= 1e-12, category + " numeric mismatch")
        else:
            check(actual == expected, category + " value mismatch")

    for name, base, manifest in [
        ("artifacts", output, complete["artifacts"]),
        ("inputs", previous, protocol["input_hashes"]),
        ("live_code", root, protocol["code_sha256"]),
        ("frozen_parent", Path(protocol["parent"]), protocol["parent_hashes"]),
    ]:
        mismatch = [file for file, sha in manifest.items() if digest(base / file) != sha]
        hashes[name] = dict(checked=len(manifest), mismatches=mismatch)
        check(not mismatch, name + " hashes")
    check(protocol["input_hashes"] == old_complete["artifacts"], "previous manifest equality")
    with zipfile.ZipFile(output / "code_snapshot.zip") as archive:
        check(set(archive.namelist()) == set(protocol["code_sha256"]), "snapshot members")
        mismatch = [n for n, sha in protocol["code_sha256"].items()
                    if hashlib.sha256(archive.read(n)).hexdigest() != sha]
        hashes["snapshot_members"] = dict(checked=len(archive.namelist()), mismatches=mismatch)
        check(not mismatch, "snapshot digests")
    source = root / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
    hashes["source"] = dict(path=str(source), sha256=digest(source))
    check(hashes["source"]["sha256"] == protocol["source_sha256"], "source hash")
    check(records == read_json(previous / "records.json"), "records copied exactly")
    check(folds == read_json(previous / "folds.json"), "folds copied exactly")
    check(len(records) == complete["opportunities"], "opportunity count")
    check(set(shot_ids) == set(ids) and len(set(ids)) == len(ids), "IDs")
    check(collections.Counter(s for f in folds for s in f["held"]) ==
          collections.Counter(ids), "held partition")
    check(all(set(f["fit"]).isdisjoint(f["held"]) and
              set(f["fit"]) | set(f["held"]) == set(ids) for f in folds), "fold separation")
    check(all(r["origin"] >= 127 and (r["origin"]-127) % 25 == 0 and
              r["maturity"] == r["origin"]+25 and r["query"] == r["origin"]+50
              for r in records), "clock")
    check(len({(r["shot"], r["origin"], r["query"]) for r in records}) ==
          len(records), "unique record clocks")
    expected_files = {f"{kind}_fold{f['fold']}_{sid}.npz"
                      for kind in KINDS for f in folds for sid in ids}
    check({p.name for p in output.glob("*.npz")} == expected_files, "NPZ file coverage")
    observation = torch.load(previous / "observations.pt", map_location="cpu", weights_only=True)
    context = observation["context"].numpy()
    support = observation["support"].numpy()
    check(context.shape == (len(records), 358) and np.isfinite(context).all(), "contexts")
    check(np.array_equal(support, (context[:, 128:].reshape(-1, 5, 46)[:, :, 23:] > 0).
                         any(axis=(1, 2))), "support from available fractions")
    check(np.all(context[:, 128:].reshape(-1, 5, 46)[:, :, [10, 33]] == 0),
          "blocked channel10")
    predictions, donor_report, training_report, scale_report = {}, [], [], []
    for spec in folds:
        fold = spec["fold"]
        fit, held = set(spec["fit"]), set(spec["held"])
        check(sum(np.any((shot_ids == s) & (group_ids == "safe")) for s in held) == 1,
              "one held safe")
        prior = torch.load(previous / f"representation_fold{fold}.pt",
                           map_location="cpu", weights_only=True)
        saved = torch.load(output / f"anchored_fold{fold}.pt",
                           map_location="cpu", weights_only=True)
        check(prior["spec"] == spec and saved["spec"] == spec, "saved fold")
        check(set(prior["state_dict"]) == set(saved["state_dict"]), "parameter names")
        for model in (prior, saved):
            check(sum(v.numel() for v in model["state_dict"].values()) ==
                  protocol["controller_parameters"], "parameter count")
            check(all(bool(torch.isfinite(v).all()) for v in model["state_dict"].values()),
                  "finite model tensors")
        check(torch.equal(prior["mean"], saved["mean"]), "mean reused exactly")
        check(all(torch.equal(v, saved["scale"][n]) for n, v in prior["scale"].items()),
              "scale reused exactly")
        stat = {n: v.numpy() for n, v in prior["scale"].items()}
        selected = np.isin(shot_ids, spec["fit"])
        x = context[selected]
        shot_counts = collections.Counter(shot_ids[selected])
        weights = np.array([1/(len(shot_counts)*shot_counts[s]) for s in shot_ids[selected]])
        hidden = x[:, :128].astype(float)
        center = (hidden*weights[:, None]).sum(axis=0)
        dev = np.sqrt(((hidden-center)**2*weights[:, None]).sum(axis=0))
        blocks = x[:, 128:].reshape(-1, 5, 46)
        valid = blocks[:, :, 23:] > 0
        means = np.where(valid, blocks[:, :, :23], 0).astype(float)
        denominator = (valid*weights[:, None, None]).sum(axis=(0, 1))
        rms = np.sqrt((means**2*weights[:, None, None]).sum(axis=(0, 1)) /
                      np.where(denominator > 0, denominator, 1.))
        expected_scale = dict(
            hcenter=center.astype(np.float32),
            hscale=np.where(dev <= 1e-6, 1., dev).astype(np.float32),
            h_constant=dev <= 1e-6,
            change_scale=np.where(rms <= 1e-6, 1., rms).astype(np.float32),
            change_constant=rms <= 1e-6, change_unobservable=denominator == 0)
        scale_error = max(float(np.max(np.abs(stat[n].astype(float)-v.astype(float))))
                          for n, v in expected_scale.items())
        check(scale_error < 1e-6, "fit-only scale statistics")
        scaled = context.copy()
        scaled[:, :128] = (context[:, :128]-stat["hcenter"])/stat["hscale"]
        b = scaled[:, 128:].reshape(-1, 5, 46)
        b[:, :, :23] = np.where(b[:, :, 23:] > 0, b[:, :, :23]/stat["change_scale"], 0.)
        mean = np.stack([scaled[shot_ids == s].mean(axis=0) for s in spec["fit"]]).mean(axis=0)
        mean_error = float(np.max(np.abs(mean-prior["mean"].numpy())))
        check(mean_error < 3e-6, "fit-only control mean")
        scale_report.append(dict(fold=fold, fit_rows=int(selected.sum()),
                                 scale_max_error=scale_error, mean_max_error=mean_error))

        donors = np.asarray(read_json(output / f"fold{fold}_donors.json"))
        check(donors.shape == (len(records),) and donors.dtype.kind in "iu" and
              ((donors >= 0) & (donors < len(records))).all(), "donor indices")
        check(np.isin(shot_ids[donors], spec["fit"]).all(), "donors fit only")
        check((shot_ids[donors] != shot_ids).all(), "donors exclude same shot")
        # Full float64 distance vectors; no labels, clocks, feedback or model calls.
        origins = scaled[:, :128].astype(np.float64)
        candidate_by_shot = {sid: np.flatnonzero(selected & (shot_ids != sid)) for sid in ids}
        chosen, ties, margins = [], 0, []
        for row, sid in enumerate(shot_ids):
            candidates = candidate_by_shot[sid]  # ascending global index
            distance = ((origins[candidates]-origins[row])**2).mean(axis=1)
            minimum = distance.min()
            tied = np.flatnonzero(distance == minimum)
            ties += int(len(tied) > 1)
            chosen.append(int(candidates[tied[0]]))
            ordered = np.partition(distance, 1)
            margins.append(float(ordered[1]-ordered[0]))
        mismatch = int(np.count_nonzero(donors != np.asarray(chosen)))
        check(mismatch == 0, "donor full nearest-neighbor recomputation")
        count["donor_assignments"] += len(donors)
        donor_report.append(dict(fold=fold, checked=len(donors), mismatches=mismatch,
                                 fit_rows=int(selected.sum()), tied_minima=ties,
                                 min_first_second_gap=min(margins)))

        tr = read_json(output / f"anchored_fold{fold}_training.json")
        old = read_json(previous / f"representation_fold{fold}_training.json")
        draw = np.asarray(tr["draws"])
        check(np.array_equal(draw, np.asarray(old["draws"])), "draws identical to ordinary run")
        check(draw.shape == (protocol["steps"], protocol["batch_size"]), "draw shape")
        check(np.isin(shot_ids[draw], spec["fit"]).all(), "teaching fit only")
        check((group_ids[draw[:, :12]] == "safe").all() and
              (group_ids[draw[:, 12:]] == "danger").all(), "12+12 teaching classes")
        pools = {g: {s: np.flatnonzero((shot_ids == s) & (group_ids == g))
                     for s in sorted(fit) if ((shot_ids == s) & (group_ids == g)).any()}
                 for g in ("safe", "danger")}
        rng = np.random.default_rng(131)
        expected_draws = []
        for _ in range(protocol["steps"]):
            one = []
            for g in ("safe", "danger"):
                for _ in range(12):
                    sid = int(rng.choice(sorted(pools[g])))
                    one.append(int(rng.choice(pools[g][sid])))
            expected_draws.append(one)
        check(np.array_equal(draw, np.asarray(expected_draws)), "independent seed131 draws")
        count["training_steps"] += len(tr["history"])
        count["draw_positions"] += draw.size
        check([h["step"] for h in tr["history"]] == list(range(1, protocol["steps"]+1)),
              "training step history")
        check(all(np.isfinite([h["loss"], h["gradient_norm"], h["mean_matrix_norm"]]).all()
                  for h in tr["history"]), "finite history")
        training_report.append(dict(
            fold=fold, first32_loss=float(np.mean([h["loss"] for h in tr["history"][:32]])),
            last32_loss=float(np.mean([h["loss"] for h in tr["history"][-32:]]))))

        for kind in KINDS:
            parts = []
            for sid in ids:
                filename = f"{kind}_fold{fold}_{sid}.npz"
                values = load_npz(output / filename)
                ix = shot_ids == sid
                check(set(values) == EXPECTED_KEYS, "NPZ keys " + filename)
                for name, value in values.items():
                    check(value.shape == (int(ix.sum()),) and np.isfinite(value).all(),
                          "NPZ shape/finite " + filename + "/" + name)
                check(np.array_equal(values["support"], support[ix]), "NPZ observation support")
                parts.append(values)
                count["npz_files"] += 1
                count["arrays"] += len(values)
                count["prediction_rows"] += int(ix.sum())
                old_npz = load_npz(previous / f"representation_{sid}.npz")
                check(np.array_equal(values["cold"], old_npz["cold"]), "cold exact across folds")
                count["cold_comparisons"] += 1
                if kind == "ordinary" and sid in held:
                    shared = set(old_npz) & set(values)
                    for name in shared:
                        check(np.array_equal(values[name], old_npz[name]),
                              "ordinary held baseline exact " + name)
                    count["ordinary_prior_shared_arrays"] += len(shared)
            predictions[kind, fold] = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
        for name in ("cold", "support"):
            check(np.array_equal(predictions["ordinary", fold][name],
                                 predictions["anchored", fold][name]), "paired " + name)

    all_rows, held_groups, held_gates, fold_summaries, controls = [], {}, {}, [], []
    for kind in KINDS:
        held_parts = []
        for sid in ids:
            fold = next(f["fold"] for f in folds if sid in f["held"])
            ix = shot_ids == sid
            held_parts.append({n: v[ix] for n, v in predictions[kind, fold].items()})
        packed = {n: np.concatenate([p[n] for p in held_parts]) for n in held_parts[0]}
        check([s for s in ids for r in records if r["shot"] == s] == shot_ids.tolist(),
              "record order for packing")
        held_rows = shot_rows(records, packed)
        groups = group_rows(held_rows)
        held_groups[kind] = groups
        saved_rows = read_json(output / f"{kind}_held_per_shot.json")
        lookup = {(r["shot"], r["group"], r["arm"]): r for r in held_rows}
        check(len(saved_rows) == len(held_rows), "held row count")
        for r in saved_rows:
            compare(lookup[r["shot"], r["group"], r["arm"]], r, "held_per_shot")
        count["held_rows_compared"] += len(saved_rows)
        compare(groups, complete["summaries"][kind]["groups"], "held_groups")
        count["held_group_cells"] += sum(len(v) for v in groups.values())
        status, conditions = gate(groups, packed)
        compare(status, complete["summaries"][kind]["gate"], "held_gate")
        held_gates[kind] = dict(**status, **conditions)

        for spec in folds:
            fold = spec["fold"]
            prediction = predictions[kind, fold]
            raw_nonzero = (prediction["real_raw_norm"] > 0) & (prediction["reverse_raw_norm"] > 0)
            check(np.array_equal(raw_nonzero, prediction["direction_nonzero"]), "direction mask")
            norm_errors = {}
            for arm in ARMS:
                check((prediction[arm+"_raw_norm"] >= 0).all() and
                      (prediction[arm+"_raw_norm"] <= .100001).all(), "raw norm bounds")
                check((prediction[arm+"_norm"] >= 0).all() and
                      (prediction[arm+"_norm"] <= .100001).all(), "norm bounds")
                expected = (prediction["real_raw_norm"] if arm == "real" else
                            np.where(prediction[arm+"_raw_norm"] > 0,
                                     prediction["real_raw_norm"], 0.))
                norm_errors[arm] = float(np.max(np.abs(prediction[arm+"_norm"]-expected)))
                check(norm_errors[arm] <= 1e-6, "matched control norms")
                check((np.abs(prediction[arm+"_cos_real"]) <= 1.000001).all(), "cosine bounds")
            if kind == "anchored":
                for name in ("mean_feedback_norm", "mean_feedback_raw_norm", "mean_feedback_cos_real"):
                    check(np.count_nonzero(prediction[name]) == 0, "anchored mean exact zero " + name)
                check(np.array_equal(prediction["mean_feedback"], prediction["cold"]),
                      "anchored mean logits exactly cold")
                count["anchored_zero_mean_rows"] += len(records)
            controls.append(dict(kind=kind, fold=fold, support=int(prediction["support"].sum()),
                                 nonzero_reverse=int(raw_nonzero.sum()), norm_errors=norm_errors,
                                 real_norm_range=[float(prediction["real_norm"].min()),
                                                  float(prediction["real_norm"].max())],
                                 zero_direction_counts={a:int((prediction[a+"_raw_norm"] == 0).sum())
                                                        for a in ARMS}))
            for split in ("fit", "held"):
                selected = np.isin(shot_ids, spec[split])
                rr = [r for r, keep in zip(records, selected) if keep]
                pp = {n: v[selected] for n, v in prediction.items()}
                rows = shot_rows(rr, pp)
                gg = group_rows(rows)
                for row in rows:
                    all_rows.append(dict(kind=kind, fold=fold, split=split, **row))
                count[split+"_per_shot_rows"] += len(rows)
                count[split+"_arm_window_BCEs"] += sum(r["windows"] for r in rows)
                if split == "fit":
                    target = next(v["groups"] for v in fit_saved
                                  if v["kind"] == kind and v["fold"] == fold)
                    compare(gg, target, "fit_groups")
                    count["fit_group_cells"] += sum(len(v) for v in gg.values())
                    if kind == "ordinary":
                        old_groups = next(v["groups"] for v in old_train
                                          if v["kind"] == "representation" and v["fold"] == fold)
                        for arm in ("real", "reverse", "mean_feedback"):
                            compare(gg[arm], old_groups[arm], "ordinary_prior_fit_groups")
                fold_summaries.append(dict(kind=kind, fold=fold, split=split, groups=gg))
    paired = {}
    for group in GROUPS:
        ordinary = {r["shot"]:r["delta_bce"] for r in all_rows if
                    r["kind"]=="ordinary" and r["split"]=="held" and
                    r["group"]==group and r["arm"]=="real"}
        anchored = {r["shot"]:r["delta_bce"] for r in all_rows if
                    r["kind"]=="anchored" and r["split"]=="held" and
                    r["group"]==group and r["arm"]=="real"}
        differences = np.array([anchored[s]-ordinary[s] for s in sorted(ordinary)])
        paired[group] = dict(mean_anchored_minus_ordinary=float(differences.mean()),
                             improved=int((differences < -1e-6).sum()),
                             worsened=int((differences > 1e-6).sum()),
                             per_shot={str(s):anchored[s]-ordinary[s] for s in sorted(ordinary)})
    check(len(fit_saved) == len(folds)*len(KINDS), "fit summaries count")
    report = dict(
        issues=issues, coverage=dict(count), hashes=hashes,
        max_absolute_errors={k:max(v, default=0.) for k,v in errors.items()},
        record_groups=dict(collections.Counter(group_ids)),
        source_shots={"safe":len({r["shot"] for r in records if r["group"]=="safe"}),
                      "danger":len({r["shot"] for r in records if r["group"]=="danger"})},
        held_groups=held_groups, held_gates=held_gates, fold_summaries=fold_summaries,
        donors=donor_report, scales=scale_report, histories=training_report,
        controls=controls, paired_held=paired,
        primary_exceptions=[r for r in all_rows if r["split"]=="held" and r["arm"]=="real"
                            and r["group"] in ("safe","danger") and r["delta_bce"] > 1e-6])
    if include_rows:
        report["per_shot_rows"] = all_rows
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--rows", action="store_true",
                        help="include every fit/held shot/group/arm BCE row in stdout")
    args = parser.parse_args()
    result = audit(args.root, args.rows)
    print(json.dumps(result, allow_nan=False, indent=2))
    raise SystemExit(bool(result["issues"]))

