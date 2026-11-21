"""Recount sealed direction-span curves against HDF5 and post-hoc menu bounds.

The only write is a new report, outside the run. This verifier cannot prove that
the producer obeyed its causal declarations; it checks their sealed receipts,
dependencies, ordering, saved curves, and independently read HDF5 outcomes.
Menu bounds use revealed outcomes and are not implementable online policies.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_layer0_action_utility_20261106 import (
    first_is_timely,
    miss_reason,
    score_first,
)
from layer0_action_utility_20261106 import alarm_envelope
from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    RAW_ROOT,
    counts,
    need,
    raw_truth,
    read,
    score,
)
from verify_layer0_signed_projection_20261106 import FIT, MAP, SOURCE, TAU
from verify_moving_peak_replay_20261105 import ATOL, check_curve, parity, valid_sha


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "probe_outputs"
REFS = {
    "fixed": (ARCHIVE / "layer0_signed_fixed360_20261106_v1",
              ARCHIVE / "layer0_signed_fixed360_verification_20261106_v1.json",
              360, 21, (180, 180), "fixed360"),
    "conditional": (ARCHIVE / "layer0_signed_conditional835_20261106_v1",
                    ARCHIVE / "layer0_signed_conditional835_verification_20261106_v1.json",
                    835, 1, (510, 325), "conditional835"),
}
ORACLE = ARCHIVE / "layer0_action_oracle_20261106_v1.json"
FIRST, STRIDE, RELATIVE_EPS = 135, 32, 1e-10
FAMILIES = ("full", "in_span", "out_span", "random")
ARMS = ("static", "intra", *(f"{family}_{sign}" for family in FAMILIES for sign in ("pos", "neg")))
MENUS = {"binary": ("static", "intra"), **{
    family: ("static", "intra", f"{family}_pos", f"{family}_neg") for family in FAMILIES
}}
RECEIPTS = ("anchor_norms", "anchor_gradient_norms", "anchor_valid")
SEAL_KEYS = {"rank", "shot", "raw_sha256", "curve_sha256", "previous_seal_sha256", "active", "seal_ns"}


def integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def is_good(row, *, capped):
    return bool(row["status"] == "tp" and (not capped or row["lead_ms"] <= 3000))


def compact(row):
    return {key: row[key] for key in ("status", "first_index", "lead_ms")}


def audit_anchors(a, n_times):
    """Check saved direction receipts, and count inversions at actual anchors."""
    idx = np.arange(FIRST, n_times, STRIDE, dtype=np.int64)
    n = len(idx)
    norms = np.asarray(a["anchor_norms"])
    grad = np.asarray(a["anchor_gradient_norms"])
    valid = np.asarray(a["anchor_valid"])
    need(norms.shape == (n, 5) and norms.dtype == np.dtype("float32")
         and grad.shape == (n, 3) and grad.dtype == np.dtype("float64")
         and valid.shape == (n, 4) and valid.dtype == np.dtype("bool")
         and np.isfinite(norms).all() and np.isfinite(grad).all()
         and (norms >= 0).all() and (grad >= 0).all(),
         "invalid anchor receipt shapes, dtypes, or values")
    original, full = norms[:, 0], grad[:, 0]
    expected = np.empty((n, 4), dtype=bool)
    expected[:, 0] = (original > 0) & (full > 0)
    expected[:, 1] = expected[:, 0] & (grad[:, 1] > RELATIVE_EPS * full)
    expected[:, 2] = expected[:, 0] & (grad[:, 2] > RELATIVE_EPS * full)
    expected[:, 3] = original > 0
    need(np.array_equal(valid, expected), "incorrect direction validity flags")
    target = norms[:, 1:].astype(np.float64)
    difference = np.abs(target - original[:, None].astype(np.float64))
    need((difference[valid] <= 1e-6).all() and (target[~valid] == 0).all(),
         "direction norm mismatch or nonzero invalid direction")
    # Scale per row to avoid underflow/overflow when checking squared energy.
    scale = np.max(grad, axis=1, initial=0)
    scaled = grad / np.where(scale > 0, scale, 1)[:, None]
    lhs = scaled[:, 0] ** 2
    rhs = scaled[:, 1] ** 2 + scaled[:, 2] ** 2
    need(np.isclose(lhs, rhs, rtol=1e-9, atol=0).all(),
         "gradient squared energy does not decompose")
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.square(grad).sum(axis=0)
    need(np.isfinite(sums).all(), "gradient energy overflow")
    baseline = np.asarray(a["static"], dtype=np.float64)[idx]
    inversions = {}
    for col, family in enumerate(FAMILIES):
        pos = np.asarray(a[f"{family}_pos"], dtype=np.float64)[idx]
        neg = np.asarray(a[f"{family}_neg"], dtype=np.float64)[idx]
        mask = valid[:, col]
        positive, negative = pos < baseline - 1e-6, neg > baseline + 1e-6
        inversions[family] = dict(denominator=n, positive=int(positive.sum()),
                                  negative=int(negative.sum()), valid_denominator=int(mask.sum()),
                                  valid_positive=int(np.count_nonzero(mask & positive)),
                                  valid_negative=int(np.count_nonzero(mask & negative)))
    return dict(n_anchors=n, sign_inversions=inversions,
                gradient_energy=dict(full_squared_sum=float(sums[0]),
                                     in_span_squared_sum=float(sums[1]),
                                     out_span_squared_sum=float(sums[2]),
                                     nonzero_full=int(np.count_nonzero(full))))


def menu_result(times, probs, truth, *, first=FIRST, stride=STRIDE):
    """Independently optimize strict/capped first hits, or minimize safe FP."""
    possible = alarm_envelope(times, probs, disruptive=truth["disruptive"],
                              down=truth["down"], threshold=TAU, first=first, stride=stride)
    result = {}
    static = score(times, probs[0], truth, TAU)
    for label, capped in (("strict", False), ("capped", True)):
        if truth["disruptive"]:
            chosen = next((int(i) for i in possible
                           if first_is_timely(times, truth["down"], int(i), capped=capped)),
                          int(possible[0]))
        else:
            chosen = len(times) if possible[-1] == len(times) else int(possible[0])
        result[label] = score_first(times, chosen, truth)
        if truth["disruptive"] and not is_good(static, capped=capped):
            why = miss_reason(times, probs, truth, chosen, capped=capped)
            need((why == "selectable_rescue") == is_good(result[label], capped=capped),
                 "oracle first-hit/reason mismatch")
            result[label + "_relative_to_static_miss"] = why
    return result


def check_basis(path, expected, fit_path, h):
    need(valid_sha(expected) and h(path) == expected, "basis hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        need(set(data.files) == {"basis", "singular_values"}, "invalid basis fields")
        q, singular = data["basis"], data["singular_values"]
        need(q.shape == (128, 33) and singular.shape == (33,)
             and q.dtype.kind == singular.dtype.kind == "f"
             and np.isfinite(q).all() and np.isfinite(singular).all()
             and (singular > 0).all()
             and (singular[:-1] >= singular[1:]).all()
             and np.allclose(q.T.astype(np.float64) @ q.astype(np.float64),
                             np.eye(33), rtol=0, atol=1e-6),
             "invalid basis rank, orthogonality, or singular values")
        with np.load(fit_path, allow_pickle=False) as fit:
            w, b = fit["layers.2.weight"], fit["layers.2.bias"]
            need(w.shape == (256, 32) and b.shape == (256,)
                 and w.dtype.kind == b.dtype.kind == "f"
                 and np.isfinite(w).all() and np.isfinite(b).all(),
                 "invalid frozen generator output layer")
            columns = np.c_[w[:128].astype(np.float64), b[:128].astype(np.float64)]
        fit_q, fit_s, _ = np.linalg.svd(columns, full_matrices=False)
        cutoff = fit_s[0] * max(columns.shape) * np.finfo(np.float64).eps
        need(np.count_nonzero(fit_s > cutoff) == 33
             and np.allclose(singular, fit_s, rtol=1e-8, atol=1e-12),
             "basis singular values or rank differ from frozen fit")
        residual = q.astype(np.float64) - fit_q @ (fit_q.T @ q.astype(np.float64))
        error = float(np.max(np.linalg.norm(residual, axis=0), initial=0))
        need(error <= 1e-6, "basis span differs from frozen fit")
        return error


def check_config(cfg, done, ref_cfg, ref_done, ref_audit, n, active_rank, mode, h, run):
    order = cfg["order"]
    raw = cfg["raw_sha256"]
    deps = cfg["input_sha256"]
    need(isinstance(order, list) and len(order) == len(set(order)) == n
         and all(integer(sid, 1) for sid in order)
         and order == ref_cfg["order"] and raw == ref_cfg["raw_sha256"]
         and isinstance(raw, dict) and set(raw) == {str(sid) for sid in order}
         and all(valid_sha(v) for v in raw.values())
         and tuple(cfg["arms"]) == ARMS
         and cfg["mode"] == ref_cfg["mode"] == mode
         and all(type(cfg[k]) is int for k in
                 ("active_rank", "first", "stride", "basis_rank", "random_seed"))
         and cfg["active_rank"] == active_rank
         and (cfg["first"], cfg["stride"], cfg["basis_rank"], cfg["random_seed"])
         == (FIRST, STRIDE, 33, 131)
         and cfg["relative_direction_epsilon"] == RELATIVE_EPS
         and cfg["threshold"] == ref_cfg["threshold"] == TAU
         and (ref_cfg["first"], ref_cfg["stride"]) == (FIRST, STRIDE)
         and cfg["source_state_sha256"] == ref_cfg["source_state_sha256"] == SOURCE
         and cfg["source_fit_sha256"] == ref_cfg["source_fit_sha256"] == FIT
         and cfg["source_map_sha256"] == MAP
         and cfg["development_only"] is True
         and all(type(cfg[k]) is int and cfg[k] == 0 for k in
                 ("new_fit_labels", "inter_updates", "outcome_reads_during_prediction"))
         and integer(cfg["initial_ns"], 1), "invalid protocol, cohort, or causal declarations")
    need(isinstance(deps, dict) and deps and FIT in deps.values() and MAP in deps.values(),
         "missing source fit or map input binding")
    for name, digest in deps.items():
        need(isinstance(name, str) and Path(name).is_absolute() and valid_sha(digest),
             "invalid absolute input hash binding")
        need(h(name) == digest, f"input hash changed: {name}")
    bound = {Path(name).resolve() for name in deps}
    ref = REFS[mode][0]
    need({(ref / "protocol.json").resolve(), (ref / "completed.json").resolve()} <= bound,
         "signed reference protocol and completion must be bound inputs")
    fit_path = next(Path(name) for name, digest in deps.items() if digest == FIT)
    basis_fit_error = check_basis(run / "basis.npz", cfg["basis_sha256"], fit_path, h)
    need(done["status"] == "COMPLETE_LAYER0_DIRECTION_SPAN" and done["mode"] == mode
         and done["protocol_sha256"] == h(run / "protocol.json")
         and done["n_shots"] == len(done["sealed"]) == n
         and done["n_curves"] == n * len(ARMS)
         and integer(done["completed_ns"], cfg["initial_ns"])
         and isinstance(done["seconds"], (int, float)) and not isinstance(done["seconds"], bool)
         and np.isfinite(done["seconds"]) and done["seconds"] >= 0
         and set(done["max_reference_error"]) == {"static", "intra"}
         and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 and np.isfinite(v) and 0 <= v <= ATOL
                 for v in done["max_reference_error"].values()), "invalid completion receipt")
    need(ref_done["status"] == "COMPLETE_LAYER0_SIGNED_PROJECTION"
         and ref_done["mode"] == mode
         and ref_done["n_shots"] == len(ref_done["sealed"]) == n
         and ref_done["n_curves"] == n * 6
         and ref_audit["status"] == "PASS_SIGNED_PROJECTION_HDF5_RECOUNT"
         and ref_audit["producer_sha256"] == h(REFS[mode][0] / "completed.json"),
         "reference signed archive is not verified")
    return basis_fit_error


def preflight_shots(run, ref, cfg, done, ref_cfg, old, h):
    """Verify all prediction files/seals before opening any HDF5 outcome."""
    previous, old_previous = None, None
    last, old_last = cfg["initial_ns"], ref_cfg["initial_ns"]
    for rank, sid in enumerate(cfg["order"], 1):
        stem = f"{rank:04d}_{sid}"
        curve, seal = run / "shots" / f"{stem}.npz", run / "shots" / f"{stem}_seal.json"
        old_curve, old_seal = ref / "shots" / f"{stem}.npz", ref / "shots" / f"{stem}_seal.json"
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        h(seal)
        h(old_seal)
        note, prior = read(seal), read(old_seal)
        active = rank >= cfg["active_rank"]
        need(set(note) == SEAL_KEYS and done["sealed"][rank - 1] == dict(note, seal_sha256=h(seal))
             and (note["rank"], note["shot"]) == (rank, sid) and note["active"] is active
             and note["raw_sha256"] == cfg["raw_sha256"][str(sid)] == h(raw)
             and note["curve_sha256"] == h(curve)
             and note["previous_seal_sha256"] == previous
             and integer(note["seal_ns"], last) and note["seal_ns"] <= done["completed_ns"],
             f"invalid new seal, sequence, or raw binding: {sid}")
        need(old["sealed"][rank - 1] == dict(prior, seal_sha256=h(old_seal))
             and (prior["rank"], prior["shot"]) == (rank, sid)
             and prior["raw_sha256"] == note["raw_sha256"]
             and prior["curve_sha256"] == h(old_curve)
             and prior["previous_seal_sha256"] == old_previous
             and prior["bank_enabled"] is active
             and integer(prior["seal_ns"], old_last) and prior["seal_ns"] <= old["completed_ns"],
             f"invalid reference seal or sequence: {sid}")
        previous, last = h(seal), note["seal_ns"]
        old_previous, old_last = h(old_seal), prior["seal_ns"]


def summary(rows, capped):
    base = counts(rows)
    tp = base["tp_30_to_3000"] if capped else base["tp"]
    return dict(n_disruptive=base["n_disruptive"], n_safe=base["n_safe"],
                max_tp=tp, min_fp=base["fp"],
                max_tpr=tp / base["n_disruptive"] if base["n_disruptive"] else None,
                min_fpr=base["fpr"])


def actual_summary(rows):
    result = counts(rows)
    result["tpr_30_to_3000"] = (result["tp_30_to_3000"] / result["n_disruptive"]
                                if result["n_disruptive"] else None)
    return result


def cohort_summary(records):
    cohort = [row for row in records if row["prior_extended_capped_reason"] == "no_eligible_candidate_crossing"]
    menu_counts = {}
    for menu in MENUS:
        reasons = [row["menus"][menu]["capped_relative_to_static_miss"] for row in cohort]
        menu_counts[menu] = dict(
            newly_reachable_capped_first_alarms=reasons.count("selectable_rescue"),
            eligible_crossing_cannot_be_first=reasons.count("eligible_crossing_cannot_be_first"),
            no_eligible_candidate_crossing=reasons.count("no_eligible_candidate_crossing"))
        need(sum(menu_counts[menu].values()) == len(cohort), "incomplete cohort reason classification")
    need(menu_counts["binary"]["no_eligible_candidate_crossing"] == len(cohort),
         "prior no-crossing cohort is inconsistent with binary baseline")
    comparisons = {}
    pairs = [(family, "binary") for family in FAMILIES] + [(family, "random") for family in FAMILIES[:-1]]
    for a, b in pairs:
        good_a = {row["shot"] for row in cohort if row["menus"][a]["capped_relative_to_static_miss"] == "selectable_rescue"}
        good_b = {row["shot"] for row in cohort if row["menus"][b]["capped_relative_to_static_miss"] == "selectable_rescue"}
        comparisons[f"{a}_vs_{b}"] = dict(
            newly_reachable_difference=len(good_a) - len(good_b),
            gained_shots=sorted(good_a - good_b), lost_shots=sorted(good_b - good_a),
            cannot_be_first_difference=(menu_counts[a]["eligible_crossing_cannot_be_first"]
                                        - menu_counts[b]["eligible_crossing_cannot_be_first"]))
    return dict(n_shots=len(cohort), shots=[row["shot"] for row in cohort],
                menus=menu_counts, comparisons=comparisons)


def verify(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    need(not output.resolve().is_relative_to(run.resolve()), "output must be outside the run directory")
    h = Hashes()
    h(run / "protocol.json")
    h(run / "completed.json")
    cfg, done = (read(run / name) for name in ("protocol.json", "completed.json"))
    mode = cfg["mode"]
    need(mode in REFS, "invalid run mode")
    ref, checked, n, active_rank, expected_counts, cohort_key = REFS[mode]
    for path in (ref / "protocol.json", ref / "completed.json", checked, ORACLE):
        h(path)
    ref_cfg, old, ref_audit = (read(p) for p in (ref / "protocol.json", ref / "completed.json", checked))
    need(old["protocol_sha256"] == h(ref / "protocol.json"), "reference protocol hash mismatch")
    basis_fit_error = check_config(cfg, done, ref_cfg, old, ref_audit, n, active_rank, mode, h, run)
    parent_sha = h(ref / "completed.json")
    prior = read(ORACLE)
    cohort = prior["cohorts"][cohort_key]
    need(prior["status"] == "PASS_POSTHOC_ACTION_MENU_HDF5"
         and cohort["parent_sha256"] == parent_sha and cohort["n_shots"] == n
         and len(cohort["per_shot"]) == n, "prior oracle provenance mismatch")
    preflight_shots(run, ref, cfg, done, ref_cfg, old, h)
    arm_rows = {arm: [] for arm in ARMS}
    menu_rows = {menu: {kind: [] for kind in ("strict", "capped")} for menu in MENUS}
    records, max_error, anchor_total = [], dict(static=0.0, intra=0.0), []
    for rank, (sid, old_row) in enumerate(zip(cfg["order"], cohort["per_shot"], strict=True), 1):
        stem = f"{rank:04d}_{sid}"
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        truth, clock = raw_truth(raw, sid)
        need((old_row["rank"], old_row["shot"], old_row["disruptive"], old_row["down"])
             == (rank, sid, truth["disruptive"], truth["down"]), "prior oracle shot or HDF5 outcome mismatch")
        with np.load(run / "shots" / f"{stem}.npz", allow_pickle=False) as a, \
             np.load(ref / "shots" / f"{stem}.npz", allow_pickle=False) as b:
            active = rank >= active_rank
            need(set(a.files) == {"times", *ARMS, *(RECEIPTS if active else ())}
                 and np.array_equal(a["times"], clock) and np.array_equal(b["times"], clock),
                 f"invalid fields or HDF5 clock: {sid}")
            curves = {arm: a[arm] for arm in ARMS}
            for arm, values in curves.items():
                need(values.dtype.kind == "f", f"invalid curve dtype: {sid}/{arm}")
                check_curve(values, clock, f"{sid}/{arm}")
                end = min(FIRST, len(clock)) if active else len(clock)
                need(np.array_equal(values[:end], curves["static"][:end]),
                     f"pre-action or inactive curve changed: {sid}/{arm}")
                arm_rows[arm].append(score(clock, values, truth, TAU))
            for arm in ("static", "intra"):
                max_error[arm] = max(max_error[arm], parity(curves[arm], b[arm], TAU, f"{sid}/{arm}"))
            need(old_row["static"] == arm_rows["static"][-1], "prior oracle baseline mismatch")
            ref_arms = ("static", "intra", "signed", "blind_signed")
            for arm in ref_arms:
                check_curve(b[arm], clock, f"reference {sid}/{arm}")
            old_menu = menu_result(clock, np.stack([b[arm] for arm in ref_arms]), truth)
            old_reason = old_row["menus"]["extended"].get("capped_relative_to_static_miss")
            need(old_reason == old_menu.get("capped_relative_to_static_miss"), "prior oracle reason mismatch")
            audit = audit_anchors({**curves, **{key: a[key] for key in RECEIPTS}}, len(clock)) if active else None
            if audit is not None:
                anchor_total.append(audit)
            shot_menus = {}
            for menu, arms in MENUS.items():
                result = menu_result(clock, np.stack([curves[arm] for arm in arms]), truth)
                for kind in ("strict", "capped"):
                    menu_rows[menu][kind].append(result[kind])
                shot_menus[menu] = {key: compact(value) if key in ("strict", "capped") else value
                                    for key, value in result.items()}
            records.append(dict(shot=sid, rank=rank, disruptive=truth["disruptive"], down=truth["down"],
                                n_times=len(clock), prior_extended_capped_reason=old_reason,
                                arms={arm: compact(arm_rows[arm][-1]) for arm in ARMS},
                                menus=shot_menus, anchor_audit=audit))
    for arm in ("static", "intra"):
        need(np.isclose(max_error[arm], done["max_reference_error"][arm], rtol=0, atol=1e-12),
             f"reported reference error mismatch: {arm}")
    actual = {arm: actual_summary(rows) for arm, rows in arm_rows.items()}
    actual_active = {arm: actual_summary(rows[active_rank - 1:]) for arm, rows in arm_rows.items()}
    need((actual["static"]["n_disruptive"], actual["static"]["n_safe"]) == expected_counts,
         "incorrect HDF5 outcome population")
    menus = {menu: {kind: summary(rows, kind == "capped") for kind, rows in sub.items()}
             for menu, sub in menu_rows.items()}
    menus_active = {menu: {kind: summary(rows[active_rank - 1:], kind == "capped")
                           for kind, rows in sub.items()} for menu, sub in menu_rows.items()}
    for family in FAMILIES:
        for kind in ("strict", "capped"):
            need(menus[family][kind]["max_tp"] >= menus["binary"][kind]["max_tp"]
                 and menus[family][kind]["min_fp"] <= menus["binary"][kind]["min_fp"],
                 "expanded oracle menu cannot lose options")
    pairs = [(family, "binary") for family in FAMILIES] + [(family, "random") for family in FAMILIES[:-1]]
    menu_comparisons = {
        f"{a}_vs_{b}": {
            kind: {f"{metric}_difference": menus[a][kind][metric] - menus[b][kind][metric]
                   for metric in ("max_tp", "min_fp")}
            for kind in ("strict", "capped")
        } for a, b in pairs
    }
    signs = {family: {key: sum(row["sign_inversions"][family][key] for row in anchor_total)
                      for key in ("denominator", "positive", "negative", "valid_denominator",
                                  "valid_positive", "valid_negative")} for family in FAMILIES}
    energy = {key: sum(row["gradient_energy"][key] for row in anchor_total)
              for key in ("full_squared_sum", "in_span_squared_sum", "out_span_squared_sum", "nonzero_full")}
    energy["in_span_fraction"] = (energy["in_span_squared_sum"] / energy["full_squared_sum"]
                                  if energy["full_squared_sum"] else None)
    energy["out_span_fraction"] = (energy["out_span_squared_sum"] / energy["full_squared_sum"]
                                   if energy["full_squared_sum"] else None)
    payload = dict(status="PASS_LAYER0_DIRECTION_SPAN_HDF5_RECOUNT", mode=mode,
                   development_only=True, n_shots=n, n_curves=n * len(ARMS),
                   scope=("Actual arms are fixed-curve HDF5 scores; menu results are post-hoc "
                          "per-shot bounds, not policies or independent validation. Causal "
                          "declarations are checked as receipts, not independently proven."),
                   hashes=dict(verifier_sha256=h(Path(__file__)), producer_sha256=h(run / "completed.json"),
                               protocol_sha256=h(run / "protocol.json"), basis_sha256=h(run / "basis.npz"),
                               reference_completed_sha256=parent_sha,
                               reference_verification_sha256=h(checked), prior_oracle_sha256=h(ORACLE)),
                   max_reference_error=max_error, basis_fit_max_subspace_error=basis_fit_error,
                   actual_arms=actual, actual_arms_active=actual_active,
                   menus=menus, menus_active=menus_active, menu_comparisons=menu_comparisons,
                   prior_extended_no_eligible_capped=cohort_summary(records),
                   anchor_sign_inversions=signs, gradient_energy=energy, per_shot=records)
    h.check()
    with output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir, args.output_json)
    print(f"verified {result['mode']}: {result['n_shots']} shots / {result['n_curves']} curves")


if __name__ == "__main__":
    main()
