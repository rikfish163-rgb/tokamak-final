"""Canonical causal stream experiment runner for zero-cold-start evaluation."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from . import backbone, data, engine, metrics

RESULTS_DIR = Path(__file__).with_name("results")


def _provenance():
    pkg = Path(__file__).parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "source_model_sha256": sha(data.ARTIFACTS / "jtext_source.pt"),
        "modules_sha256": {p.name: sha(p) for p in sorted(pkg.glob("*.py"))},
    }


def load_cache(shots, device="EAST"):
    out = {}
    for s in shots:
        sig = data.read_signals(device, s)
        dis, dt = data.read_outcome(device, s)
        out[int(s)] = (sig["times"], sig["values"], sig["available"], dis, dt)
    return out


def source_coords(ckpt, align_mode):
    if align_mode == "efd":
        return np.asarray(ckpt["source_mean"], float), np.asarray(ckpt["source_std"], float)
    return np.asarray(ckpt["robust_median"], float), np.asarray(ckpt["robust_iqr"], float)


def _timely_auc(rows, timely_scores):
    y = [r["disruptive"] for r in rows]
    return float(roc_auc_score(y, timely_scores)) if any(y) and not all(y) else None


def run_stream(shots, cache, model, ckpt, cfg: engine.EngineConfig, *, meta=None, device="cuda",
               deadline_ms=30.0, checkpoints=(1, 5, 10, 20, 50, 100, 180, 360),
               return_rows=False):
    t0 = time.perf_counter()
    sc, ss = source_coords(ckpt, cfg.align_mode)
    st = engine.InterShotState.from_source(sc, ss, cfg)
    rows, timely_scores = [], []
    causes, applied_count = Counter(), 0
    unreachable = Counter()
    for s in shots:
        sid = int(s)
        t, v, a, dis, dt = cache[sid]
        res = engine.run_shot_causal(
            sid, t, v, a,
            lambda vv, aa: backbone.predict_shot_proba(model, vv, aa, device=device),
            st, cfg, sc, ss,
        )
        m = meta.get(sid, {}) if meta is not None else {}
        rec = engine.reflect_and_update(res, v, a, dis, dt, st, cfg, sc, ss)
        causes[rec.root_cause] += 1
        applied_count += int(rec.applied)
        row = metrics.score_shot(
            res.times, res.risk_curve, res.threshold_curve, dis, dt,
            hold_ms=cfg.hold_ms, deadline_ms=deadline_ms,
            adv_time_ms=m.get("adv_time_ms"),
        )
        if cfg.threshold_mode == "score_ratio":
            after_reference = (t - t[0]) * 1000 >= cfg.ref_ms
            if after_reference.any() and res.threshold_curve[after_reference][0] > 1:
                unreachable[row["outcome"]] += 1
        row["shot"] = sid
        row["dbc_ip2"] = m.get("dbc_ip2")
        row["root_cause"] = rec.root_cause
        row["applied"] = rec.applied
        rows.append(row)
        if dis:
            cut = res.times < dt - deadline_ms / 1000.0
            timely_scores.append(float(res.risk_curve[cut].max()) if cut.any() else 0.0)
        else:
            timely_scores.append(float(res.risk_curve.max()) if len(res.risk_curve) else 0.0)
    summary = metrics.summarize_stream(rows, deadline_ms=deadline_ms, checkpoints=checkpoints)
    summary["timely_roc_auc"] = _timely_auc(rows, timely_scores)
    summary["elapsed_s"] = round(time.perf_counter() - t0, 3)
    summary["reflection_causes"] = dict(causes)
    summary["threshold_updates_applied"] = applied_count
    summary["final_state"] = {
        "multiplier": float(st.multiplier),
        "shots_seen": int(st.shots_seen),
    }
    if len(rows) >= 100:
        slices = {"first20": rows[:20], "mid80": rows[20:100], "after100": rows[100:]}
        summary["phases"] = {
            k: {mk: mv for mk, mv in metrics.summarize_stream(v, deadline_ms=deadline_ms, checkpoints=()).items()
                if mk in ("n_shots", "n_disruptive", "n_normal", "timely_tpr", "fpr", "youden_J", "f1_timely", "roc_auc", "tp_timely", "tardy", "miss", "fp")}
            for k, v in slices.items() if v
        }
    if cfg.threshold_mode == "score_ratio":
        summary["post_reference_unreachable_by_outcome"] = dict(unreachable)
    summary["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
    if return_rows:
        summary["rows"] = rows
    return summary


def save_result(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    stamped = {"provenance": _provenance(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(stamped, indent=2))
    return path


def locked_configs(tau_src: float) -> dict[str, engine.EngineConfig]:
    """Final locked configurations (selected strictly on J-TEXT + dev360)."""
    from dataclasses import replace
    primary = engine.EngineConfig(
        align_mode="shrinkage_robust", align_channels=engine.COHERENT6,
        align_prefix_ms=200, shrinkage_k=5.0, align_ema=0.02,
        threshold_mode="logit_margin", ref_ms=100, init_multiplier=100.0,
        source_threshold=tau_src, fixed_threshold=tau_src,
        r_floor=1e-4, ref_cap=0.45, tau_min=1e-4, tau_max=0.995,
        k_spread=2.0, hold_ms=1,
        enable_reflection=True, reflect_align=True, reflect_threshold=True,
        diagnose_reflection=True, max_log_mult_step=0.15, fp_weight=2.5, fn_weight=1.0,
    )
    efd2 = replace(primary, align_mode="efd", align_channels=engine.EFD2, align_prefix_ms=100, align_ema=0.05)
    return {
        "ZeroShot_Unaligned_Fixed": replace(primary, align_mode="none", threshold_mode="fixed", enable_reflection=False),
        "COH6_AlignOnly_SourceTau": replace(primary, threshold_mode="fixed", reflect_threshold=False),
        "COH6_IntraOnly_M100": replace(primary, enable_reflection=False),
        "COH6_InterAlignOnly_M100": replace(primary, reflect_threshold=False),
        "COH6_Blind_M100": replace(primary, diagnose_reflection=False),
        "COH6_Reflective_M100": primary,
        "COH6_Reflective_M15": replace(primary, init_multiplier=15.0),
        "COH6_Reflective_M909_JTEXT": replace(primary, init_multiplier=909.0),
        "EFD2_Reflective_M100": efd2,
    }


def _fmt_row(label: str, s: dict) -> str:
    adv = s.get("adv_time_subset") or {}
    adv_s = f" adv={adv['beat_adv_time_rate']:.3f}({adv['n_covered']})" if adv.get("n_covered") else ""
    return (
        f"{label:28s} | AUC={s['roc_auc']:.4f} tAUC={s['timely_roc_auc']:.4f} | "
        f"TPR={s['timely_tpr']:.3f} FPR={s['fpr']:.3f} J={s['youden_J']:.4f} "
        f"F1={s['f1_timely']:.4f} b3s={s['bounded3000_tpr']:.3f} prem={s['premature_gt3000_count']:3d} "
        f"DBC={s['dbc_weighted_cost']:.4f} lead={s['lead_ms_median'] or 0:5.1f}ms | "
        f"Mf={s['final_state']['multiplier']:6.1f} upd={s['threshold_updates_applied']:3d}{adv_s} ({s['elapsed_s']:.1f}s)"
    )


def run_locked_eval(cohort: str = "one_shot", device: str = "cuda"):
    """Run locked suite on dev360 or one-shot independent holdout east_indep2593."""
    model, ckpt = backbone.load_source_model(device=device)
    tau_src = float(ckpt["val_threshold_fpr5"])
    meta = data.build_east_meta()
    cohorts = data.load_east_cohorts(meta)
    cfgs = locked_configs(tau_src)

    if cohort == "one_shot":
        from dataclasses import replace
        shots = cohorts["dev360"]
        cfg = replace(cfgs["COH6_Reflective_M100"], align_mode="one_shot")
        cache = load_cache(shots)
        trials = {}
        cases = [("fixed", 100, 1)]
        cases += [("logit_margin", m, h) for m, h in ((15, 1), (100, 1), (909, 1), (5000, 1), (15, 10), (100, 10), (15, 25), (100, 25), (100, 50), (100, 100))]
        cases.extend(("score_ratio", m, h) for m in (2, 15, 100, 909) for h in (1, 10, 25))
        for mode, multiplier, hold_ms in cases:
            trial = replace(cfg, threshold_mode=mode, init_multiplier=float(multiplier), hold_ms=hold_ms)
            result = run_stream(shots, cache, model, ckpt, trial, meta=meta, device=device, return_rows=True)
            rows = result.pop("rows")
            result["fp_before_reference"] = sum(
                row["outcome"] == "FP" and 1000 * (row["alarm_time"] - float(cache[row["shot"]][0][0])) < cfg.ref_ms
                for row in rows
            )
            key = f"{mode}_M{multiplier}_hold{hold_ms}"
            trials[key] = result
            print(_fmt_row(key, result), flush=True)
        feasible = [m for m, r in trials.items() if r["fpr"] < 0.2]
        selected = max(feasible, key=lambda m: (trials[m]["timely_tpr"], -trials[m]["fpr"]), default=None)
        return save_result("dev360_one_shot", {"trials": trials, "selected_config": selected})

    if cohort == "one_shot_extension":
        dev = json.loads((RESULTS_DIR / "dev360_one_shot.json").read_text())
        selected = dev["selected_config"]
        if selected is None:
            raise ValueError("No development configuration met the FPR constraint.")
        if dev["provenance"]["source_model_sha256"] != _provenance()["source_model_sha256"]:
            raise ValueError("Source checkpoint differs from the selected development run.")
        cfg = engine.EngineConfig(**dev["trials"][selected]["config"])
        shots = cohorts["east_indep2593"]
        result = run_stream(shots, load_cache(shots), model, ckpt, cfg, meta=meta, device=device)
        print(_fmt_row(selected, result), flush=True)
        return save_result("extension2593_one_shot", {"dev_selection": selected, "evaluation": result})

    if cohort == "dev360":
        shots = cohorts["dev360"]
        cache = load_cache(shots)
        cps = (1, 5, 10, 20, 50, 100, 180, 360)
        out = {"cohort": "dev360", "n_shots": len(shots), "ablation": {}, "ordering": {}}
        print("=== DEV360 LOCKED ABLATION (archived360 order) ===", flush=True)
        for name, cfg in cfgs.items():
            s = run_stream(shots, cache, model, ckpt, cfg, meta=meta, device=device, checkpoints=cps)
            out["ablation"][name] = s
            print(_fmt_row(name, s), flush=True)
        print("=== DEV360 ORDERING ROBUSTNESS (COH6_Reflective_M100) ===", flush=True)
        for order in ("archived360", "chrono", "ip_asc", "ip_desc", "ip_stratified", "dbc_asc", "dbc_desc"):
            ordered = data.sort_shots(shots, meta, order)
            s = run_stream(ordered, cache, model, ckpt, cfgs["COH6_Reflective_M100"], meta=meta, device=device, checkpoints=cps)
            out["ordering"][order] = s
            print(_fmt_row(f"order:{order}", s), flush=True)
        return save_result("dev360_locked_comparison", out)

    if cohort == "indep2593":
        shots = cohorts["east_indep2593"]
        cache = load_cache(shots)
        cps = (1, 5, 10, 20, 50, 100, 200, 360, 500, 1000, 1500, 2000, len(shots))
        holdout_arms = [
            "ZeroShot_Unaligned_Fixed",
            "COH6_AlignOnly_SourceTau",
            "COH6_InterAlignOnly_M100",
            "COH6_Blind_M100",
            "COH6_Reflective_M100",
            "COH6_Reflective_M909_JTEXT",
            "EFD2_Reflective_M100",
        ]
        out = {"cohort": "east_indep2593", "n_shots": len(shots), "ablation_chrono": {}, "ordering_primary": {}}
        print(f"=== LOCKED ONE-SHOT HOLDOUT: east_indep2593 ({len(shots)} shots, chrono order) ===", flush=True)
        for name in holdout_arms:
            s = run_stream(shots, cache, model, ckpt, cfgs[name], meta=meta, device=device, checkpoints=cps)
            out["ablation_chrono"][name] = s
            print(_fmt_row(name, s), flush=True)
        out["ordering_primary"]["chrono"] = out["ablation_chrono"]["COH6_Reflective_M100"]
        print("=== HOLDOUT ORDERING ROBUSTNESS (COH6_Reflective_M100) ===", flush=True)
        for order in ("dbc_asc", "ip_stratified"):
            ordered = data.sort_shots(shots, meta, order)
            s = run_stream(ordered, cache, model, ckpt, cfgs["COH6_Reflective_M100"], meta=meta, device=device, checkpoints=cps)
            out["ordering_primary"][order] = s
            print(_fmt_row(f"order:{order}", s), flush=True)
        return save_result("final_holdout_indep2593", out)

    raise ValueError(f"unknown cohort: {cohort}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "one_shot"
    p = run_locked_eval(target)
    print(f"Saved: {p}", flush=True)
