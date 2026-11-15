"""Fit an eight-sample reached-shadow action selector and gate any replay."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_patch_utility_intergroup_20261106 as phase
import run_patch_utility_multiview_20261106 as m
from run_patch_utility_shadow_timing_20261106 import wait
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis


b, sel, sparse = m.sel.bank, m.sel, m.sparse
PLAN = b.r.ROOT / "PATCH_UTILITY_SHADOW_RESPONSE_PLAN_20261106.md"
IMPL = b.r.ROOT / "PATCH_UTILITY_SHADOW_RESPONSE_IMPLEMENTATION_20261106.md"
DELAY = 8
ARMS = ("blind", "amplitude", "real", "reverse")


def commit_mask(allowed, reached, can_act, previous_peak, tau):
    if (reached.ndim != 3 or reached.shape[1:] != (b.NA, DELAY)
            or allowed.shape != reached.shape[:2] or not allowed[:,0].all()
            or not np.array_equal(allowed[:,1::2],allowed[:,2::2])
            or not np.isfinite(reached).all()):
        raise ValueError("invalid symmetric reached shadow menu")
    static = reached[:,0].astype(np.float64)
    risk_now = reached[:,1:,-1].astype(np.float64) - static[:,-1:]
    response = np.abs(risk_now).reshape(-1, (b.NA-1)//2, 2)
    protected = (response.max(axis=2) <= b.pair.LIMIT-b.pair.GUARD_HEADROOM).repeat(2,axis=1)
    choice = allowed.copy()
    choice[:,1:] &= (
        protected & can_act[:,None]
        & (static.max(axis=1)[:,None] < tau)
        & (previous_peak.astype(np.float64)[:,None] < tau)
    )
    return choice


def design(base, reached, arm, tau):
    """The only new observation argument is exactly eight reached samples."""
    if reached.ndim != 3 or reached.shape[-1] != DELAY or reached.shape[:2] != base.shape[:2]:
        raise ValueError("exactly eight reached shadow values required")
    if arm not in ARMS:
        raise ValueError("unknown information arm")
    observed = reached.astype(np.float64)
    static = observed[:,:1]
    common = np.broadcast_to(static-tau,observed.shape)
    response = observed-static
    mu = response.mean(axis=2,keepdims=True)
    centered = response-mu
    amp = np.stack(
        (mu[...,0], np.sqrt(np.mean(response**2,axis=2)), np.sqrt(np.mean(centered**2,axis=2)),
         response.min(axis=2), response.max(axis=2)), axis=2,
    )
    if arm == "blind":
        amp = np.zeros_like(amp)
    shape = centered if arm=="real" else centered[:,:,::-1] if arm=="reverse" else np.zeros_like(centered)
    interaction = shape * (static-static.mean(axis=2,keepdims=True))
    return np.concatenate((base,common,amp,shape,interaction),axis=2).astype(np.float32)


def summary(tape, hard, ids, flags):
    benefit, harm = (np.unique(tape.owner[(hard*k>0).any(axis=1)]) for k in (1,-1))
    return dict(late_single=ids[benefit[flags[benefit]]].tolist(),
                false_remove_single=ids[benefit[~flags[benefit]]].tolist(),
                false_add_single=ids[harm[~flags[harm]]].tolist(),
                timely_at_risk=ids[harm[flags[harm]]].tolist())


def prep(tapes, ids, flags, independent):
    ready, reached, scalar = {}, {}, {}
    for name, t in tapes.items():
        old = t.data
        reached[name] = old["curves"][:,:,:DELAY].copy()
        delayed = wait(old["curves"],DELAY)
        original = old["allowed"]
        room, baseline_peak = [], []
        for q in t.shots:
            c = old["clock"][q.blocks]
            room.append(c+DELAY < len(q.native)-1)
            baseline_peak.append(np.maximum.accumulate(q.native)[c])
        before = np.concatenate(baseline_peak)
        if not np.array_equal(before,old["risk"][:,8]):
            raise ValueError("recorded prefix peak changed")
        avail = commit_mask(original,reached[name],np.concatenate(room),before,b.TAU)
        if int(avail[:,1:].sum()) != independent["views"][name]["action_counts"]["protected"]:
            raise ValueError("eight-sample guard did not reproduce independent support")
        labels, ctrl, ctrl_labels = [], [], []
        for q in t.shots:
            sl = q.blocks
            c, a = old["clock"][sl], avail[sl]
            matched = wait(sel.scalar_curves(old["curves"][sl],old["effect"][sl],c,len(q.native)),DELAY)
            real_result = menu_analysis(q,c,delayed[sl],a,b.TAU)
            scalar_result = menu_analysis(q,c,matched,a,b.TAU)
            if real_result["baseline"] != q.teacher["baseline"] or scalar_result["baseline"] != real_result["baseline"]:
                raise ValueError("delaying an action cannot change the frozen baseline")
            labels.append(real_result["hard"])
            ctrl_labels.append(scalar_result["hard"])
            ctrl.append(matched)
        hard, alternate = np.concatenate(labels), np.concatenate(ctrl_labels)
        expected = independent["views"][name]["guarded_at_8"]
        if summary(t,hard,ids,flags)!=expected["real"] or summary(t,alternate,ids,flags)!=expected["scalar"]:
            raise ValueError("eight-sample protected efficacy differs from independent proof")
        ready[name] = m.Tape(t.shots, dict(old,curves=delayed,allowed=avail), hard, t.owner,avail)
        scalar[name] = np.concatenate(ctrl)
    return ready, reached, scalar


def late_difference(reports):
    finds = {a:{i for v in m.VIEWS for i in reports[a][v]["late_top"]} for a in ARMS}
    return sorted(finds["real"]-set.union(*(finds[a] for a in ARMS if a!="real")))


def learned(tapes, reached, ids, flags, folds):
    result, logs, ranks = {}, {}, {}
    owner = tapes[m.VIEWS[0]].owner
    base = {v:sel.features(tapes[v].data,"risk") for v in m.VIEWS}
    for arm in ARMS:
        x = {v:design(base[v],reached[v],arm,b.TAU) for v in m.VIEWS}
        fit = []
        pred = {v:np.empty(tapes[v].hard.shape) for v in m.VIEWS}
        for k in range(sel.N_FOLDS):
            tr, _, hd = m.rows_for_fold(folds,owner,k)
            model = m.train_model(tapes,x,tr,m.VIEWS)
            fit.append(model)
            for v in m.VIEWS:
                pred[v][hd] = sparse.event_scores(model,x[v][hd])
        logs[arm], result[arm] = fit, {}
        for v,t in tapes.items():
            result[arm][v], a, rank = phase.describe(
                t,pred[v],{"supported":t.allowed[:,1:]},ids,flags,
            )
            ranks[f"{arm}__{v}"] = np.stack((a,rank),axis=1)
            for rec in result[arm][v]["late_blocks"]:
                rec["commit"] = rec["clock"]+DELAY
        print(arm,{v:result[arm][v]["late_top"] for v in m.VIEWS},flush=True)
    return result, logs, ranks, base


def policy(tapes, reached, scalar, ids, flags, folds, logs, base):
    owner = tapes[m.VIEWS[0]].owner
    reports, logs_cal, archive = {}, {}, {}
    for arm in ARMS:
        x = {v:design(base[v],reached[v],arm,b.TAU) for v in m.VIEWS}
        choices = {v:np.full((len(owner),2),-1,dtype=np.int64) for v in m.VIEWS}
        records = []
        for k,model in enumerate(logs[arm]):
            _, cal, held = m.rows_for_fold(folds,owner,k)
            score = {v:sparse.event_scores(model,x[v][cal]) for v in m.VIEWS}
            all_max = np.concatenate([sparse.top(score[v],tapes[v].allowed[cal])[1] for v in m.VIEWS])
            finite = all_max[np.isfinite(all_max)]
            if not len(finite):
                raise ValueError("no writable rows to calibrate")
            grid = np.unique(np.r_[-np.inf,np.quantile(finite,sparse.QUANTILES),np.inf])

            def effects(cut):
                return {v:sparse.hard_credit(
                    t.hard[cal],sparse.choose_once(score[v],t.allowed[cal],owner[cal],cut),
                    owner[cal],ids,flags,
                ) for v,t in tapes.items()}

            cut, curve = m.calibrate(effects,grid)
            records.append(dict(fold=k,cut=m.safe_margin(cut),curve=curve))
            for v,t in tapes.items():
                s = sparse.event_scores(model,x[v][held])
                for j,gate in enumerate((-np.inf,cut)):
                    choices[v][held,j] = sparse.choose_once(s,t.allowed[held],owner[held],gate)
        report = {"open":{},"guarded":{},"open_scalar":{},"guarded_scalar":{}}
        for v,t in tapes.items():
            choice = choices[v]
            if (choice<0).any():
                raise ValueError("unscored held windows")
            archive[f"{arm}__{v}"] = choice
            for j,label in enumerate(("open","guarded")):
                report[label][v] = m.replay(t,choice[:,j],ids,flags)
                shadow = sel.score_policy(t.shots,dict(t.data,curves=scalar[v]),choice[:,j])
                report[f"{label}_scalar"][v] = sel.summarize(shadow) | {"writes":int(np.count_nonzero(choice[:,j]))}
        reports[arm], logs_cal[arm] = report, records
    return reports, logs_cal, archive


def promoted(scores):
    real = scores["real"]["guarded"]
    controls = [scores[k]["guarded"] for k in ARMS if k!="real"] + [scores["real"]["guarded_scalar"]]
    unique = sorted(set(m.union(real,"recovered"))-{i for c in controls for i in m.union(c,"recovered")})
    return bool(unique and not m.union(real,"lost") and not m.union(real,"fp_added")), unique


def run(native,masked,native_result,proof,out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    independent = b.r.read(proof)
    if (not independent["commit_guard_gate"] or independent["commit_plan_sha256"]!=h(PLAN)
            or independent["new_val_test_east_reads"]):
        raise ValueError("requires the fixed, independently checked eight-frame capacity")
    for p in (Path(__file__),Path(m.__file__),Path(sel.__file__),Path(phase.__file__),IMPL,proof):
        h(p)
    raw, ids, flags = m.load(native,masked,native_result,h)
    tapes, reached, scalar = prep(raw,ids,flags,independent)
    folds = sel.fit.shot_folds(ids,sel.N_FOLDS)
    ranks, fits, archive, base = learned(tapes,reached,ids,flags,folds)
    distinct = late_difference(ranks)
    result = dict(status="SCORED_SHADOW_HELD_RANK_NOT_EAST", delay=DELAY, source_shots=len(ids),
                  ranks=ranks,rank_gate=bool(distinct),rank_distinct=distinct, first_alarm_replays=0,
                  source_val_gate=False,new_val_test_east_reads=0)
    if distinct:
        scores, cal, choices = policy(tapes,reached,scalar,ids,flags,folds,fits,base)
        proceed, wins = promoted(scores)
        result.update(status="SCORED_SHADOW_TRAIN_REPLAY_NOT_EAST",scores=scores,
                      first_alarm_replays=len(ids)*len(ARMS)*len(m.VIEWS)*4,
                      source_val_gate=proceed,replay_distinct=wins)
    h.check()
    out.mkdir(parents=True,exist_ok=False)
    rank_path,fit_path = out/"rankings.npz",out/"fits.json"
    owner = tapes[m.VIEWS[0]].owner
    b.r.save_arrays(rank_path,ids=ids,folds=folds,owner=owner,**archive)
    splits = [dict(held=ids[folds==k].tolist(),calibrate=ids[folds==(k+1)%5].tolist(),
                   fit=ids[(folds!=k)&(folds!=(k+1)%5)].tolist()) for k in range(sel.N_FOLDS)]
    b.r.write_new(fit_path,dict(fits=fits,splits=splits))
    if distinct:
        path,cp = out/"choices.npz",out/"calibration.json"
        b.r.save_arrays(path,ids=ids,folds=folds,owner=owner,**choices)
        b.r.write_new(cp,cal)
        result.update(choices_sha256=h(path),calibration_sha256=h(cp))
    result.update(rank_sha256=h(rank_path),fit_sha256=h(fit_path),threshold=b.TAU,
                  input_sha256={str(p):v[1] for p,v in h.cache.items()},
                  elapsed_seconds=time.monotonic()-start)
    b.r.write_new(out/"result.json",result)
    print({k:result[k] for k in ("rank_gate","rank_distinct","first_alarm_replays","source_val_gate")},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("native","masked","native_result","guard_proof","output"):
        p.add_argument(k,type=Path)
    q = p.parse_args()
    run(q.native,q.masked,q.native_result,q.guard_proof,q.output)
