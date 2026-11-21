"""Cross-fit a label-free-at-inference state probe on reached TRAIN landmarks."""

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import run_patch_utility_pivotal_support_20261106 as p
from run_patch_utility_shadow_effect_20261106 import weights
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds


b, m, s = p.b, p.m, p.s
PLAN = b.r.ROOT / "PATCH_UTILITY_LANDMARK_STATE_PROTOCOL_20261106.md"
ARMS = ("risk", "sorted", "real", "reverse")
TARGETS = ("original", "125ms")
REGIMES = ("all", "writable")


def future_label(times, clock, disruptive, down, horizon):
    if not disruptive:
        return np.zeros(len(clock), dtype=bool)
    grid = np.asarray(clock)[:,None]+np.arange(p.DELAY+1,p.WIDTH+1)
    actual = grid < len(times)
    lead = np.round((down-times[np.minimum(grid,len(times)-1)])*1000,9)
    return (actual & (lead>10) & (lead<=horizon+1e-6)).any(axis=1)


def feature_matrix(risk, reached, ordered, kind):
    n = len(risk)
    if risk.shape!=(n,10) or reached.shape!=(n,8) or ordered.shape!=(n,2,27):
        raise ValueError("require ten prefix features and exactly eight reached scores")
    if kind == "risk":
        extra = np.zeros((n,27))
    elif kind == "sorted":
        extra = np.sort(ordered[:,0].reshape(n,9,3),axis=2).reshape(n,27)
    elif kind in ("real","reverse"):
        extra = ordered[:,int(kind=="reverse")]
    else:
        raise ValueError("unknown state input")
    common = risk.astype(np.float64,copy=True)
    common[:,1:9] -= b.TAU
    x = np.column_stack((common,reached-b.TAU,extra))
    if not np.isfinite(x).all():
        raise ValueError("non-finite reached state")
    return x


def class_weights(keep, label, owner):
    out = np.zeros(keep.shape, dtype=float)
    for y in (False,True):
        rows = keep & (label==y)
        if not rows.any():
            raise ValueError("both landmark classes must have independent support")
        out += weights(rows[:,:,None],owner)[:,:,0]/2
    if abs(out.sum()-1)>1e-10:
        raise ValueError("unbalanced physical shot weights")
    return out


def fit_head(x,label,w):
    center = np.average(x,axis=0,weights=w)
    variance = np.average((x-center)**2,axis=0,weights=w)
    scale = np.where(variance>1e-12,np.sqrt(variance),1.)
    model = LogisticRegression(C=1/s.sel.ALPHA,solver="lbfgs",max_iter=300,tol=1e-9)
    model.fit((x-center)/scale,label,sample_weight=w)
    if model.n_iter_.max() >= 300 or not np.array_equal(model.classes_,[False,True]):
        raise RuntimeError("failed fixed logistic fit")
    return dict(center=center.tolist(),scale=scale.tolist(),coef=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]),steps=int(model.n_iter_[0]))


def logit(head,x):
    return ((x-head["center"])/head["scale"])@np.asarray(head["coef"])+head["intercept"]


def extract(tapes, ids, flags):
    original = tapes[m.VIEWS[0]]
    owner = original.owner
    static_target = {k:[] for k in TARGETS}
    observed = []
    for q in original.shots:
        clock = original.data["clock"][q.blocks]
        observed.append(clock+p.DELAY < len(q.native)-1)
        for label,h in zip(TARGETS,(q.horizon,125.),strict=True):
            static_target[label].append(future_label(q.times,clock,q.disruptive,q.down,h))
    look = np.concatenate(observed)
    y = {k:np.concatenate(z) for k,z in static_target.items()}
    rescue, hazard, available = [], [], []
    for name,tape in tapes.items():
        d = tape.data
        allowed = s.commit_mask(tape.allowed,d["curves"][:,:,:p.DELAY],look,d["risk"][:,8],b.TAU)
        available.append(allowed[:,1:].any(axis=1))
        hard = np.zeros(allowed.shape,dtype=np.int8)
        for q in tape.shots:
            sl = q.blocks
            value,valid,*_ = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,d["clock"][sl],b.TAU)
            hard[sl] = p.effect(d["curves"][sl,:,p.DELAY:],value,valid,b.TAU)[0]
        good = (allowed & (hard>0)).any(axis=1) & flags[owner]
        bad = (allowed & (hard<0)).any(axis=1) & ~flags[owner]
        print(name,"pivotal case shots",len(set(ids[owner[good]])),len(set(ids[owner[bad]])),flush=True)
        rescue.append(good)
        hazard.append(bad)
    return y,np.broadcast_to(look,(len(m.VIEWS),len(owner))),np.stack(available),np.stack(rescue),np.stack(hazard)


def metrics(pred, keep, labels, owner):
    weight = class_weights(keep,labels,owner)[keep]
    y = np.broadcast_to(labels,keep.shape)[keep]
    return dict(
        auc=float(roc_auc_score(y,pred[keep],sample_weight=weight)),
        ap=float(average_precision_score(y,pred[keep],sample_weight=weight)),
        shots_by_class=[int(np.unique(owner[(keep & (labels==k)).any(axis=0)]).size) for k in (False,True)],
        rows_by_class=[int((keep & (labels==k)).sum()) for k in (False,True)],
    )


def screen(report, held_rescues):
    key = "all__125ms__real"
    baseline = ["all__original__real"] + [f"all__125ms__{arm}" for arm in ARMS if arm!="real"]
    a = report[key]
    comparisons = {}
    for other in baseline:
        q = report[other]
        x = {k: a["writable"]["125ms"][k]-q["writable"]["125ms"][k] for k in ("auc","ap")}
        x["case_auc"] = a["case"]["auc"]-q["case"]["auc"]
        x["case_loo_min"] = min(a["case_loo_auc"][str(i)]-q["case_loo_auc"][str(i)] for i in held_rescues)
        comparisons[other] = x
    return all(all(d>0 for d in x.values()) for x in comparisons.values()),comparisons


def run(native, masked, baseline, proof, support_path, out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    for path in (Path(__file__),Path(p.__file__),Path(s.__file__),PLAN,p.PLAN,proof,support_path):
        h(path)
    proof_data, support = b.r.read(proof), b.r.read(support_path)
    if (not proof_data["commit_guard_gate"] or proof_data["commit_plan_sha256"]!=h(s.PLAN)
            or support["status"]!="EXACT_OFFLINE_FIRST_ALARM_FACTOR_AND_SOURCE_SUPPORT"
            or support["input_sha256"].get(str(proof.resolve()))!=h(proof)):
        raise ValueError("requires audited, unchanged decision support")
    tapes,ids,flags = m.load(native,masked,baseline,h)
    owner = tapes[m.VIEWS[0]].owner
    folds = deterministic_folds(ids)
    y, observed, write, positive, negative = extract(tapes,ids,flags)
    for vi,v in enumerate(m.VIEWS):
        q = support["views"][v]
        if (set(ids[owner[positive[vi]]]) != {z["shot"] for z in q["rescue_actions"]}
                or (positive[vi] & ~y["125ms"]).any()):
            raise ValueError("expanded teaching target misses the audited rescue population")
    held_rescues = sorted(set(ids[owner[positive.any(axis=0)]].tolist()))
    case = positive | negative
    if not np.array_equal(positive,case & flags[owner]):
        raise ValueError("a safe hazard case must be separate from recoverable disruption")
    state,report,fits = {},{},{}
    for kind in ARMS:
        x = np.stack([feature_matrix(t.data["risk"],t.data["curves"][:,0,:p.DELAY],
                                     t.data["ordered"],kind) for t in tapes.values()])
        for label in TARGETS:
            for regime,rows in zip(REGIMES,(observed,write),strict=True):
                key = f"{regime}__{label}__{kind}"
                value, fit_log = np.full(rows.shape,np.nan),[]
                for k in range(s.sel.N_FOLDS):
                    tr,_,held = m.rows_for_fold(folds,owner,k)
                    use = rows[:,tr]
                    w = class_weights(use,y[label][tr],owner[tr])[use]
                    target = np.broadcast_to(y[label][tr],use.shape)[use]
                    head = fit_head(x[:,tr][use],target,w)
                    fit_log.append(head)
                    value[:,held] = logit(head,x[:,held])
                if not np.isfinite(value).all():
                    raise ValueError("incomplete held predictions")
                fits[key],state[key] = fit_log,value
                report[key] = dict(
                    writable={lab:metrics(value,write,yy,owner) for lab,yy in y.items()},
                    case=metrics(value,case,flags[owner],owner),
                    case_loo_auc={str(i):metrics(value,case & (ids[owner]!=i)[None],
                                               flags[owner],owner)["auc"] for i in held_rescues},
                    case_rescues={str(i):{
                        v:value[vi,positive[vi] & (ids[owner]==i)].tolist()
                        for vi,v in enumerate(m.VIEWS)
                    } for i in held_rescues},
                )
                print(key,report[key]["writable"]["125ms"]["auc"],report[key]["case"]["auc"],flush=True)
    gate, delta = screen(report,held_rescues)
    h.check()
    out.mkdir(parents=True,exist_ok=False)
    fp,pr = out/"fits.json",out/"held_states.npz"
    b.r.write_new(fp,dict(fits=fits))
    b.r.save_arrays(pr,ids=ids,owner=owner,folds=folds,known=observed,writable=write,
                    rescue=positive,safe_harm=negative,flags=flags,
                    y_original=y["original"],y_125ms=y["125ms"],**state)
    support_counts = {
        k:{target:{v:int(np.unique(owner[mask[vi]&label]).size) for vi,v in enumerate(m.VIEWS)}
           for target,label in y.items()}
        for k,mask in (("all",observed),("writable",write))
    }
    b.r.write_new(out/"result.json",dict(
        status="SOURCE_STATE_CLASSIFICATION_NOT_POLICY_REPLAY",support=support_counts,
        report=report,state_evidence_gate=gate,all_real_differences=delta,
        rescue_shots=held_rescues,fit_sha256=h(fp),held_sha256=h(pr),
        new_val_test_east_reads=0,first_alarm_policy_replays=0,
        input_sha256={str(a):z[1] for a,z in h.cache.items()},elapsed_seconds=time.monotonic()-start,
    ))
    print("state_evidence_gate",gate,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","proof","support","output"):
        cli.add_argument(key,type=Path)
    args = cli.parse_args()
    run(args.native,args.masked,args.baseline,args.proof,args.support,args.output)
