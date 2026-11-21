"""Held-shot forecast of dense action effects; no fitting to event labels."""

import argparse
from pathlib import Path
import time

import numpy as np

import run_patch_utility_shadow_response_20261106 as s
from verify_confirmation_raw_replay_20261105 import Hashes


b, m = s.b, s.m
PLAN = b.r.ROOT / "PATCH_UTILITY_SHADOW_EFFECT_FORECAST_PROTOCOL_20261106.md"
WIDTH = 32
CENTERS = np.array([4.5,12.5,20.5])


def targets(curves):
    """Absolute changes of candidate-vs-static relative to last reached effect."""
    if curves.ndim != 3 or curves.shape[-1] != WIDTH:
        raise ValueError("complete sealed 32-sample candidate curves required")
    a = curves.astype(np.float64)
    d = a-a[:,:1]
    later = d[:,:,s.DELAY:]-d[:,:,s.DELAY-1:s.DELAY]
    return later.reshape(*later.shape[:2],3,8).mean(axis=3)


def slope(reached):
    if reached.ndim != 3 or reached.shape[-1] != s.DELAY:
        raise ValueError("only eight reached samples are allowed for extrapolation")
    a = reached.astype(np.float64)
    d = a-a[:,:1]
    return ((d[:,:,-1]-d[:,:,0])/7)[:,:,None]*CENTERS[None,None,:]


def weights(allowed,owner):
    """One mass per physical shot, divided over supported views/blocks/actions."""
    actions = allowed.sum(axis=2)
    blocks = np.stack([np.bincount(owner,weights=(a>0),minlength=int(owner.max())+1)
                       for a in actions])
    views = (blocks>0).sum(axis=0)
    den = np.count_nonzero(views)*views[owner][None,:]*blocks[:,owner]*actions
    return np.divide(allowed,den[:,:,None],out=np.zeros(allowed.shape),where=den[:,:,None]>0)


def fit(x,y,w):
    x = x.astype(np.float64)
    c = np.average(x,axis=0,weights=w)
    yc = np.average(y,axis=0,weights=w)
    var = np.average((x-c)**2,axis=0,weights=w)
    scale = np.where(var>1e-12,np.sqrt(var),1.)
    z = (x-c)/scale
    weight = w[:,None]
    matrix = z.T@(z*weight) + s.sel.ALPHA*np.eye(z.shape[1])
    coefficient = np.linalg.solve(matrix,z.T@((y-yc)*weight))
    return dict(center=c.tolist(),scale=scale.tolist(),target=yc.tolist(),
                coefficient=coefficient.tolist())


def predict(model,x):
    return ((x.astype(np.float64)-model["center"])/model["scale"]
            )@np.asarray(model["coefficient"])+model["target"]


def shot_error(prediction, truth, keep, owner, n):
    keep = keep.astype(bool)
    count = keep.sum(axis=1)
    live = count>0
    err = np.abs(prediction-truth)*keep[:,:,None]
    per_block = err.sum(axis=1)[live]/count[live,None]
    total = np.bincount(owner[live],minlength=n)
    sums = np.stack([np.bincount(owner[live],weights=col,minlength=n) for col in per_block.T],axis=1)
    return np.divide(sums,total[:,None],out=np.full((n,3),np.nan),where=total[:,None]>0)


def pooled(errors):
    used = np.isfinite(errors)
    count = used.sum(axis=0)
    return np.divide(np.nansum(errors,axis=0),count,out=np.full(errors.shape[1:],np.nan),where=count>0)


def run(native,masked,baseline,proof,out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    for p in (Path(__file__),Path(s.__file__),Path(m.__file__),PLAN,s.PLAN,s.IMPL,proof):
        h(p)
    guard = b.r.read(proof)
    if not guard["commit_guard_gate"] or guard["commit_plan_sha256"]!=h(s.PLAN):
        raise ValueError("unbound eight-sample protection audit")
    tapes,ids,_ = m.load(native,masked,baseline,h)
    owner = tapes[m.VIEWS[0]].owner
    folds = s.sel.fit.shot_folds(ids,s.sel.N_FOLDS)
    base, reached, y, mask = {}, {}, {}, {}
    for v,t in tapes.items():
        c = t.data["curves"]
        reached[v] = c[:,:,:s.DELAY].copy()
        y[v] = targets(c)
        full = np.concatenate([t.data["clock"][q.blocks]+WIDTH<len(q.native) for q in t.shots])
        mask[v] = s.commit_mask(t.data["allowed"],reached[v],full,t.data["risk"][:,8],b.TAU)
        mask[v][:,0] = False
        base[v] = s.sel.features(t.data,"risk")
    available = np.stack([mask[v] for v in m.VIEWS])
    errors, fits, predictions = {}, {}, {}
    for arm in ("hold","linear",*s.ARMS):
        pred = {}
        if arm in ("hold","linear"):
            for v in m.VIEWS:
                pred[v] = np.zeros_like(y[v]) if arm=="hold" else slope(reached[v])
        else:
            fits[arm] = []
            x = {v:s.design(base[v],reached[v],arm,b.TAU) for v in m.VIEWS}
            pred = {v:np.zeros_like(y[v]) for v in m.VIEWS}
            for k in range(s.sel.N_FOLDS):
                tr, _, held = m.rows_for_fold(folds,owner,k)
                w = weights(available[:,tr],owner[tr])
                assert abs(w.sum()-1)<1e-12
                train = {v:mask[v][tr] for v in m.VIEWS}
                model = fit(
                    np.concatenate([x[v][tr][train[v]] for v in m.VIEWS]),
                    np.concatenate([y[v][tr][train[v]] for v in m.VIEWS]),
                    np.concatenate([w[i][train[v]] for i,v in enumerate(m.VIEWS)]),
                )
                fits[arm].append(model)
                for v in m.VIEWS:
                    pred[v][held] = predict(model,x[v][held])
        errors[arm] = np.stack([shot_error(pred[v],y[v],mask[v],owner,len(ids)) for v in m.VIEWS])
        print(arm, np.nanmean(pooled(errors[arm]),axis=0).tolist(),flush=True)
        if arm=="real":
            predictions = {v:pred[v][mask[v]] for v in m.VIEWS}
    def metrics(a):
        return dict(mae=float(np.nanmean(a)),segment_mae=np.nanmean(a,axis=0).tolist(),
                    shots=int(np.isfinite(a[:,0]).sum()))
    result = {
        k:dict(views={v:metrics(a[j]) for j,v in enumerate(m.VIEWS)},pooled=metrics(pooled(a)))
        for k,a in errors.items()
    }
    ref = pooled(errors["real"]).mean(1)
    comparisons = {}
    for k,e in errors.items():
        if k=="real":
            continue
        diff = ref-pooled(e).mean(1)
        use = diff[np.isfinite(diff)]
        comparisons[k] = dict(real_minus_control=float(use.mean()),real_better=int((use<0).sum()),
                              real_worse=int((use>0).sum()),tied=int((use==0).sum()))
    h.check()
    out.mkdir(parents=True,exist_ok=False)
    receipt, err_path = out/"fits.json",out/"held_errors.npz"
    b.r.write_new(receipt,dict(fits=fits))
    b.r.save_arrays(err_path,ids=ids,folds=folds,**errors,**{f"real_prediction__{v}":a for v,a in predictions.items()})
    b.r.write_new(out/"result.json",dict(
        status="SCORED_HELD_DENSE_EFFECT_NOT_FIRST_ALARM",metrics=result,real_comparisons=comparisons,
        available_actions={v:int(mask[v].sum()) for v in m.VIEWS},
        input_sha256={str(p):d[1] for p,d in h.cache.items()},fits_sha256=h(receipt),
        errors_sha256=h(err_path),elapsed_seconds=time.monotonic()-start,
        first_alarm_replays=0,fit_reads_disruption_labels=False,new_val_test_east_reads=0,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("native","masked","native_result","guard_proof","output"):
        p.add_argument(k,type=Path)
    q = p.parse_args()
    run(q.native,q.masked,q.native_result,q.guard_proof,q.output)
