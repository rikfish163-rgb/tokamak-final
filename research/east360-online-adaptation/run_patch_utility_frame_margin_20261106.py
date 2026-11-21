"""Shot-held, pointwise separation of static logit and paired action effect."""

import argparse
from pathlib import Path
import time

import numpy as np

import patch_margin_landmarks_20261106 as windows
import run_patch_utility_pivotal_support_20261106 as p
from run_patch_utility_shadow_effect_20261106 import weights
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds


b,m,s = p.b,p.m,p.s
PLAN = b.r.ROOT / "PATCH_UTILITY_FRAME_MARGIN_PROTOCOL_20261106.md"
STEPS = (1,2,6,12)
BOUND = b.pair.LIMIT-b.pair.GUARD_HEADROOM


def ridge(x,y,length,weight):
    """One masked ridge per future frame; nested support reuses Gram matrices."""
    x,w = np.asarray(x,np.float64),np.asarray(weight,np.float64)
    w = w/w.sum()
    center = w@x
    var = w@((x-center)**2)
    cols = np.flatnonzero(var>1e-12)
    scale = np.sqrt(var[cols])
    z = np.column_stack((np.ones(len(x)),(x[:,cols]-center[cols])/scale))
    width = y.shape[1]
    gram = np.zeros((z.shape[1],z.shape[1]))
    cross = np.zeros((z.shape[1],width))
    penalty = np.eye(z.shape[1])*s.sel.ALPHA
    penalty[0,0] = 0
    beta = np.empty_like(cross)
    mass = 0.
    for k in range(width,0,-1):
        new = length==k
        if new.any():
            zz = z[new]
            wz = zz*w[new,None]
            known = y[new,:k]
            if not np.isfinite(known).all():
                raise ValueError("missing observed frame target")
            gram += zz.T@wz
            cross[:,:k] += wz.T@known
            mass += float(w[new].sum())
        if not mass>0:
            raise ValueError("no training target at this horizon")
        beta[:,k-1] = np.linalg.solve(gram+mass*penalty,cross[:,k-1])
    return dict(cols=cols.tolist(),center=center[cols].tolist(),scale=scale.tolist(),
                intercept=beta[0].tolist(),coef=beta[1:].tolist())


def predict(head,x):
    use = np.asarray(head["cols"],np.intp)
    coef = np.asarray(head["coef"],np.float64).reshape(len(use),-1) if len(use) else np.empty((0,len(head["intercept"])))
    return (x[...,use]-head["center"])/head["scale"]@coef+head["intercept"]


def compose(static,delta):
    output = static[:,None,:]+delta
    output[:,0] = static
    return output


def align_tail(x,delay):
    i = np.arange(24)[None,:]+delay[:,None]-8
    return np.where(i<24,np.take_along_axis(x,np.minimum(i,23),axis=1),0)


def common_features(risk,reached,delay):
    r = risk.astype(np.float64,copy=True)
    r[:,1:9] -= b.TAU
    z = reached[:,0].astype(np.float64)-b.TAU
    return np.column_stack((r,delay/32.,z,z[:,-1]-z[:,-4],z[:,-1]-z[:,0])).astype(np.float32)


def action_features(common,reached,old,arm):
    n,a,_ = reached.shape
    z = reached.astype(np.float64)
    d = z-z[:,:1]
    current = d[:,:,-1:]
    mu = d.mean(axis=-1,keepdims=True)
    shape = d-mu
    amp = np.stack((mu[...,0],(d**2).mean(2)**.5,(shape**2).mean(2)**.5,
                    d.min(2),d.max(2)),axis=2)
    if arm=="blind":
        amp = np.zeros_like(amp)
    if arm in ("blind","amplitude"):
        shape = np.zeros_like(shape)
    elif arm=="reverse":
        shape = shape[:,:,::-1]
    elif arm!="real":
        raise ValueError("unknown reached information arm")
    shared = np.broadcast_to(common[:,None],(n,a,common.shape[-1]))
    names = np.broadcast_to(np.eye(a)[None,:,1:],(n,a,a-1))
    interaction = shape*(z[:,:1]-z[:,:1].mean(axis=2,keepdims=True))
    return np.concatenate((shared,names,old,current,amp,shape,interaction),axis=-1).astype(np.float32)


def fit_panel(x,y,length,mask,owner,folds):
    prediction = np.empty(y.shape,dtype=np.float64)
    fits = []
    for k in range(s.sel.N_FOLDS):
        tr,_,held = m.rows_for_fold(folds,owner,k)
        train = mask[:,tr]
        n = np.broadcast_to(length[None,tr,None],train.shape)
        w = weights(train,owner[tr])[train]
        model = ridge(x[:,tr][train],y[:,tr][train],n[train],w)
        fits.append(model)
        prediction[:,held] = predict(model,x[:,held])
    return prediction,fits


def payoff_panel(tapes,delay):
    payoff,valid = [],[]
    for v in m.VIEWS:
        tape = tapes[v]
        g,ok = [],[]
        for q in tape.shots:
            z,a,*_ = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,tape.data["clock"][q.blocks],b.TAU)
            g.append(z)
            ok.append(a)
        payoff.append(align_tail(np.tile(np.concatenate(g),(2,1)),delay))
        valid.append(align_tail(np.tile(np.concatenate(ok),(2,1)),delay).astype(bool))
    return np.stack(payoff),np.stack(valid)


def point_metrics(static,delta,z,d,mask,length,owner,rows):
    result = {}
    for h in STEPS:
        j = h-1
        use = mask & (rows&(length>=h))[None,:,None]
        truth = (z[...,j,None]+d[...,j]>=b.TAU).astype(np.int8)-(z[...,j,None]>=b.TAU)
        forecast = (static[...,j,None]+delta[...,j]>=b.TAU).astype(np.int8)-(static[...,j,None]>=b.TAU)
        st = use.any(2)[...,None]
        entry = dict(static_mae=float((weights(st,owner)[...,0]*np.abs(static[...,j]-z[...,j])).sum()),
                     effect_mae=float((weights(use,owner)*np.abs(delta[...,j]-d[...,j])).sum()))
        for sign in (-1,0,1):
            chosen = use&(truth==sign)
            entry[str(sign)] = dict(
                shots=int(np.unique(owner[chosen.any(axis=(0,2))]).size),actions=int(chosen.sum()),
                rate=float(weights(chosen,owner)[(forecast!=0 if sign==0 else forecast==sign)].sum())
                if chosen.any() else None)
        result[str(h)] = entry
    return result


def first_credit(static,delta,reward,valid):
    return np.stack([p.effect(compose(z,d),g,t,b.TAU)[0]
                     for z,d,g,t in zip(static,delta,reward,valid,strict=True)])


def describe(credit,truth,mask,owner,ids,flags,rows):
    out = {}
    for i,v in enumerate(m.VIEWS):
        eligible = mask[i]&rows[:,None]
        real = truth[i]
        pred = credit[i]
        def count(x):
            use = eligible&x
            shot = np.unique(owner[use.any(1)])
            return dict(actions=int(use.sum()),late=ids[shot[flags[shot]]].tolist(),
                        safe=ids[shot[~flags[shot]]].tolist())
        out[v] = dict(kept=count((pred>0)&(real>0)),true_benefit=count(real>0),
                      invented_benefit=count((pred>0)&(real<=0)),
                      kept_harm=count((pred<0)&(real<0)),invented_harm=count((pred<0)&(real>=0)))
    return out


def case_predictions(tapes,ids,folds,last,static_heads,arm_heads):
    """Already known rescue cases are never model-selection or fit inputs."""
    result = {}
    for v in m.VIEWS:
        tape = tapes[v]
        byid = {q.sid:q for q in tape.shots}
        observed = tape.data
        result[v] = []
        for old in last["views"][v]:
            q = byid[old["shot"]]
            i = q.blocks.start+int(np.flatnonzero(observed["clock"][q.blocks]==old["release"])[0])
            col = old["crossing_delay"]-1
            c = observed["curves"][i].astype(np.float64)
            a = old["action"]
            fold = int(folds[np.flatnonzero(ids==q.sid)[0]])
            next_observed = col+1<min(32,len(q.native)-1-old["release"])
            next_hit = bool(c[a,col+1]>=b.TAU) if next_observed else None
            next_timely = (
                round(float((q.down-q.times[old["release"]+col+2])*1000),9)>10 if next_observed else None
            )
            row = dict(shot=int(q.sid),action=a,release=old["release"],
                       next_crossing=next_hit,next_timely=next_timely,
                       true_static_margin=float(c[0,col]-b.TAU),
                       true_delta=float(c[a,col]-c[0,col]),landmarks={})
            for d in sorted({8,old["latest"]}):
                read = observed["curves"][i:i+1,:,d-8:d]
                xx = common_features(observed["risk"][i:i+1],read,np.array([d]))
                step = col-d
                zz = float(predict(static_heads[fold],xx)[0,step]+c[0,d-1]-b.TAU)
                anchor = c[a,d-1]-c[0,d-1]
                land = dict(horizon=step+1,pred_static_margin=zz,delta={})
                for arm in ("hold",*s.ARMS):
                    predicted = anchor
                    if arm!="hold":
                        x = action_features(xx,read,observed["effect"][i:i+1],arm)
                        predicted += float(predict(arm_heads[arm][fold],x[:,a])[0,step])
                    land["delta"][arm] = dict(effect=float(predicted),both_cross=bool(zz+predicted>=0),
                        exact_static_cross=bool(row["true_static_margin"]+predicted>=0),
                        exact_effect_cross=bool(zz+row["true_delta"]>=0))
                row["landmarks"][str(d)] = land
            result[v].append(row)
    return result


def run(native,masked,baseline,crossing,last_path,out):
    if out.exists():
        raise FileExistsError(out)
    t0,h = time.monotonic(),Hashes()
    prev = b.r.read(crossing/"result.json")
    proof = b.r.read(crossing/"verification.json")
    last = b.r.read(last_path)
    for f in (Path(__file__),Path(windows.__file__),PLAN,crossing/"verification.json"):
        h(f)
    if (proof["source_result_sha256"]!=h(crossing/"result.json")
            or proof["latest_write_sha256"]!=h(last_path)
            or prev["predictions_sha256"]!=h(crossing/"held_crossings.npz")
            or prev["forecast_support_gate"]):
        raise ValueError("source/clock witness must match the fixed previous comparison")
    tapes,ids,flags = m.load(native,masked,baseline,h)
    reference = tapes[m.VIEWS[0]]
    size = len(reference.owner)
    folds = deterministic_folds(ids)
    owner = np.tile(reference.owner,2)
    late = windows.sample_late(ids,reference.owner,reference.data["clock"])
    delay = np.r_[np.full(size,8),late]
    packed = [[windows.select(tapes[v],d,b.TAU,BOUND) for d in (8,late)] for v in m.VIEWS]
    def panel(field):
        return np.stack([np.concatenate([getattr(q,field) for q in v]) for v in packed])
    mask = panel("allowed")
    with np.load(crossing/"held_crossings.npz",allow_pickle=False) as q:
        if not np.array_equal(mask[:,:size],q["allowed"]) or not np.array_equal(folds,q["folds"]):
            raise ValueError("changed eight-frame guard or held folds")
    mask[...,0] = False
    length = panel("length")
    if not np.array_equal(length,np.broadcast_to(length[0],length.shape)):
        raise ValueError("physical view clocks differ")
    length = length[0]
    reached,tail = panel("reached"),panel("tail").astype(np.float64)
    reward,valid = payoff_panel(tapes,delay)
    if (valid & (np.arange(24)[None,None,:]>=length[None,:,None])).any():
        raise ValueError("padded suffix became a valid scoring sample")
    g = np.stack([common_features(np.tile(tapes[v].data["risk"],(2,1)),reached[i],delay)
                  for i,v in enumerate(m.VIEWS)])
    z = tail[:,:,0]
    actual_delta = tail-z[:,:,None]
    anchor = reached[...,-1].astype(np.float64)
    z_anchor = anchor[:,:,0]
    d_anchor = anchor-z_anchor[:,:,None]
    z_diff = z-z_anchor[:,:,None]
    d_diff = actual_delta-d_anchor[...,None]
    pred,static_heads = fit_panel(g[:,:,None],z_diff[:,:,None],length,mask.any(2)[...,None],owner,folds)
    pred = pred[:,:,0]+z_anchor[:,:,None]
    print("static",int(mask.any(2).sum()),flush=True)
    truth = np.stack([p.effect(c,g,t,b.TAU)[0] for c,g,t in zip(tail,reward,valid,strict=True)])
    with np.load(crossing/"held_crossings.npz",allow_pickle=False) as q:
        if not np.array_equal(np.where(mask[:,:size],truth[:,:size],0),q["hard"]):
            raise ValueError("frame truth differs from exact first-alarm witness")
    regimes = dict(early=np.arange(len(owner))<size,late=np.arange(len(owner))>=size)
    report,arms = {},{}
    for arm in ("hold",*s.ARMS):
        if arm=="hold":
            change = np.broadcast_to(d_anchor[...,None],actual_delta.shape)
        else:
            x = np.stack([action_features(g[i],reached[i],np.tile(tapes[v].data["effect"],(2,1,1)),arm)
                          for i,v in enumerate(m.VIEWS)])
            change,arms[arm] = fit_panel(x,d_diff,length,mask,owner,folds)
            change += d_anchor[...,None]
        combos = dict(estimated=first_credit(pred,change,reward,valid),
                      exact_static=first_credit(z,change,reward,valid),
                      exact_effect=first_credit(pred,actual_delta,reward,valid))
        report[arm] = {k:dict(points=point_metrics(pred,change,z,actual_delta,mask,length,owner,rows),
                             first={c:describe(y,truth,mask,owner,ids,flags,rows) for c,y in combos.items()})
                       for k,rows in regimes.items()}
        print(arm,{key:m.union({v:{"late":x["kept"]["late"]} for v,x in a.items()},"late")
                   for key,a in report[arm]["early"]["first"].items()},flush=True)
    def late_ids(arm):
        return {i for v in report[arm]["early"]["first"]["estimated"].values() for i in v["kept"]["late"]}
    distinct = sorted(late_ids("real")-set.union(*(late_ids(a) for a in ("hold","blind","amplitude","reverse"))))
    cases = case_predictions(tapes,ids,folds,last,static_heads,arms)
    h.check()
    out.mkdir(parents=True,exist_ok=False)
    fit_path = out/"fits.json"
    b.r.write_new(fit_path,dict(static=static_heads,action=arms))
    b.r.write_new(out/"result.json",dict(
        status="FRAME_MARGIN_SEALED_PAIRED_DIAGNOSIS_NOT_POLICY",shots=len(ids),report=report,
        case_landmarks=cases,distinct_true_late=distinct,source_policy_candidate=bool(distinct),
        writable={k:{v:int(mask[i,rows].sum()) for i,v in enumerate(m.VIEWS)} for k,rows in regimes.items()},
        new_val_test_east_reads=0,policy_replays=0,fit_sha256=h(fit_path),
        input_sha256={str(x):r[1] for x,r in h.cache.items()},elapsed_seconds=time.monotonic()-t0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","crossing","last","output"):
        cli.add_argument(key,type=Path)
    a = cli.parse_args()
    run(a.native,a.masked,a.baseline,a.crossing,a.last,a.output)
