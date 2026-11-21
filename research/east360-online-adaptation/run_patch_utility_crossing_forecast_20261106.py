"""Predict a paired score's first crossing from an eight-frame reached shadow."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score

import run_patch_utility_pivotal_support_20261106 as p
from run_patch_utility_shadow_effect_20261106 import weights
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds


b, m, s = p.b, p.m, p.s
PLAN = b.r.ROOT / "PATCH_UTILITY_CROSSING_FORECAST_PROTOCOL_20261106.md"
HORIZONS = (8,16,24)


def first_offsets(curves,length,tau):
    if (curves.ndim != 3 or curves.shape[-1]!=24 or length.shape!=(len(curves),)
            or (length<0).any() or (length>24).any()):
        raise ValueError("invalid recorded crossing window")
    hit = (curves.astype(np.float64)>=tau)&(np.arange(24)[None,None,:]<length[:,None,None])
    return np.where(hit.any(axis=-1),hit.argmax(axis=-1)+1,0).astype(np.int8)


def exposure(first,length):
    length = np.asarray(length)
    if (first.ndim!=2 or length.shape!=(len(first),) or (length<0).any()
            or (length>24).any() or (first<0).any() or (first>length[:,None]).any()):
        raise ValueError("first crossing outside observed at-risk interval")
    last = np.where(first>0,first,length[:,None])
    start = np.array([0,8,16])
    n = np.clip(last[...,None]-start,0,8)
    e = ((first[...,None]>start)&(first[...,None]<=start+8)).astype(np.int8)
    return n,e


def delta_target(first,h):
    hit = ((first>0)&(first<=h)).astype(np.int8)
    return hit-hit[...,:1]


def first_mass(hazard):
    x = np.asarray(hazard,dtype=np.float64)
    if x.shape[-1]!=3 or not np.isfinite(x).all() or (x<0).any() or (x>1).any():
        raise ValueError("invalid piecewise hazard")
    h = np.repeat(x,8,axis=-1)
    surv = np.cumprod(1-h[...,:-1],axis=-1)
    return np.concatenate((h[...,:1],h[...,1:]*surv),axis=-1)


def fit_hazard(x,n,event,w):
    x,n,event,w = (np.asarray(a,dtype=np.float64) for a in (x,n,event,w))
    active = n>0
    x,n,event,w = x[active],n[active],event[active],w[active]
    total = w@n
    hit = w@event
    if not 0<hit<total or (event>n).any() or (event<0).any():
        raise ValueError("insufficient observed event/no-event exposure")
    count = w*n/total
    yes = w*event/total
    center = count@x
    var = count@((x-center)**2)
    used = np.flatnonzero(var>1e-12)
    scale = np.sqrt(var[used])
    z = np.ascontiguousarray((x[:,used]-center[used])/scale)

    def loss(theta):
        intercept,coef = theta[0],theta[1:]
        logit = intercept+z@coef
        err = count*expit(logit)-yes
        alpha = s.sel.ALPHA
        val = count@np.logaddexp(0,logit)-yes@logit+alpha/2*(coef@coef)
        return val,np.r_[err.sum(),z.T@err+alpha*coef]

    init = np.r_[np.log(hit/(total-hit)),np.zeros(len(used))]
    result = minimize(loss,init,jac=True,method="L-BFGS-B",
                      options=dict(ftol=1e-13,gtol=1e-8,maxiter=300))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"hazard optimizer did not converge: {result.message}")
    return dict(used=used.tolist(),center=center[used].tolist(),scale=scale.tolist(),
                intercept=float(result.x[0]),coef=result.x[1:].tolist(),steps=int(result.nit))


def hazard_predict(head,x):
    used = np.asarray(head["used"],dtype=np.intp)
    return expit(head["intercept"]+((x[...,used]-head["center"])/head["scale"])@head["coef"])


def pair_predict(head,x):
    z = ((x.astype(np.float64)-head["center"])/head["scale"])@np.asarray(head["coefficient"]).T+head["intercept"]
    probs = softmax(z,axis=-1)
    return probs[...,2]-probs[...,0]


def local_rank(score,target,allowed,direction):
    positives = allowed & (target==direction)
    others = allowed & (target!=direction)
    blocks = np.flatnonzero(positives.any(axis=1)&others.any(axis=1))
    acc = []
    for i in blocks:
        gap = direction*(score[i][positives[i],None]-score[i][None,others[i]])
        acc.append(np.mean((gap>0)+.5*(gap==0)))
    return np.asarray(acc),blocks


def read_trials(tapes):
    ref = tapes[m.VIEWS[0]]
    length = np.concatenate([
        np.clip(len(q.native)-1-p.DELAY-ref.data["clock"][q.blocks],0,24) for q in ref.shots
    ])
    first, allow, reached = [], [], {}
    for name,t in tapes.items():
        d = t.data
        reached[name] = d["curves"][:,:,:p.DELAY]
        allowed = s.commit_mask(t.allowed,reached[name],length>0,d["risk"][:,8],b.TAU)
        first.append(first_offsets(d["curves"][:,:,p.DELAY:],length,b.TAU))
        allow.append(allowed)
    return np.stack(first),length,np.stack(allow),reached


def physical_rank(score,target,allowed,owner):
    ans = {}
    for direction in (-1,1):
        per = np.full((len(m.VIEWS),int(owner.max())+1),np.nan)
        for i in range(len(per)):
            a,ix = local_rank(score[i],target[i],allowed[i],direction)
            shots,groups,counts = np.unique(owner[ix],return_inverse=True,return_counts=True)
            per[i,shots] = np.bincount(groups,weights=a)/counts
        support = np.isfinite(per).sum(axis=0)
        flip = allowed & (target==direction)
        avg = np.nansum(per,axis=0)[support>0]/support[support>0]
        sign = float(weights(flip,owner)[direction*score>0].sum()) if flip.any() else None
        ans[str(direction)] = dict(rank=float(avg.mean()) if avg.size else None,rank_shots=int(len(avg)),
                                   flip_shots=int(np.unique(owner[flip.any(axis=(0,2))]).size),
                                   strict_sign=sign,
                                   pairs=int(flip.sum()))
    return ans


def summarize_prediction(pred,first,length,eligible,owner,*,paired=None):
    cdf = None if pred is None else 1-(np.cumprod((1-pred.astype(np.float64))**8,axis=-1))
    result = {}
    for j,hh in enumerate(HORIZONS):
        if paired is not None and not np.isfinite(paired[...,j]).any():
            result[str(hh)] = {"supported":False}
            continue
        keep = eligible & (length[None,:,None]>=hh)
        delta = delta_target(first,hh)
        pair_ok = keep.copy()
        pair_ok[:,:,0] = False
        score = cdf[...,j]-cdf[:,:,:1,j] if paired is None else paired[...,j]
        row = dict(supported=True,paired=physical_rank(score,delta,pair_ok,owner))
        if cdf is not None:
            event = (first>0)&(first<=hh)
            row["brier"] = float((weights(keep,owner)*(cdf[...,j]-event)**2).sum())
            weight = (weights(keep&event,owner)+weights(keep&~event,owner))/2
            row["balanced_ap"] = float(average_precision_score(event[keep],cdf[...,j][keep],
                                                                sample_weight=weight[keep]))
            row["observed_event_shots"] = int(np.unique(owner[(keep&event).any(axis=(0,2))]).size)
        result[str(hh)] = row
    return result


def oracle_data(tapes,eligible,proof):
    rewards,hard,scalar = [], [], []
    for i,(name,tape) in enumerate(tapes.items()):
        d = tape.data
        z = d["curves"].astype(np.float64)
        fixed = z[:,[0],8:]+(z[:,:,7]-z[:,[0],7])[...,None]
        g,a,c = [],[],[]
        for q in tape.shots:
            sl = q.blocks
            reward,valid,*_ = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,d["clock"][sl],b.TAU)
            g.append(reward)
            a.append(p.effect(z[sl,:,8:],reward,valid,b.TAU)[0])
            c.append(p.effect(fixed[sl],reward,valid,b.TAU)[0])
        labels = np.concatenate(a)
        allowed = eligible[i]
        reached_ids = {q.sid for q in tape.shots if q.disruptive and ((labels[q.blocks]>0)&allowed[q.blocks]).any()}
        if reached_ids!={a["shot"] for a in proof["views"][name]["rescue_actions"]}:
            raise ValueError("pivotal labels do not reproduce audited recoverable shots")
        rewards.append(np.concatenate(g))
        hard.append(np.where(allowed,labels,0))
        scalar.append(np.concatenate(c))
    return np.stack(rewards),np.stack(hard),np.stack(scalar)


def oracle_rank(score,hard,allowed,ids,flags,owner):
    result = {}
    for i,v in enumerate(m.VIEWS):
        act,best = s.sparse.top(score[i],allowed[i])
        available = np.isfinite(best)
        credit = hard[i,np.arange(len(act)),act]
        disruptive = flags[owner]
        shots = ids[owner]
        good = allowed[i] & (hard[i]>0)
        witnesses = []
        for j in np.flatnonzero(disruptive & good.any(axis=1)):
            wins = np.flatnonzero(good[j])
            z = np.where(allowed[i,j],score[i,j],-np.inf)
            z[0] = -np.inf
            order = np.argsort(-z,kind="stable")
            rank = int(np.flatnonzero(np.isin(order,wins))[0]+1)
            witnesses.append(dict(shot=int(shots[j]),block=int(j),top=int(act[j]),
                                  helpful=wins.tolist(),rank=rank,score=float(best[j])))
        result[v] = dict(
            late_top=np.unique(shots[available & (credit>0) & disruptive]).tolist(),
            safe_help_top=np.unique(shots[available & (credit>0) & ~disruptive]).tolist(),
            safe_harm_top=np.unique(shots[available & (credit<0) & ~disruptive]).tolist(),
            late_blocks=witnesses,
        )
    return result


def fit_arm(x,owner,folds,eligible,first,length):
    hazard = np.empty(eligible.shape+(3,),np.float32)
    pair = np.full(hazard.shape,np.nan,np.float32)
    observed = [exposure(f,length) for f in first]
    n,e = (np.stack(z) for z in zip(*observed,strict=True))
    d = [delta_target(first,h) for h in HORIZONS]
    log = []
    for k in range(s.sel.N_FOLDS):
        tr,_,held = m.rows_for_fold(folds,owner,k)
        keep = eligible[:,tr]
        w = weights(keep,owner[tr])[keep]
        x_train = x[:,tr]
        use = x_train[keep]
        xi = x[:,held]
        heads,dirs = [],[]
        for j,h in enumerate(HORIZONS):
            model = fit_hazard(use,n[...,j][:,tr][keep],e[...,j][:,tr][keep],w)
            hazard[...,j][:,held] = hazard_predict(model,xi)
            heads.append(model)
            pair_ok = keep & (length[None,tr,None]>=h)
            pair_ok[:,:,0] = False
            yy = d[j][:,tr]
            support = {str(q):int(np.unique(owner[tr][(pair_ok&(yy==q)).any(axis=(0,2))]).size) for q in (-1,0,1)}
            if min(support.values())==0:
                dirs.append(dict(supported=False,support=support))
                continue
            views = np.repeat(np.arange(len(m.VIEWS),dtype=np.int8),int(tr.sum()))
            blocks = np.tile(owner[tr],len(m.VIEWS))
            model = m.fit(x_train.reshape(-1,b.NA,x.shape[-1]),yy.reshape(-1,b.NA),
                          pair_ok.reshape(-1,b.NA),blocks,views)
            pair[...,j][:,held] = pair_predict(model,xi)
            dirs.append(dict(supported=True,support=support,model=model))
        log.append(dict(hazard=heads,paired=dirs))
    for j in range(3):
        if not np.isfinite(pair[...,j]).all():
            pair[...,j] = np.nan
    return hazard,pair,log


def diagnostic_gate(metrics,oracle):
    controls = [k for k in ("hold",*s.ARMS) if k!="real"]
    def avg(name,sign):
        values = [metrics[name][str(h)]["paired"][sign]["rank"] for h in HORIZONS]
        return None if None in values else float(np.mean(values))
    means = {a:{sg:avg(a,sg) for sg in ("-1","1")} for a in ("hold",*s.ARMS)}
    rank_ok = all(means["real"][sg] is not None and means[a][sg] is not None
                  and means["real"][sg]>=means[a][sg] for sg in ("-1","1") for a in controls)
    def wins(a):
        return set(m.union(oracle[a],"late_top"))
    unique = sorted(wins("real")-set.union(*(wins(a) for a in controls)))
    return bool(rank_ok and unique),unique,means


def run(native, masked, baseline, guard_path, support_path, out):
    if out.exists():
        raise FileExistsError(out)
    start,h = time.monotonic(),Hashes()
    for path in (Path(__file__),Path(s.__file__),Path(p.__file__),PLAN,guard_path,support_path):
        h(path)
    guard,proof = b.r.read(guard_path),b.r.read(support_path)
    if (not guard["commit_guard_gate"] or guard["commit_plan_sha256"]!=h(s.PLAN)
            or proof["status"]!="EXACT_OFFLINE_FIRST_ALARM_FACTOR_AND_SOURCE_SUPPORT"
            or proof["input_sha256"].get(str(guard_path.resolve()))!=h(guard_path)):
        raise ValueError("requires the exact held write and payoff contract")
    tapes,ids,flags = m.load(native,masked,baseline,h)
    owner = tapes[m.VIEWS[0]].owner
    folds = deterministic_folds(ids)
    first,length,allowed,reached = read_trials(tapes)
    for i,v in enumerate(m.VIEWS):
        if int(allowed[i,:,1:].sum())!=guard["views"][v]["action_counts"]["protected"]:
            raise ValueError("first crossing sample has a different write boundary")
    eligible = allowed & allowed[:,:,1:].any(axis=2)[...,None]
    reward,hard,scalar = oracle_data(tapes,allowed,proof)
    oracle = {k:oracle_rank(z,hard,allowed,ids,flags,owner) for k,z in (("true_curve",hard),("scalar_curve",scalar))}
    metrics,direct,models,archive = {},{},{},{}
    base = {v:s.sel.features(t.data,"risk") for v,t in tapes.items()}
    for arm in ("hold",*s.ARMS):
        if arm=="hold":
            pos = np.stack([reached[v][:,:,-1].astype(np.float64)>=b.TAU for v in m.VIEWS])
            hazard = np.repeat(pos[...,None],3,axis=-1).astype(np.float32)
        else:
            x = np.stack([s.design(base[v],reached[v],arm,b.TAU) for v in m.VIEWS])
            hazard,pair,models[arm] = fit_arm(x,owner,folds,eligible,first,length)
            direct[arm] = summarize_prediction(None,first,length,eligible,owner,paired=pair)
            archive[f"pair__{arm}"] = pair[eligible]
        archive[f"hazard__{arm}"] = hazard[eligible]
        metrics[arm] = summarize_prediction(hazard,first,length,eligible,owner)
        valued = np.einsum("vbat,vbt->vba",first_mass(hazard),reward,optimize=True)
        oracle[arm] = oracle_rank(valued-valued[:,:,:1],hard,allowed,ids,flags,owner)
        print(arm,{d:metrics[arm]["24"]["paired"][d]["rank"] for d in ("-1","1")},
              {v:oracle[arm][v]["late_top"] for v in m.VIEWS},flush=True)
    gate,unique,avg = diagnostic_gate(metrics,oracle)
    h.check()
    out.mkdir(parents=True,exist_ok=False)
    fitted,pred = out/"fits.json",out/"held_crossings.npz"
    b.r.write_new(fitted,models)
    b.r.save_arrays(pred,ids=ids,owner=owner,folds=folds,length=length,first=first,
                    allowed=allowed,eligible=eligible,hard=hard,**archive)
    b.r.write_new(out/"result.json",dict(
        status="FIRST_CROSSING_FORECAST_WITH_ORACLE_STATE_DIAGNOSIS_NOT_POLICY",metrics=metrics,
        direct_pair=direct,oracle_need=oracle,forecast_support_gate=gate,unique_oracle_late=unique,
        crossing_rank_average=avg,first_alarm_policy_replays=0,
        shots=len(ids),source_outcome_used_in_fit=False,new_val_test_east_reads=0,
        writable_complete_blocks=[int((x.any(1)&(length==24)).sum()) for x in eligible],
        writable_partial_blocks=[int((x.any(1)&(length<24)).sum()) for x in eligible],
        input_sha256={str(p): v[1] for p,v in h.cache.items()},fits_sha256=h(fitted),
        predictions_sha256=h(pred),elapsed_seconds=time.monotonic()-start,
    ))
    print("forecast_support_gate",gate,unique,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","guard","support","output"):
        cli.add_argument(key,type=Path)
    args = cli.parse_args()
    run(args.native,args.masked,args.baseline,args.guard,args.support,args.output)
