"""Independent fold, action truth, first-passage scoring and witness checks."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

import run_patch_utility_pivotal_support_20261106 as p
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_crossing_truth_20261106 import verify_truth
from verify_patch_utility_selection_20261106 import deterministic_folds, menu_analysis


b,m,s = p.b,p.m,p.s
ROOT = b.r.ROOT / "probe_outputs"
HORIZONS = (8,16,24)


def count_weights(ok,owner):
    """Average over actions then blocks, views and distinct shots."""
    q = np.zeros(ok.shape)
    sizes = ok.sum(-1)
    active = sizes>0
    count = np.array([np.bincount(owner[row],minlength=owner.max()+1) for row in active])
    views = (count>0).sum(axis=0)
    people = int(np.count_nonzero(views))
    for v in range(len(q)):
        rows = np.flatnonzero(active[v])
        shot = owner[rows]
        q[v,rows] = ok[v,rows]/(sizes[v,rows]*count[v,shot]*views[shot]*people)[:,None]
    return q


def ap(y,score,w):
    order = np.argsort(-score,kind="stable")
    yy,v,zz = y[order],w[order],score[order]
    end = np.r_[np.flatnonzero(zz[1:]!=zz[:-1]),len(zz)-1]
    tp,all_ = np.cumsum(v*yy)[end],np.cumsum(v)[end]
    return float((np.diff(tp,prepend=0)*tp/all_).sum()/tp[-1])


def check_metrics(score,auc_prob,delta,mask,owner,reported):
    errors = []
    if auc_prob is not None:
        event = delta
        err = np.sum(count_weights(mask,owner)*(auc_prob-event)**2)
        weight = (count_weights(mask&event,owner)+count_weights(mask&~event,owner))/2
        errors.extend([abs(err-reported["brier"]),
                       abs(ap(event[mask],auc_prob[mask],weight[mask])-reported["balanced_ap"])])
        return errors
    ok = mask.copy()
    ok[...,0] = False
    for sign in (-1,1):
        by_shot = {}
        for v in range(len(ok)):
            yes = ok[v]&(delta[v]==sign)
            other = ok[v]&(delta[v]!=sign)
            for j in np.flatnonzero(yes.any(1)&other.any(1)):
                good,bad = score[v,j,yes[j]],score[v,j,other[j]]
                acc = .5+.5*np.sign(sign*(good[:,None]-bad)).mean()
                by_shot.setdefault(owner[j],{}).setdefault(v,[]).append(acc)
        ranks = [np.mean([np.mean(f) for f in a.values()]) for a in by_shot.values()]
        hit = ok&(delta==sign)
        assert reported[str(sign)]["rank_shots"]==len(ranks)
        assert reported[str(sign)]["flip_shots"]==np.unique(owner[hit.any(axis=(0,2))]).size
        assert reported[str(sign)]["pairs"]==int(hit.sum())
        if ranks:
            errors.append(abs(np.mean(ranks)-reported[str(sign)]["rank"]))
        if hit.any():
            strict = count_weights(hit,owner)[score*sign>0].sum()
            errors.append(abs(strict-reported[str(sign)]["strict_sign"]))
    return errors


def audit_witness(tape,view,cases,truth,late):
    by_id = {q.sid:q for q in tape.shots}
    retained = 0
    for x in cases:
        q = by_id[x["shot"]]
        clock = tape.data["clock"][q.blocks]
        i,a,k = x["block"]-q.blocks.start,x["action"],x["offset"]
        original = tape.data["curves"][q.blocks].astype(float).copy()
        exact = original.copy()
        exact[:,:,:8] = exact[:,:1,:8]
        ok = np.zeros(original.shape[:2],bool)
        ok[:,0] = True
        ok[i,a] = True
        raw = menu_analysis(q,clock,exact,ok,b.TAU)
        assert raw["hard"][i,a]==1 and truth[view,x["block"],a]==1
        for j in range(8,32,8):
            gap = original[i,a,j:j+8]-original[i,0,j:j+8]
            exact[i,a,j:j+8] = original[i,0,j:j+8]+sum(gap)/8
        matched = menu_analysis(q,clock,exact,ok,b.TAU)["hard"][i,a]>0
        assert matched==x["alternatives"]["exact_bin_mean"]["kept"]
        retained += int(matched)
        assert k == next(i+1 for i,z in enumerate(original[i,a,8:]) if z>=b.TAU)
    for x in late:
        q = by_id[x["shot"]]
        idx = np.flatnonzero(tape.data["clock"]==x["release"])
        idx = idx[(idx>=q.blocks.start)&(idx<q.blocks.stop)]
        assert len(idx)==1
        c = tape.data["curves"][idx[0]].astype(float)
        a = x["action"]
        other = a+1 if a%2 else a-1
        delays = []
        for d in range(8,x["crossing_delay"]):
            z = c[0,d-1]
            if (max(abs(c[a,d-1]-z),abs(c[other,d-1]-z))<=x["bound"]
                    and not any(val>=b.TAU for val in q.native[:x["release"]+d+1].astype(float))):
                delays.append(d)
        assert delays==x["delays"]
        assert x["crossing_delay"]-max(delays)==x["necessary_frames"]
    return retained


def run(write):
    source = ROOT/"patch_utility_crossing_forecast_20261106_v1"
    resolution = ROOT/"patch_utility_crossing_resolution_20261106_v1.json"
    late = ROOT/"patch_utility_crossing_last_write_20261106_v1.json"
    h = Hashes()
    r = b.r.read(source/"result.json")
    rr,rl = b.r.read(resolution),b.r.read(late)
    for rec in (r,rr,rl):
        for path,sha in rec["input_sha256"].items():
            assert h(path)==sha,path
    assert r["fits_sha256"]==h(source/"fits.json")
    assert r["predictions_sha256"]==h(source/"held_crossings.npz")
    tapes,ids,flags = m.load(ROOT/"patch_utility_bank_20261106_v1",ROOT/"patch_utility_target_views_20261106_v1",
                            ROOT/"patch_utility_selection_20261106_v1/result.json",h)
    first,length,allowed = verify_truth(tapes,ids,b.TAU)
    eligible = allowed & allowed[:,:,1:].any(2)[...,None]
    owner = tapes[m.VIEWS[0]].owner
    folds = deterministic_folds(ids)
    models = b.r.read(source/"fits.json")
    artifact = np.load(source/"held_crossings.npz",allow_pickle=False)
    for k,x in (("first",first),("length",length),("allowed",allowed),("eligible",eligible),
                ("owner",owner),("ids",ids),("folds",folds)):
        assert np.array_equal(artifact[k],x),k
    hard = artifact["hard"]
    reward = np.empty(eligible.shape[:2]+(24,),dtype=np.int8)
    for v,name in enumerate(m.VIEWS):
        t = tapes[name]
        for q in t.shots:
            sl = q.blocks
            clock = t.data["clock"][sl]
            reward[v,sl] = p.payoff(q.native,q.times,q.disruptive,q.down,q.horizon,clock,b.TAU)[0]
            c = t.data["curves"][sl].copy()
            c[:,:,:8] = c[:,:1,:8]
            assert np.array_equal(hard[v,sl],menu_analysis(q,clock,c,allowed[v,sl],b.TAU)["hard"])
        assert audit_witness(t,v,rr["views"][name]["witnesses"],hard,rl["views"][name])==0
    worst,metrics = 0.,0.
    n = 0
    for arm in ("hold",*s.ARMS):
        hz = np.zeros(first.shape+(3,))
        hz[eligible] = artifact[f"hazard__{arm}"]
        if arm=="hold":
            for v,t in enumerate(tapes.values()):
                hit = t.data["curves"][:,:,7].astype(float)>=b.TAU
                assert np.array_equal(hz[v][eligible[v]],np.repeat(hit[...,None],3,axis=-1)[eligible[v]])
        else:
            dr = np.zeros_like(hz)
            dr[eligible] = artifact[f"pair__{arm}"]
            x = np.stack([s.design(s.sel.features(t.data,"risk"),t.data["curves"][:,:,:8],arm,b.TAU)
                          for t in tapes.values()])
            for k,fit in enumerate(models[arm]):
                held = folds[owner]==k
                data = x[:,held]
                ok = eligible[:,held]
                for j,a in enumerate(fit["hazard"]):
                    z = (data[...,a["used"]]-a["center"])/a["scale"]
                    y = expit(np.einsum("...d,d->...",z,a["coef"])+a["intercept"])
                    worst = max(worst,float(np.max(np.abs(y[ok]-hz[...,j][:,held][ok]))))
                    n += int(ok.sum())
                    if fit["paired"][j]["supported"] and np.isfinite(dr[...,j][eligible]).all():
                        g = fit["paired"][j]["model"]
                        z = (data-g["center"])/g["scale"]
                        q = np.einsum("...d,cd->...c",z,g["coefficient"])+g["intercept"]
                        e = np.exp(q-q.max(axis=-1,keepdims=True))
                        pred = (e[...,2]-e[...,0])/e.sum(axis=-1)
                        worst = max(worst,float(np.max(np.abs(pred[ok]-dr[...,j][:,held][ok]))))
                        n += int(ok.sum())
        remain,score = np.ones(first.shape),np.zeros(first.shape)
        for step in range(24):
            gain = hz[...,step//8]*remain
            score += gain*reward[:,:,None,step]
            remain -= gain
            if (step+1) in HORIZONS:
                d = str(step+1)
                col = step//8
                observed = eligible & (length[None,:,None]>=step+1)
                event = (first>0)&(first<=step+1)
                delta = event.astype(np.int8)-event[:,:,:1].astype(np.int8)
                cdf = 1-remain
                ref = r["metrics"][arm][d]
                errors = check_metrics(None,cdf,event,observed,owner,ref)
                errors += check_metrics(cdf-cdf[:,:,:1],None,delta,observed,owner,ref["paired"])
                if arm!="hold" and r["direct_pair"][arm][d]["supported"]:
                    errors += check_metrics(dr[...,col],None,delta,observed,owner,
                                            r["direct_pair"][arm][d]["paired"])
                metrics = max(metrics,*errors)
        for v,name in enumerate(m.VIEWS):
            ok = eligible[v].copy()
            ok[:,0] = False
            rows = np.flatnonzero(ok.any(1))
            take = np.argmax(np.where(ok,score[v],-np.inf),axis=1)
            labels = hard[v,rows,take[rows]]
            f = flags[owner[rows]]
            for key,mask in (("late_top",f&(labels>0)),("safe_help_top",~f&(labels>0)),
                             ("safe_harm_top",~f&(labels<0))):
                assert np.unique(ids[owner[rows[mask]]]).tolist()==r["oracle_need"][arm][name][key],(arm,name,key)
        print("verified",arm,flush=True)
    assert worst<7e-8 and metrics<1e-10,(worst,metrics)
    h.check()
    result = dict(status="INDEPENDENT_CROSSING_FOLDS_TRUTH_METRICS_AND_CASE_CHECK",
                  source_result_sha256=h(source/"result.json"),resolution_sha256=h(resolution),
                  latest_write_sha256=h(late),blocks=len(length),
                  guarded_actions=int(allowed[...,1:].sum()),forecast_values=n,
                  max_prediction_error=worst,max_metric_error=metrics,
                  complete_rescue_action_witnesses=sum(len(x["witnesses"]) for x in rr["views"].values()),
                  latest_write_witnesses=sum(map(len,rl["views"].values())),
                  new_val_test_east_reads=0)
    if write is not None:
        b.r.write_new(write,result)
    print(result,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--write-receipt",type=Path)
    run(cli.parse_args().write_receipt)
